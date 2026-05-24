#pragma once

#include "AudioProcessingStage.h"
#include "Tanh2xLUT.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_core/juce_core.h>
#include <juce_dsp/juce_dsp.h>
#include <vector>

namespace dsp_core::audio_pipeline {

/**
 * N-stage Moog-style Ladder lowpass — proper TPT integrators with closed-form
 * ZDF feedback solve via Newton iteration on the tanh nonlinearity.
 *
 * Math (Zavalishin, ch. 5):
 *   Each 1-pole stage uses TPT: v = G*(u - s), y = v + s, s_new = y + v,
 *   where G = g/(1+g) and g = tan(pi*fc/fs).
 *
 *   The N stages compose linearly between the input u1 and output yN:
 *     yN = G^N * u1 + S
 *   For N=4: S = alpha * (G^3*s_1 + G^2*s_2 + G*s_3 + s_4)
 *   For N=2: S = alpha * (G*s_1 + s_2)
 *   where alpha = 1/(1+g) = 1-G.
 *
 *   Feedback equation (zero-delay, nonlinear):
 *     u1 = x - tanh(2*R*yN)
 *   Combined: F(yN) = yN - G^N*(x - tanh(2*R*yN)) - S = 0
 *
 *   Solved by Newton iteration with the linear ZDF result as initial guess:
 *     yN_0 = (G^N*x + S) / (1 + 2*R*G^N)
 *     yN_{n+1} = yN_n - F(yN_n) / F'(yN_n)
 *     F'(y) = 1 + G^N * 2*R * (1 - tanh^2(2*R*y))   (always >= 1; unconditionally stable)
 *
 * Compile-time stage count via int template parameter:
 *   - N=2 → 12 dB/oct ("LadderTPT-12dB"), softer / Roland-flavored
 *   - N=4 → 24 dB/oct ("LadderTPT-24dB"), classic Moog
 *
 * Header-only because each instantiation needs its own cascade unroll;
 * the body is short enough that putting it inline costs nothing.
 *
 * Thread Safety: same atomic + smoothed-value pattern as LadderStage.
 */
template <int N>
class LadderTPTStage : public AudioProcessingStage {
    static_assert(N == 2 || N == 4, "LadderTPTStage only supports N=2 (12dB) or N=4 (24dB)");

  public:
    LadderTPTStage() = default;

    void prepareToPlay(double sampleRate, int /*samplesPerBlock*/, int numChannels) override {
        sampleRate_ = sampleRate;
        piOverFs_ = juce::MathConstants<double>::pi / sampleRate;

        const int ch = std::max(1, numChannels);
        channels_.assign(static_cast<std::size_t>(ch), ChannelState{});

        const double maxCutoff = std::max(kMinCutoffHz, sampleRate_ * kNyquistMargin);
        const double clampedCutoff =
            juce::jlimit(kMinCutoffHz, maxCutoff, cutoffHz_.load(std::memory_order_acquire));
        cutoffHz_.store(clampedCutoff, std::memory_order_release);

        smoothCutoff_.reset(sampleRate, kSmoothingTimeSec);
        smoothResonance_.reset(sampleRate, kSmoothingTimeSec);
        smoothCutoff_.setCurrentAndTargetValue(clampedCutoff);
        smoothResonance_.setCurrentAndTargetValue(resonance_.load(std::memory_order_acquire));
        lastCutoffTarget_ = clampedCutoff;
        lastResonanceTarget_ = resonance_.load(std::memory_order_acquire);
    }

    void process(juce::AudioBuffer<double>& buffer) override {
        // Flush denormals to zero for the per-sample loop. TPT integrator
        // states decay asymptotically; without FTZ they enter the denormal
        // range and cost 10-100x normal arithmetic on x86. JUCE plugin hosts
        // usually set FTZ in processBlock, but standalone uses (tests, future
        // headless paths) don't, so be defensive.
        const juce::ScopedNoDenormals noDenormals;

        if (!enabled_.load(std::memory_order_acquire)) {
            return;
        }

        const double cutTarget = cutoffHz_.load(std::memory_order_acquire);
        const double resTarget = resonance_.load(std::memory_order_acquire);
        if (cutTarget != lastCutoffTarget_) {
            smoothCutoff_.setTargetValue(cutTarget);
            lastCutoffTarget_ = cutTarget;
        }
        if (resTarget != lastResonanceTarget_) {
            smoothResonance_.setTargetValue(resTarget);
            lastResonanceTarget_ = resTarget;
        }

        const int numChannels =
            std::min(buffer.getNumChannels(), static_cast<int>(channels_.size()));
        const int numSamples = buffer.getNumSamples();

        for (int i = 0; i < numSamples; ++i) {
            const double cutoff = smoothCutoff_.getNextValue();
            const double R = smoothResonance_.getNextValue();

            const double g = std::tan(piOverFs_ * cutoff);
            const double alpha = 1.0 / (1.0 + g); // = 1 - G
            const double G = g * alpha;
            const double twoR = 2.0 * R;

            double GN = G * G;
            if constexpr (N == 4) {
                GN = GN * GN; // G^4
            }
            // (for N == 2, GN is already G^2)

            for (int ch = 0; ch < numChannels; ++ch) {
                auto& st = channels_[static_cast<std::size_t>(ch)];
                const double x = buffer.getWritePointer(ch)[i];

                double S;
                if constexpr (N == 2) {
                    S = alpha * (G * st.s[0] + st.s[1]);
                } else { // N == 4
                    const double G2 = G * G;
                    const double G3 = G2 * G;
                    S = alpha * (G3 * st.s[0] + G2 * st.s[1] + G * st.s[2] + st.s[3]);
                }

                // Linear-feedback ZDF as initial guess; exact when tanh ≈ identity.
                double yN = (GN * x + S) / (1.0 + twoR * GN);

                // Newton iteration on F(y) = y - G^N*(x - tanh(2Ry)) - S.
                // 3 iterations is conservative; F' >= 1 so Newton is monotonic.
                // g_tanh2xLUT.lookup(R*y) = tanh(2*R*y); the chain-rule 2R in
                // F' is d(2Ry)/dy, not a duplicate of the LUT's internal *2.
                for (int iter = 0; iter < 3; ++iter) {
                    const double fb = g_tanh2xLUT.lookup(R * yN);
                    const double Fy = yN - GN * (x - fb) - S;
                    const double Fp = 1.0 + GN * twoR * (1.0 - fb * fb);
                    yN -= Fy / Fp;
                }

                // Forward pass: states resolved, walk the cascade and commit.
                double u = x - g_tanh2xLUT.lookup(R * yN);
                for (int n = 0; n < N; ++n) {
                    const auto k = static_cast<std::size_t>(n);
                    const double v = G * (u - st.s[k]);
                    const double y = v + st.s[k];
                    st.s[k] = y + v;
                    u = y;
                }

                buffer.getWritePointer(ch)[i] = u;
            }
        }
    }

    void reset() override {
        for (auto& c : channels_) {
            c = ChannelState{};
        }
    }

    juce::String getName() const override {
        return N == 2 ? "LadderTPT-12dB" : "LadderTPT-24dB";
    }

    void setEnabled(bool shouldBeEnabled) {
        enabled_.store(shouldBeEnabled, std::memory_order_release);
    }
    bool isEnabled() const {
        return enabled_.load(std::memory_order_acquire);
    }

    void setCutoffFrequency(double frequencyHz) {
        const double maxCutoff = std::max(kMinCutoffHz, sampleRate_ * kNyquistMargin);
        frequencyHz = juce::jlimit(kMinCutoffHz, maxCutoff, frequencyHz);
        cutoffHz_.store(frequencyHz, std::memory_order_release);
    }
    double getCutoffFrequency() const {
        return cutoffHz_.load(std::memory_order_acquire);
    }

    /**
     * Current smoothed cutoff value (live audio-thread state).
     * Intended for tests / metering; not RT-safe for cross-thread reads without
     * external synchronization, but reading from the audio thread or between
     * blocks is fine.
     */
    double getCurrentSmoothedCutoff() const {
        return smoothCutoff_.getCurrentValue();
    }

    /** Resonance, clamped to [0, 4]. Self-oscillation onset ≈ 3.95. */
    void setResonance(double r) {
        r = juce::jlimit(kMinResonance, kMaxResonance, r);
        resonance_.store(r, std::memory_order_release);
    }
    double getResonance() const {
        return resonance_.load(std::memory_order_acquire);
    }

  private:
    static constexpr double kMinCutoffHz = 20.0;
    static constexpr double kNyquistMargin = 0.45;
    static constexpr double kMinResonance = 0.0;
    static constexpr double kMaxResonance = 4.0;
    static constexpr double kSmoothingTimeSec = 0.002;

    struct ChannelState {
        std::array<double, N> s{};
    };

    std::atomic<bool> enabled_{true};
    std::atomic<double> cutoffHz_{20000.0};
    std::atomic<double> resonance_{0.0};

    double sampleRate_ = 48000.0;
    double piOverFs_ = juce::MathConstants<double>::pi / 48000.0;
    double lastCutoffTarget_ = -1.0;
    double lastResonanceTarget_ = -1.0;

    juce::SmoothedValue<double, juce::ValueSmoothingTypes::Linear> smoothCutoff_;
    juce::SmoothedValue<double, juce::ValueSmoothingTypes::Linear> smoothResonance_;

    std::vector<ChannelState> channels_;
};

using LadderTPT12dBStage = LadderTPTStage<2>;
using LadderTPT24dBStage = LadderTPTStage<4>;

} // namespace dsp_core::audio_pipeline
