#pragma once

#include "AudioProcessingStage.h"
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
 *   Feedback equation (zero-delay, nonlinear, textbook Moog):
 *     u1 = x - tanh(R*yN)
 *   Combined: F(yN) = yN - G^N*(x - tanh(R*yN)) - S = 0
 *
 *   Solved by Newton iteration with the linear ZDF result as initial guess:
 *     yN_0 = (G^N*x + S) / (1 + R*G^N)
 *     yN_{n+1} = yN_n - F(yN_n) / F'(yN_n)
 *     F'(y) = 1 + G^N * R * (1 - tanh^2(R*y))   (always >= 1; unconditionally stable)
 *
 *   Note: small-signal loop gain at DC is R. Self-oscillation onset at R=4
 *   (textbook 4-pole Moog). kMaxResonance is capped at 3.9 to keep the knob
 *   top safely below the threshold (resonance peak is huge, but ringing decays).
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

            double GN = G * G;
            if constexpr (N == 4) {
                GN = GN * GN; // G^4
            }
            // (for N == 2, GN is already G^2)

            // Bass compensation: pre-amp input by (1+R) so the cascade's
            // small-signal DC gain — 1/(1+R) without compensation — lands at
            // unity. Coefficient derived from the linearized feedback equation
            // y_N = x_eff - tanh(R*y_N): for tanh ≈ identity, y_N = x_eff/(1+R);
            // setting x_eff = (1+R)*x yields y_N = x.
            //
            // History: commit ae8ba9a tried (1 + 2R) — twice the correct value —
            // and was reverted (48854a3, "sounded worse"). Tests in this file
            // (LadderTPTStage.BassCompensation_*) pin the coefficient down.
            //
            // Saturation note: at hot input levels (R*x > 1) the tanh feedback
            // saturates and compensation slightly over-shoots, adding a touch
            // of tanh-driven harmonic warmth at large amplitudes — musically
            // appropriate inside a distortion plugin.
            const double bassMakeup = 1.0 + R;

            for (int ch = 0; ch < numChannels; ++ch) {
                auto& st = channels_[static_cast<std::size_t>(ch)];
                const double x = buffer.getWritePointer(ch)[i] * bassMakeup;

                double S;
                if constexpr (N == 2) {
                    S = alpha * (G * st.s[0] + st.s[1]);
                } else { // N == 4
                    const double G2 = G * G;
                    const double G3 = G2 * G;
                    S = alpha * (G3 * st.s[0] + G2 * st.s[1] + G * st.s[2] + st.s[3]);
                }

                // Linear-feedback ZDF as initial guess; exact when tanh ≈ identity.
                double yN = (GN * x + S) / (1.0 + R * GN);

                // Newton iteration on F(y) = y - G^N*(x - tanh(R*y)) - S.
                // Feedback is tanh(R*y) (not tanh(2R*y) — that would double the
                // small-signal loop gain and move self-osc from R=4 to R=2,
                // which is mid-knob and audibly broken). F'(y) = 1 + G^N*R*(1-tanh²);
                // F' >= 1 so Newton is unconditionally monotonic and the step
                // magnitude bounds the residual.
                for (int iter = 0; iter < kNewtonMaxIter; ++iter) {
                    const double fb = std::tanh(R * yN);
                    const double Fy = yN - GN * (x - fb) - S;
                    const double Fp = 1.0 + GN * R * (1.0 - fb * fb);
                    const double dy = Fy / Fp;
                    yN -= dy;
                    if (std::abs(dy) < kNewtonTol) {
                        break;
                    }
                }

                // Forward pass: states resolved, walk the cascade and commit.
                double u = x - std::tanh(R * yN);
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

    /** Public for tests so they can compute the half-ramp window without hardcoding. */
    static constexpr double kSmoothingTimeSec = 0.002;

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
    // Self-oscillation threshold (under the corrected tanh(R*y) feedback math)
    // is R=4 — the textbook 4-pole Moog loop gain. Cap below that so the knob
    // top can't sustain oscillation; user does not want self-osc.
    static constexpr double kMaxResonance = 3.9;

    // Newton step magnitude below this triggers early exit. F' >= 1 (proven
    // earlier) means the step size is an upper bound on the residual; 1e-9 is
    // ~30 dB below the smallest 24-bit audio quantum, so any residual smaller
    // than this is inaudible. At low R the linear ZDF guess is near-exact and
    // we exit after 1-2 iterations.
    static constexpr double kNewtonTol = 1e-9;
    static constexpr int kNewtonMaxIter = 4;

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

    // Cutoff uses Multiplicative (exponential / log-in-Hz) — pitch perception
    // is logarithmic, so equal-time steps cover equal musical distance. Resonance
    // stays Linear: no analogous log scale, and Multiplicative can't cross zero.
    juce::SmoothedValue<double, juce::ValueSmoothingTypes::Multiplicative> smoothCutoff_;
    juce::SmoothedValue<double, juce::ValueSmoothingTypes::Linear> smoothResonance_;

    std::vector<ChannelState> channels_;
};

using LadderTPT12dBStage = LadderTPTStage<2>;
using LadderTPT24dBStage = LadderTPTStage<4>;

} // namespace dsp_core::audio_pipeline
