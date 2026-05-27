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
 * N-stage Moog-style Ladder highpass — proper TPT integrators with closed-form
 * ZDF feedback solve via Newton iteration on the tanh nonlinearity. Sibling of
 * LadderTPTStage; shares the same Newton + smoothing scaffolding, swaps the
 * per-stage output from LP (yLP = v + s) to HP (yHP = u - yLP = alpha*(u - s)).
 *
 * Math (Zavalishin, ch. 5; HP cascade derivation):
 *   Each 1-pole TPT stage produces both LP and HP outputs simultaneously:
 *     v   = G*(u - s)
 *     yLP = v + s            = G*u + alpha*s
 *     yHP = u - yLP          = alpha*(u - s)
 *     s_new = yLP + v        (same TPT integrator update as the LP cascade)
 *   where G = g/(1+g), alpha = 1/(1+g) = 1 - G, g = tan(pi*fc/fs).
 *
 *   Cascading HP outputs y_{k+1} = alpha*(y_k - s_{k+1}) unrolls linearly
 *   between u1 and yN_HP:
 *     yN = alpha^N * u1 - S_HP
 *   For N=4: S_HP = alpha*(alpha^3*s_1 + alpha^2*s_2 + alpha*s_3 + s_4)
 *   For N=2: S_HP = alpha*(alpha*s_1 + s_2)
 *
 *   Feedback equation (zero-delay, nonlinear; same Moog structure as the LPF):
 *     u1 = x - tanh(R*yN)
 *   Combined: F(yN) = yN - alpha^N*(x - tanh(R*yN)) + S_HP = 0
 *
 *   Solved by Newton with the linear ZDF result as initial guess:
 *     yN_0 = (alpha^N*x - S_HP) / (1 + R*alpha^N)
 *     yN_{n+1} = yN_n - F(yN_n) / F'(yN_n)
 *     F'(y) = 1 + alpha^N * R * (1 - tanh^2(R*y))   (always >= 1; unconditionally stable)
 *
 *   Note on self-osc: the 4-pole HP cascade contributes +180 deg phase at the
 *   cutoff (each pole +45 deg), so the negative feedback u = x - tanh(R*y)
 *   closes the loop with the same +/-360 deg rotation as the LP case at cutoff.
 *   Self-oscillation onset is therefore at R=4 (textbook Moog), same as the
 *   LPF. kMaxResonance caps at 3.9 to keep the knob top safely below threshold.
 *
 * Compile-time stage count via int template parameter:
 *   - N=2 -> 12 dB/oct ("LadderHPTPT-12dB"), softer, gentler bass-removal
 *   - N=4 -> 24 dB/oct ("LadderHPTPT-24dB"), classic Moog HP
 *
 * Thread Safety: same atomic + smoothed-value pattern as LadderTPTStage.
 */
template <int N>
class LadderHighpassTPTStage : public AudioProcessingStage {
    static_assert(N == 2 || N == 4, "LadderHighpassTPTStage only supports N=2 (12dB) or N=4 (24dB)");

  public:
    LadderHighpassTPTStage() = default;

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
        // Flush denormals — TPT integrator states decay asymptotically; without
        // FTZ they enter the denormal range and cost 10-100x normal arithmetic.
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

            double alphaN = alpha * alpha;
            if constexpr (N == 4) {
                alphaN = alphaN * alphaN; // alpha^4
            }
            // (for N == 2, alphaN is already alpha^2)

            // Passband (HF) makeup: pre-amp input by (1+R) so the cascade's
            // small-signal passband gain — 1/(1+R) without compensation — lands
            // at unity. Same derivation as the LPF bass makeup: linearised
            // feedback equation y_N = x_eff - tanh(R*y_N), tanh ≈ identity gives
            // y_N = x_eff/(1+R); setting x_eff = (1+R)*x yields y_N = x in the
            // passband (HF for HPF, LF for LPF).
            //
            // Saturation behaviour at hot input mirrors the LPF: when R*x > 1
            // the tanh feedback saturates and compensation slightly over-shoots,
            // adding a touch of tanh-driven harmonic warmth — musically right
            // inside a distortion plugin.
            const double passbandMakeup = 1.0 + R;

            for (int ch = 0; ch < numChannels; ++ch) {
                auto& st = channels_[static_cast<std::size_t>(ch)];
                const double x = buffer.getWritePointer(ch)[i] * passbandMakeup;

                // HPF cascade state sum: each stage's HP output is
                // alpha*(u_k - s_k), so unrolling y_{k+1} = alpha*(y_k - s_{k+1})
                // collects state contributions with descending alpha powers.
                double S_HP;
                if constexpr (N == 2) {
                    S_HP = alpha * (alpha * st.s[0] + st.s[1]);
                } else { // N == 4
                    const double a2 = alpha * alpha;
                    const double a3 = a2 * alpha;
                    S_HP = alpha * (a3 * st.s[0] + a2 * st.s[1] + alpha * st.s[2] + st.s[3]);
                }

                // Linear-feedback ZDF as initial guess; exact when tanh ≈ identity.
                double yN = (alphaN * x - S_HP) / (1.0 + R * alphaN);

                // Newton iteration on F(y) = y - alpha^N*(x - tanh(R*y)) + S_HP.
                // Feedback is tanh(R*y) (not tanh(2R*y) — that would double the
                // small-signal loop gain and halve the self-osc threshold).
                // F'(y) = 1 + alpha^N*R*(1 - tanh^2) >= 1, so Newton is
                // unconditionally monotonic and the step magnitude bounds the
                // residual.
                for (int iter = 0; iter < kNewtonMaxIter; ++iter) {
                    const double fb = std::tanh(R * yN);
                    const double Fy = yN - alphaN * (x - fb) + S_HP;
                    const double Fp = 1.0 + alphaN * R * (1.0 - fb * fb);
                    const double dy = Fy / Fp;
                    yN -= dy;
                    if (std::abs(dy) < kNewtonTol) {
                        break;
                    }
                }

                // Forward pass: states resolved, walk the cascade taking HP
                // outputs and committing TPT integrator state updates.
                double u = x - std::tanh(R * yN);
                double yHP = 0.0;
                for (int n = 0; n < N; ++n) {
                    const auto k = static_cast<std::size_t>(n);
                    const double v = G * (u - st.s[k]);
                    const double yLP = v + st.s[k];
                    yHP = u - yLP; // = alpha * (u - st.s[k])
                    st.s[k] = yLP + v;
                    u = yHP;
                }

                buffer.getWritePointer(ch)[i] = yHP;
            }
        }
    }

    void reset() override {
        for (auto& c : channels_) {
            c = ChannelState{};
        }
    }

    juce::String getName() const override {
        return N == 2 ? "LadderHPTPT-12dB" : "LadderHPTPT-24dB";
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

    double getCurrentSmoothedCutoff() const {
        return smoothCutoff_.getCurrentValue();
    }

    static constexpr double kSmoothingTimeSec = 0.002;

    /** Resonance, clamped to [0, 3.9]. Self-oscillation onset ≈ 4 (same as LP cascade). */
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
    // Self-oscillation threshold under the tanh(R*y) feedback math is R=4 —
    // matches the LPF cascade by the symmetric phase-shift argument (4-pole
    // HP at cutoff is +180°, paired with the -1 feedback sign closes the loop
    // with the same total rotation as 4-pole LP at cutoff). Cap below so the
    // knob top can't sustain oscillation.
    static constexpr double kMaxResonance = 3.9;

    // Same Newton convergence parameters as the LP cascade: F' >= 1 makes step
    // magnitude an upper bound on residual; 1e-9 is ~30 dB below the smallest
    // 24-bit audio quantum.
    static constexpr double kNewtonTol = 1e-9;
    static constexpr int kNewtonMaxIter = 4;

    struct ChannelState {
        std::array<double, N> s{};
    };

    std::atomic<bool> enabled_{true};
    std::atomic<double> cutoffHz_{200.0};
    std::atomic<double> resonance_{0.0};

    double sampleRate_ = 48000.0;
    double piOverFs_ = juce::MathConstants<double>::pi / 48000.0;
    double lastCutoffTarget_ = -1.0;
    double lastResonanceTarget_ = -1.0;

    juce::SmoothedValue<double, juce::ValueSmoothingTypes::Multiplicative> smoothCutoff_;
    juce::SmoothedValue<double, juce::ValueSmoothingTypes::Linear> smoothResonance_;

    std::vector<ChannelState> channels_;
};

using LadderHPTPT12dBStage = LadderHighpassTPTStage<2>;
using LadderHPTPT24dBStage = LadderHighpassTPTStage<4>;

} // namespace dsp_core::audio_pipeline
