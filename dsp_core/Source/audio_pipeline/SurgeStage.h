#pragma once

#include "AudioProcessingStage.h"
#include <juce_audio_basics/juce_audio_basics.h>
#include <algorithm>
#include <array>
#include <atomic>

namespace dsp_core::audio_pipeline {

/**
 * Input-side soft-dynamic surge pre-processor.
 *
 * Transforms each sample x via
 *     effective_x = (1 − w) · x + w · clamp(x, −1, 1)
 * where w ∈ [0,1] is a per-channel, per-rail weight that ramps up on sustained
 * overshoot (|x| > 1) and decays during in-range samples. Full saturation
 * (w → 1) takes `surgeDurationSec` of wall-clock time.
 *
 * When paired with a downstream LUT in LinearExtrap mode, this reproduces the
 * same surge response as the old in-LUT ExtrapolationMode::Surge for any
 * memoryless stack. Unlike the in-LUT approach, it also produces a time-
 * varying H for a downstream Jiles-Atherton hysteresis stage, which is the
 * whole reason for making surge a standalone upstream stage.
 *
 * Pipeline position: immediately before the LUT/Hysteresis consumer.
 *
 * Thread Safety:
 *   - enabled_: atomic (UI writes, audio reads)
 *   - setSurgeDurationSec: UI thread only, between process() calls
 *   - prepareToPlay: UI thread only
 *   - process: audio thread only
 */
class SurgeStage : public AudioProcessingStage {
  public:
    SurgeStage() = default;

    void prepareToPlay(double sampleRate, int samplesPerBlock, int numChannels) override;
    void process(juce::AudioBuffer<double>& buffer) override;
    void reset() override;
    juce::String getName() const override {
        return "Surge";
    }

    void setEnabled(bool shouldBeEnabled) {
        enabled_.store(shouldBeEnabled, std::memory_order_release);
    }

    bool isEnabled() const {
        return enabled_.load(std::memory_order_acquire);
    }

    // Wall-clock time from first overshoot to full saturation. Re-derives
    // stepPerSample from the currently-prepared sample rate.
    void setSurgeDurationSec(double seconds);

    double getSurgeDurationSec() const {
        return surgeDurationSec_;
    }

    // Kumaraswamy CDF (a=2, b=8): front-loaded asymmetric S-curve.
    // Lifted verbatim from the old in-LUT implementation.
    static double shapedWeight(double phase) noexcept {
        const double v = 1.0 - phase * phase;
        const double v2 = v * v;
        const double v4 = v2 * v2;
        const double v8 = v4 * v4;
        return 1.0 - v8;
    }

  private:
    struct ChannelState {
        double wPos{0.0};
        double wNeg{0.0};
    };

    void recomputeStep();

    std::array<ChannelState, 2> state_{};
    std::atomic<bool> enabled_{false};
    double sampleRate_{48000.0};
    double surgeDurationSec_{0.001};
    double stepPerSample_{0.0};
};

} // namespace dsp_core::audio_pipeline
