#pragma once

#include "AudioProcessingStage.h"
#include <juce_audio_basics/juce_audio_basics.h>
#include <array>
#include <atomic>

namespace dsp_core::audio_pipeline {

/**
 * Passthrough envelope follower — measures input level and publishes it
 * to an atomic owned by the processor for morph modulation.
 *
 * Algorithm per sample (single-pole peak detector):
 *     x = max(|L|, |R|) * inputGainLin * sensitivityLin
 *     target = min(x, 1.0)
 *     coef = (target > env) ? attackCoef : releaseCoef
 *     env += coef * (target - env)
 *
 * The published envelope value is in [0, 1]. Audio is never modified — this
 * is purely a measurement tap.
 *
 * Thread safety:
 *   - enabled_, inputGainLinear_, sensitivityLinear_: atomics (UI writes, audio reads)
 *   - setAttackReleaseSec, prepareToPlay: UI thread only, between process() calls
 *   - process: audio thread only
 */
class EnvelopeFollowerStage : public AudioProcessingStage {
  public:
    explicit EnvelopeFollowerStage(std::atomic<double>& envelopeStorage);

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void process(juce::AudioBuffer<double>& buffer) override;
    void reset() override;
    juce::String getName() const override {
        return "EnvelopeFollower";
    }

    void setEnabled(bool shouldBeEnabled) {
        enabled_.store(shouldBeEnabled, std::memory_order_release);
    }
    bool isEnabled() const {
        return enabled_.load(std::memory_order_acquire);
    }

    // Pre-detection gain applied to |sample| before the peak detector.
    // UI thread writes; audio thread reads at block boundaries.
    void setInputGainLinear(double linear) {
        inputGainLinear_.store(linear, std::memory_order_release);
    }
    void setSensitivityLinear(double linear) {
        sensitivityLinear_.store(linear, std::memory_order_release);
    }

    // Wall-clock attack/release times. Recomputes one-pole coefficients
    // against the currently-prepared sample rate.
    void setAttackReleaseSec(double attackSec, double releaseSec);

    double getAttackSec() const {
        return attackSec_;
    }
    double getReleaseSec() const {
        return releaseSec_;
    }

  private:
    void recomputeCoefficients();

    std::atomic<double>& envelopeStorage_;

    std::atomic<bool> enabled_{false};
    std::atomic<double> inputGainLinear_{1.0};
    std::atomic<double> sensitivityLinear_{1.0};

    double sampleRate_{48000.0};
    double attackSec_{0.005};
    double releaseSec_{0.150};
    double attackCoef_{0.0};
    double releaseCoef_{0.0};

    double env_{0.0};
};

} // namespace dsp_core::audio_pipeline
