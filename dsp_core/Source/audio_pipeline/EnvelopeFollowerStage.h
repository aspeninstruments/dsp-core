#pragma once

#include "AudioProcessingStage.h"
#include <juce_audio_basics/juce_audio_basics.h>
#include <atomic>

namespace dsp_core::audio_pipeline {

/**
 * Passthrough envelope follower — measures input level and publishes it
 * to an atomic owned by the processor for modulation routing. Audio is
 * never modified; the stage is purely a measurement tap.
 *
 * Per-sample detector pipeline:
 *   1. Peak (max |x| across channels) OR RMS (leaky integrator on x²,
 *      ~30 ms averaging window). Stereo RMS uses sum-of-squares linking.
 *   2. Apply pre-detection sensitivity, clamp to 1.0.
 *   3. Convert to dB and smooth with an attack/release one-pole in the
 *      log domain. Release is nonlinear: the release coefficient scales
 *      with dB-drop magnitude so big drops decay quickly while sustained
 *      tails track smoothly.
 *   4. Publish smoothed value back as linear [0, 1] for downstream consumers.
 *
 * Thread safety:
 *   - enabled_, sensitivityLinear_, detectionMode_: atomics (UI writes,
 *     audio reads at block boundaries)
 *   - setAttackReleaseSec, prepareToPlay, reset: UI thread only, between
 *     process() calls
 *   - process: audio thread only
 */
class EnvelopeFollowerStage : public AudioProcessingStage {
  public:
    enum class DetectionMode { Peak = 0, Rms = 1 };

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

    // Pre-detection sensitivity applied before the detector.
    // UI thread writes; audio thread reads at block boundaries.
    void setSensitivityLinear(double linear) {
        sensitivityLinear_.store(linear, std::memory_order_release);
    }

    // Wall-clock attack/release times — used as the dB-domain time constants
    // of the smoother. Recomputes one-pole + RMS coefficients against the
    // currently-prepared sample rate.
    void setAttackReleaseSec(double attackSec, double releaseSec);

    void setDetectionMode(DetectionMode mode) {
        detectionMode_.store(static_cast<int>(mode), std::memory_order_release);
    }

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
    std::atomic<double> sensitivityLinear_{1.0};
    std::atomic<int> detectionMode_{static_cast<int>(DetectionMode::Peak)};

    double sampleRate_{48000.0};
    double attackSec_{0.005};
    double releaseSec_{0.150};
    double attackCoef_{0.0};
    double releaseCoef_{0.0};
    double rmsCoef_{0.0};

    // Audio-thread-only state.
    double envDb_{kEnvDbFloor_};   // smoother state in dB domain
    double ms2_{0.0};              // RMS leaky-integrator state (mean-square)

    // dB floor — silence sits here. Used both as the smoother's lower bound
    // and as the minus-infinity reference for JUCE's gain<->dB helpers.
    static constexpr double kEnvDbFloor_ = -120.0;
};

} // namespace dsp_core::audio_pipeline
