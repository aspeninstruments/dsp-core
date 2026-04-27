#pragma once

#include "AudioProcessingStage.h"
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_dsp/juce_dsp.h>
#include <atomic>
#include <vector>

namespace dsp_core::audio_pipeline {

/**
 * Passthrough envelope follower — measures input level and publishes it
 * to an atomic owned by the processor for modulation routing. Audio is
 * never modified; the stage is purely a measurement tap.
 *
 * Per-sample detector pipeline:
 *   1. Optional sidechain HPF (per-channel 1st-order Butterworth) to keep
 *      bass energy from dominating the modulation signal. Useful even when
 *      a sidechain bus is in use — saves a routing detour.
 *   2. RMS detector: leaky integrator on x² (~15 ms averaging window).
 *      Stereo uses sum-of-squares linking.
 *   3. Apply pre-detection sensitivity, clamp to 1.0.
 *   4. Convert to dB and smooth with an attack/release one-pole in the
 *      log domain. Release is nonlinear: the release coefficient scales
 *      with dB-drop magnitude so big drops decay quickly while sustained
 *      tails track smoothly.
 *   5. Publish smoothed value back as linear [0, 1] for downstream consumers.
 *
 * Thread safety:
 *   - enabled_, sensitivityLinear_, hpfEnabled_, hpfCutoffHz_:
 *     atomics (UI writes, audio reads at block boundaries)
 *   - setAttackReleaseSec, setSidechainHpfCutoffHz, prepareToPlay, reset:
 *     UI thread only, between process() calls
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

    // Pre-detection sensitivity applied before the detector.
    // UI thread writes; audio thread reads at block boundaries.
    void setSensitivityLinear(double linear) {
        sensitivityLinear_.store(linear, std::memory_order_release);
    }

    // Wall-clock attack/release times — used as the dB-domain time constants
    // of the smoother. Recomputes one-pole + RMS coefficients against the
    // currently-prepared sample rate.
    void setAttackReleaseSec(double attackSec, double releaseSec);

    void setSidechainHpfEnabled(bool shouldBeEnabled) {
        hpfEnabled_.store(shouldBeEnabled, std::memory_order_release);
    }

    // Sidechain HPF cutoff in Hz. Allocation-free — safe to call per block
    // from updatePipelineParameters().
    void setSidechainHpfCutoffHz(double cutoffHz);

    // When non-null, process() reads samples from this buffer instead of the
    // in-pipeline buffer arg. Used to feed the detector from a sidechain bus
    // while the audio path still flows through the main pipeline. Caller
    // must keep the pointee alive for the duration of the next process() call;
    // pointer is set on the audio thread between sub-stages, no atomic needed.
    void setExternalInputBuffer(const juce::AudioBuffer<double>* buffer) {
        externalInputBuffer_ = buffer;
    }

    double getAttackSec() const {
        return attackSec_;
    }
    double getReleaseSec() const {
        return releaseSec_;
    }

  private:
    void recomputeCoefficients();
    void updateHpfCoefficients();
    void resizeHpfFiltersForChannels(int numChannels);

    std::atomic<double>& envelopeStorage_;

    std::atomic<bool> enabled_{false};
    std::atomic<double> sensitivityLinear_{1.0};
    std::atomic<bool> hpfEnabled_{false};
    std::atomic<double> hpfCutoffHz_{100.0};

    // Plain pointer — read and written on the audio thread only.
    const juce::AudioBuffer<double>* externalInputBuffer_{nullptr};

    double sampleRate_{48000.0};
    double attackSec_{0.005};
    double releaseSec_{0.150};
    double attackCoef_{0.0};
    double releaseCoef_{0.0};
    double rmsCoef_{0.0};

    // Audio-thread-only state.
    double envDb_{kEnvDbFloor_}; // smoother state in dB domain
    double ms2_{0.0};            // RMS leaky-integrator state (mean-square)

    // Per-channel sidechain HPF filters, all sharing one Coefficients object
    // (mutated in-place on cutoff changes — allocation-free per block).
    std::vector<juce::dsp::IIR::Filter<double>> hpfFilters_;
    juce::dsp::IIR::Coefficients<double>::Ptr hpfCoefficients_;

    // dB floor — silence sits here. Used both as the smoother's lower bound
    // and as the minus-infinity reference for JUCE's gain<->dB helpers.
    static constexpr double kEnvDbFloor_ = -120.0;
};

} // namespace dsp_core::audio_pipeline
