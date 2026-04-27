#pragma once

#include "AudioProcessingStage.h"
#include "../SeamlessTransferFunction.h"
#include "../HysteresisProcessor.h"
#include <array>
#include <atomic>

namespace dsp_core::audio_pipeline {

/**
 * Pipeline stage adapter wrapping HysteresisProcessor for use in AudioPipeline.
 *
 * Follows the solver/stage separation pattern (like SoftClippingSolver/SoftClippingStage).
 *
 * When hysteresis is enabled: per-sample, per-channel processing through
 * HysteresisProcessor (Jiles-Atherton with custom NL injection).
 *
 * When hysteresis is disabled: delegates to transferFunction_->processBuffer()
 * for memoryless waveshaping fallback (identical to WaveshapingStage).
 *
 * Thread Safety:
 *   - hysteresisEnabled_: atomic (UI writes, audio reads)
 *   - setDrive/setSaturation/setWidth: UI thread, between process() calls
 *   - process(): audio thread only
 */
class HysteresisStage : public AudioProcessingStage {
  public:
    // Non-const ref: HysteresisStage::prepareToPlay forwards the oversampled
    // sample rate into transferFunction.prepareToPlay, so surge timing stays
    // wall-time accurate across oversampling-order changes.
    explicit HysteresisStage(dsp_core::SeamlessTransferFunction& tf);

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void process(juce::AudioBuffer<double>& buffer) override;
    void reset() override;
    juce::String getName() const override { return "Hysteresis"; }

    void setHysteresisEnabled(bool enabled);
    void requestWarmup();
    void setDrive(double drive);
    void setSaturation(double sat);
    void setWidth(double width);
    void setMakeupGain(double gain);
    void setOperatingPoint(double Ms);

    // Returns makeup gain that compensates for width-dependent peak loss so a
    // sine at default sat=0.5 exits hysteresis at roughly the input peak.
    // Tuned empirically — see MakeupGain_WidthSweep_PeakWithinFivePercent.
    static double computeMakeupForWidth(double width);

  private:
    enum class CrossfadeState { Inactive, WarmingUp, CrossfadingIn, CrossfadingOut };

    static double smoothstep(double t) { return t * t * (3.0 - 2.0 * t); }

    void processWarmup(juce::AudioBuffer<double>& buffer, int startSample, int numSamples);
    void processCrossfadeIn(juce::AudioBuffer<double>& buffer, int startSample, int numSamples);
    void processCrossfadeOut(juce::AudioBuffer<double>& buffer, int startSample, int numSamples);
    void processSteadyHysteresis(juce::AudioBuffer<double>& buffer, int startSample, int numSamples);
    void processSteadyWaveshaping(juce::AudioBuffer<double>& buffer, int startSample, int numSamples);

    // Detect enabled/disabled toggle and start a transition crossfade if needed.
    void detectTransition(bool enabled);
    // Process one crossfade phase chunk; returns samples consumed.
    int advanceCrossfadePhase(juce::AudioBuffer<double>& buffer, int pos, int remaining, bool enabled);

    dsp_core::SeamlessTransferFunction* transferFunction_;
    std::array<dsp_core::HysteresisProcessor, 2> processors_; // stereo
    std::atomic<bool> hysteresisEnabled_{true};
    juce::SmoothedValue<double, juce::ValueSmoothingTypes::Linear> smoothedMakeupGain_{1.0};

    CrossfadeState crossfadeState_{CrossfadeState::Inactive};
    int crossfadePosition_{0};
    int phaseSamples_{480};
    bool previousEnabled_{false};
};

} // namespace dsp_core::audio_pipeline
