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
    explicit HysteresisStage(const dsp_core::SeamlessTransferFunction& tf);

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void process(juce::AudioBuffer<double>& buffer) override;
    void reset() override;
    juce::String getName() const override { return "Hysteresis"; }

    void setHysteresisEnabled(bool enabled);
    void setDrive(double drive);
    void setSaturation(double sat);
    void setWidth(double width);
    void setMakeupGain(double gain);
    void setOperatingPoint(double Ms);

  private:
    const dsp_core::SeamlessTransferFunction* transferFunction_;
    std::array<dsp_core::HysteresisProcessor, 2> processors_; // stereo
    std::atomic<bool> hysteresisEnabled_{true};
    double makeupGain_{1.0};
};

} // namespace dsp_core::audio_pipeline
