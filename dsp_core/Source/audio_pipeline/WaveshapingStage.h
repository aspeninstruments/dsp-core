#pragma once

#include "AudioProcessingStage.h"
#include "../SeamlessTransferFunction.h"
#include "../LayeredTransferFunction.h"

namespace dsp_core::audio_pipeline {

/**
 * Applies waveshaping using SeamlessTransferFunction or LayeredTransferFunction.
 * Optimized for stereo processing (no threading overhead).
 */
class WaveshapingStage : public AudioProcessingStage {
  public:
    /**
     * @param tf Reference to seamless transfer function (production use)
     */
    // Non-const ref: WaveshapingStage::prepareToPlay forwards the oversampled
    // sample rate into transferFunction.prepareToPlay, so surge timing stays
    // wall-time accurate across oversampling-order changes.
    explicit WaveshapingStage(dsp_core::SeamlessTransferFunction& tf);

    /**
     * @param ltf Reference to layered transfer function (testing/profiling use)
     */
    explicit WaveshapingStage(dsp_core::LayeredTransferFunction& ltf);

    void prepareToPlay(double sampleRate, int samplesPerBlock, int numChannels) override;
    void process(juce::AudioBuffer<double>& buffer) override;
    void reset() override;
    juce::String getName() const override {
        return "Waveshaping";
    }

  private:
    dsp_core::SeamlessTransferFunction* seamlessTransferFunction_{nullptr};
    dsp_core::LayeredTransferFunction* layeredTransferFunction_{nullptr};
};

} // namespace dsp_core::audio_pipeline
