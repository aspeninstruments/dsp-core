#pragma once

#include "AudioProcessingStage.h"
#include "AutoGainState.h"

namespace dsp_core::audio_pipeline {

/**
 * AutoRestoreStage — back half of the auto-normalize ("squash and restore")
 * feature.
 *
 * Reads the per-sample gain written by AutoSquashStage into the shared state
 * and applies its exact reciprocal. This preserves the outer input→output
 * dynamics regardless of how dramatically the input level varies, while still
 * letting the nonlinear stages see a consistent internal level.
 */
class AutoRestoreStage : public AudioProcessingStage {
  public:
    explicit AutoRestoreStage(AutoGainState& state) : state_(state) {}

    void prepareToPlay(double sampleRate, int samplesPerBlock, int numChannels) override;
    void process(juce::AudioBuffer<double>& buffer) override;
    void reset() override;
    juce::String getName() const override {
        return "AutoRestore";
    }

  private:
    AutoGainState& state_;
};

} // namespace dsp_core::audio_pipeline
