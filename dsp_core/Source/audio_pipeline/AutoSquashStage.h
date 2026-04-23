#pragma once

#include "AudioProcessingStage.h"
#include "AutoGainState.h"

namespace dsp_core::audio_pipeline {

/**
 * AutoSquashStage — front half of the auto-normalize ("squash and restore")
 * feature.
 *
 * Tracks a stereo-linked peak envelope (max(|L|, |R|)) with a one-pole
 * attack/release follower, derives a per-sample gain that drives the envelope
 * toward the target peak (~ -1 dBFS), clamps it within a max-gain ceiling, and
 * applies that gain in place. The per-sample gain is recorded into the shared
 * state's gainHistory buffer so AutoRestoreStage can apply the exact inverse
 * downstream.
 *
 * When the Auto feature is disabled, process() becomes a no-op after fading
 * gain back to 1.0 over an enable-fade window (to prevent clicks on toggle).
 */
class AutoSquashStage : public AudioProcessingStage {
  public:
    explicit AutoSquashStage(AutoGainState& state) : state_(state) {}

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void process(juce::AudioBuffer<double>& buffer) override;
    void reset() override;
    juce::String getName() const override {
        return "AutoSquash";
    }

  private:
    AutoGainState& state_;
};

} // namespace dsp_core::audio_pipeline
