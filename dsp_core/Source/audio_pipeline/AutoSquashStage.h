#pragma once

#include "AudioProcessingStage.h"
#include "AutoGainState.h"

#include <juce_audio_basics/juce_audio_basics.h>

namespace dsp_core::audio_pipeline {

/**
 * AutoSquashStage — front half of the auto-normalize ("squash and restore")
 * feature.
 *
 * Tracks a stereo-linked peak envelope (max(|L|, |R|)) with a one-pole
 * attack/release follower, derives a per-sample gain that drives the envelope
 * toward state_.targetPeakLinear (set by the host from InputGain - 0.1 dB),
 * clamps it within a max-gain ceiling, and applies that gain in place. The
 * per-sample gain is recorded into the shared state's gainHistory buffer so
 * AutoRestoreStage can apply the exact inverse downstream.
 *
 * When the Auto feature is disabled, process() becomes a no-op after fading
 * gain back to 1.0 over an enable-fade window (to prevent clicks on toggle).
 */
class AutoSquashStage : public AudioProcessingStage {
  public:
    explicit AutoSquashStage(AutoGainState& state) : state_(state) {}

    void prepareToPlay(double sampleRate, int samplesPerBlock, int numChannels) override;
    void process(juce::AudioBuffer<double>& buffer) override;
    void reset() override;
    juce::String getName() const override {
        return "AutoSquash";
    }

  private:
    AutoGainState& state_;

    // Per-sample ramp on the squash target. Matches GainStage's 10 ms ramp so
    // InputGain's smoothed gain trajectory and the squash target trajectory
    // line up — without this, InputGain sweeps cause clicks because the
    // waveshaper between squash and restore sees mismatched ramps.
    juce::SmoothedValue<double, juce::ValueSmoothingTypes::Linear> smoothedTargetPeak_;
};

} // namespace dsp_core::audio_pipeline
