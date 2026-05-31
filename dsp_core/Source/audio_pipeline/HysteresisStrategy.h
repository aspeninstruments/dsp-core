#pragma once

#include "ToneFilterStrategy.h"

namespace dsp_core::audio_pipeline {

/**
 * Hysteresis tone-filter strategy — a pure marker. Its process() is a no-op.
 *
 * The actual hysteresis DSP (the Jiles-Atherton magnetisation loop) runs at the
 * pipeline-level HysteresisStage, which wraps the waveshaper at a fixed central
 * position. Selecting Hysteresis as the type on the pre or post ToneStage merely
 * signals PluginProcessor to toggle HysteresisStage::setHysteresisEnabled; the
 * ToneStage itself is left disabled (it contributes no audio) so the opposite
 * side is free to host a normal filter before or after the saturator.
 *
 * Hysteresis is mutually exclusive across the two sides — see
 * PluginProcessor::parameterChanged. All universal setters here are no-ops.
 */
class HysteresisStrategy : public ToneFilterStrategy {
  public:
    HysteresisStrategy() = default;

    void prepareToPlay(double /*sampleRate*/, int /*samplesPerBlock*/, int /*numChannels*/) override {}
    void process(juce::AudioBuffer<double>& /*buffer*/) override {}
    void reset() override {}

    juce::String getName() const override {
        return "Hysteresis";
    }

    // ToneFilterStrategy universal setters — all no-ops (marker strategy).
    void setFrequency(double /*frequencyHz*/) override {}
    void setResonance(double /*zeroToOne*/) override {}
    void setShelfGainDb(double /*gainDb*/) override {}
    void setFat(double /*percent*/) override {}
    void setLowShelfRatio(double /*ratio*/) override {}
    void setQ(double /*q*/) override {}
};

} // namespace dsp_core::audio_pipeline
