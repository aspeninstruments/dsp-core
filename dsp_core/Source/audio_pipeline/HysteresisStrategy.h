#pragma once

#include "ToneFilterStrategy.h"

namespace dsp_core::audio_pipeline {

/**
 * Hysteresis tone-filter strategy — a NO-OP MARKER.
 *
 * Unlike the other tone strategies (Lowpass, LowShelf, HighShelf, Smile,
 * Bell), hysteresis cannot be implemented inside `ToneStage::process()`: the
 * Jiles-Atherton magnetisation loop uses the waveshaper LUT as its
 * anhysteretic nonlinearity, so it IS the waveshaping stage. Moving its DSP
 * into a buffer-in/buffer-out tone strategy would force the pipeline to
 * disable `HysteresisStage` (which already owns the waveshaper) and would
 * double-apply if both the pre- and post-tone-stage instances tried to run it.
 *
 * Instead, this strategy is selected via `Tone_Type == Hysteresis` purely as
 * a marker. The plugin processBlock reads the tone type and drives the
 * pipeline-level `HysteresisStage::setHysteresisEnabled(...)` accordingly.
 * When Hysteresis is the active tone type, `ToneStage::process` early-returns
 * because this strategy's `process()` does nothing — the real magnetic-tape
 * memory effect happens downstream in the pipeline at the waveshaper position.
 *
 * For implementers adding new tone types: if the new type is a wrap-around-
 * the-waveshaper effect rather than a buffer filter, follow this pattern
 * (no-op strategy + drive the pipeline-level stage from Tone_Type).
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

    // ToneFilterStrategy universal setters — all no-ops. Hysteresis's only
    // user-facing param (Width) is owned by the pipeline-level HysteresisStage,
    // not this strategy.
    void setFrequency(double /*frequencyHz*/) override {}
    void setResonance(double /*zeroToOne*/) override {}
    void setShelfGainDb(double /*gainDb*/) override {}
    void setFat(double /*percent*/) override {}
    void setLowShelfRatio(double /*ratio*/) override {}
    void setQ(double /*q*/) override {}
};

} // namespace dsp_core::audio_pipeline
