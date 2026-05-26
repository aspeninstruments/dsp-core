#pragma once

#include "HighShelfStage.h"
#include "ToneFilterStrategy.h"

namespace dsp_core::audio_pipeline {

/**
 * High-shelf tone-filter strategy: standalone user-tunable RBJ high shelf
 * (frequency in Hz, gain in dB, Butterworth Q). Boosts/cuts the band above
 * the corner frequency, leaving the low band largely intact.
 *
 * Mirrors LowShelfStrategy exactly — the only difference is which RBJ-shelf
 * direction is applied. Both share the Tone_Cutoff / Tone_Gain APVTS params
 * and the inline-panel Frequency / Gain controls.
 */
class HighShelfStrategy : public ToneFilterStrategy {
  public:
    HighShelfStrategy() = default;

    void prepareToPlay(double sampleRate, int samplesPerBlock, int numChannels) override {
        shelf_.prepareToPlay(sampleRate, samplesPerBlock, numChannels);
    }

    void process(juce::AudioBuffer<double>& buffer) override {
        shelf_.process(buffer);
    }

    void reset() override {
        shelf_.reset();
    }

    juce::String getName() const override {
        return "HighShelf";
    }

    // ToneFilterStrategy universal setters
    void setFrequency(double frequencyHz) override {
        shelf_.setCutoffFrequency(frequencyHz);
    }

    void setResonance(double /*zeroToOne*/) override {
        // No-op. The RBJ high shelf uses a fixed Butterworth Q.
    }

    void setShelfGainDb(double gainDb) override {
        shelf_.setGainDb(gainDb);
    }

    void setFat(double /*percent*/) override {
        // No-op. Fat is a ladder-specific bass-restoration concept.
    }

    void setLowShelfRatio(double /*ratio*/) override {
        // No-op. The LS-to-HS frequency ratio is a Smile-strategy concept.
    }

    void setQ(double /*q*/) override {
        // No-op. Bell Q is a Bell-strategy concept; this shelf uses fixed Butterworth Q.
    }

    void setEmph(double /*zeroToOne*/) override {
        // No-op. Emph is a Hysteresis pre/de-emphasis concept.
    }

  private:
    HighShelfStage shelf_;
};

} // namespace dsp_core::audio_pipeline
