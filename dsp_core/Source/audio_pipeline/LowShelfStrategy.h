#pragma once

#include "LowShelfStage.h"
#include "ToneFilterStrategy.h"

namespace dsp_core::audio_pipeline {

/**
 * Low-shelf tone-filter strategy: standalone user-tunable RBJ low shelf
 * (frequency in Hz, gain in dB, Butterworth Q). Boosts/cuts the band below
 * the corner frequency, leaving the high band largely intact.
 *
 * Unlike LowpassStrategy, this strategy does NOT own a Fat sub-stage. Fat is
 * a ladder-specific bass-restoration concept; a user-driven low shelf already
 * exposes "boost the bass" directly, so stacking Fat on top would just be a
 * second shelf in series.
 */
class LowShelfStrategy : public ToneFilterStrategy {
  public:
    LowShelfStrategy() = default;

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
        return "LowShelf";
    }

    // ToneFilterStrategy universal setters
    void setFrequency(double frequencyHz) override {
        shelf_.setCutoffFrequency(frequencyHz);
    }

    void setResonance(double /*zeroToOne*/) override {
        // No-op. The RBJ low shelf uses a fixed Butterworth Q.
    }

    void setShelfGainDb(double gainDb) override {
        shelf_.setGainDb(gainDb);
    }

    void setFat(double /*percent*/) override {
        // No-op. Fat is a ladder-specific bass-restoration concept; users
        // already control the shelf gain directly here.
    }

    void setLowShelfRatio(double /*ratio*/) override {
        // No-op. The LS-to-HS frequency ratio is a Smile-strategy concept.
    }

    void setQ(double /*q*/) override {
        // No-op. Bell Q is a Bell-strategy concept; this shelf uses fixed Butterworth Q.
    }

  private:
    LowShelfStage shelf_;
};

} // namespace dsp_core::audio_pipeline
