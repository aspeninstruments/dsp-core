#pragma once

#include "BellStage.h"
#include "ToneFilterStrategy.h"

namespace dsp_core::audio_pipeline {

/**
 * Bell (parametric peaking) tone-filter strategy: a single RBJ peaking
 * biquad with user-tunable centre frequency, peak gain, and Q (bandwidth).
 *
 * Bell-specific control: Q. All other strategies treat the universal
 * setQ() as a no-op; only Bell honours it.
 */
class BellStrategy : public ToneFilterStrategy {
  public:
    BellStrategy() = default;

    void prepareToPlay(double sampleRate, int samplesPerBlock, int numChannels) override {
        bell_.prepareToPlay(sampleRate, samplesPerBlock, numChannels);
    }

    void process(juce::AudioBuffer<double>& buffer) override {
        bell_.process(buffer);
    }

    void reset() override {
        bell_.reset();
    }

    juce::String getName() const override {
        return "Bell";
    }

    // ToneFilterStrategy universal setters
    void setFrequency(double frequencyHz) override {
        bell_.setCutoffFrequency(frequencyHz);
    }

    void setResonance(double /*zeroToOne*/) override {
        // No-op. Ladder resonance has no Bell analogue; Q controls bandwidth instead.
    }

    void setShelfGainDb(double gainDb) override {
        // Reused as the bell's peak gain in dB. Sharing Tone_Gain with the
        // shelves keeps presets / automation portable across Shelf <-> Bell.
        bell_.setGainDb(gainDb);
    }

    void setFat(double /*percent*/) override {
        // No-op. Fat is a ladder-specific bass-restoration concept.
    }

    void setLowShelfRatio(double /*ratio*/) override {
        // No-op. The LS-to-HS frequency ratio is a Smile-strategy concept.
    }

    void setQ(double q) override {
        bell_.setQ(q);
    }

    void setEmph(double /*zeroToOne*/) override {
        // No-op. Emph is a Hysteresis pre/de-emphasis concept.
    }

  private:
    BellStage bell_;
};

} // namespace dsp_core::audio_pipeline
