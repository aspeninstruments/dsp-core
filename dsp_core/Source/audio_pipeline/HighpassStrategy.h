#pragma once

#include "LadderHighpassTPTStage.h"
#include "ToneFilterStrategy.h"

namespace dsp_core::audio_pipeline {

/**
 * Highpass tone-filter strategy: a ladder highpass (TPT-ZDF, tanh feedback).
 * Templated on the ladder order:
 *   - N=2: 12 dB/oct ladder ("Highpass 12 dB")
 *   - N=4: 24 dB/oct ladder ("Highpass 24 dB")
 *
 * Unlike LowpassStrategy this owns no FatStage — Fat exists in the LP family
 * to compensate for resonance stealing DC gain, and the HP cascade has no
 * symmetric problem (passband makeup `(1+R)` already restores HF level).
 * Any HF tilt / sparkle effect is a separate concept, not a VA HPF property.
 */
template <int N>
class HighpassStrategy : public ToneFilterStrategy {
  public:
    HighpassStrategy() = default;

    void prepareToPlay(double sampleRate, int samplesPerBlock, int numChannels) override {
        ladder_.prepareToPlay(sampleRate, samplesPerBlock, numChannels);
    }

    void process(juce::AudioBuffer<double>& buffer) override {
        ladder_.process(buffer);
    }

    void reset() override {
        ladder_.reset();
    }

    juce::String getName() const override {
        return N == 2 ? "Highpass12dB" : "Highpass24dB";
    }

    // ToneFilterStrategy universal setters
    void setFrequency(double frequencyHz) override {
        ladder_.setCutoffFrequency(frequencyHz);
    }

    void setResonance(double zeroToOne) override {
        // The [0, 1] -> [0, kToneMaxResonance] mapping is performed by
        // ToneStage::setResonance before forwarding; we receive the already-
        // mapped value as the raw ladder resonance.
        ladder_.setResonance(zeroToOne);
    }

    void setShelfGainDb(double /*gainDb*/) override {
        // No-op. Shelf gain is a Low/High-Shelf strategy concept.
    }

    void setFat(double /*percent*/) override {
        // No-op. Fat is LP-only (bass restoration for resonance-driven loss);
        // HP has nothing analogous to compensate.
    }

    void setLowShelfRatio(double /*ratio*/) override {
        // No-op. The LS-to-HS frequency ratio is a Smile-strategy concept.
    }

    void setQ(double /*q*/) override {
        // No-op. Bell Q is a Bell-strategy concept; ladder resonance has its own knob.
    }

    void setEmph(double /*zeroToOne*/) override {
        // No-op. Emph is a Hysteresis pre/de-emphasis concept.
    }

  private:
    LadderHighpassTPTStage<N> ladder_;
};

using HighpassStrategy12dB = HighpassStrategy<2>;
using HighpassStrategy24dB = HighpassStrategy<4>;

} // namespace dsp_core::audio_pipeline
