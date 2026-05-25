#pragma once

#include "FatStage.h"
#include "LadderTPTStage.h"
#include "ToneFilterStrategy.h"

namespace dsp_core::audio_pipeline {

/**
 * Lowpass tone-filter strategy: a LowShelf (Fat, bass restoration) feeding a
 * ladder lowpass. Templated on the ladder order:
 *   - N=2: 12 dB/oct ladder ("Lowpass 12 dB")
 *   - N=4: 24 dB/oct ladder ("Lowpass 24 dB")
 *
 * Conceptually the "real" lowpass topology is `lowshelf(fat) -> ladder`. Fat
 * lives inside this strategy because it is a property of the ladder family
 * (it compensates for resonance-driven bass loss), not a top-level tone-stage
 * feature. Other strategy types (LowShelf, future High Shelf, ...) do not own
 * a Fat sub-stage.
 *
 * Fat bypass invariant (preserved from the pre-strategy ToneStage):
 *   - When Fat percent is 0, FatStage::process is not called and the IIR
 *     biquad accrues no state.
 *   - When Fat percent transitions 0 -> nonzero, fat_ is reset before its
 *     first active block so frozen-stale state doesn't replay.
 */
template <int N>
class LowpassStrategy : public ToneFilterStrategy {
  public:
    LowpassStrategy() = default;

    void prepareToPlay(double sampleRate, int samplesPerBlock, int numChannels) override {
        ladder_.prepareToPlay(sampleRate, samplesPerBlock, numChannels);
        fat_.prepareToPlay(sampleRate, samplesPerBlock, numChannels);
        lastFatActive_ = (fat_.getFat() > 0.0);
    }

    void process(juce::AudioBuffer<double>& buffer) override {
        // Strict 0% bypass: single atomic load, single branch. Skips biquad
        // work entirely when Fat is at zero (audible-RMS-preserving bypass).
        const double fatPct = fat_.getFat();
        const bool fatActive = (fatPct > 0.0);
        if (fatActive) {
            if (!lastFatActive_) {
                // Emerged from bypass — flush stale shelf state so re-entry
                // is click-free.
                fat_.reset();
            }
            fat_.process(buffer);
        }
        lastFatActive_ = fatActive;

        ladder_.process(buffer);
    }

    void reset() override {
        ladder_.reset();
        fat_.reset();
        lastFatActive_ = false;
    }

    juce::String getName() const override {
        return N == 2 ? "Lowpass12dB" : "Lowpass24dB";
    }

    // ToneFilterStrategy universal setters
    void setFrequency(double frequencyHz) override {
        ladder_.setCutoffFrequency(frequencyHz);
    }

    void setResonance(double zeroToOne) override {
        // The [0, 1] -> [0, kToneMaxResonance] mapping is performed by
        // ToneStage::setResonance before forwarding; we receive the already-
        // mapped value as the raw ladder resonance. (See ToneStage.cpp for
        // why the cap sits at 3.5, safely below self-osc onset.)
        ladder_.setResonance(zeroToOne);
    }

    void setShelfGainDb(double /*gainDb*/) override {
        // No-op. Shelf gain is a Low-Shelf strategy concept; Fat has its own
        // knob (setFat) and must NOT be driven from setShelfGainDb.
    }

    void setFat(double percent) override {
        fat_.setFat(percent);
    }

    void setLowShelfRatio(double /*ratio*/) override {
        // No-op. The LS-to-HS frequency ratio is a Smile-strategy concept.
    }

    void setQ(double /*q*/) override {
        // No-op. Bell Q is a Bell-strategy concept; ladder resonance has its own knob.
    }

    /** Read current Fat percent — used by ToneStage::getFat to preserve the
     *  pre-refactor public API. */
    double getFatPercent() const {
        return fat_.getFat();
    }

  private:
    LadderTPTStage<N> ladder_;
    FatStage fat_;

    // Audio-thread cache for Fat bypass gate (mirrors the pre-refactor
    // ToneStage::lastFatActive_ behaviour).
    bool lastFatActive_ = false;
};

using LowpassStrategy12dB = LowpassStrategy<2>;
using LowpassStrategy24dB = LowpassStrategy<4>;

} // namespace dsp_core::audio_pipeline
