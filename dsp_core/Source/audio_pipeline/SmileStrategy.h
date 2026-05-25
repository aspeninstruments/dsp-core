#pragma once

#include "HighShelfStage.h"
#include "LowShelfStage.h"
#include "ToneFilterStrategy.h"
#include <atomic>
#include <juce_core/juce_core.h>

namespace dsp_core::audio_pipeline {

/**
 * Smile-curve tone strategy: low shelf + high shelf in series, with both
 * shelves sharing a single gain value (linked) and the low-shelf corner
 * frequency expressed as a ratio of the high-shelf corner. Positive gain
 * produces a classic smile (bass + treble boost); negative gain produces
 * a frown (cut at both ends).
 *
 * Control surface:
 *   - setFrequency(hz)        — high-shelf corner; LS corner recomputes as hz × ratio
 *   - setShelfGainDb(db)      — applied to both shelves
 *   - setLowShelfRatio(ratio) — LS-to-HS frequency ratio, clamped [0.015, 0.5]
 *
 * This is the only strategy that owns BOTH a LowShelfStage and a HighShelfStage
 * (the standalone LowShelf / HighShelf strategies own only their respective
 * stage). Owning the stages directly — rather than composing LowShelfStrategy
 * and HighShelfStrategy — keeps the LS-frequency derivation logic local and
 * avoids forwarding setFrequency through wrappers that would interpret it as
 * the per-shelf frequency.
 */
class SmileStrategy : public ToneFilterStrategy {
  public:
    SmileStrategy() = default;

    void prepareToPlay(double sampleRate, int samplesPerBlock, int numChannels) override {
        lowShelf_.prepareToPlay(sampleRate, samplesPerBlock, numChannels);
        highShelf_.prepareToPlay(sampleRate, samplesPerBlock, numChannels);
    }

    void process(juce::AudioBuffer<double>& buffer) override {
        // Series order is arbitrary — both shelves are linear/time-invariant
        // biquads and commute. LS first is a stylistic pick.
        lowShelf_.process(buffer);
        highShelf_.process(buffer);
    }

    void reset() override {
        lowShelf_.reset();
        highShelf_.reset();
    }

    juce::String getName() const override {
        return "Smile";
    }

    // ToneFilterStrategy universal setters
    void setFrequency(double frequencyHz) override {
        hsHz_.store(frequencyHz, std::memory_order_release);
        highShelf_.setCutoffFrequency(frequencyHz);
        applyLowShelfFrequency();
    }

    void setResonance(double /*zeroToOne*/) override {
        // No-op. Both shelves use fixed Butterworth Q.
    }

    void setShelfGainDb(double gainDb) override {
        // Linked: both shelves get the same dB value. The user changes the
        // smile/frown depth by sweeping a single knob; sign picks direction.
        lowShelf_.setGainDb(gainDb);
        highShelf_.setGainDb(gainDb);
    }

    void setFat(double /*percent*/) override {
        // No-op. Fat is a ladder-specific bass-restoration concept.
    }

    void setLowShelfRatio(double ratio) override {
        ratio = juce::jlimit(0.015, 0.5, ratio);
        lsRatio_.store(ratio, std::memory_order_release);
        applyLowShelfFrequency();
    }

    void setQ(double /*q*/) override {
        // No-op. Bell Q is a Bell-strategy concept; Smile uses fixed Butterworth shelves.
    }

  private:
    void applyLowShelfFrequency() {
        const double hsHz = hsHz_.load(std::memory_order_acquire);
        const double ratio = lsRatio_.load(std::memory_order_acquire);
        // LowShelfStage clamps to its valid [20, 20000] Hz range, so HS × ratio
        // combos that fall outside the audible band just leave the LS inert.
        lowShelf_.setCutoffFrequency(hsHz * ratio);
    }

    LowShelfStage lowShelf_;
    HighShelfStage highShelf_;

    // Cached so changes to either field can recompute the LS corner without
    // needing the other current value pushed again.
    std::atomic<double> hsHz_{3000.0};   // matches Tone_Cutoff default
    std::atomic<double> lsRatio_{0.0625}; // = HS / 16
};

} // namespace dsp_core::audio_pipeline
