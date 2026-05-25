#pragma once

#include "AudioProcessingStage.h"
#include "LowShelfStrategy.h"
#include "LowpassStrategy.h"
#include "ToneFilterStrategy.h"
#include <atomic>

namespace dsp_core::audio_pipeline {

/**
 * Tone filter stage: a thin host that dispatches to one of several filter
 * strategies (Lowpass 12 dB, Lowpass 24 dB, Low Shelf, ...). Each strategy
 * owns its own sub-pipeline; ToneStage holds one concrete instance of each
 * and forwards parameters to all of them so the idle strategies stay primed
 * with current values when activated.
 *
 * Adding a new filter type: see the checklist in ToneFilterStrategy.h.
 *
 * Resonance is exposed as a normalized [0, 1] dial value; internally mapped
 * to [0, kToneMaxResonance] — safely below the LadderTPT's self-oscillation
 * onset at ~3.95. The mapping happens at the ToneStage boundary so strategies
 * see the already-mapped value.
 *
 * Thread safety:
 *  - enabled_, type_: atomic (UI writes via setEnabled/setType, audio reads).
 *  - Strategy parameter setters handle their own atomicity and are forwarded
 *    to every strategy so an idle strategy is primed when it becomes active.
 *  - On detected type change, process() resets the newly-active strategy on
 *    the audio thread to clear stale state without racing UI-thread writes.
 *    Note: LP12 <-> LP24 transitions are still click-free thanks to the
 *    prime-everything forwarding above; LP <-> LS topology switches reset the
 *    new strategy and let the IIR re-stabilize (RBJ biquad settle is ms-scale,
 *    no crossfade needed for an explicit user-driven type change).
 */
class ToneStage : public AudioProcessingStage {
  public:
    // APPEND-ONLY: the integer values are persisted via APVTS choice indices
    // (preset/automation stability). Never reorder; only add at the end.
    enum class Type { Off, Lowpass12dB, Lowpass24dB, LowShelf };

    ToneStage() = default;

    // AudioProcessingStage interface
    void prepareToPlay(double sampleRate, int samplesPerBlock, int numChannels) override;
    void process(juce::AudioBuffer<double>& buffer) override;
    void reset() override;
    juce::String getName() const override {
        return "Tone";
    }

    // Control interface (thread-safe)
    void setEnabled(bool shouldBeEnabled) {
        enabled_.store(shouldBeEnabled, std::memory_order_release);
    }
    bool isEnabled() const {
        return enabled_.load(std::memory_order_acquire);
    }

    void setType(Type t) {
        type_.store(t, std::memory_order_release);
    }
    Type getType() const {
        return type_.load(std::memory_order_acquire);
    }

    /** Filter corner frequency in Hz (LP cutoff or LS frequency). Forwarded
     *  to all strategies; each clamps to its own valid range. */
    void setCutoffFrequency(double frequencyHz);

    /** Resonance from the UI as [0, 1]; mapped to the LadderTPT's internal
     *  [0, kToneMaxResonance] range (just below self-oscillation). LP-only. */
    void setResonance(double zeroToOne);

    /** Shelf gain in dB. Forwarded to LS strategies; LP strategies ignore. */
    void setShelfGainDb(double gainDb);

    /** Fat percent [0, 100] — pre-LP low-shelf bass restoration owned by the
     *  Lowpass strategies. At 0% the inner FatStage is skipped entirely (no
     *  biquad work, no IIR state). LS strategy ignores. */
    void setFat(double percent);
    double getFat() const;

  private:
    LowpassStrategy<2> lp12_;
    LowpassStrategy<4> lp24_;
    LowShelfStrategy lowShelf_;

    std::atomic<bool> enabled_{false};
    std::atomic<Type> type_{Type::Off};

    // Audio-thread cache for change detection so we can reset the newly-
    // activated strategy on the audio thread rather than from setType (which
    // would race concurrent process() reads).
    Type lastType_ = Type::Off;

    /** Returns the strategy for the given type, or nullptr for Type::Off. */
    ToneFilterStrategy* strategyFor(Type t);
};

} // namespace dsp_core::audio_pipeline
