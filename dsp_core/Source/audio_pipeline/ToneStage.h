#pragma once

#include "AudioProcessingStage.h"
#include "FatStage.h"
#include "LadderTPTStage.h"
#include <atomic>

namespace dsp_core::audio_pipeline {

/**
 * Tone filter stage: wraps the 12 dB and 24 dB LadderTPT variants behind a
 * single type-switchable interface. Owns one instance of each; only the
 * selected one runs per block. When Type == Off (or setEnabled(false)),
 * process() early-returns and the buffer passes through unchanged.
 *
 * Resonance is exposed as a normalized [0, 1] dial value; internally mapped
 * to [0, 3.5] — safely below the LadderTPT's self-oscillation onset at ~3.95.
 *
 * Thread safety:
 *  - enabled_, type_: atomic (UI writes via setEnabled/setType, audio reads).
 *  - Inner filter cutoff/resonance setters handle their own atomicity and
 *    are forwarded to both filters so the idle one is always primed with
 *    current values when it becomes active.
 *  - On detected type change, process() resets the newly-active filter on
 *    the audio thread to clear stale state without racing UI-thread writes.
 */
class ToneStage : public AudioProcessingStage {
  public:
    enum class Type { Off, Lowpass12dB, Lowpass24dB };

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

    /** Cutoff in Hz. Forwarded to both inner filters; each clamps to its
     *  own [20 Hz, 0.45 * fs] range. */
    void setCutoffFrequency(double frequencyHz);

    /** Resonance from the UI as [0, 1]; mapped to the LadderTPT's internal
     *  [0, kToneMaxResonance] range (just below self-oscillation). */
    void setResonance(double zeroToOne);

    /** Fat percent [0, 100] — pre-LP low-shelf bass restoration. At 0% the
     *  inner FatStage is skipped entirely (no biquad work, no IIR state
     *  accumulation). Only active in the lowpass branches; the Type::Off
     *  early-return prevents Fat from running outside lowpass modes. */
    void setFat(double percent);
    double getFat() const;

  private:
    LadderTPT12dBStage lp12_;
    LadderTPT24dBStage lp24_;
    FatStage fat_;

    std::atomic<bool> enabled_{false};
    std::atomic<Type> type_{Type::Off};

    // Audio-thread cache for change detection so we can reset the newly-
    // activated filter on the audio thread rather than from setType (which
    // would race concurrent process() reads).
    Type lastType_ = Type::Off;

    // Audio-thread cache for the Fat bypass gate. False until the audio
    // thread has seen fat > 0; flipping false -> true triggers fat_.reset()
    // so the shelf IIR doesn't replay stale state from before the bypass.
    bool lastFatActive_ = false;
};

} // namespace dsp_core::audio_pipeline
