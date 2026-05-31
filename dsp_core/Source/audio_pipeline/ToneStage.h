#pragma once

#include "AudioProcessingStage.h"
#include "BellStrategy.h"
#include "HighShelfStrategy.h"
#include "HighpassStrategy.h"
#include "HysteresisStrategy.h"
#include "ImpulseResponseStrategy.h"
#include "LowShelfStrategy.h"
#include "LowpassStrategy.h"
#include "SmileStrategy.h"
#include "ToneFilterStrategy.h"
#include <array>
#include <atomic>

namespace dsp_core::audio_pipeline {

// Forward declaration — full definition lives in ToneFrequencyResponse.h,
// which references ToneStage::Type and so includes this file. Methods that
// take this by reference are defined in ToneStage.cpp, which pulls in the
// full header.
struct ToneFrequencyResponseParams;


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
    //
    // Note on Hysteresis: HysteresisStrategy is a marker — its process() is a
    // no-op. The actual hysteresis DSP runs at the pipeline-level HysteresisStage
    // (which wraps the waveshaper). PluginProcessor::processBlock reads Tone_Type
    // and toggles HysteresisStage::setHysteresisEnabled accordingly. See
    // HysteresisStrategy.h for the rationale.
    enum class Type {
        Off,
        Lowpass12dB,
        Lowpass24dB,
        LowShelf,
        HighShelf,
        Smile,
        Bell,
        Hysteresis,
        Highpass12dB,
        Highpass24dB,
        ImpulseResponse
    };

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

    /** Low-shelf-to-high-shelf frequency ratio for the Smile strategy
     *  (LS Hz = HS Hz × ratio). Clamped to [0.015, 0.5]. Smile-only;
     *  other strategies ignore. */
    void setLowShelfRatio(double ratio);

    /** Bell Q (bandwidth). Clamped to [0.1, 10]. Bell-only; other
     *  strategies ignore. */
    void setQ(double q);

    /** Load the .wav file used by the ImpulseResponse strategy. Routes only
     *  to ir_ (the IR is strategy-specific, unlike Frequency/Gain/etc. which
     *  are forwarded to all strategies to keep idle ones primed). Must be
     *  called from the message thread; the convolver hot-swaps the IR
     *  asynchronously so it is safe while audio is running. */
    void setImpulseResponseFile(const juce::File& file) {
        ir_.setImpulseResponseFile(file);
    }

    // ── Frequency-response LUT (UI-thread pull source for the visualizer) ──
    //
    // Sampled magnitude response of the *currently selected* strategy at
    // log-spaced frequencies from 10 Hz to 20 kHz, in dB. Published via
    // atomic version bump.
    //
    // Threading: the editor's 60 Hz timer reads APVTS directly (with
    // modulation math applied) and calls recomputeFrequencyResponse(params)
    // when the param fingerprint changes. The audio thread is NOT involved
    // in the visualization path — that's what makes the trace update while
    // the host transport is stopped. Audio-thread parameter setters still
    // mutate the strategies' DSP state for actual filtering, but they no
    // longer drive the LUT.
    //
    // 1024 samples (≈93 per octave across 11 octaves) is dense enough that
    // a Bell at Q=10 or Moog ladder at R≈3.5 — both with sub-octave peak
    // bandwidths — render smoothly at typical plot widths and stay stable
    // as cutoff modulates (sparse sampling would let the peak's apex slip
    // between bins frame-to-frame, producing visible jitter).
    //
    // Hysteresis: a flat (0 dB) trace — the saturation is nonlinear and lives
    // at the pipeline-level HysteresisStage, so the tone curve is identity.

    static constexpr int kFrequencyResponseSize = 1024;
    static constexpr double kFrequencyResponseMinHz = 10.0;
    static constexpr double kFrequencyResponseMaxHz = 20000.0;
    static constexpr double kFrequencyResponseDbRange = 24.0; // matches Tone_Gain extremes

    const double* getFrequencyResponseLUT() const {
        return frequencyResponseBuffer_.data();
    }
    const std::atomic<uint64_t>* getFrequencyResponseVersion() const {
        return &frequencyResponseVersion_;
    }
    int getFrequencyResponseSize() const {
        return kFrequencyResponseSize;
    }
    /** Log-spaced frequency grid corresponding 1:1 to the LUT entries (Hz).
     *  Editor uses this as the bin-mapping target when running an FFT of the
     *  IR — keeps the IR trace on exactly the same x-axis as the analytic
     *  traces so the visualizer doesn't have to interpolate at paint time. */
    const double* getFrequencyResponseFrequenciesHz() const {
        return frequencyResponseHz_.data();
    }
    double getFrequencyResponseMinHz() const {
        return kFrequencyResponseMinHz;
    }
    double getFrequencyResponseMaxHz() const {
        return kFrequencyResponseMaxHz;
    }
    double getFrequencyResponseDbRange() const {
        return kFrequencyResponseDbRange;
    }

    /** Recompute the frequency-response LUT from caller-supplied params.
     *  The editor reads APVTS directly (including modulation math) and feeds
     *  the result here so the trace updates regardless of whether the audio
     *  thread is running. MUST be called from a non-audio thread (allocates
     *  via juce::dsp::IIR::Coefficients::makeXxx for biquad strategies).
     *  Thread-safe vs. concurrent audio-thread writes to the cached params —
     *  this method does not read or write any of the cachedXxx atomics. */
    void recomputeFrequencyResponse(const ToneFrequencyResponseParams& params);

  private:
    LowpassStrategy<2> lp12_;
    LowpassStrategy<4> lp24_;
    LowShelfStrategy lowShelf_;
    HighShelfStrategy highShelf_;
    SmileStrategy smile_;
    BellStrategy bell_;
    HysteresisStrategy hysteresis_;
    HighpassStrategy<2> hp12_;
    HighpassStrategy<4> hp24_;
    ImpulseResponseStrategy ir_;

    std::atomic<bool> enabled_{false};
    std::atomic<Type> type_{Type::Off};

    // Audio-thread cache for change detection so we can reset the newly-
    // activated strategy on the audio thread rather than from setType (which
    // would race concurrent process() reads).
    Type lastType_ = Type::Off;

    /** Returns the strategy for the given type, or nullptr for Type::Off. */
    ToneFilterStrategy* strategyFor(Type t);

    // Cached params for the audio-thread strategies. Each setter writes its
    // corresponding atomic so the strategies' DSP state stays current. The
    // visualization path bypasses these (editor reads APVTS directly) so the
    // trace stays live when the host transport is stopped. Defaults match
    // the per-strategy defaults documented in
    // PluginProcessor / APVTS so the LUT is sensible at construction.
    std::atomic<double> cachedCutoffHz_{3000.0};
    std::atomic<double> cachedResonanceNorm_{0.0};
    std::atomic<double> cachedShelfGainDb_{0.0};
    std::atomic<double> cachedFatPercent_{0.0};
    std::atomic<double> cachedLowShelfRatio_{0.0625};
    std::atomic<double> cachedBellQ_{1.0};
    std::atomic<double> cachedSampleRate_{48000.0};

    // Single-buffered frequency-response LUT. The only writer is the editor's
    // 60 Hz timer (message thread); the only reader is the visualizer's
    // VBlank-driven paint (also message thread). With both on the same thread
    // a single buffer is sufficient, and the published pointer never changes —
    // so the visualizer's overlay registry can keep its captured pointer
    // indefinitely. The version atomic alone signals "data changed" for the
    // overlay's pull-mode path-rebuild check.
    std::array<double, kFrequencyResponseSize> frequencyResponseBuffer_{};
    std::atomic<uint64_t> frequencyResponseVersion_{0};
    std::array<double, kFrequencyResponseSize> frequencyResponseHz_{};

    void initFrequencyResponseFrequencies();
};

} // namespace dsp_core::audio_pipeline
