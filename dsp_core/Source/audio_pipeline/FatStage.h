#pragma once

#include "AudioProcessingStage.h"
#include "LowShelfStage.h"
#include <atomic>

namespace dsp_core::audio_pipeline {

/**
 * "Fat" pre-filter low-shelf, intended to sit immediately before a ladder
 * (or other resonant-feedback) filter to compensate for the low-band loss
 * the ladder introduces as resonance rises.
 *
 * Design Rationale:
 *   A reviewer suggested inline make-up gain like x *= (1 + R) inside the
 *   ladder. That was rejected: program material entering the filter is much
 *   more dynamic than a static VCO waveform, so a single make-up factor
 *   either over- or under-compensates depending on input level. Exposing
 *   bass restoration as a user-facing low-shelf is more honest — it's
 *   program-dependent EQ, not a magic correction — and decouples it from R.
 *
 * Topology:
 *   Composes a LowShelfStage with hardcoded cutoff (200 Hz) and Q (1/sqrt(2),
 *   Butterworth — no resonant overshoot at the corner). Only the boost
 *   amount is user-controllable, mapped from the "Fat" percent knob.
 *
 *   fat=0   → 0 dB shelf gain → mathematically identity (transparent)
 *   fat=100 → +6 dB shelf gain (the cap)
 *   Mapping is linear: gain_dB = (fat / 100) * 6.
 *
 * Pipeline Position: pre-filter (before LadderTPT12dBStage,
 * LadderTPT24dBStage, VirtualAnalogFilterStage). Wiring is handled by the
 * pipeline builder downstream of this stage's introduction.
 *
 * Thread Safety:
 *   - enabled_, fat_: atomic (UI writes, audio reads).
 *   - Coefficient updates run through LowShelfStage::setGainDb() on the UI
 *     thread (allocates via makeLowShelf — same constraint as LowShelfStage
 *     and DCBlockingFilter). No per-sample smoothing in this stage; the
 *     LowShelfStage IIR state carries through, so abrupt knob jumps will
 *     produce small transients but slow user movement is click-free.
 */
class FatStage : public AudioProcessingStage {
  public:
    FatStage() = default;

    // AudioProcessingStage interface
    void prepareToPlay(double sampleRate, int samplesPerBlock, int numChannels) override;
    void process(juce::AudioBuffer<double>& buffer) override;
    void reset() override;
    juce::String getName() const override {
        return "Fat";
    }

    // Control interface (thread-safe)
    void setEnabled(bool shouldBeEnabled) {
        enabled_.store(shouldBeEnabled, std::memory_order_release);
    }

    bool isEnabled() const {
        return enabled_.load(std::memory_order_acquire);
    }

    /**
     * Set Fat amount in percent. Clamped to [0, 100].
     * 0   → transparent (0 dB shelf gain).
     * 100 → +6 dB low-shelf boost.
     * Thread Safety: UI thread only (delegates to LowShelfStage::setGainDb,
     * which allocates new biquad coefficients).
     */
    void setFat(double percent);

    double getFat() const {
        return fat_.load(std::memory_order_acquire);
    }

  private:
    std::atomic<bool> enabled_{true};
    std::atomic<double> fat_{0.0}; // Transparent on construction

    // Audio thread reads, UI thread writes via setGainDb / setCutoffFrequency.
    LowShelfStage lowShelf_;
};

} // namespace dsp_core::audio_pipeline
