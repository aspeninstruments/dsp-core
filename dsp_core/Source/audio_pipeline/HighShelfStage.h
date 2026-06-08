#pragma once

#include "AudioProcessingStage.h"
#include <juce_dsp/juce_dsp.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <atomic>
#include <vector>

namespace dsp_core::audio_pipeline {

/**
 * 2nd-order high-shelf filter (RBJ biquad).
 *
 * Boosts or cuts frequencies ABOVE the shelf cutoff while leaving the low
 * band approximately unchanged. Intended as a static tone-shaping control
 * for a parametric EQ — NOT for rapid modulation (RBJ biquads have small
 * zipper-noise risk when coefficients are updated per-sample).
 *
 * Topology: juce::dsp::IIR::Coefficients<double>::makeHighShelf with Q fixed
 * at 1/sqrt(2) (Butterworth slope). Q is intentionally NOT user-controllable
 * to keep the parametric EQ surface simple.
 *
 * Design Rationale:
 *   - RBJ biquad: industry-standard shelf, smooth musical character
 *   - Q = 1/sqrt(2): maximally flat (no resonant overshoot at shelf corner)
 *   - Per-channel filter instances (state vector independence)
 *
 * Pipeline Position: TBD — planned as part of post-waveshaper EQ.
 *
 * Thread Safety:
 *   - enabled_, cutoffHz_, gainDb_: atomic (UI writes, audio reads)
 *   - filters_: per-instance, audio thread only
 *   - Coefficient updates: UI thread only (set...() calls), because
 *     juce::dsp::IIR::Coefficients::makeHighShelf returns a ReferenceCounted
 *     Ptr (shared_ptr — allocates). Mirrors DCBlockingFilter pattern.
 */
class HighShelfStage : public AudioProcessingStage {
  public:
    HighShelfStage() = default;

    // AudioProcessingStage interface
    void prepareToPlay(double sampleRate, int samplesPerBlock, int numChannels) override;
    void process(juce::AudioBuffer<double>& buffer) override;
    void reset() override;
    juce::String getName() const override {
        return "HighShelf";
    }

    // Control interface (thread-safe)
    void setEnabled(bool shouldBeEnabled) {
        enabled_.store(shouldBeEnabled, std::memory_order_release);
    }

    bool isEnabled() const {
        return enabled_.load(std::memory_order_acquire);
    }

    /**
     * Set shelf cutoff frequency in Hz. Clamped to [20, 20000].
     * Thread-safe: stores the target; the audio thread smooths toward it.
     */
    void setCutoffFrequency(double frequencyHz);

    double getCutoffFrequency() const {
        return cutoffHz_.load(std::memory_order_acquire);
    }

    /**
     * Set shelf gain in dB. Clamped to [-24, +24]. 0 dB = transparent.
     * Thread-safe: stores the target; the audio thread smooths toward it.
     */
    void setGainDb(double gainDb);

    double getGainDb() const {
        return gainDb_.load(std::memory_order_acquire);
    }

    // Smoothing ramp time for cutoff/gain. Matches the Ladder filters' 2 ms —
    // long enough that per-sample coefficient updates are click-free, short
    // enough not to smear cutoff/gain modulation. Public for tests.
    static constexpr double kSmoothingTimeSec = 0.002;

  private:
    std::atomic<bool> enabled_{true};
    std::atomic<double> cutoffHz_{8000.0}; // Musically sensible default for a high shelf
    std::atomic<double> gainDb_{0.0};      // Transparent on construction

    // Per-channel IIR filters (audio thread only)
    std::vector<juce::dsp::IIR::Filter<double>> filters_;
    double sampleRate_ = 48000.0;

    // Per-sample parameter smoothing (audio thread only). Cutoff uses
    // Multiplicative (log/pitch) smoothing; gain (dB) uses Linear. The smoothers
    // chase the atomic targets so coefficients move continuously instead of
    // jumping each block — that discontinuity was the click source.
    juce::SmoothedValue<double, juce::ValueSmoothingTypes::Multiplicative> smoothCutoff_;
    juce::SmoothedValue<double, juce::ValueSmoothingTypes::Linear> smoothGainDb_;
    double lastCutoffTarget_ = -1.0;
    double lastGainTarget_ = -1000.0;

    // Allocating coefficient sizer — used only in prepareToPlay and the
    // channel-count-change path. Sizes each filter's coefficient array to a
    // 2nd-order biquad so writeCoefficients() can update it in place afterwards.
    void updateFilterCoefficients();

    // Recompute the high-shelf-biquad coefficients in place from explicit
    // (already-smoothed) parameters. No allocation — audio-thread safe. Matches
    // juce::dsp::IIR::Coefficients<double>::makeHighShelf exactly.
    void writeCoefficients(double cutoffHz, double gainDb);
};

} // namespace dsp_core::audio_pipeline
