#pragma once

#include "AudioProcessingStage.h"
#include <juce_dsp/juce_dsp.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <atomic>
#include <vector>

namespace dsp_core::audio_pipeline {

/**
 * 2nd-order low-shelf filter (RBJ biquad).
 *
 * Boosts or cuts frequencies BELOW the shelf cutoff while leaving the high
 * band approximately unchanged. Intended as a static tone-shaping control
 * for a parametric EQ — NOT for rapid modulation (RBJ biquads have small
 * zipper-noise risk when coefficients are updated per-sample).
 *
 * Topology: juce::dsp::IIR::Coefficients<double>::makeLowShelf with Q fixed
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
 *     juce::dsp::IIR::Coefficients::makeLowShelf returns a ReferenceCounted
 *     Ptr (shared_ptr — allocates). Mirrors DCBlockingFilter pattern.
 */
class LowShelfStage : public AudioProcessingStage {
  public:
    LowShelfStage() = default;

    // AudioProcessingStage interface
    void prepareToPlay(double sampleRate, int samplesPerBlock, int numChannels) override;
    void process(juce::AudioBuffer<double>& buffer) override;
    void reset() override;
    juce::String getName() const override {
        return "LowShelf";
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
     * Thread Safety: UI thread only (allocates new coefficients).
     */
    void setCutoffFrequency(double frequencyHz);

    double getCutoffFrequency() const {
        return cutoffHz_.load(std::memory_order_acquire);
    }

    /**
     * Set shelf gain in dB. Clamped to [-24, +24]. 0 dB = transparent.
     * Thread Safety: UI thread only (allocates new coefficients).
     */
    void setGainDb(double gainDb);

    double getGainDb() const {
        return gainDb_.load(std::memory_order_acquire);
    }

  private:
    std::atomic<bool> enabled_{true};
    std::atomic<double> cutoffHz_{100.0}; // Musically sensible default for a low shelf
    std::atomic<double> gainDb_{0.0};     // Transparent on construction

    // Per-channel IIR filters (audio thread only)
    std::vector<juce::dsp::IIR::Filter<double>> filters_;
    double sampleRate_ = 48000.0;

    // Update all filter coefficients (call after sampleRate, cutoff, or gain changes).
    // UI thread only — allocates via makeLowShelf.
    void updateFilterCoefficients();
};

} // namespace dsp_core::audio_pipeline
