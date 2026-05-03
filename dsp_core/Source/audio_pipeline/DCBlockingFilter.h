#pragma once

#include "AudioProcessingStage.h"
#include <juce_dsp/juce_dsp.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <atomic>
#include <vector>

namespace dsp_core::audio_pipeline {

/**
 * DC blocking filter for removing DC offset from audio signals.
 *
 * Problem: When transfer functions map silence (x=0) to non-zero output, the
 * audio signal contains DC offset which can cause speaker damage, headroom loss,
 * and subsonic cone excursion.
 *
 * Solution: 1st-order Butterworth highpass filter at 5Hz (industry standard, inaudible).
 *
 * Design Rationale:
 *   - 5Hz cutoff: Industry precedent (UAD, Waves, FabFilter)
 *   - Butterworth: Maximally flat passband (no ripple)
 *   - 1st-order: -6dB/octave rolloff, minimal phase shift
 *   - Inaudible: 5Hz is below typical monitor range (20-20kHz)
 *   - Preserves harmonics: No interaction with musical content
 *
 * Pipeline Position: After WaveshapingStage, before output gain.
 *
 * Thread Safety:
 *   - enabled_: atomic (UI writes, audio reads)
 *   - cutoffFrequency_: atomic (UI writes, audio reads)
 *   - Filter state: per-instance (audio thread only)
 *   - Coefficient updates: UI thread only (prepareToPlay/setCutoffFrequency)
 *
 * CPU Cost: Minimal (~0.1% overhead per channel)
 */
class DCBlockingFilter : public AudioProcessingStage {
  public:
    DCBlockingFilter() = default;

    // AudioProcessingStage interface
    void prepareToPlay(double sampleRate, int samplesPerBlock, int numChannels) override;
    void process(juce::AudioBuffer<double>& buffer) override;
    void reset() override;
    juce::String getName() const override {
        return "DCBlockingFilter";
    }

    // Control interface (thread-safe)
    void setEnabled(bool shouldBeEnabled) {
        enabled_.store(shouldBeEnabled, std::memory_order_release);
    }

    bool isEnabled() const {
        return enabled_.load(std::memory_order_acquire);
    }

    /**
     * Set DC blocking filter cutoff frequency.
     *
     * @param frequencyHz Cutoff frequency in Hz (clamped to 1-20Hz range)
     *
     * Thread Safety: UI thread only (updates filter coefficients)
     */
    void setCutoffFrequency(double frequencyHz);

    /**
     * Get current cutoff frequency.
     */
    double getCutoffFrequency() const {
        return cutoffFrequency_.load(std::memory_order_acquire);
    }

  private:
    std::atomic<bool> enabled_{true};          // Default: ON (safety feature)
    std::atomic<double> cutoffFrequency_{5.0}; // Hz

    // Per-channel IIR filters (audio thread only)
    std::vector<juce::dsp::IIR::Filter<double>> filters_;
    double sampleRate_ = 48000.0;

    // Update all filter coefficients (call after sampleRate or cutoff changes)
    void updateFilterCoefficients();
};

} // namespace dsp_core::audio_pipeline
