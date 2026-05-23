#pragma once

#include "AudioProcessingStage.h"
#include <juce_dsp/juce_dsp.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <atomic>

namespace dsp_core::audio_pipeline {

/**
 * 2nd-order parametric lowpass filter (-12 dB/octave).
 *
 * Uses a Topology-Preserving Transform / Zero-Delay Feedback (TPT/ZDF) state
 * variable structure, which is designed for smooth response to modulated
 * cutoff and resonance — coefficient updates are cheap and artifact-free,
 * unlike RBJ biquads where rapid coefficient changes produce zipper noise.
 *
 * Design Rationale:
 *   - TPT/ZDF: future-proof for cutoff/Q modulation (LFO, envelope follower)
 *   - 2nd-order: musical 12 dB/oct slope, single-sample latency
 *   - Resonance up to Q=10 enables filter-sweep character without runaway
 *
 * Pipeline Position: TBD (deferred). Likely post-waveshaper to tame harmonics.
 *
 * Thread Safety:
 *   - enabled_, cutoffHz_, resonance_: atomic (UI writes, audio reads)
 *   - filter_: audio thread only. UI setters write atomics; audio thread reads
 *     them at the top of each process() block.
 */
class LowpassStage : public AudioProcessingStage {
  public:
    LowpassStage() = default;

    // AudioProcessingStage interface
    void prepareToPlay(double sampleRate, int samplesPerBlock, int numChannels) override;
    void process(juce::AudioBuffer<double>& buffer) override;
    void reset() override;
    juce::String getName() const override {
        return "Lowpass";
    }

    // Control interface (thread-safe)
    void setEnabled(bool shouldBeEnabled) {
        enabled_.store(shouldBeEnabled, std::memory_order_release);
    }

    bool isEnabled() const {
        return enabled_.load(std::memory_order_acquire);
    }

    /**
     * Set lowpass cutoff frequency in Hz.
     * Clamped to [20 Hz, 0.45 * sampleRate]. The Nyquist margin guards against
     * the TPT prewarp tan(pi*fc/fs) approaching infinity.
     */
    void setCutoffFrequency(double frequencyHz);

    double getCutoffFrequency() const {
        return cutoffHz_.load(std::memory_order_acquire);
    }

    /**
     * Set resonance (JUCE-native Q). 1/sqrt(2) ~= 0.7071 is Butterworth (flat).
     * Clamped to [0.5, 10.0].
     */
    void setResonance(double q);

    double getResonance() const {
        return resonance_.load(std::memory_order_acquire);
    }

  private:
    std::atomic<bool> enabled_{true};
    std::atomic<double> cutoffHz_{20000.0};
    std::atomic<double> resonance_{0.7071067811865476}; // 1/sqrt(2), Butterworth

    juce::dsp::StateVariableTPTFilter<double> filter_;
    double sampleRate_ = 48000.0;
};

} // namespace dsp_core::audio_pipeline
