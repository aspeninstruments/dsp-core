#pragma once

#include <juce_audio_basics/juce_audio_basics.h>
#include <atomic>
#include <cmath>

namespace dsp_core::audio_pipeline {

/**
 * Shared state between AutoSquashStage and AutoRestoreStage.
 *
 * Owned by the plugin processor; both stages hold a reference. The squash stage
 * writes the per-sample gain applied; the restore stage reads that same block
 * and applies its exact reciprocal. Because the two stages bracket the
 * waveshaper within a single process() chain (on the audio thread, serially),
 * no synchronization is required on gainHistory_.
 *
 * enabled_ is atomic because the UI thread writes it.
 */
struct AutoGainState {
    // Tuning constants (hardcoded — goal: fast but transparent).
    //
    // The detector is a classic peak follower: instant attack (any sample
    // whose magnitude exceeds the envelope snaps the envelope up to it) plus
    // smooth, held release. Instant attack is necessary because we have no
    // lookahead — any smoothed attack would let the first sample of a sudden
    // transient through at the previous (too-high) gain, causing large
    // internal overshoot. Opto-style release preserves transparency.
    struct Constants {
        double targetPeak = 0.9;          // -1 dB headroom
        double maxGainLinear = 100.0;     // +40 dB cap to prevent noise runaway
        double noiseFloorLinear = 0.001;  // -60 dB: below this, slew gain toward 1
        double releaseTauSeconds = 0.120; // 120 ms — opto-style release
        double holdSeconds = 0.010;       // 10 ms hold before release can start
        double enableFadeSeconds = 0.020; // 20 ms crossfade on toggle
    };

    Constants k{};

    // Thread-shared enable flag. UI writes, audio reads.
    std::atomic<bool> enabled{false};

    // Thread-shared peak target (linear). UI writes per parameter update; audio
    // reads at the top of each process() block. Driven by InputGain so that the
    // user's input-gain knob effectively sets the post-squash peak (with a small
    // headroom below it to swallow envelope wiggle without clipping).
    std::atomic<double> targetPeakLinear{0.9};

    // Per-sample gain history written by AutoSquash, consumed by AutoRestore.
    // Size matches the maximum block size set in prepareToPlay.
    juce::AudioBuffer<double> gainHistory;

    // Envelope-follower state (audio thread only).
    double envelope = 0.0; // peak envelope
    int holdCounter = 0;   // samples remaining in hold before release kicks in

    // Enable crossfade mix: 0.0 = fully bypassed, 1.0 = fully engaged.
    // Ramps on enable/disable to avoid clicks.
    double enableMix = 0.0;

    // Pre-computed one-pole release coefficient set in prepareToPlay.
    // Attack is instantaneous (peak follower) so no attack alpha is needed.
    double releaseAlpha = 0.0;
    int holdSamples = 0;
    int enableFadeSamples = 1;

    double sampleRate = 44100.0;

    /**
     * Recompute release / hold / fade lengths for the current sample rate.
     * Must be called from prepareToPlay on both stages — it is idempotent so
     * calling it twice is fine.
     *
     * alpha = 1 - exp(-1 / (tau * fs))
     */
    void recomputeCoefficients(double newSampleRate) {
        sampleRate = newSampleRate;
        const double fs = sampleRate > 0.0 ? sampleRate : 44100.0;
        releaseAlpha = 1.0 - std::exp(-1.0 / (k.releaseTauSeconds * fs));
        holdSamples = static_cast<int>(k.holdSeconds * fs);
        enableFadeSamples = std::max(1, static_cast<int>(k.enableFadeSeconds * fs));
    }

    /**
     * Zero the run-time state (envelope, hold, enable mix). Called on reset.
     * Does NOT touch enabled (that is owned by the UI thread).
     */
    void resetRuntime() {
        envelope = 0.0;
        holdCounter = 0;
        enableMix = enabled.load(std::memory_order_acquire) ? 1.0 : 0.0;
        gainHistory.clear();
    }

    /**
     * Ensure gainHistory is allocated for the given block size. Called from
     * prepareToPlay (UI thread) — realtime-safe because we never allocate in
     * process().
     */
    void prepare(double newSampleRate, int samplesPerBlock) {
        recomputeCoefficients(newSampleRate);
        gainHistory.setSize(1, samplesPerBlock, false, true, true);
        gainHistory.clear();
        // Align the enable crossfade to the steady-state of the current flag,
        // so the very first block after prepareToPlay is consistent.
        enableMix = enabled.load(std::memory_order_acquire) ? 1.0 : 0.0;
        envelope = 0.0;
        holdCounter = 0;
    }
};

} // namespace dsp_core::audio_pipeline
