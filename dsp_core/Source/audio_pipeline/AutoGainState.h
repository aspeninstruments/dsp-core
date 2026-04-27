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
    // Tuning constants (hardcoded — goal: smooth control signal without
    // audio-rate gain modulation).
    //
    // The detector is an asymmetric RMS follower: a one-pole low-pass on
    // sample² with separate fast-attack / slow-release time constants. RMS
    // (rather than peak) is what removes the per-cycle peak re-attacks that
    // produced the audio-rate ring-mod / zipper artifact in the prior peak-
    // follower design. The fast attack stops the gain from sitting absurdly
    // high through a sudden level jump; the slow release keeps quiet-section
    // recovery from pumping. We deliberately do NOT enforce a strict no-
    // overshoot guarantee — short-lived overshoot is allowed to clip / hit
    // the waveshaper saturation, which sounds far better than chasing every
    // peak with audio-rate gain changes.
    struct Constants {
        double targetPeak = 0.9;             // Now an RMS target. -1 dB headroom on
                                             // sustained tones (peak ≈ target × √2);
                                             // peakier program material will overshoot
                                             // and saturate in the waveshaper.
        double maxGainLinear = 100.0;        // +40 dB cap to prevent noise runaway
        double noiseFloorLinear = 0.001;     // -60 dB: below this, slew gain toward 1
        double rmsAttackTauSeconds = 0.005;  // 5 ms — fast enough to keep step-up
                                             // overshoot bounded without re-introducing
                                             // per-cycle peak reactions
        double rmsReleaseTauSeconds = 0.100; // 100 ms — smooth release on level drops
                                             // (pumping-free)
        double enableFadeSeconds = 0.020;    // 20 ms crossfade on toggle
        // Asymmetric one-pole on the applied gain: instant when gain drops
        // (so the RMS-detected level reduction takes effect immediately on
        // the audio), smoothed when gain rises (kills any residual zipper
        // through the 1/envelope nonlinearity at the noise-floor edge).
        double gainSmoothTauSeconds = 0.002;
    };

    Constants k{};

    // Thread-shared enable flag. UI writes, audio reads.
    std::atomic<bool> enabled{false};

    // Thread-shared target level (linear). UI writes per parameter update; audio
    // reads at the top of each process() block. Driven by InputGain. With the
    // RMS detector, this is the target *RMS* level — actual sample peaks will
    // exceed it for any signal with crest factor > 1.
    std::atomic<double> targetPeakLinear{0.9};

    // Per-sample gain history written by AutoSquash, consumed by AutoRestore.
    // Size matches the maximum block size set in prepareToPlay.
    juce::AudioBuffer<double> gainHistory;

    // RMS detector state (audio thread only) — one-pole on stereo-linked
    // sample² (max(L², R²)).
    double meanSquare = 0.0;

    // Enable crossfade mix: 0.0 = fully bypassed, 1.0 = fully engaged.
    // Ramps on enable/disable to avoid clicks.
    double enableMix = 0.0;

    // Last applied gain after asymmetric smoothing — carried sample-to-sample
    // so the smoother is stateful across blocks.
    double smoothedGain = 1.0;

    // Pre-computed one-pole coefficients set in prepareToPlay.
    double rmsAttackAlpha = 0.0;
    double rmsReleaseAlpha = 0.0;
    double gainSmoothAlpha = 0.0;
    int enableFadeSamples = 1;

    double sampleRate = 44100.0;

    /**
     * Recompute detector / fade coefficients for the current sample rate.
     * Must be called from prepareToPlay on both stages — idempotent.
     *
     * alpha = 1 - exp(-1 / (tau * fs))
     */
    void recomputeCoefficients(double newSampleRate) {
        sampleRate = newSampleRate;
        const double fs = sampleRate > 0.0 ? sampleRate : 44100.0;
        rmsAttackAlpha = 1.0 - std::exp(-1.0 / (k.rmsAttackTauSeconds * fs));
        rmsReleaseAlpha = 1.0 - std::exp(-1.0 / (k.rmsReleaseTauSeconds * fs));
        gainSmoothAlpha = 1.0 - std::exp(-1.0 / (k.gainSmoothTauSeconds * fs));
        enableFadeSamples = std::max(1, static_cast<int>(k.enableFadeSeconds * fs));
    }

    /**
     * Zero the run-time state. Called on reset. Does NOT touch enabled (that
     * is owned by the UI thread).
     */
    void resetRuntime() {
        meanSquare = 0.0;
        enableMix = enabled.load(std::memory_order_acquire) ? 1.0 : 0.0;
        smoothedGain = 1.0;
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
        meanSquare = 0.0;
        smoothedGain = 1.0;
    }
};

} // namespace dsp_core::audio_pipeline
