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
    // Tuning constants (hardcoded — goal: bound internal peak at target in
    // steady state while keeping the gain trajectory smooth enough to avoid
    // audio-rate ring modulation).
    //
    // The detector is a peak follower with *smoothed* attack: an asymmetric
    // one-pole that chases sample magnitude on rises (attackTauSeconds) and
    // exponentially decays on falls (after holdSeconds). Smoothed attack
    // trades a brief overshoot above target on big level steps (saturated by
    // the waveshaper — clipping for a few ms is fine) for a non-instantaneous
    // envelope, which collapses per-sample gain steps and kills the zipper
    // / ring-mod artifact that an instantaneous attack produced. Steady-state
    // peak tracking is preserved by the long hold: each cycle peak refreshes
    // the hold window, so the envelope rides at peak between cycles instead
    // of releasing.
    //
    // Smoothness is further reinforced on the *gain-rise* side (release
    // direction) through gainSmoothTauSeconds.
    struct Constants {
        double targetPeak = 0.9;             // -1 dB headroom in steady state
        double maxGainLinear = 100.0;        // +40 dB cap to prevent noise runaway
        double noiseFloorLinear = 0.001;     // -60 dB: below this, slew gain toward 1
        double attackTauSeconds = 0.0005;    // 0.5 ms — short enough that big
                                             // level jumps overshoot for ~1–2 ms
                                             // (waveshaper clips), but long
                                             // enough to spread per-sample env
                                             // step on small level changes
        double releaseTauSeconds = 0.200;    // 200 ms — opto-style release for a
                                             // gentle "breathing" recovery
        double holdSeconds = 0.040;          // 40 ms — long enough to bridge
                                             // typical percussive gaps so the
                                             // envelope doesn't release-and-re-
                                             // attack between consecutive bursts
        double enableFadeSeconds = 0.020;    // 20 ms crossfade on toggle
        // Asymmetric one-pole on the applied gain: instant when gain drops
        // (preserves headroom on attacks — the smoothed attack already gives
        // it a 0.5 ms ramp), smoothed when gain rises (further attenuates
        // per-sample zipper through the 1/envelope nonlinearity during
        // release-side recovery).
        double gainSmoothTauSeconds = 0.002;
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

    // Last applied gain after asymmetric smoothing — carried sample-to-sample
    // so the smoother is stateful across blocks.
    double smoothedGain = 1.0;

    // Pre-computed one-pole coefficients set in prepareToPlay.
    double attackAlpha = 0.0;
    double releaseAlpha = 0.0;
    // One-pole coefficient for the applied-gain rise smoother.
    double gainSmoothAlpha = 0.0;
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
        attackAlpha = 1.0 - std::exp(-1.0 / (k.attackTauSeconds * fs));
        releaseAlpha = 1.0 - std::exp(-1.0 / (k.releaseTauSeconds * fs));
        gainSmoothAlpha = 1.0 - std::exp(-1.0 / (k.gainSmoothTauSeconds * fs));
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
        envelope = 0.0;
        holdCounter = 0;
        smoothedGain = 1.0;
    }
};

} // namespace dsp_core::audio_pipeline
