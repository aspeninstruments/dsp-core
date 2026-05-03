#pragma once

#include <juce_audio_basics/juce_audio_basics.h>

namespace dsp_core::audio_pipeline {

/**
 * Base interface for audio processing stages in a serial pipeline.
 *
 * Design principles:
 * - Processes audio buffer in-place (modify input buffer)
 * - MUST be real-time safe (no allocations, no locks in process())
 * - Stateless or manages internal state (filters, delays, etc.)
 * - Composable (can be chained in any order)
 *
 * Thread safety:
 * - prepareToPlay() called from UI thread
 * - process() called from audio thread
 * - reset() can be called from either thread (use atomics if stateful)
 */
class AudioProcessingStage {
  public:
    virtual ~AudioProcessingStage() = default;

    /**
     * Prepare for playback.
     * Called when audio engine starts or sample rate changes.
     *
     * @param sampleRate The sample rate (may be oversampled rate)
     * @param samplesPerBlock Maximum expected buffer size (worst case)
     * @param numChannels Max main-bus channel count expected at process()
     */
    virtual void prepareToPlay(double sampleRate, int samplesPerBlock, int numChannels) = 0;

    /**
     * Process audio buffer in-place.
     * MUST be real-time safe (no allocations, no locks).
     *
     * @param buffer Audio buffer to process (modified in-place)
     */
    virtual void process(juce::AudioBuffer<double>& buffer) = 0;

    /**
     * Reset internal state (e.g., filter coefficients, delay lines).
     * Called when audio stops or parameters change dramatically.
     */
    virtual void reset() = 0;

    /**
     * Get stage name for debugging/profiling.
     */
    virtual juce::String getName() const = 0;

    /**
     * Get latency introduced by this stage (in samples).
     * Default: 0 (no latency)
     */
    virtual int getLatencySamples() const {
        return 0;
    }
};

} // namespace dsp_core::audio_pipeline
