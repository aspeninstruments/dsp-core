#pragma once

#include "LaneMixer.h"
#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <juce_core/juce_core.h>
#include <juce_audio_basics/juce_audio_basics.h>

namespace dsp_core {

// Forward declarations
class EventDrivenRenderer;
class VisualizerUpdateTimer;

/**
 * SeamlessConfig - Configuration constants for seamless transfer function updates
 */
struct SeamlessConfig {
    static constexpr int DSP_LUT_SIZE = 16384;
    static constexpr int VISUALIZER_LUT_SIZE = 1024;
    static constexpr double MIN_VALUE = -1.0;
    static constexpr double MAX_VALUE = 1.0;
    static constexpr double CROSSFADE_DURATION_MS = 5.0;
    static constexpr int DSP_TIMER_HZ = 20;
    static constexpr int VISUALIZER_TIMER_HZ = 120;
};

// Legacy aliases for backward compatibility
static constexpr int TABLE_SIZE = SeamlessConfig::DSP_LUT_SIZE;
static constexpr int VISUALIZER_LUT_SIZE = SeamlessConfig::VISUALIZER_LUT_SIZE;
static constexpr double MIN_VALUE = SeamlessConfig::MIN_VALUE;
static constexpr double MAX_VALUE = SeamlessConfig::MAX_VALUE;

// Buffer roles for triple-buffered LUT system
enum class BufferRole {
    Primary = 0,      // Active LUT for playback (audio thread reads)
    Secondary = 1,    // Previous LUT used during crossfade (audio thread reads)
    WorkerTarget = 2  // Worker thread writes here (isolated from audio)
};

// LUTBuffer - Triple-buffered lookup table
struct LUTBuffer {
    std::array<double, TABLE_SIZE> data;
    uint64_t version{0};
    LaneMixer::ExtrapolationMode extrapolationMode{
        LaneMixer::ExtrapolationMode::Clamp};
};

/**
 * AudioEngine - Audio thread component for seamless transfer function updates
 *
 * Architecture:
 *   - Triple buffering prevents data race during crossfade
 *   - lutBuffers[0,1]: Used for crossfading (audio thread reads)
 *   - lutBuffers[2]: Worker thread writes here (safe from audio thread)
 *   - 10ms linear crossfade (sample-rate adaptive)
 *
 * Thread Safety:
 *   - Audio thread: reads primaryIndex, secondaryIndex, checks newLUTReady
 *   - Worker thread: writes to lutBuffers[workerTargetIndex], sets newLUTReady
 *   - No locks, all communication via atomics
 *
 * Memory Ordering:
 *   - acquire/release for index swaps (ensures LUT data visibility)
 *   - relaxed for worker target (worker thread only)
 */
class AudioEngine {
  public:
    AudioEngine();

    /**
     * Prepare for playback (audio thread)
     *
     * Calculates sample-rate-adaptive crossfade duration.
     * Called from audio thread in prepareToPlay().
     *
     * @param sampleRate Sample rate in Hz
     * @param samplesPerBlock Maximum block size
     */
    void prepareToPlay(double sampleRate, int samplesPerBlock);

    /**
     * Apply transfer function to single sample (audio thread)
     *
     * Uses active LUT or crossfades between old and new LUT.
     *
     * @param x Input sample
     * @return Output sample
     */
    double applyTransferFunction(double x) const;

    /**
     * Process multi-channel buffer in-place (audio thread)
     *
     * Processes all channels with shared crossfade state.
     * Checks for new LUT once at start, then processes all channels.
     * Crossfade position advances correctly (once per sample, not per channel).
     *
     * @param buffer Multi-channel audio buffer (modified in-place)
     */
    void processBuffer(juce::AudioBuffer<double>& buffer) const;

    /**
     * Get reference to worker target index (for worker thread)
     *
     * Worker thread loads this index to know where to write.
     *
     * @return Atomic reference to worker target index
     */
    std::atomic<int>& getWorkerTargetIndexReference() {
        return workerTargetIndex;
    }

    /**
     * Get reference to new LUT ready flag (for worker thread)
     *
     * Worker thread sets this to true after rendering LUT.
     *
     * @return Atomic reference to ready flag
     */
    std::atomic<bool>& getNewLUTReadyFlag() {
        return newLUTReady;
    }

    /**
     * Get LUT buffers pointer (for worker thread writes)
     *
     * Worker thread writes to lutBuffers[workerTargetIndex].
     *
     * @return Pointer to LUT buffer array
     */
    LUTBuffer* getLUTBuffers() {
        return lutBuffers;
    }

    /**
     * Check if audio engine is currently crossfading (for worker thread)
     *
     * Worker thread uses this to decide whether to render DSP LUT:
     * - If crossfading: Skip DSP render (audio thread can't accept new LUT)
     * - If not crossfading: Render DSP LUT (audio thread ready)
     *
     * This implements the two-speed worker strategy:
     * - Always render visualizer LUT (2K samples, ~2ms)
     * - Only render DSP LUT when audio thread can accept it
     *
     * @return true if crossfade in progress, false otherwise
     */
    bool isCrossfading() const {
        return crossfading;
    }

  private:
    /**
     * Check for new LUT from worker thread (audio thread)
     *
     * Called once per processBlock(). If new LUT is ready:
     *   1. Aborts any active crossfade
     *   2. Rotates buffer indices (worker → primary, primary → secondary, secondary → worker)
     *   3. Starts new crossfade from old primary to new primary
     *
     * Uses acquire memory ordering to ensure worker's LUT data is visible.
     */
    void checkForNewLUT() const;

    /**
     * Evaluate LUT with Catmull-Rom interpolation
     *
     * @param lut LUT buffer to evaluate
     * @param x Input value
     * @return Interpolated output value
     */
    double evaluateLUT(const LUTBuffer* lut, double x) const;

    /**
     * Evaluate crossfade between two LUTs (OPTIMIZED)
     *
     * CRITICAL OPTIMIZATION: Mix table values BEFORE interpolation, not after.
     * - Old approach: interpolate(oldLUT) + interpolate(newLUT) = 2 interpolations
     * - New approach: interpolate(mix(oldLUT, newLUT)) = 1 interpolation
     *
     * Both LUTs share the same fractional index, so we can:
     * 1. Map x to fractional index once (j + delta)
     * 2. Fetch 4 samples from each LUT at same positions
     * 3. Mix corresponding samples: mixed[i] = gainOld * old[i] + gainNew * new[i]
     * 4. Do ONE Catmull-Rom interpolation on mixed samples
     *
     * Saves one complex polynomial evaluation per sample during crossfade.
     *
     * @param oldLUT Old LUT buffer
     * @param newLUT New LUT buffer
     * @param x Input value
     * @param gainOld Gain for old LUT [0, 1]
     * @param gainNew Gain for new LUT [0, 1]
     * @return Crossfaded output value
     */
    double evaluateCrossfade(const LUTBuffer* oldLUT, const LUTBuffer* newLUT,
                            double x, double gainOld, double gainNew) const;

    /**
     * Catmull-Rom interpolation on 4 pre-fetched samples
     *
     * @param y0 Sample at index-1
     * @param y1 Sample at index
     * @param y2 Sample at index+1
     * @param y3 Sample at index+2
     * @param t Fractional position [0, 1]
     * @return Interpolated value
     */
    static double interpolateCatmullRom(double y0, double y1, double y2, double y3, double t);

    // TRIPLE BUFFERING (prevents data race during crossfade):
    // - lutBuffers[0,1]: Used for crossfading (audio thread reads)
    // - lutBuffers[2]: Worker thread writes here (safe from audio thread)
    mutable LUTBuffer lutBuffers[3];  // mutable: blend snapshot writes in checkForNewLUT()

    // Atomics are mutable because they're modified in const methods (thread-safe state)
    mutable std::atomic<int> primaryIndex{static_cast<int>(BufferRole::Primary)};      // Active LUT for playback
    mutable std::atomic<int> secondaryIndex{static_cast<int>(BufferRole::Secondary)};    // Previous LUT (used during crossfade)
    mutable std::atomic<int> workerTargetIndex{static_cast<int>(BufferRole::WorkerTarget)}; // Worker writes here
    mutable std::atomic<bool> newLUTReady{false};

    // Crossfade state (audio thread local - mutable for const methods)
    double sampleRate{44100.0};
    mutable int crossfadeSamples{441};  // Recalculated in prepareToPlay()
    mutable int crossfadePosition{0};
    mutable std::atomic<bool> crossfading{false};  // Atomic for worker thread reads
    mutable const LUTBuffer* oldLUT{nullptr};
    mutable const LUTBuffer* newLUT{nullptr};
};

/**
 * EventDrivenRenderer - Event-driven DSP LUT renderer with rate limiting
 *
 * Replaces LUTRenderTimer + LUTRendererThread with a single class that:
 *   - Triggers renders immediately via AsyncUpdater (no polling delay)
 *   - Rate-limits expensive curve changes (60Hz max)
 *   - Renders mix-only changes (amplitude, scan) without rate limiting
 *   - Writes directly to the triple-buffered LUT (no worker thread needed)
 *   - Falls back to a 5Hz safety timer as a guaranteed delivery net
 *
 * THREADING:
 *   - triggerAsyncUpdate() called from any thread (message or audio via AutomationSlot)
 *   - handleAsyncUpdate() runs on message thread (JUCE guarantee)
 *   - timerCallback() runs on message thread (JUCE Timer contract)
 *   - doRender() writes to triple buffer atomics (safe from message thread)
 *
 * Two-Tier Version Tracking:
 *   - Full version (versionCounter_): incremented on ANY change
 *   - Mix version (mixVersionCounter_): incremented only for amplitude/scan changes
 *   - If only mix version changed: render immediately (cheap computeSum)
 *   - If curve content changed: rate-limited to 60Hz
 */
class EventDrivenRenderer : public juce::AsyncUpdater, public juce::Timer {
  public:
    EventDrivenRenderer(LaneMixer& mixer, AudioEngine& engine);
    ~EventDrivenRenderer() override;

    /** Force synchronous render (for initialization and testing) */
    void forceRender();

    /** AsyncUpdater callback — dispatched on message thread */
    void handleAsyncUpdate() override;

    /** Safety timer callback (5Hz fallback) */
    void timerCallback() override;

  private:
    void doRender();

    LaneMixer& laneMixer;
    AudioEngine& audioEngine;

    uint64_t lastRenderedFullVersion{0};
    uint64_t lastRenderedMixVersion{0};
    double lastRenderedScanPosition{0.0};
    double lastCurveRenderTimeMs{0.0};

    static constexpr double CURVE_RENDER_MIN_INTERVAL_MS = 16.7;  // 60Hz max
    static constexpr int SAFETY_TIMER_HZ = 5;                      // 200ms fallback
};

/**
 * VisualizerUpdateTimer - Fast timer for direct model sampling (60Hz)
 *
 * Separate from DSP LUT rendering to decouple visualizer responsiveness
 * from audio update frequency. Samples the editing model directly on the
 * message thread without using the worker thread.
 *
 * Threading:
 *   - MUST run on message thread (JUCE Timer contract)
 *   - Reads from editingModel directly (safe - message thread only)
 *   - Writes to visualizerLUT directly (safe - message thread only)
 *
 * Performance:
 *   - 60Hz update rate for smooth UI
 *   - ~0.5ms per update (1024 points sampled)
 *   - No worker thread overhead
 */
class VisualizerUpdateTimer : public juce::Timer {
  public:
    /**
     * Construct visualizer timer (message thread only)
     *
     * @param mixer Reference to lane mixer (primary data source for sum visualization)
     */
    explicit VisualizerUpdateTimer(LaneMixer& mixer);

    /**
     * Destructor - stops timer
     */
    ~VisualizerUpdateTimer() override;

    /**
     * Set visualizer target buffer and callback
     *
     * @param lutPtr Pointer to visualizer LUT buffer
     * @param callback Callback to invoke after update
     */
    void setVisualizerTarget(std::array<double, VISUALIZER_LUT_SIZE>* lutPtr,
                             std::function<void()> callback);

    /**
     * Timer callback - samples model and updates visualizer (60Hz)
     */
    void timerCallback() override;

    /**
     * Force immediate update (for initialization)
     */
    void forceUpdate();

    /**
     * Set lane LUT target buffer and selected lane pointer
     *
     * @param lutPtr Pointer to lane LUT buffer (1024 samples)
     * @param selectedLanePtr Pointer to selected lane index (-1 = none)
     */
    void setLaneLUTTarget(std::array<double, VISUALIZER_LUT_SIZE>* lutPtr, int* selectedLanePtr);

  private:
    LaneMixer& laneMixer;
    std::array<double, VISUALIZER_LUT_SIZE>* visualizerLUTPtr{nullptr};
    std::function<void()> onVisualizerUpdate;
    uint64_t lastSeenVersion{0};

    std::array<double, VISUALIZER_LUT_SIZE>* laneLUTPtr_{nullptr};
    int* selectedLanePtr_{nullptr};
};

} // namespace dsp_core
