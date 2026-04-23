#pragma once

#include "LaneMixer.h"
#include <array>
#include <functional>
#include <memory>

namespace dsp_core {

/**
 * SeamlessTransferFunction - Reusable click-free transfer function with event-driven LUT rendering
 *
 * GOAL: Eliminate clicks/pops during ALL transfer function edits while maintaining
 *       <10ms latency for automation and interactive operations.
 *
 * ARCHITECTURE: Event-driven two-thread design
 *   - Message thread: UI/controller mutates LaneMixer, onVersionChanged triggers render
 *   - Audio thread: Triple-buffered LUT with 5ms smoothstep crossfade (sample-rate-adaptive)
 *
 * KEY DESIGN DECISIONS:
 *   - Event-driven via AsyncUpdater (no polling delay — renders within 1ms of mutation)
 *   - Two-tier version counter: mix-only changes (amplitude, scan) render immediately,
 *     curve content changes rate-limited to 60Hz
 *   - Direct triple-buffer writes from message thread (no worker thread needed)
 *   - 5ms crossfade with interruption support for rapid automation
 *   - Hardcoded constants (16384 table size, -1.0 to 1.0 range) for performance
 *   - Visualizer shows latest model directly at 120Hz (independent of DSP rendering)
 *
 * PERFORMANCE:
 *   - Latency: ~7-10ms for amplitude/scan automation, ~20-28ms for curve edits
 *   - CPU overhead: <0.3% (event-driven + 5Hz safety timer)
 *   - Memory overhead: ~400KB (393KB triple-buffered LUTs)
 *
 * REUSABILITY:
 *   - Renders the weighted sum of LaneMixer lanes into a production LUT
 *   - Can be reused across multiple Aspen Instruments plugins
 *   - No plugin-specific dependencies
 */
class SeamlessTransferFunction {
  public:
    // Constants (hardcoded for performance - not configurable)
    static constexpr int TABLE_SIZE = 16384;          // DSP LUT size (audio thread)
    static constexpr int VISUALIZER_LUT_SIZE = 1024;  // Visualizer LUT size (UI thread)
    static constexpr double MIN_VALUE = -1.0;
    static constexpr double MAX_VALUE = 1.0;

    /**
     * INITIALIZATION SEQUENCE (REQUIRED ORDER):
     *
     * 1. Construct SeamlessTransferFunction
     *    - LaneMixer initialized to defaults (H1=1.0 → y=x)
     *    - AudioEngine initialized to identity LUTs
     *    - Worker/poller NOT started yet
     *
     * 2. Create controller: new CurveEditorController(stf.getLaneMixer(), ...)
     *    - Controller can now mutate lane mixer
     *    - Still no async updates happening
     *
     * 3. Call startSeamlessUpdates() (message thread)
     *    - Starts worker thread and 25Hz poller
     *    - Triggers initial LUT render for visualizer
     *    - From this point, edits trigger async LUT renders
     *    - MUST be called AFTER controller creation
     *
     * 4. Call prepareToPlay(sampleRate, samplesPerBlock) (audio thread)
     *    - Sets up crossfade duration based on sample rate
     *    - Can be called before or after startSeamlessUpdates()
     *
     * 5. Normal operation: process audio, make edits
     *
     * 6. Call releaseResources() before destruction
     *    - Stops worker thread cleanly
     */
    SeamlessTransferFunction();
    ~SeamlessTransferFunction();

    // Non-copyable, non-movable (manages threads)
    SeamlessTransferFunction(const SeamlessTransferFunction&) = delete;
    SeamlessTransferFunction& operator=(const SeamlessTransferFunction&) = delete;
    SeamlessTransferFunction(SeamlessTransferFunction&&) = delete;
    SeamlessTransferFunction& operator=(SeamlessTransferFunction&&) = delete;

    /**
     * Access lane mixer (message thread only)
     *
     * The lane mixer holds the 41-lane model that drives the audio render pipeline.
     *
     * @return Reference to lane mixer
     */
    LaneMixer& getLaneMixer();
    const LaneMixer& getLaneMixer() const;

    /**
     * Apply transfer function to single sample (audio thread)
     *
     * Uses active LUT or crossfades between old and new LUT.
     *
     * @param x Input sample
     * @param channel Channel index (0 = L, 1 = R)
     * @return Output sample
     */
    double applyTransferFunction(double x, int channel = 0) const;

    /**
     * Analytical derivative dy/dx of the transfer function at x (audio thread).
     *
     * Uses the currently active primary LUT; does not crossfade. Accounts for
     * soft-clip via chain rule when enabled.
     */
    double applyTransferFunctionDerivative(double x, int channel = 0) const;

    /**
     * Process multi-channel buffer in-place (audio thread)
     *
     * Processes all channels with shared crossfade state.
     * Checks for new LUT once at start, then processes all channels.
     * Crossfade position advances correctly (once per sample, not per channel).
     *
     * CRITICAL: This method processes all channels together to ensure
     * stereo consistency. All channels see identical crossfade positions.
     *
     * @param buffer Multi-channel audio buffer (modified in-place)
     */
    void processBuffer(juce::AudioBuffer<double>& buffer) const;

    /**
     * Check for new LUT at the start of a processing block (audio thread)
     *
     * Must be called once per block when using applyTransferFunction() per-sample
     * instead of processBuffer(). processBuffer() calls this internally.
     */
    void beginBlock() const;

    /**
     * Advance crossfade position by one sample (audio thread)
     *
     * Must be called once per sample when using applyTransferFunction() per-sample
     * instead of processBuffer(). processBuffer() calls this internally.
     */
    void advanceCrossfadeSample() const;

    /**
     * Prepare for playback (audio thread)
     *
     * Calculates sample-rate-adaptive crossfade duration.
     * Can be called before or after startSeamlessUpdates().
     *
     * @param sampleRate Sample rate in Hz
     * @param samplesPerBlock Maximum block size
     */
    void prepareToPlay(double sampleRate, int samplesPerBlock);

    /**
     * Release resources (audio thread)
     *
     * Delegates to stopSeamlessUpdates() asynchronously on message thread.
     * Safe to call from audio thread.
     */
    void releaseResources();

    /**
     * Start seamless updates (message thread, after controller created)
     *
     * Starts worker thread and 25Hz poller.
     * Triggers initial LUT render for visualizer.
     * MUST be called AFTER controller creation.
     *
     * Asserts:
     *   - Called on message thread
     *   - Not already started
     */
    void startSeamlessUpdates();

    /**
     * Stop seamless updates (message thread)
     *
     * Stops poller and worker thread.
     * Safe to call multiple times (idempotent).
     */
    void stopSeamlessUpdates();

    /**
     * Render LUT immediately and synchronously (message thread only)
     *
     * Directly renders the current editing model state to the audio thread's
     * LUT buffer WITHOUT using the worker thread queue. This is intended for
     * state restoration scenarios where:
     *   - Audio playback is assumed to not be active yet
     *   - We need the audio LUT to reflect the restored state immediately
     *   - The normal 20Hz timer-based update path would be too slow
     *
     * IMPORTANT: This bypasses the seamless crossfade system. Use only during
     * state restoration, not during interactive editing.
     *
     * Thread Safety:
     *   - MUST be called from message thread
     *   - Safe to call even if seamless updates haven't started yet
     *   - The next processBlock() will use the newly rendered LUT
     *
     * Typical Usage:
     *   transferFunction.getLaneMixer().fromValueTree(savedState);
     *   transferFunction.renderLUTImmediate();  // Audio LUT now matches restored state
     */
    void renderLUTImmediate();

    /**
     * Get visualizer LUT (message thread only)
     *
     * Returns the latest rendered LUT (target curve that audio is crossfading toward
     * or has reached). This may be ahead of what's currently playing if a crossfade
     * is still in progress.
     *
     * SIZE: VISUALIZER_LUT_SIZE (2048 samples) - optimized for UI rendering
     *
     * @return Const reference to visualizer LUT buffer (2048 samples)
     */
    const std::array<double, VISUALIZER_LUT_SIZE>& getVisualizerLUT() const;

    /**
     * Set visualizer callback (message thread only)
     *
     * Called after LUT render completes (via MessageManager::callAsync).
     * Use this to trigger visualizer repaint.
     *
     * @param callback Callback function for visualizer repaint
     */
    void setVisualizerCallback(std::function<void()> callback);

    /**
     * Synchronously recompute the visualizer LUT from current mixer state and
     * fire the visualizer callback (message thread only).
     *
     * Call on editor reopen so a freshly constructed UI receives the current
     * curve. The dispatcher normally only fires on mixer version changes, and
     * nothing changes across editor close/reopen — without this, the new
     * visualizer renders empty. No-op if startSeamlessUpdates() hasn't been
     * called yet.
     */
    void forceVisualizerUpdate();

    /**
     * Get lane LUT for secondary visualizer overlay (message thread only)
     *
     * Returns the selected lane's raw curve data (NOT amplitude-scaled),
     * downsampled to VISUALIZER_LUT_SIZE for UI rendering.
     */
    const std::array<double, VISUALIZER_LUT_SIZE>& getLaneLUT() const;

    /**
     * Get/set which lane the visualizer timer should sample for the lane LUT.
     * -1 = no lane selected (lane LUT not computed).
     */
    int getSelectedVisualizerLane() const;
    void setSelectedVisualizerLane(int laneIndex);

    /**
     * Get the render trigger for AutomationSlot integration.
     * Returns the EventDrivenRenderer as an AsyncUpdater* so AutomationSlot
     * can call triggerAsyncUpdate() from the audio thread.
     * Returns nullptr if seamless updates haven't been started yet.
     */
    juce::AsyncUpdater* getRenderTrigger() const;

  private:
    class Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace dsp_core
