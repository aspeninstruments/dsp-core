#pragma once

#include "LaneMixer.h"
#include "SeamlessTransferFunction.h"
#include "audio_pipeline/SoftClippingStage.h"
#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <juce_core/juce_core.h>
#include <juce_audio_basics/juce_audio_basics.h>

namespace dsp_core {

// Forward declarations
class EventDrivenRenderer;
class VisualizerUpdateDispatcher;

/**
 * SeamlessConfig - Configuration constants for seamless transfer function updates.
 * DSP_LUT_SIZE is sourced from LaneMixer::TABLE_SIZE so audio/render/storage stay in sync.
 */
struct SeamlessConfig {
    static constexpr int DSP_LUT_SIZE = LaneMixer::TABLE_SIZE;
    static constexpr int VISUALIZER_LUT_SIZE = 1024;
    static constexpr double MIN_VALUE = -1.0;
    static constexpr double MAX_VALUE = 1.0;
    static constexpr double CROSSFADE_DURATION_MS = 5.0;
    static constexpr int DSP_TIMER_HZ = 20;
    static constexpr int VISUALIZER_TIMER_HZ = 120;
};

// Cross-class invariant: every TABLE_SIZE definition in the codebase must match.
// If this fails, the audio thread, renderer, and lane storage will silently disagree.
static_assert(SeamlessConfig::DSP_LUT_SIZE == LaneMixer::TABLE_SIZE,
              "DSP_LUT_SIZE must match LaneMixer::TABLE_SIZE");

// Legacy aliases for backward compatibility
static constexpr int TABLE_SIZE = SeamlessConfig::DSP_LUT_SIZE;
static constexpr int VISUALIZER_LUT_SIZE = SeamlessConfig::VISUALIZER_LUT_SIZE;
static constexpr double MIN_VALUE = SeamlessConfig::MIN_VALUE;
static constexpr double MAX_VALUE = SeamlessConfig::MAX_VALUE;

// Buffer roles for triple-buffered LUT system
enum class BufferRole {
    Primary = 0,     // Active LUT for playback (audio thread reads)
    Secondary = 1,   // Previous LUT used during crossfade (audio thread reads)
    WorkerTarget = 2 // Worker thread writes here (isolated from audio)
};

// LUTBuffer - Triple-buffered lookup table
struct LUTBuffer {
    std::array<double, TABLE_SIZE> data;
    uint64_t version{0};
    LaneMixer::ExtrapolationMode extrapolationMode{LaneMixer::ExtrapolationMode::Clamp};
    double leftSlope{0.0};       // Precomputed, clamped slope at left edge (for Linear extrapolation)
    double rightSlope{0.0};      // Precomputed, clamped slope at right edge (for Linear extrapolation)
    bool softClipEnabled{false}; // When true, input is soft-clipped before LUT lookup
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
     * @param channel Channel index (0 = L, 1 = R)
     * @return Output sample
     */
    double applyTransferFunction(double x, int channel = 0) const;

    /**
     * Analytical derivative of the transfer function at x (audio thread, const).
     *
     * Uses the currently active primary LUT (does not crossfade). Intended for
     * numerical solvers (hysteresis RK4) that need a smooth, deterministic slope.
     */
    double applyTransferFunctionDerivative(double x, int channel = 0) const;

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
     * Check for new LUT at block start (audio thread)
     * Call once per block when using applyTransferFunction() per-sample.
     */
    void beginBlock() const {
        checkForNewLUT();
    }

    /**
     * Advance crossfade by one sample (audio thread)
     * Call once per sample when using applyTransferFunction() per-sample.
     */
    void advanceCrossfadeSample() const {
        if (crossfading) {
            if (++crossfadePosition >= crossfadeSamples) {
                crossfading = false;
            }
        }
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
     * Analytical derivative dy/dx of the LUT evaluation at x.
     *
     * Matches evaluateLUT's branches one-for-one (Clamp, Mirror, Linear-extrap).
     * If softClipEnabled, multiplies by the soft clipper's analytical derivative
     * at the original x (chain rule).
     */
    double evaluateLUTDerivative(const LUTBuffer* lut, double x) const;

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
    double evaluateCrossfade(const LUTBuffer* oldLUT, const LUTBuffer* newLUT, double x, double gainOld,
                             double gainNew) const;

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

    // Soft clipper for input bounding (stateless, const-safe)
    audio_pipeline::SoftClippingSolver softClipper_{0.95};

    // TRIPLE BUFFERING (prevents data race during crossfade):
    // - lutBuffers[0,1]: Used for crossfading (audio thread reads)
    // - lutBuffers[2]: Worker thread writes here (safe from audio thread)
    mutable LUTBuffer lutBuffers[3]; // mutable: blend snapshot writes in checkForNewLUT()

    // Atomics are mutable because they're modified in const methods (thread-safe state)
    mutable std::atomic<int> primaryIndex{static_cast<int>(BufferRole::Primary)}; // Active LUT for playback
    mutable std::atomic<int> secondaryIndex{
        static_cast<int>(BufferRole::Secondary)}; // Previous LUT (used during crossfade)
    mutable std::atomic<int> workerTargetIndex{static_cast<int>(BufferRole::WorkerTarget)}; // Worker writes here
    mutable std::atomic<bool> newLUTReady{false};

    // Crossfade state (audio thread local - mutable for const methods)
    double sampleRate{44100.0};
    mutable int crossfadeSamples{441}; // Recalculated in prepareToPlay()
    mutable int crossfadePosition{0};
    mutable std::atomic<bool> crossfading{false}; // Atomic for worker thread reads
    mutable const LUTBuffer* oldLUT{nullptr};
    mutable const LUTBuffer* newLUT{nullptr};
};

/**
 * EventDrivenRenderer - DSP LUT renderer with priority-isolated worker thread
 *
 * Architecture:
 *   - A high-priority worker thread (juce::Thread::Priority::high) owns the
 *     actual O(TABLE_SIZE) lane-mix compute and the 120 Hz rate gate.
 *   - Audio-thread / version-changed callers wake the worker DIRECTLY via
 *     requestRender() → juce::Thread::notify() (a WaitableEvent::signal).
 *     The message thread is not involved in the wake-up path, so editor
 *     close / App Nap / host main-loop throttling cannot stall renders
 *     while modulators are driving fast parameter changes.
 *   - The 5 ms LUT crossfade smooths the resulting parameter steps. Audio-
 *     rate modulation via the LUT path is intentionally not supported in
 *     exchange for predictable CPU on slow hardware.
 *   - Writes directly to the audio engine's triple-buffered LUT.
 *   - AsyncUpdater + 5 Hz Timer are retained as message-thread fallbacks
 *     (e.g. UI-side mutations); both forward to notify() and no longer hold
 *     rate-limit state.
 *
 * THREADING:
 *   - requestRender() callable from any thread; just signals the worker.
 *   - handleAsyncUpdate() runs on message thread; calls notify().
 *   - timerCallback() runs on message thread (5 Hz safety net); calls notify().
 *   - run() is the worker thread loop. It applies the 120 Hz gate via
 *     wait(remainingMs) before calling renderIfNeeded() / doRender().
 *   - forceRender() runs synchronously on the calling thread (init + tests only).
 *   - Triple-buffer atomics in AudioEngine handle worker→audio handoff safely.
 *   - lastRenderedSum_ is shared between worker (writer) and message thread
 *     (visualizer dispatcher reader); both sides take lastRenderedSumMutex_.
 *
 * Two-Tier Version Tracking (used for scan-mode amplitude-skip; both tiers
 * share the same 120 Hz rate limit):
 *   - Full version (versionCounter_): incremented on ANY change
 *   - Mix version (mixVersionCounter_): incremented only for amplitude/scan changes
 */
class EventDrivenRenderer : public juce::AsyncUpdater,
                            public juce::Timer,
                            public IRenderTrigger,
                            private juce::Thread {
  public:
    EventDrivenRenderer(LaneMixer& mixer, AudioEngine& engine);
    ~EventDrivenRenderer() override;

    /** Audio-thread-safe direct wake. Signals the worker's WaitableEvent —
     *  no message-thread hop, no locks, no allocations. The worker owns the
     *  120 Hz rate gate, so over-notifying is harmless. */
    void requestRender() noexcept override {
        notify();
    }

    /** Force synchronous render on the calling thread. Init/test only — must
     *  not be called concurrently with the worker (typically called before
     *  audio starts or after stopThread). */
    void forceRender();

    /** AsyncUpdater callback — runs on the message thread. Forwards to
     *  notify() as a fallback wake path; the worker owns the rate gate. */
    void handleAsyncUpdate() override;

    /** Safety timer callback (5 Hz fallback) — runs on the message thread. */
    void timerCallback() override;

    /** Set the visualizer dispatcher to notify after each render.
     *  triggerAsyncUpdate() is cross-thread-safe by JUCE contract, so this
     *  is callable from the worker thread. */
    void setVisualizerDispatcher(juce::AsyncUpdater* dispatcher) {
        visualizerDispatcher_ = dispatcher;
    }

    /** Copy the most recent worker-rendered full-resolution sum into `dest`
     *  under a mutex (worker may be writing concurrently). Returns false if
     *  no render has ever completed. Thread-safe. */
    bool copyLastRenderedSum(std::array<double, TABLE_SIZE>& dest) const;

  private:
    /** Worker-thread loop (juce::Thread::run override). */
    void run() override;

    /** Worker-thread render gate: re-checks version, scan-mode amplitude-skip,
     *  and isCrossfading() before calling doRender(). */
    void renderIfNeeded();

    /** The actual O(TABLE_SIZE) compute. Runs on the worker thread (or the
     *  calling thread for forceRender). */
    void doRender();

    LaneMixer& laneMixer;
    AudioEngine& audioEngine;
    juce::AsyncUpdater* visualizerDispatcher_ = nullptr;

    // Serializes forceRender (caller's thread) with the worker's
    // renderIfNeeded path. Both write to the audio engine's worker-target
    // buffer and to the worker-local lastRendered* state below; without
    // this lock, a forceRender call during a preset restore could race
    // with an in-flight worker render and corrupt the LUT. Held only for
    // the duration of one render (~5-10 ms on slow hardware), so message-
    // thread contention is negligible.
    mutable std::mutex renderMutex_;

    // Worker-thread state — read/written only by the worker (or by
    // forceRender on the calling thread). Always accessed under
    // renderMutex_.
    uint64_t lastRenderedFullVersion{0};
    uint64_t lastRenderedMixVersion{0};
    double lastRenderedScanPosition{0.0};

    // Worker-thread-local state — read/written only inside run(). Tracks
    // when the most recent render completed so the worker can self-throttle
    // to RENDER_MIN_INTERVAL_MS without involving the message thread.
    double lastRenderTimeMs_{0.0};

    // Cross-thread snapshot of the most recent render output, read by the
    // visualizer dispatcher on the message thread. Worker writes under mutex
    // after each successful doRender; reader copies under mutex via
    // copyLastRenderedSum.
    mutable std::mutex lastRenderedSumMutex_;
    std::array<double, TABLE_SIZE> lastRenderedSum_{};
    bool hasLastRenderedSum_{false};

    // 120 Hz cap on all DSP LUT renders. Yields 6 samples/cycle for the
    // worst-case 20 Hz LFO target — above the 4-samples/cycle "perceptually
    // smooth with crossfade" floor. The 5 ms CROSSFADE_DURATION_MS smooths
    // remaining stair-stepping. Going higher (240 Hz) buys headroom for
    // higher-frequency modulation at 2x the worker wake-up cost.
    static constexpr double RENDER_MIN_INTERVAL_MS = 8.33; // 120Hz max
    static constexpr int SAFETY_TIMER_HZ = 5;              // 200ms fallback
};

/**
 * VisualizerUpdateDispatcher - Event-driven visualizer updates
 *
 * Mirrors the DSP EventDrivenRenderer pattern: AsyncUpdater dispatches the
 * curve/lane compute on the message thread in response to LaneMixer version
 * changes, rate-limited to 60Hz. A slow safety timer (5Hz) picks up any edge
 * missed by the event path.
 *
 * Idle cost is zero — nothing runs until a mutation fires onVersionChanged
 * or a lane-selection change calls triggerAsyncUpdate directly.
 *
 * Threading:
 *   - triggerAsyncUpdate() callable from any thread (JUCE guarantee)
 *   - handleAsyncUpdate() runs on message thread
 *   - timerCallback() runs on message thread
 */
class VisualizerUpdateDispatcher : public juce::AsyncUpdater, public juce::Timer {
  public:
    explicit VisualizerUpdateDispatcher(LaneMixer& mixer);
    ~VisualizerUpdateDispatcher() override;

    void setVisualizerTarget(std::array<double, VISUALIZER_LUT_SIZE>* lutPtr, std::function<void()> callback);

    void setLaneLUTTarget(std::array<double, VISUALIZER_LUT_SIZE>* lutPtr, int* selectedLanePtr);

    /** Wire the dispatcher to the renderer that produces the same sum.
     *  When set, runUpdate() reads the renderer's cached buffer instead of
     *  recomputing — saving a full 16k LaneMixer mix per modulation tick. */
    void setSourceRenderer(EventDrivenRenderer* renderer) {
        sourceRenderer_ = renderer;
    }

    /** Force synchronous update (for initialization). */
    void forceUpdate();

    /** AsyncUpdater callback — dispatched on message thread. */
    void handleAsyncUpdate() override;

    /** Safety-timer fallback callback (catches edges missed by the event path). */
    void timerCallback() override;

  private:
    void runUpdate();

    LaneMixer& laneMixer;
    EventDrivenRenderer* sourceRenderer_{nullptr};
    std::array<double, VISUALIZER_LUT_SIZE>* visualizerLUTPtr{nullptr};
    std::function<void()> onVisualizerUpdate;
    uint64_t lastSeenVersion{0};

    std::array<double, VISUALIZER_LUT_SIZE>* laneLUTPtr_{nullptr};
    int* selectedLanePtr_{nullptr};
    int lastSeenSelectedLane{-1};

    double lastUpdateTimeMs{0.0};

    // 60 Hz cap on visualizer dispatcher invocations. Each invocation does a
    // 16k→1k downsample plus the onVisualizerUpdate callback chain (which
    // triggers a paint). Without this cap, the dispatcher fires at the same
    // rate as the renderer (up to 120 Hz), doubling UI-side message-thread
    // queue depth and starving DSP renders on slow hardware. 16.7 ms is one
    // 60 Hz frame — below human discrimination for a continuous knob drag,
    // and matches the perceptual smoothness floor for waveform animation.
    static constexpr double RENDER_MIN_INTERVAL_MS = 16.67;
    static constexpr int SAFETY_TIMER_HZ = 5; // 200ms fallback for missed edges
};

} // namespace dsp_core
