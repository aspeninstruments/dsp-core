#include "SeamlessTransferFunction.h"
#include "SeamlessTransferFunctionImpl.h"
#include <juce_core/juce_core.h>

namespace dsp_core {

/**
 * SeamlessTransferFunction::Impl - Private implementation with all components
 *
 * Components:
 *   - editingModel: LayeredTransferFunction (message thread only)
 *   - audioEngine: AudioEngine (audio thread)
 *   - renderer: LUTRendererThread (worker thread, DSP LUT only)
 *   - lutRenderTimer: LUTRenderTimer (20Hz, enqueues DSP render jobs)
 *   - visualizerTimer: VisualizerUpdateTimer (60Hz, direct model reads)
 *   - visualizerLUT: UI-owned LUT buffer (message thread only)
 *   - visualizerCallback: Callback for visualizer repaint (message thread)
 *
 * Two-Timer Architecture:
 *   - VisualizerUpdateTimer (60Hz): Samples model directly, updates visualizer
 *   - LUTRenderTimer (20Hz): Enqueues render jobs, guaranteed final delivery
 *
 * Lifecycle:
 *   1. Constructor: Creates editingModel and audioEngine (both initialized to identity)
 *   2. startSeamlessUpdates(): Creates renderer and both timers, triggers initial render
 *   3. prepareToPlay(): Configures audioEngine crossfade duration
 *   4. processBlock(): Audio thread processes samples with crossfade
 *   5. stopSeamlessUpdates(): Stops timers and renderer
 *   6. Destructor: Ensures synchronous cleanup
 */
class SeamlessTransferFunction::Impl {
  public:
    Impl()
        : editingModel(TABLE_SIZE, MIN_VALUE, MAX_VALUE) {
        // editingModel initialized to identity
        // laneMixer initialized to defaults (H1=1.0 → y=x)
        // audioEngine initialized to identity LUTs (in AudioEngine constructor)
        // renderer and timers are null (created in startSeamlessUpdates)

        // Initialize visualizer LUT to identity (1024 samples)
        for (int i = 0; i < VISUALIZER_LUT_SIZE; ++i) {
            const double x = MIN_VALUE + (i / static_cast<double>(VISUALIZER_LUT_SIZE - 1)) * (MAX_VALUE - MIN_VALUE);
            visualizerLUT[i] = x;
        }
    }

    // Editing model (message thread only — controller/modes still write here)
    LayeredTransferFunction editingModel;

    // Lane mixer (message thread only — render pipeline reads from here)
    LaneMixer laneMixer;

    // Audio engine (audio thread)
    AudioEngine audioEngine;

    // Worker thread (DSP LUT only, created in startSeamlessUpdates)
    std::unique_ptr<LUTRendererThread> renderer;

    // Two separate timers for decoupled update rates
    std::unique_ptr<LUTRenderTimer> lutRenderTimer;        // 20Hz, DSP LUT (guaranteed delivery)
    std::unique_ptr<VisualizerUpdateTimer> visualizerTimer; // 60Hz, direct model reads

    // Visualizer state (message thread only)
    // NOTE: Visualizer shows the editing model directly (may be slightly ahead
    // of audio during crossfade, but user sees their edit immediately).
    // VISUALIZER_LUT_SIZE = 1024 samples
    std::array<double, VISUALIZER_LUT_SIZE> visualizerLUT;
    std::function<void()> visualizerCallback;

    // Lane LUT for secondary visualizer overlay (selected lane's raw curve)
    std::array<double, VISUALIZER_LUT_SIZE> laneLUT{};
    int selectedVisualizerLane = -1;  // -1 = no lane selected
};

SeamlessTransferFunction::SeamlessTransferFunction()
    : pimpl(std::make_unique<Impl>()) {
    // editingModel already initialized in Impl constructor
    // audioEngine already initialized (identity LUTs)
    // Worker thread and poller NOT created yet (deferred to startSeamlessUpdates)
}

SeamlessTransferFunction::~SeamlessTransferFunction() {
    // Stop async operations before destruction

    // Stop timers first (no more jobs enqueued, no more visualizer updates)
    if (pimpl->visualizerTimer) {
        pimpl->visualizerTimer->stopTimer();
    }
    if (pimpl->lutRenderTimer) {
        pimpl->lutRenderTimer->stopTimer();
    }

    // Stop worker thread (finish current job, then exit)
    if (pimpl->renderer) {
        pimpl->renderer->stopThread(2000); // 2 second timeout
    }

    // pimpl destruction handles the rest
}

LayeredTransferFunction& SeamlessTransferFunction::getEditingModel() {
    return pimpl->editingModel;
}

const LayeredTransferFunction& SeamlessTransferFunction::getEditingModel() const {
    return pimpl->editingModel;
}

LaneMixer& SeamlessTransferFunction::getLaneMixer() {
    return pimpl->laneMixer;
}

const LaneMixer& SeamlessTransferFunction::getLaneMixer() const {
    return pimpl->laneMixer;
}

double SeamlessTransferFunction::applyTransferFunction(double x) const {
    return pimpl->audioEngine.applyTransferFunction(x);
}

void SeamlessTransferFunction::processBuffer(juce::AudioBuffer<double>& buffer) const {
    // Audio thread: process entire multi-channel buffer with shared crossfade state
    // Change detection happens on the Editor's timer (25Hz) via notifyEditingModelChanged()
    pimpl->audioEngine.processBuffer(buffer);
}

void SeamlessTransferFunction::prepareToPlay(double sampleRate, int samplesPerBlock) {
    // Can be called from audio thread - no message thread check
    pimpl->audioEngine.prepareToPlay(sampleRate, samplesPerBlock);
}

void SeamlessTransferFunction::releaseResources() {
    // Do NOT stop seamless updates here!
    //
    // DAWs call releaseResources() unpredictably (e.g., when stopping playback,
    // or even during initialization). The seamless update system should stay alive
    // for the plugin's entire lifetime because:
    //
    // 1. Poller runs on message thread (UI-related, not an audio resource)
    // 2. Worker thread is idle when not rendering (no CPU overhead)
    // 3. Audio engine is always needed (not just during playback)
    //
    // The only time we should stop seamless updates is in the destructor.
    //
    // Previous bug: Calling stopSeamlessUpdates() here destroyed the poller,
    // breaking UI updates after DAW called releaseResources().
}

void SeamlessTransferFunction::startSeamlessUpdates() {
    // VERIFY: Called on message thread
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());

    // VERIFY: Not already started
    jassert(pimpl->renderer == nullptr && pimpl->lutRenderTimer == nullptr);

    // Create worker thread (DSP LUT only, no visualizer callback)
    pimpl->renderer = std::make_unique<LUTRendererThread>(
        pimpl->audioEngine,
        pimpl->audioEngine.getWorkerTargetIndexReference(),
        pimpl->audioEngine.getNewLUTReadyFlag());

    // Pass LUT buffers pointer to worker thread (for direct writes)
    pimpl->renderer->setLUTBuffersPointer(pimpl->audioEngine.getLUTBuffers());

    // Start worker thread
    pimpl->renderer->startThread(juce::Thread::Priority::normal);

    // Create LUT render timer (20Hz, guaranteed delivery via two-version tracking)
    pimpl->lutRenderTimer = std::make_unique<LUTRenderTimer>(pimpl->editingModel, pimpl->laneMixer, *pimpl->renderer);
    // Timer starts automatically in constructor at 20Hz

    // Create visualizer timer (60Hz, direct model reads)
    pimpl->visualizerTimer = std::make_unique<VisualizerUpdateTimer>(pimpl->editingModel, pimpl->laneMixer);
    pimpl->visualizerTimer->setVisualizerTarget(&pimpl->visualizerLUT, pimpl->visualizerCallback);
    pimpl->visualizerTimer->setLaneLUTTarget(&pimpl->laneLUT, &pimpl->selectedVisualizerLane);
    // Timer starts automatically in constructor at 60Hz

    // Trigger initial render for both (ensures correct state immediately)
    pimpl->lutRenderTimer->forceRender();
    pimpl->visualizerTimer->forceUpdate();
}

void SeamlessTransferFunction::stopSeamlessUpdates() {
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());

    // Stop timers first (no more jobs enqueued, no more visualizer updates)
    if (pimpl->visualizerTimer) {
        pimpl->visualizerTimer->stopTimer();
        pimpl->visualizerTimer.reset();
    }

    if (pimpl->lutRenderTimer) {
        pimpl->lutRenderTimer->stopTimer();
        pimpl->lutRenderTimer.reset();
    }

    // Stop worker thread
    if (pimpl->renderer) {
        pimpl->renderer->stopThread(1000); // 1 second timeout
        pimpl->renderer.reset();
    }
}

void SeamlessTransferFunction::notifyEditingModelChanged() {
    // DEPRECATED: This method is no longer needed.
    // The LUTRenderTimer (20Hz) and VisualizerUpdateTimer (60Hz) now run
    // independently and detect version changes automatically.
    //
    // This method is kept as a no-op for backwards compatibility.
    // It can be removed once all callers are updated.
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());
}

const std::array<double, VISUALIZER_LUT_SIZE>&
SeamlessTransferFunction::getVisualizerLUT() const {
    return pimpl->visualizerLUT;
}

void SeamlessTransferFunction::setVisualizerCallback(std::function<void()> callback) {
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());
    pimpl->visualizerCallback = callback;

    // Route callback to visualizer timer if it exists
    if (pimpl->visualizerTimer) {
        pimpl->visualizerTimer->setVisualizerTarget(&pimpl->visualizerLUT, std::move(callback));
        pimpl->visualizerTimer->setLaneLUTTarget(&pimpl->laneLUT, &pimpl->selectedVisualizerLane);
    }
}

const std::array<double, VISUALIZER_LUT_SIZE>&
SeamlessTransferFunction::getLaneLUT() const {
    return pimpl->laneLUT;
}

int SeamlessTransferFunction::getSelectedVisualizerLane() const {
    return pimpl->selectedVisualizerLane;
}

void SeamlessTransferFunction::setSelectedVisualizerLane(int laneIndex) {
    pimpl->selectedVisualizerLane = laneIndex;
}

void SeamlessTransferFunction::renderLUTImmediate() {
    // VERIFY: Called on message thread
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());

    const auto& model = pimpl->editingModel;
    auto& mixer = pimpl->laneMixer;

    // Sync LTF editing model state into LaneMixer lanes
    // (Same logic as LUTRenderTimer::syncLaneMixerFromLTF)
    const auto coeffs = model.getHarmonicCoefficients();
    for (int i = 0; i < LaneMixer::NUM_LANES; ++i) {
        mixer.setLaneAmplitude(i, coeffs[static_cast<size_t>(i)]);
    }

    // Copy LTF base layer → lane 0 curve data, only if lane 0 hasn't been
    // directly edited (Phase 5 guard)
    const auto& lane0Content = mixer.getLane(0).contentType;
    if (lane0Content == LaneContentType::Harmonic) {
        auto& lane0 = mixer.getMutableLane(0);
        for (int i = 0; i < LaneMixer::TABLE_SIZE; ++i) {
            lane0.curveData[static_cast<size_t>(i)] = model.getBaseLayerValue(i);
        }
    }

    // Always-on mixer: normalization enabled when LTF says so
    mixer.setNormalizationEnabled(model.isNormalizationEnabled());

    // LEGACY: Spline override only reachable when no lane mixer is present.
    // Lane-scoped spline editing writes directly to lane curveData and never sets
    // RenderingMode::Spline, so this block is dead code in production.
    if (model.getRenderingMode() == RenderingMode::Spline) {
        if (lane0Content == LaneContentType::Harmonic) {
            const auto& splineLayer = model.getSplineLayer();
            auto& lane0 = mixer.getMutableLane(0);
            for (int i = 0; i < LaneMixer::TABLE_SIZE; ++i) {
                const double x = mixer.normalizeIndex(i);
                lane0.curveData[static_cast<size_t>(i)] = splineLayer.evaluate(x);
            }
        }
        mixer.setLaneAmplitude(0, 1.0);
        for (int i = 1; i < LaneMixer::NUM_LANES; ++i) {
            mixer.setLaneAmplitude(i, 0.0);
        }
        mixer.setNormalizationEnabled(false);
    }
    // Paint, Harmonic, Equation: all lanes active (mixer always drives audio)

    // Compute the mixed sum
    std::array<double, TABLE_SIZE> sumBuffer{};
    mixer.computeSum(sumBuffer.data(), TABLE_SIZE);

    // Get the worker target buffer (where we'll write the LUT)
    const int targetIdx = pimpl->audioEngine.getWorkerTargetIndexReference().load(std::memory_order_relaxed);
    LUTBuffer* outputBuffer = &pimpl->audioEngine.getLUTBuffers()[targetIdx];

    // Copy sum to output buffer
    std::copy(sumBuffer.begin(), sumBuffer.end(), outputBuffer->data.begin());

    // Set metadata
    outputBuffer->version = model.getVersion();
    outputBuffer->extrapolationMode = model.getExtrapolationMode();

    // Signal audio thread that new LUT is ready (using release to ensure LUT writes are visible)
    pimpl->audioEngine.getNewLUTReadyFlag().store(true, std::memory_order_release);

    // Also update the visualizer LUT to match (so UI is consistent)
    // Downsample from TABLE_SIZE to VISUALIZER_LUT_SIZE
    for (int i = 0; i < VISUALIZER_LUT_SIZE; ++i) {
        const double frac = i / static_cast<double>(VISUALIZER_LUT_SIZE - 1);
        const double srcIdx = frac * (TABLE_SIZE - 1);
        const int idx = static_cast<int>(srcIdx);
        const int nextIdx = std::min(idx + 1, TABLE_SIZE - 1);
        const double t = srcIdx - idx;
        pimpl->visualizerLUT[static_cast<size_t>(i)] =
            sumBuffer[static_cast<size_t>(idx)] * (1.0 - t) +
            sumBuffer[static_cast<size_t>(nextIdx)] * t;
    }

    // Also update the lane LUT if a lane is selected
    const int laneIdx = pimpl->selectedVisualizerLane;
    if (laneIdx >= 0 && laneIdx < LaneMixer::NUM_LANES) {
        const auto& lane = mixer.getLane(laneIdx);
        for (int i = 0; i < VISUALIZER_LUT_SIZE; ++i) {
            const double frac = i / static_cast<double>(VISUALIZER_LUT_SIZE - 1);
            const double srcIdx = frac * (LaneMixer::TABLE_SIZE - 1);
            const int idx = static_cast<int>(srcIdx);
            const int nextIdx = std::min(idx + 1, LaneMixer::TABLE_SIZE - 1);
            const double t = srcIdx - idx;
            pimpl->laneLUT[static_cast<size_t>(i)] =
                lane.curveData[static_cast<size_t>(idx)] * (1.0 - t) +
                lane.curveData[static_cast<size_t>(nextIdx)] * t;
        }
    }

    // Invoke visualizer callback if set (so UI updates)
    if (pimpl->visualizerCallback) {
        pimpl->visualizerCallback();
    }
}

} // namespace dsp_core
