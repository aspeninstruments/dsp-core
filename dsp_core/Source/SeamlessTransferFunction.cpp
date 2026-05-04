#include "SeamlessTransferFunction.h"
#include "SeamlessTransferFunctionImpl.h"
#include <juce_core/juce_core.h>
#include <algorithm>

namespace dsp_core {

/**
 * SeamlessTransferFunction::Impl - Private implementation with all components
 *
 * Components:
 *   - laneMixer: LaneMixer (message thread only — authoritative curve data)
 *   - audioEngine: AudioEngine (audio thread — triple-buffered LUT playback)
 *   - eventRenderer: EventDrivenRenderer (replaces LUTRenderTimer + LUTRendererThread)
 *   - visualizerDispatcher: VisualizerUpdateDispatcher (event-driven, 60Hz rate-limited)
 *
 * Event-Driven Architecture:
 *   - LaneMixer::onVersionChanged callback triggers EventDrivenRenderer
 *   - EventDrivenRenderer writes directly to triple-buffered LUT (no worker thread)
 *   - Mix-only changes (amplitude, scan) render immediately
 *   - Curve changes rate-limited to 60Hz with 5Hz safety timer fallback
 *   - 5ms crossfade with interruption support for snappy automation response
 *
 * Lifecycle:
 *   1. Constructor: Creates laneMixer and audioEngine (both initialized to identity)
 *   2. startSeamlessUpdates(): Creates EventDrivenRenderer + visualizer timer
 *   3. prepareToPlay(): Configures audioEngine crossfade duration
 *   4. processBlock(): Audio thread processes samples with crossfade
 *   5. stopSeamlessUpdates(): Stops renderer and timers
 *   6. Destructor: Ensures synchronous cleanup
 */
class SeamlessTransferFunction::Impl {
  public:
    Impl() {
        // laneMixer initialized to defaults (H1=1.0 → y=x)
        // audioEngine initialized to identity LUTs (in AudioEngine constructor)
        // eventRenderer and timers are null (created in startSeamlessUpdates)

        // Initialize visualizer LUT to identity (1024 samples)
        for (int i = 0; i < VISUALIZER_LUT_SIZE; ++i) {
            const double x = MIN_VALUE + (i / static_cast<double>(VISUALIZER_LUT_SIZE - 1)) * (MAX_VALUE - MIN_VALUE);
            visualizerLUT[i] = x;
        }
    }

    // Lane mixer (message thread only — render pipeline reads from here)
    LaneMixer laneMixer;

    // Audio engine (audio thread)
    AudioEngine audioEngine;

    // Event-driven renderer (replaces LUTRenderTimer + LUTRendererThread)
    std::unique_ptr<EventDrivenRenderer> eventRenderer;

    // Visualizer dispatcher (event-driven, 60Hz rate-limited — independent of DSP rendering)
    std::unique_ptr<VisualizerUpdateDispatcher> visualizerDispatcher;

    // Visualizer state (message thread only)
    std::array<double, VISUALIZER_LUT_SIZE> visualizerLUT{};
    std::function<void()> visualizerCallback;

    // Lane LUT for secondary visualizer overlay (selected lane's raw curve)
    std::array<double, VISUALIZER_LUT_SIZE> laneLUT{};
    int selectedVisualizerLane = -1; // -1 = no lane selected
};

SeamlessTransferFunction::SeamlessTransferFunction() : pimpl(std::make_unique<Impl>()) {
    // laneMixer initialized to defaults in Impl constructor
    // audioEngine already initialized (identity LUTs)
    // Worker thread and poller NOT created yet (deferred to startSeamlessUpdates)
}

SeamlessTransferFunction::~SeamlessTransferFunction() {
    // Stop async operations before destruction

    // Clear the version change callback first to prevent dangling references
    // during subsequent teardown
    pimpl->laneMixer.setOnVersionChanged(nullptr);

    // Disconnect renderer → visualizer link before destroying the dispatcher
    if (pimpl->eventRenderer) {
        pimpl->eventRenderer->setVisualizerDispatcher(nullptr);
    }

    // Stop visualizer dispatcher (cancels async updates + stops safety timer)
    if (pimpl->visualizerDispatcher) {
        pimpl->visualizerDispatcher->cancelPendingUpdate();
        pimpl->visualizerDispatcher->stopTimer();
    }

    // Stop event-driven renderer (cancels async updates + stops safety timer)
    if (pimpl->eventRenderer) {
        pimpl->eventRenderer->cancelPendingUpdate();
        pimpl->eventRenderer->stopTimer();
    }

    // pimpl destruction handles the rest
}

LaneMixer& SeamlessTransferFunction::getLaneMixer() {
    return pimpl->laneMixer;
}

const LaneMixer& SeamlessTransferFunction::getLaneMixer() const {
    return pimpl->laneMixer;
}

double SeamlessTransferFunction::applyTransferFunction(double x, int channel) const {
    return pimpl->audioEngine.applyTransferFunction(x, channel);
}

double SeamlessTransferFunction::applyTransferFunctionDerivative(double x, int channel) const {
    return pimpl->audioEngine.applyTransferFunctionDerivative(x, channel);
}

void SeamlessTransferFunction::processBuffer(juce::AudioBuffer<double>& buffer) const {
    // Audio thread: process entire multi-channel buffer with shared crossfade state
    // Change detection happens on the Editor's timer (25Hz) via notifyEditingModelChanged()
    pimpl->audioEngine.processBuffer(buffer);
}

void SeamlessTransferFunction::beginBlock() const {
    pimpl->audioEngine.beginBlock();
}

void SeamlessTransferFunction::advanceCrossfadeSample() const {
    pimpl->audioEngine.advanceCrossfadeSample();
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
    jassert(pimpl->eventRenderer == nullptr);

    // Create event-driven renderer (writes directly to triple buffer, no worker thread)
    pimpl->eventRenderer = std::make_unique<EventDrivenRenderer>(pimpl->laneMixer, pimpl->audioEngine);

    // Create visualizer dispatcher so the version-changed callback can wake it
    pimpl->visualizerDispatcher = std::make_unique<VisualizerUpdateDispatcher>(pimpl->laneMixer);
    pimpl->visualizerDispatcher->setVisualizerTarget(&pimpl->visualizerLUT, pimpl->visualizerCallback);
    pimpl->visualizerDispatcher->setLaneLUTTarget(&pimpl->laneLUT, &pimpl->selectedVisualizerLane);

    // Let the renderer notify the visualizer after each doRender(), so
    // automation-driven changes (which bypass onVersionChanged) update the UI.
    pimpl->eventRenderer->setVisualizerDispatcher(pimpl->visualizerDispatcher.get());

    // Reverse link: dispatcher reads the renderer's cached sum instead of
    // recomputing, halving the message-thread cost per modulation tick.
    pimpl->visualizerDispatcher->setSourceRenderer(pimpl->eventRenderer.get());

    // Wire the version change callback to trigger both the DSP renderer and
    // the visualizer dispatcher. Both use AsyncUpdater and are rate-limited.
    pimpl->laneMixer.setOnVersionChanged([this]() {
        if (pimpl->eventRenderer) {
            pimpl->eventRenderer->triggerAsyncUpdate();
        }
        if (pimpl->visualizerDispatcher) {
            pimpl->visualizerDispatcher->triggerAsyncUpdate();
        }
    });

    // Trigger initial render for both (ensures correct state immediately)
    pimpl->eventRenderer->forceRender();
    pimpl->visualizerDispatcher->forceUpdate();
}

void SeamlessTransferFunction::stopSeamlessUpdates() {
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());

    // Clear callback first (prevent triggering after renderer destroyed)
    pimpl->laneMixer.setOnVersionChanged(nullptr);

    // Disconnect renderer → visualizer link before destroying the dispatcher
    if (pimpl->eventRenderer) {
        pimpl->eventRenderer->setVisualizerDispatcher(nullptr);
    }

    // Stop visualizer dispatcher
    if (pimpl->visualizerDispatcher) {
        pimpl->visualizerDispatcher->setSourceRenderer(nullptr);
        pimpl->visualizerDispatcher->cancelPendingUpdate();
        pimpl->visualizerDispatcher->stopTimer();
        pimpl->visualizerDispatcher.reset();
    }

    // Stop event-driven renderer
    if (pimpl->eventRenderer) {
        pimpl->eventRenderer->cancelPendingUpdate();
        pimpl->eventRenderer->stopTimer();
        pimpl->eventRenderer.reset();
    }
}

const std::array<double, VISUALIZER_LUT_SIZE>& SeamlessTransferFunction::getVisualizerLUT() const {
    return pimpl->visualizerLUT;
}

void SeamlessTransferFunction::setVisualizerCallback(std::function<void()> callback) {
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());
    pimpl->visualizerCallback = callback;

    // Route callback to visualizer dispatcher if it exists
    if (pimpl->visualizerDispatcher) {
        pimpl->visualizerDispatcher->setVisualizerTarget(&pimpl->visualizerLUT, std::move(callback));
        pimpl->visualizerDispatcher->setLaneLUTTarget(&pimpl->laneLUT, &pimpl->selectedVisualizerLane);
    }
}

void SeamlessTransferFunction::forceVisualizerUpdate() {
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());
    if (pimpl->visualizerDispatcher) {
        pimpl->visualizerDispatcher->forceUpdate();
    }
}

const std::array<double, VISUALIZER_LUT_SIZE>& SeamlessTransferFunction::getLaneLUT() const {
    return pimpl->laneLUT;
}

int SeamlessTransferFunction::getSelectedVisualizerLane() const {
    return pimpl->selectedVisualizerLane;
}

void SeamlessTransferFunction::setSelectedVisualizerLane(int laneIndex) {
    if (pimpl->selectedVisualizerLane == laneIndex) {
        return;
    }
    pimpl->selectedVisualizerLane = laneIndex;
    if (pimpl->visualizerDispatcher) {
        pimpl->visualizerDispatcher->triggerAsyncUpdate();
    }
}

juce::AsyncUpdater* SeamlessTransferFunction::getRenderTrigger() const {
    return pimpl->eventRenderer.get();
}

void SeamlessTransferFunction::renderLUTImmediate() {
    // VERIFY: Called on message thread
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());

    auto& mixer = pimpl->laneMixer;

    // Compute the output curve based on mixer mode
    std::array<double, TABLE_SIZE> sumBuffer{};
    const auto mode = mixer.getMixerMode();
    if (mode == LaneMixer::MixerMode::Scan) {
        mixer.computeScan(sumBuffer.data(), TABLE_SIZE);
    } else if (mode == LaneMixer::MixerMode::Series) {
        mixer.computeSeries(sumBuffer.data(), TABLE_SIZE);
    } else {
        mixer.computeSum(sumBuffer.data(), TABLE_SIZE);
    }

    // Get the worker target buffer (where we'll write the LUT)
    const int targetIdx = pimpl->audioEngine.getWorkerTargetIndexReference().load(std::memory_order_relaxed);
    LUTBuffer* outputBuffer = &pimpl->audioEngine.getLUTBuffers()[targetIdx];

    // Copy sum to output buffer
    std::copy(sumBuffer.begin(), sumBuffer.end(), outputBuffer->data.begin());

    // Set metadata
    outputBuffer->version = mixer.getVersion();
    outputBuffer->extrapolationMode = mixer.getExtrapolationMode();
    outputBuffer->softClipEnabled = mixer.getSoftClipEnabled();

    // Precompute edge slopes for Linear extrapolation. Keep this in sync with
    // EventDrivenRenderer::doRender — both paths populate the same LUT.
    if (outputBuffer->extrapolationMode == LaneMixer::ExtrapolationMode::Linear) {
        constexpr double MAX_SLOPE = 16.0;
        const double leftSlope = outputBuffer->data[1] - outputBuffer->data[0];
        const double rightSlope = outputBuffer->data[TABLE_SIZE - 1] - outputBuffer->data[TABLE_SIZE - 2];
        outputBuffer->leftSlope = std::clamp(leftSlope, -MAX_SLOPE, MAX_SLOPE);
        outputBuffer->rightSlope = std::clamp(rightSlope, -MAX_SLOPE, MAX_SLOPE);
    } else {
        outputBuffer->leftSlope = 0.0;
        outputBuffer->rightSlope = 0.0;
    }

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
            sumBuffer[static_cast<size_t>(idx)] * (1.0 - t) + sumBuffer[static_cast<size_t>(nextIdx)] * t;
    }

    // Also update the lane LUT if a lane is selected
    const int laneIdx = pimpl->selectedVisualizerLane;
    if (laneIdx >= 0 && laneIdx < mixer.getActiveLaneCount()) {
        const auto& lane = mixer.getLane(laneIdx);
        for (int i = 0; i < VISUALIZER_LUT_SIZE; ++i) {
            const double frac = i / static_cast<double>(VISUALIZER_LUT_SIZE - 1);
            const double srcIdx = frac * (LaneMixer::TABLE_SIZE - 1);
            const int idx = static_cast<int>(srcIdx);
            const int nextIdx = std::min(idx + 1, LaneMixer::TABLE_SIZE - 1);
            const double t = srcIdx - idx;
            pimpl->laneLUT[static_cast<size_t>(i)] =
                lane.curveData[static_cast<size_t>(idx)] * (1.0 - t) + lane.curveData[static_cast<size_t>(nextIdx)] * t;
        }
    }

    // Invoke visualizer callback if set (so UI updates)
    if (pimpl->visualizerCallback) {
        pimpl->visualizerCallback();
    }
}

} // namespace dsp_core
