#include "SeamlessTransferFunction.h"
#include "SeamlessTransferFunctionImpl.h"
#include <juce_core/juce_core.h>

namespace dsp_core {

/**
 * SeamlessTransferFunction::Impl - Private implementation with all components
 *
 * Components:
 *   - laneMixer: LaneMixer (message thread only — authoritative curve data)
 *   - audioEngine: AudioEngine (audio thread — triple-buffered LUT playback)
 *   - eventRenderer: EventDrivenRenderer (replaces LUTRenderTimer + LUTRendererThread)
 *   - visualizerTimer: VisualizerUpdateTimer (120Hz, direct model reads)
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

    // Visualizer timer (120Hz, direct LaneMixer reads — independent of DSP rendering)
    std::unique_ptr<VisualizerUpdateTimer> visualizerTimer;

    // Visualizer state (message thread only)
    std::array<double, VISUALIZER_LUT_SIZE> visualizerLUT;
    std::function<void()> visualizerCallback;

    // Lane LUT for secondary visualizer overlay (selected lane's raw curve)
    std::array<double, VISUALIZER_LUT_SIZE> laneLUT{};
    int selectedVisualizerLane = -1;  // -1 = no lane selected
};

SeamlessTransferFunction::SeamlessTransferFunction()
    : pimpl(std::make_unique<Impl>()) {
    // laneMixer initialized to defaults in Impl constructor
    // audioEngine already initialized (identity LUTs)
    // Worker thread and poller NOT created yet (deferred to startSeamlessUpdates)
}

SeamlessTransferFunction::~SeamlessTransferFunction() {
    // Stop async operations before destruction

    // Stop visualizer timer
    if (pimpl->visualizerTimer) {
        pimpl->visualizerTimer->stopTimer();
    }

    // Stop event-driven renderer (cancels async updates + stops safety timer)
    if (pimpl->eventRenderer) {
        pimpl->eventRenderer->cancelPendingUpdate();
        pimpl->eventRenderer->stopTimer();
    }

    // Clear the version change callback to prevent dangling references
    pimpl->laneMixer.setOnVersionChanged(nullptr);

    // pimpl destruction handles the rest
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
    jassert(pimpl->eventRenderer == nullptr);

    // Create event-driven renderer (writes directly to triple buffer, no worker thread)
    pimpl->eventRenderer = std::make_unique<EventDrivenRenderer>(
        pimpl->laneMixer, pimpl->audioEngine);

    // Wire the version change callback to trigger async rendering
    pimpl->laneMixer.setOnVersionChanged([this]() {
        if (pimpl->eventRenderer) {
            pimpl->eventRenderer->triggerAsyncUpdate();
        }
    });

    // Create visualizer timer (120Hz, direct LaneMixer reads)
    pimpl->visualizerTimer = std::make_unique<VisualizerUpdateTimer>(pimpl->laneMixer);
    pimpl->visualizerTimer->setVisualizerTarget(&pimpl->visualizerLUT, pimpl->visualizerCallback);
    pimpl->visualizerTimer->setLaneLUTTarget(&pimpl->laneLUT, &pimpl->selectedVisualizerLane);
    // Timer starts automatically in constructor at 120Hz

    // Trigger initial render for both (ensures correct state immediately)
    pimpl->eventRenderer->forceRender();
    pimpl->visualizerTimer->forceUpdate();
}

void SeamlessTransferFunction::stopSeamlessUpdates() {
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());

    // Clear callback first (prevent triggering after renderer destroyed)
    pimpl->laneMixer.setOnVersionChanged(nullptr);

    // Stop visualizer timer
    if (pimpl->visualizerTimer) {
        pimpl->visualizerTimer->stopTimer();
        pimpl->visualizerTimer.reset();
    }

    // Stop event-driven renderer
    if (pimpl->eventRenderer) {
        pimpl->eventRenderer->cancelPendingUpdate();
        pimpl->eventRenderer->stopTimer();
        pimpl->eventRenderer.reset();
    }
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

juce::AsyncUpdater* SeamlessTransferFunction::getRenderTrigger() const {
    return pimpl->eventRenderer.get();
}

void SeamlessTransferFunction::renderLUTImmediate() {
    // VERIFY: Called on message thread
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());

    auto& mixer = pimpl->laneMixer;

    // Compute the output curve based on mixer mode
    std::array<double, TABLE_SIZE> sumBuffer{};
    if (mixer.getMixerMode() == LaneMixer::MixerMode::Scan) {
        mixer.computeScan(sumBuffer.data(), TABLE_SIZE);
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
    if (laneIdx >= 0 && laneIdx < mixer.getActiveLaneCount()) {
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
