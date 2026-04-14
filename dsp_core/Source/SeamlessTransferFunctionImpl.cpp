#include "SeamlessTransferFunctionImpl.h"
#include <algorithm>
#include <cmath>

namespace dsp_core {

namespace {
    /**
     * Smoothstep interpolation (cubic Hermite)
     *
     * Provides ease-in/ease-out S-curve with zero derivative at endpoints.
     * Used for perceptually smooth crossfading between LUTs.
     *
     * Formula: t² × (3 - 2t)
     * Properties: C¹ continuous, symmetric, computationally cheap
     *
     * @param t Normalized position [0, 1]
     * @return Smoothed position [0, 1]
     */
    inline double smoothstep(double t) {
        // Clamp to [0, 1] (defensive programming - shouldn't happen in normal operation)
        t = std::clamp(t, 0.0, 1.0);

        // Smoothstep formula: t² × (3 - 2t)
        // Expanded: 3t² - 2t³
        return t * t * (3.0 - 2.0 * t);
    }
} // namespace

// AudioEngine Implementation

AudioEngine::AudioEngine() {
    for (int bufIdx = 0; bufIdx < 3; ++bufIdx) {
        for (int i = 0; i < TABLE_SIZE; ++i) {
            const double x = MIN_VALUE + (i / static_cast<double>(TABLE_SIZE - 1)) * (MAX_VALUE - MIN_VALUE);
            lutBuffers[bufIdx].data[i] = x;
        }
        lutBuffers[bufIdx].version = 0;
        lutBuffers[bufIdx].extrapolationMode = LaneMixer::ExtrapolationMode::Clamp;
        lutBuffers[bufIdx].softClipEnabled = false;
    }
}

void AudioEngine::prepareToPlay(double sampleRate, int samplesPerBlock) {
    this->sampleRate = sampleRate;
    // 50ms crossfade = 1.5× DC blocking time constant (balances smoothness vs latency)
    constexpr double msToSeconds = 1000.0;
    crossfadeSamples = static_cast<int>(sampleRate * SeamlessConfig::CROSSFADE_DURATION_MS / msToSeconds);

    if (crossfading && crossfadePosition >= crossfadeSamples) {
        crossfading = false;
    }
}

double AudioEngine::applyTransferFunction(double x) const {
    if (crossfading) {
        const double t = static_cast<double>(crossfadePosition) / crossfadeSamples;
        const double alpha = smoothstep(t);
        const double gainOld = 1.0 - alpha;
        const double gainNew = alpha;
        return evaluateCrossfade(oldLUT, newLUT, x, gainOld, gainNew);
    } else {
        const int idx = primaryIndex.load(std::memory_order_acquire);
        return evaluateLUT(&lutBuffers[idx], x);
    }
}

void AudioEngine::processBuffer(juce::AudioBuffer<double>& buffer) const {
    checkForNewLUT();

    const int numChannels = buffer.getNumChannels();
    const int numSamples = buffer.getNumSamples();

    // Crossfade position advances once per sample (not per channel)
    for (int i = 0; i < numSamples; ++i) {
        if (crossfading) {
            const double t = static_cast<double>(crossfadePosition) / crossfadeSamples;
            const double alpha = smoothstep(t);
            const double gainOld = 1.0 - alpha;
            const double gainNew = alpha;

            for (int ch = 0; ch < numChannels; ++ch) {
                double* channelData = buffer.getWritePointer(ch);
                const double input = channelData[i];
                channelData[i] = evaluateCrossfade(oldLUT, newLUT, input, gainOld, gainNew);
            }

            if (++crossfadePosition >= crossfadeSamples) {
                crossfading = false;
            }
        } else {
            const int idx = primaryIndex.load(std::memory_order_acquire);

            for (int ch = 0; ch < numChannels; ++ch) {
                double* channelData = buffer.getWritePointer(ch);
                const double input = channelData[i];
                channelData[i] = evaluateLUT(&lutBuffers[idx], input);
            }
        }
    }
}

void AudioEngine::checkForNewLUT() const {
    if (newLUTReady.load(std::memory_order_acquire)) {
        const int oldPrimaryIdx = primaryIndex.load(std::memory_order_acquire);
        const int oldSecondaryIdx = secondaryIndex.load(std::memory_order_acquire);
        const int workerIdx = workerTargetIndex.load(std::memory_order_acquire);

        if (crossfading) {
            // Blend snapshot: merge the current crossfade state into the secondary
            // buffer so the new crossfade starts from exactly where the audio is,
            // not from the crossfade target. Eliminates pops on fast automation.
            //
            // Buffer roles before: B=secondary (oldLUT), A=primary (newLUT), C=workerTarget (incoming)
            // We write blend(B, A, position) into B, then rotate: primary=C, secondary=B, workerTarget=A
            const double t = static_cast<double>(crossfadePosition) / crossfadeSamples;
            const double alpha = smoothstep(t);
            const double gainNew = alpha;
            const double gainOld = 1.0 - alpha;

            auto& secBuf = lutBuffers[oldSecondaryIdx];
            const auto& priBuf = lutBuffers[oldPrimaryIdx];
            for (int i = 0; i < TABLE_SIZE; ++i) {
                secBuf.data[static_cast<size_t>(i)] = gainOld * secBuf.data[static_cast<size_t>(i)]
                                                    + gainNew * priBuf.data[static_cast<size_t>(i)];
            }

            // Blend edge slopes for Linear extrapolation continuity
            secBuf.leftSlope = gainOld * secBuf.leftSlope + gainNew * priBuf.leftSlope;
            secBuf.rightSlope = gainOld * secBuf.rightSlope + gainNew * priBuf.rightSlope;
            secBuf.extrapolationMode = lutBuffers[workerIdx].extrapolationMode;
            secBuf.softClipEnabled = lutBuffers[workerIdx].softClipEnabled;

            // Rotate: secondary stays as blend snapshot, old primary becomes workerTarget
            primaryIndex.store(workerIdx, std::memory_order_release);
            // secondaryIndex stays at oldSecondaryIdx (now holds blend snapshot)
            workerTargetIndex.store(oldPrimaryIdx, std::memory_order_release);

            oldLUT = &lutBuffers[oldSecondaryIdx]; // blend snapshot
            newLUT = &lutBuffers[workerIdx];        // new incoming LUT
        } else {
            // Normal rotation: no crossfade in progress
            primaryIndex.store(workerIdx, std::memory_order_release);
            secondaryIndex.store(oldPrimaryIdx, std::memory_order_release);
            workerTargetIndex.store(oldSecondaryIdx, std::memory_order_release);

            oldLUT = &lutBuffers[oldPrimaryIdx];
            newLUT = &lutBuffers[workerIdx];
        }

        newLUTReady.store(false, std::memory_order_release);
        crossfading = true;
        crossfadePosition = 0;
    }
}

double AudioEngine::evaluateLUT(const LUTBuffer* lut, double x) const {
    if (lut->softClipEnabled) {
        x = softClipper_.process(x);
    }
    const double x_proj = (x - MIN_VALUE) / (MAX_VALUE - MIN_VALUE) * (TABLE_SIZE - 1);
    const int index = static_cast<int>(x_proj);
    const double t = x_proj - index;

    if (lut->extrapolationMode == LaneMixer::ExtrapolationMode::Clamp) {
        const int idx0 = std::clamp(index - 1, 0, TABLE_SIZE - 1);
        const int idx1 = std::clamp(index, 0, TABLE_SIZE - 1);
        const int idx2 = std::clamp(index + 1, 0, TABLE_SIZE - 1);
        const int idx3 = std::clamp(index + 2, 0, TABLE_SIZE - 1);

        const double y0 = lut->data[idx0];
        const double y1 = lut->data[idx1];
        const double y2 = lut->data[idx2];
        const double y3 = lut->data[idx3];

        // NOLINTBEGIN(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
        // Standard Catmull-Rom interpolation formula
        return 0.5 * ((2.0 * y1) + (-y0 + y2) * t + (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) * t * t +
                      (-y0 + 3.0 * y1 - 3.0 * y2 + y3) * t * t * t);
        // NOLINTEND(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
    }

    if (lut->extrapolationMode == LaneMixer::ExtrapolationMode::Mirror) {
        const int idx0 = LaneMixer::mirrorIndex(index - 1, TABLE_SIZE);
        const int idx1 = LaneMixer::mirrorIndex(index, TABLE_SIZE);
        const int idx2 = LaneMixer::mirrorIndex(index + 1, TABLE_SIZE);
        const int idx3 = LaneMixer::mirrorIndex(index + 2, TABLE_SIZE);

        const double y0 = lut->data[idx0];
        const double y1 = lut->data[idx1];
        const double y2 = lut->data[idx2];
        const double y3 = lut->data[idx3];

        // NOLINTBEGIN(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
        return 0.5 * ((2.0 * y1) + (-y0 + y2) * t + (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) * t * t +
                      (-y0 + 3.0 * y1 - 3.0 * y2 + y3) * t * t * t);
        // NOLINTEND(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
    }

    // Linear extrapolation using precomputed, clamped edge slopes
    constexpr double OUTPUT_LIMIT = 15.848931924611134; // +24dB

    auto getSample = [lut](int i) -> double {
        if (i < 0) {
            return lut->data[0] + lut->leftSlope * i;
        }
        if (i >= TABLE_SIZE) {
            return lut->data[TABLE_SIZE - 1] + lut->rightSlope * (i - TABLE_SIZE + 1);
        }
        return lut->data[i];
    };

    const double y0 = getSample(index - 1);
    const double y1 = getSample(index);
    const double y2 = getSample(index + 1);
    const double y3 = getSample(index + 2);

    // NOLINTBEGIN(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
    // Standard Catmull-Rom interpolation formula
    double result = 0.5 * ((2.0 * y1) + (-y0 + y2) * t + (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) * t * t +
                           (-y0 + 3.0 * y1 - 3.0 * y2 + y3) * t * t * t);
    // NOLINTEND(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)

    if (std::isnan(result) || std::isinf(result)) return 0.0;
    return std::clamp(result, -OUTPUT_LIMIT, OUTPUT_LIMIT);
}

double AudioEngine::interpolateCatmullRom(double y0, double y1, double y2, double y3, double t) {
    // NOLINTBEGIN(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
    return 0.5 * ((2.0 * y1) + (-y0 + y2) * t + (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) * t * t +
                  (-y0 + 3.0 * y1 - 3.0 * y2 + y3) * t * t * t);
    // NOLINTEND(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
}

double AudioEngine::evaluateCrossfade(const LUTBuffer* oldLUT, const LUTBuffer* newLUT,
                                     double x, double gainOld, double gainNew) const {
    if (newLUT->softClipEnabled) {
        x = softClipper_.process(x);
    }
    // Mix samples BEFORE interpolation (saves one polynomial eval per sample)
    const double x_proj = (x - MIN_VALUE) / (MAX_VALUE - MIN_VALUE) * (TABLE_SIZE - 1);
    const int index = static_cast<int>(x_proj);
    const double t = x_proj - index;

    const auto extrapMode = newLUT->extrapolationMode;

    if (extrapMode == LaneMixer::ExtrapolationMode::Clamp) {
        const int idx0 = std::clamp(index - 1, 0, TABLE_SIZE - 1);
        const int idx1 = std::clamp(index, 0, TABLE_SIZE - 1);
        const int idx2 = std::clamp(index + 1, 0, TABLE_SIZE - 1);
        const int idx3 = std::clamp(index + 2, 0, TABLE_SIZE - 1);

        const double old_y0 = oldLUT->data[idx0];
        const double old_y1 = oldLUT->data[idx1];
        const double old_y2 = oldLUT->data[idx2];
        const double old_y3 = oldLUT->data[idx3];

        const double new_y0 = newLUT->data[idx0];
        const double new_y1 = newLUT->data[idx1];
        const double new_y2 = newLUT->data[idx2];
        const double new_y3 = newLUT->data[idx3];

        const double mixed_y0 = gainOld * old_y0 + gainNew * new_y0;
        const double mixed_y1 = gainOld * old_y1 + gainNew * new_y1;
        const double mixed_y2 = gainOld * old_y2 + gainNew * new_y2;
        const double mixed_y3 = gainOld * old_y3 + gainNew * new_y3;

        return interpolateCatmullRom(mixed_y0, mixed_y1, mixed_y2, mixed_y3, t);
    }

    if (extrapMode == LaneMixer::ExtrapolationMode::Mirror) {
        const int idx0 = LaneMixer::mirrorIndex(index - 1, TABLE_SIZE);
        const int idx1 = LaneMixer::mirrorIndex(index, TABLE_SIZE);
        const int idx2 = LaneMixer::mirrorIndex(index + 1, TABLE_SIZE);
        const int idx3 = LaneMixer::mirrorIndex(index + 2, TABLE_SIZE);

        const double old_y0 = oldLUT->data[idx0];
        const double old_y1 = oldLUT->data[idx1];
        const double old_y2 = oldLUT->data[idx2];
        const double old_y3 = oldLUT->data[idx3];

        const double new_y0 = newLUT->data[idx0];
        const double new_y1 = newLUT->data[idx1];
        const double new_y2 = newLUT->data[idx2];
        const double new_y3 = newLUT->data[idx3];

        const double mixed_y0 = gainOld * old_y0 + gainNew * new_y0;
        const double mixed_y1 = gainOld * old_y1 + gainNew * new_y1;
        const double mixed_y2 = gainOld * old_y2 + gainNew * new_y2;
        const double mixed_y3 = gainOld * old_y3 + gainNew * new_y3;

        return interpolateCatmullRom(mixed_y0, mixed_y1, mixed_y2, mixed_y3, t);
    }

    // Linear extrapolation using precomputed, clamped edge slopes
    constexpr double OUTPUT_LIMIT = 15.848931924611134; // +24dB

    auto getSample = [](const LUTBuffer* lut, int i) -> double {
        if (i < 0) {
            return lut->data[0] + lut->leftSlope * i;
        }
        if (i >= TABLE_SIZE) {
            return lut->data[TABLE_SIZE - 1] + lut->rightSlope * (i - TABLE_SIZE + 1);
        }
        return lut->data[i];
    };

    const double old_y0 = getSample(oldLUT, index - 1);
    const double old_y1 = getSample(oldLUT, index);
    const double old_y2 = getSample(oldLUT, index + 1);
    const double old_y3 = getSample(oldLUT, index + 2);

    const double new_y0 = getSample(newLUT, index - 1);
    const double new_y1 = getSample(newLUT, index);
    const double new_y2 = getSample(newLUT, index + 1);
    const double new_y3 = getSample(newLUT, index + 2);

    const double mixed_y0 = gainOld * old_y0 + gainNew * new_y0;
    const double mixed_y1 = gainOld * old_y1 + gainNew * new_y1;
    const double mixed_y2 = gainOld * old_y2 + gainNew * new_y2;
    const double mixed_y3 = gainOld * old_y3 + gainNew * new_y3;

    double result = interpolateCatmullRom(mixed_y0, mixed_y1, mixed_y2, mixed_y3, t);

    if (std::isnan(result) || std::isinf(result)) return 0.0;
    return std::clamp(result, -OUTPUT_LIMIT, OUTPUT_LIMIT);
}

// EventDrivenRenderer Implementation

EventDrivenRenderer::EventDrivenRenderer(LaneMixer& mixer, AudioEngine& engine)
    : laneMixer(mixer)
    , audioEngine(engine) {
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());
    startTimerHz(SAFETY_TIMER_HZ);
}

EventDrivenRenderer::~EventDrivenRenderer() {
    cancelPendingUpdate();
    stopTimer();
}

void EventDrivenRenderer::forceRender() {
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());
    doRender();
}

void EventDrivenRenderer::handleAsyncUpdate() {
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());

    const uint64_t currentFullVersion = laneMixer.getVersion();
    const uint64_t currentMixVersion = laneMixer.getMixVersion();

    if (currentFullVersion == lastRenderedFullVersion)
        return; // Nothing changed

    // Detect whether only mix-related changes occurred (cheap to re-render)
    const bool curveChanged = (currentFullVersion - currentMixVersion)
                            != (lastRenderedFullVersion - lastRenderedMixVersion);

    // In scan mode, amplitude changes don't affect the output (computeScan ignores amplitudes).
    // Skip the render to avoid unnecessary crossfade artifacts from stale buffer data.
    if (!curveChanged && laneMixer.getMixerMode() == LaneMixer::MixerMode::Scan) {
        const double currentScanPos = laneMixer.getScanPosition();
        if (currentScanPos == lastRenderedScanPosition) {
            lastRenderedFullVersion = currentFullVersion;
            lastRenderedMixVersion = currentMixVersion;
            return;
        }
    }

    if (curveChanged) {
        // Rate limit expensive curve renders to 60Hz max
        const double now = juce::Time::getMillisecondCounterHiRes();
        const double elapsed = now - lastCurveRenderTimeMs;
        if (elapsed < CURVE_RENDER_MIN_INTERVAL_MS) {
            // Defer: start a one-shot timer for the remaining interval
            const int remainingMs = static_cast<int>(CURVE_RENDER_MIN_INTERVAL_MS - elapsed) + 1;
            startTimer(remainingMs);
            return;
        }
        lastCurveRenderTimeMs = now;
    }

    doRender();

    // Restore safety timer (in case one-shot timer was used for rate limiting)
    startTimerHz(SAFETY_TIMER_HZ);
}

void EventDrivenRenderer::timerCallback() {
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());

    if (lastRenderedFullVersion != laneMixer.getVersion()) {
        handleAsyncUpdate();
    }
}

void EventDrivenRenderer::doRender() {
    // Compute the mixed sum directly on the message thread
    std::array<double, TABLE_SIZE> sumData{};
    if (laneMixer.getMixerMode() == LaneMixer::MixerMode::Scan) {
        laneMixer.computeScan(sumData.data(), TABLE_SIZE);
    } else {
        laneMixer.computeSum(sumData.data(), TABLE_SIZE);
    }

    // Write directly to the worker target buffer (no worker thread)
    const int targetIdx = audioEngine.getWorkerTargetIndexReference().load(std::memory_order_relaxed);
    LUTBuffer* outputBuffer = &audioEngine.getLUTBuffers()[targetIdx];
    std::copy(sumData.begin(), sumData.end(), outputBuffer->data.begin());
    outputBuffer->version = laneMixer.getVersion();
    outputBuffer->extrapolationMode = laneMixer.getExtrapolationMode();
    outputBuffer->softClipEnabled = laneMixer.getSoftClipEnabled();

    // Precompute and clamp edge slopes for Linear extrapolation
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

    // Signal audio thread
    audioEngine.getNewLUTReadyFlag().store(true, std::memory_order_release);

    lastRenderedFullVersion = laneMixer.getVersion();
    lastRenderedMixVersion = laneMixer.getMixVersion();
    lastRenderedScanPosition = laneMixer.getScanPosition();
}

// VisualizerUpdateDispatcher Implementation (event-driven, 60Hz rate-limited)

VisualizerUpdateDispatcher::VisualizerUpdateDispatcher(LaneMixer& mixer)
    : laneMixer(mixer) {
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());
    startTimerHz(SAFETY_TIMER_HZ);  // slow fallback; triggerAsyncUpdate() drives the hot path
}

VisualizerUpdateDispatcher::~VisualizerUpdateDispatcher() {
    cancelPendingUpdate();
    stopTimer();
}

void VisualizerUpdateDispatcher::setVisualizerTarget(std::array<double, VISUALIZER_LUT_SIZE>* lutPtr,
                                                      std::function<void()> callback) {
    visualizerLUTPtr = lutPtr;
    onVisualizerUpdate = std::move(callback);
}

void VisualizerUpdateDispatcher::setLaneLUTTarget(std::array<double, VISUALIZER_LUT_SIZE>* lutPtr,
                                                   int* selectedLanePtr) {
    laneLUTPtr_ = lutPtr;
    selectedLanePtr_ = selectedLanePtr;
}

void VisualizerUpdateDispatcher::handleAsyncUpdate() {
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());

    const uint64_t currentVersion = laneMixer.getVersion();
    const int currentSelectedLane = selectedLanePtr_ ? *selectedLanePtr_ : -1;
    const bool curveChanged = currentVersion != lastSeenVersion;
    const bool selectionChanged = currentSelectedLane != lastSeenSelectedLane;
    if (!curveChanged && !selectionChanged)
        return;

    // No explicit rate limit: JUCE's AsyncUpdater coalesces multiple
    // triggerAsyncUpdate() calls between message-thread ticks into a single
    // handleAsyncUpdate(), which already gives us ~vsync-rate updates without
    // introducing the up-to-16ms deferral lag the previous gate caused during
    // knob drags. The 5Hz safety timer below catches any edge missed by the
    // event path (e.g. lane selection changed without a version bump).
    lastUpdateTimeMs = juce::Time::getMillisecondCounterHiRes();

    runUpdate();
    lastSeenVersion = currentVersion;
    lastSeenSelectedLane = currentSelectedLane;
}

void VisualizerUpdateDispatcher::timerCallback() {
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());

    const int currentSelectedLane = selectedLanePtr_ ? *selectedLanePtr_ : -1;
    if (lastSeenVersion != laneMixer.getVersion() || currentSelectedLane != lastSeenSelectedLane) {
        handleAsyncUpdate();
    }
}

void VisualizerUpdateDispatcher::forceUpdate() {
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());

    runUpdate();
    lastSeenVersion = laneMixer.getVersion();
    lastSeenSelectedLane = selectedLanePtr_ ? *selectedLanePtr_ : -1;
    lastUpdateTimeMs = juce::Time::getMillisecondCounterHiRes();
}

void VisualizerUpdateDispatcher::runUpdate() {
    if (visualizerLUTPtr) {
        // Compute the output curve into a temporary buffer, then downsample to visualizer resolution
        std::array<double, LaneMixer::TABLE_SIZE> sumBuffer{};
        if (laneMixer.getMixerMode() == LaneMixer::MixerMode::Scan) {
            laneMixer.computeScan(sumBuffer.data(), LaneMixer::TABLE_SIZE);
        } else {
            laneMixer.computeSum(sumBuffer.data(), LaneMixer::TABLE_SIZE);
        }

        // Downsample from TABLE_SIZE (16384) to VISUALIZER_LUT_SIZE (1024)
        for (int i = 0; i < VISUALIZER_LUT_SIZE; ++i) {
            const double frac = i / static_cast<double>(VISUALIZER_LUT_SIZE - 1);
            const double srcIdx = frac * (LaneMixer::TABLE_SIZE - 1);
            const int idx = static_cast<int>(srcIdx);
            const int nextIdx = std::min(idx + 1, LaneMixer::TABLE_SIZE - 1);
            const double t = srcIdx - idx;
            (*visualizerLUTPtr)[static_cast<size_t>(i)] =
                sumBuffer[static_cast<size_t>(idx)] * (1.0 - t) +
                sumBuffer[static_cast<size_t>(nextIdx)] * t;
        }
    }

    // Compute selected lane's raw curve for secondary visualizer overlay
    if (laneLUTPtr_ && selectedLanePtr_) {
        const int laneIdx = *selectedLanePtr_;
        if (laneIdx >= 0 && laneIdx < laneMixer.getActiveLaneCount()) {
            const auto& lane = laneMixer.getLane(laneIdx);
            for (int i = 0; i < VISUALIZER_LUT_SIZE; ++i) {
                const double frac = i / static_cast<double>(VISUALIZER_LUT_SIZE - 1);
                const double srcIdx = frac * (LaneMixer::TABLE_SIZE - 1);
                const int idx = static_cast<int>(srcIdx);
                const int nextIdx = std::min(idx + 1, LaneMixer::TABLE_SIZE - 1);
                const double t = srcIdx - idx;
                (*laneLUTPtr_)[static_cast<size_t>(i)] =
                    lane.curveData[static_cast<size_t>(idx)] * (1.0 - t) +
                    lane.curveData[static_cast<size_t>(nextIdx)] * t;
            }
        }
    }

    if (onVisualizerUpdate) {
        onVisualizerUpdate();
    }
}

} // namespace dsp_core
