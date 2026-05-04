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
    for (auto& lutBuffer : lutBuffers) {
        for (int i = 0; i < TABLE_SIZE; ++i) {
            const double x = MIN_VALUE + (i / static_cast<double>(TABLE_SIZE - 1)) * (MAX_VALUE - MIN_VALUE);
            lutBuffer.data[i] = x;
        }
        lutBuffer.version = 0;
        lutBuffer.extrapolationMode = LaneMixer::ExtrapolationMode::Clamp;
        lutBuffer.softClipEnabled = false;
    }
}

void AudioEngine::prepareToPlay(double sampleRate, int /*samplesPerBlock*/) {
    this->sampleRate = sampleRate;
    // 50ms crossfade = 1.5× DC blocking time constant (balances smoothness vs latency)
    constexpr double msToSeconds = 1000.0;
    crossfadeSamples = static_cast<int>(sampleRate * SeamlessConfig::CROSSFADE_DURATION_MS / msToSeconds);

    if (crossfading && crossfadePosition >= crossfadeSamples) {
        crossfading = false;
    }
}

double AudioEngine::applyTransferFunction(double x, int /*channel*/) const {
    if (crossfading) {
        const double t = static_cast<double>(crossfadePosition) / crossfadeSamples;
        const double alpha = smoothstep(t);
        const double gainOld = 1.0 - alpha;
        const double gainNew = alpha;
        return evaluateCrossfade(oldLUT, newLUT, x, gainOld, gainNew);
    }
    const int idx = primaryIndex.load(std::memory_order_acquire);
    return evaluateLUT(&lutBuffers[idx], x);
}

double AudioEngine::applyTransferFunctionDerivative(double x, int /*channel*/) const {
    const int idx = primaryIndex.load(std::memory_order_acquire);
    return evaluateLUTDerivative(&lutBuffers[idx], x);
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
                channelData[i] = evaluateCrossfade(oldLUT, newLUT, channelData[i], gainOld, gainNew);
            }

            if (++crossfadePosition >= crossfadeSamples) {
                crossfading = false;
            }
        } else {
            const int idx = primaryIndex.load(std::memory_order_acquire);

            for (int ch = 0; ch < numChannels; ++ch) {
                double* channelData = buffer.getWritePointer(ch);
                channelData[i] = evaluateLUT(&lutBuffers[idx], channelData[i]);
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
                secBuf.data[static_cast<size_t>(i)] =
                    gainOld * secBuf.data[static_cast<size_t>(i)] + gainNew * priBuf.data[static_cast<size_t>(i)];
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
            newLUT = &lutBuffers[workerIdx];       // new incoming LUT
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
    const double result = 0.5 * ((2.0 * y1) + (-y0 + y2) * t + (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) * t * t +
                                 (-y0 + 3.0 * y1 - 3.0 * y2 + y3) * t * t * t);
    // NOLINTEND(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)

    if (std::isnan(result) || std::isinf(result)) {
        return 0.0;
    }
    return std::clamp(result, -OUTPUT_LIMIT, OUTPUT_LIMIT);
}

double AudioEngine::evaluateLUTDerivative(const LUTBuffer* lut, double x) const {
    // Chain rule: applying soft clip first multiplies dy/dx by the soft clipper's slope at
    // the original x. Capture that factor before x is overwritten with the clipped value.
    double scDeriv = 1.0;
    if (lut->softClipEnabled) {
        scDeriv = softClipper_.derivative(x);
        x = softClipper_.process(x);
    }

    const double x_proj = (x - MIN_VALUE) / (MAX_VALUE - MIN_VALUE) * (TABLE_SIZE - 1);
    const int index = static_cast<int>(x_proj);
    const double t = x_proj - index;

    // dt/dx factor from the x → table-index projection
    constexpr double DT_DX = static_cast<double>(TABLE_SIZE - 1) / (MAX_VALUE - MIN_VALUE);

    // Catmull-Rom derivative in t given four neighborhood samples:
    //   dy/dt = 0.5·[(-y0+y2) + 2(2y0-5y1+4y2-y3)·t + 3(-y0+3y1-3y2+y3)·t²]
    auto crDeriv = [](double y0, double y1, double y2, double y3, double tt) {
        // NOLINTBEGIN(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
        return 0.5 * ((-y0 + y2) + 2.0 * (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) * tt +
                      3.0 * (-y0 + 3.0 * y1 - 3.0 * y2 + y3) * tt * tt);
        // NOLINTEND(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
    };

    double dy_dt = 0.0;

    if (lut->extrapolationMode == LaneMixer::ExtrapolationMode::Clamp) {
        const int idx0 = std::clamp(index - 1, 0, TABLE_SIZE - 1);
        const int idx1 = std::clamp(index, 0, TABLE_SIZE - 1);
        const int idx2 = std::clamp(index + 1, 0, TABLE_SIZE - 1);
        const int idx3 = std::clamp(index + 2, 0, TABLE_SIZE - 1);
        dy_dt = crDeriv(lut->data[idx0], lut->data[idx1], lut->data[idx2], lut->data[idx3], t);
    } else if (lut->extrapolationMode == LaneMixer::ExtrapolationMode::Mirror) {
        const int idx0 = LaneMixer::mirrorIndex(index - 1, TABLE_SIZE);
        const int idx1 = LaneMixer::mirrorIndex(index, TABLE_SIZE);
        const int idx2 = LaneMixer::mirrorIndex(index + 1, TABLE_SIZE);
        const int idx3 = LaneMixer::mirrorIndex(index + 2, TABLE_SIZE);
        dy_dt = crDeriv(lut->data[idx0], lut->data[idx1], lut->data[idx2], lut->data[idx3], t);
    } else {
        // Linear extrapolation
        auto getSample = [lut](int i) -> double {
            if (i < 0)
                return lut->data[0] + lut->leftSlope * i;
            if (i >= TABLE_SIZE)
                return lut->data[TABLE_SIZE - 1] + lut->rightSlope * (i - TABLE_SIZE + 1);
            return lut->data[i];
        };
        const double y0 = getSample(index - 1);
        const double y1 = getSample(index);
        const double y2 = getSample(index + 1);
        const double y3 = getSample(index + 2);
        dy_dt = crDeriv(y0, y1, y2, y3, t);
    }

    const double result = dy_dt * DT_DX * scDeriv;
    if (std::isnan(result) || std::isinf(result))
        return 0.0;
    return result;
}

double AudioEngine::interpolateCatmullRom(double y0, double y1, double y2, double y3, double t) {
    // NOLINTBEGIN(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
    return 0.5 * ((2.0 * y1) + (-y0 + y2) * t + (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) * t * t +
                  (-y0 + 3.0 * y1 - 3.0 * y2 + y3) * t * t * t);
    // NOLINTEND(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
}

double AudioEngine::evaluateCrossfade(const LUTBuffer* oldLUT, const LUTBuffer* newLUT, double x, double gainOld,
                                      double gainNew) const {
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

    const double result = interpolateCatmullRom(mixed_y0, mixed_y1, mixed_y2, mixed_y3, t);

    if (std::isnan(result) || std::isinf(result)) {
        return 0.0;
    }
    return std::clamp(result, -OUTPUT_LIMIT, OUTPUT_LIMIT);
}

// EventDrivenRenderer Implementation

EventDrivenRenderer::EventDrivenRenderer(LaneMixer& mixer, AudioEngine& engine)
    : laneMixer(mixer), audioEngine(engine) {
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

    if (currentFullVersion == lastRenderedFullVersion) {
        return; // Nothing changed
    }

    // Detect whether only mix-related changes occurred (cheap to re-render)
    const bool curveChanged =
        (currentFullVersion - currentMixVersion) != (lastRenderedFullVersion - lastRenderedMixVersion);

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

    // Rate limit ALL renders to 60Hz, not just curve-content changes. Amplitude/
    // mix-only changes (amplitude LFOs, scan automation) used to skip this gate
    // because each render was relatively cheap, but at audio-block rate (12–200 Hz)
    // they still saturate weak machines. The 5ms crossfade (CROSSFADE_DURATION_MS)
    // smooths the resulting 60Hz parameter steps. Tradeoff: lose audio-rate LUT
    // modulation (which the architecture never truly delivered) for consistent
    // slow-machine behaviour.
    {
        const double now = juce::Time::getMillisecondCounterHiRes();
        const double elapsed = now - lastRenderTimeMs;
        if (elapsed < RENDER_MIN_INTERVAL_MS) {
            // Defer: start a one-shot timer for the remaining interval.
            const int remainingMs = static_cast<int>(RENDER_MIN_INTERVAL_MS - elapsed) + 1;
            startTimer(remainingMs);
            return;
        }
        lastRenderTimeMs = now;
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
    // Compute the mixed sum directly on the message thread, into the
    // persistent member buffer so the visualizer dispatcher can read it
    // without paying a second 16k computeSum/Scan/Series.
    auto& sumData = lastRenderedSum_;
    const auto mode = laneMixer.getMixerMode();
    if (mode == LaneMixer::MixerMode::Scan) {
        laneMixer.computeScan(sumData.data(), TABLE_SIZE);
    } else if (mode == LaneMixer::MixerMode::Series) {
        laneMixer.computeSeries(sumData.data(), TABLE_SIZE);
    } else {
        laneMixer.computeSum(sumData.data(), TABLE_SIZE);
    }
    hasLastRenderedSum_ = true;

    // Write directly to the worker target buffer (no worker thread)
    const int targetIdx = audioEngine.getWorkerTargetIndexReference().load(std::memory_order_relaxed);
    LUTBuffer* outputBuffer = &audioEngine.getLUTBuffers()[targetIdx];
    std::copy(sumData.begin(), sumData.end(), outputBuffer->data.begin());
    outputBuffer->version = laneMixer.getVersion();
    outputBuffer->extrapolationMode = laneMixer.getExtrapolationMode();
    outputBuffer->softClipEnabled = laneMixer.getSoftClipEnabled();

    // Precompute and clamp edge slopes for Linear extrapolation.
    if (outputBuffer->extrapolationMode == LaneMixer::ExtrapolationMode::Linear) {
        // Slopes are stored as delta-per-LUT-step (extrapolation walks integer index space
        // outside [-1, 1]; see evaluateCrossfade getSample lambda). The historical cap of 16.0
        // was tuned at TABLE_SIZE=16384; keep the same effective dy/dx by scaling with dx.
        constexpr double MAX_SLOPE = 16.0 * (16383.0 / static_cast<double>(TABLE_SIZE - 1));
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

    // Notify visualizer dispatcher so automation-driven amplitude changes
    // (which bypass onVersionChanged) still update the UI promptly.
    // Already on the message thread; triggerAsyncUpdate coalesces naturally.
    if (visualizerDispatcher_ != nullptr) {
        visualizerDispatcher_->triggerAsyncUpdate();
    }
}

// VisualizerUpdateDispatcher Implementation (event-driven, 60Hz rate-limited)

VisualizerUpdateDispatcher::VisualizerUpdateDispatcher(LaneMixer& mixer) : laneMixer(mixer) {
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());
    startTimerHz(SAFETY_TIMER_HZ); // slow fallback; triggerAsyncUpdate() drives the hot path
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
    const int currentSelectedLane = (selectedLanePtr_ != nullptr) ? *selectedLanePtr_ : -1;
    const bool curveChanged = currentVersion != lastSeenVersion;
    const bool selectionChanged = currentSelectedLane != lastSeenSelectedLane;
    if (!curveChanged && !selectionChanged) {
        return;
    }

    // Rate-limit dispatcher invocations to 60 Hz. JUCE's AsyncUpdater only
    // coalesces calls between message-thread ticks; without an explicit gate,
    // the dispatcher fires at the same rate as the renderer (up to 120 Hz),
    // queuing twice the UI-side message-thread events per modulation tick and
    // starving DSP renders on slow hardware. The 16.7 ms cap is one 60 Hz
    // frame — below human discrimination for continuous knob drags. The 5 Hz
    // safety timer (restored below) catches any edge missed by the event path.
    const double now = juce::Time::getMillisecondCounterHiRes();
    const double elapsed = now - lastUpdateTimeMs;
    if (elapsed < RENDER_MIN_INTERVAL_MS) {
        const int remainingMs = static_cast<int>(RENDER_MIN_INTERVAL_MS - elapsed) + 1;
        startTimer(remainingMs);
        return;
    }
    lastUpdateTimeMs = now;

    runUpdate();
    lastSeenVersion = currentVersion;
    lastSeenSelectedLane = currentSelectedLane;

    // Restore safety timer (in case one-shot timer was used for rate limiting).
    startTimerHz(SAFETY_TIMER_HZ);
}

void VisualizerUpdateDispatcher::timerCallback() {
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());

    const int currentSelectedLane = (selectedLanePtr_ != nullptr) ? *selectedLanePtr_ : -1;
    if (lastSeenVersion != laneMixer.getVersion() || currentSelectedLane != lastSeenSelectedLane) {
        handleAsyncUpdate();
    }
}

void VisualizerUpdateDispatcher::forceUpdate() {
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());

    runUpdate();
    lastSeenVersion = laneMixer.getVersion();
    lastSeenSelectedLane = (selectedLanePtr_ != nullptr) ? *selectedLanePtr_ : -1;
    lastUpdateTimeMs = juce::Time::getMillisecondCounterHiRes();
}

void VisualizerUpdateDispatcher::runUpdate() {
    if (visualizerLUTPtr != nullptr) {
        // Prefer the renderer's cached buffer to avoid a redundant 16k
        // LaneMixer recompute. The renderer always runs before us in the
        // forceUpdate path, and on every modulation tick that bumps the
        // version. The fallback handles the (rare) case where the dispatcher
        // is triggered without a prior render — e.g., before init completes.
        const std::array<double, LaneMixer::TABLE_SIZE>* sumPtr =
            (sourceRenderer_ != nullptr) ? sourceRenderer_->getLastRenderedSum() : nullptr;

        std::array<double, LaneMixer::TABLE_SIZE> fallbackSum{};
        if (sumPtr == nullptr) {
            const auto mode = laneMixer.getMixerMode();
            if (mode == LaneMixer::MixerMode::Scan) {
                laneMixer.computeScan(fallbackSum.data(), LaneMixer::TABLE_SIZE);
            } else if (mode == LaneMixer::MixerMode::Series) {
                laneMixer.computeSeries(fallbackSum.data(), LaneMixer::TABLE_SIZE);
            } else {
                laneMixer.computeSum(fallbackSum.data(), LaneMixer::TABLE_SIZE);
            }
            sumPtr = &fallbackSum;
        }

        // Downsample from TABLE_SIZE (16384) to VISUALIZER_LUT_SIZE (1024)
        const auto& sum = *sumPtr;
        for (int i = 0; i < VISUALIZER_LUT_SIZE; ++i) {
            const double frac = i / static_cast<double>(VISUALIZER_LUT_SIZE - 1);
            const double srcIdx = frac * (LaneMixer::TABLE_SIZE - 1);
            const int idx = static_cast<int>(srcIdx);
            const int nextIdx = std::min(idx + 1, LaneMixer::TABLE_SIZE - 1);
            const double t = srcIdx - idx;
            (*visualizerLUTPtr)[static_cast<size_t>(i)] =
                sum[static_cast<size_t>(idx)] * (1.0 - t) + sum[static_cast<size_t>(nextIdx)] * t;
        }
    }

    // Compute selected lane's raw curve for secondary visualizer overlay
    if (laneLUTPtr_ != nullptr && selectedLanePtr_ != nullptr) {
        const int laneIdx = *selectedLanePtr_;
        if (laneIdx >= 0 && laneIdx < laneMixer.getActiveLaneCount()) {
            const auto& lane = laneMixer.getLane(laneIdx);
            for (int i = 0; i < VISUALIZER_LUT_SIZE; ++i) {
                const double frac = i / static_cast<double>(VISUALIZER_LUT_SIZE - 1);
                const double srcIdx = frac * (LaneMixer::TABLE_SIZE - 1);
                const int idx = static_cast<int>(srcIdx);
                const int nextIdx = std::min(idx + 1, LaneMixer::TABLE_SIZE - 1);
                const double t = srcIdx - idx;
                (*laneLUTPtr_)[static_cast<size_t>(i)] = lane.curveData[static_cast<size_t>(idx)] * (1.0 - t) +
                                                         lane.curveData[static_cast<size_t>(nextIdx)] * t;
            }
        }
    }

    if (onVisualizerUpdate) {
        onVisualizerUpdate();
    }
}

} // namespace dsp_core
