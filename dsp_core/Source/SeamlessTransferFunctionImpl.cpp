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
    : juce::Thread("BDD-LUTRenderer"), laneMixer(mixer), audioEngine(engine) {
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());
    startThread(juce::Thread::Priority::high);
    startTimerHz(SAFETY_TIMER_HZ);
}

EventDrivenRenderer::~EventDrivenRenderer() {
    // Stop the worker FIRST so it can no longer touch laneMixer/audioEngine
    // when the owning Pimpl tears down those members after this destructor.
    signalThreadShouldExit();
    notify();
    stopThread(2000);
    cancelPendingUpdate();
    stopTimer();
}

void EventDrivenRenderer::forceRender() {
    // Synchronous render on the calling thread. Serializes against the
    // worker via renderMutex_ — preset-load paths call this while the
    // worker may already be mid-render in response to a separate trigger,
    // so the lock is required for correctness, not just a contract.
    std::lock_guard<std::mutex> lk(renderMutex_);
    doRender();
}

void EventDrivenRenderer::handleAsyncUpdate() {
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());
    // Message-thread fallback wake — direct callers use requestRender()
    // instead. The worker owns the 120 Hz rate gate, so over-notifying is
    // harmless.
    notify();
}

void EventDrivenRenderer::timerCallback() {
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());
    // 5 Hz safety net for any wake that didn't take the direct path.
    notify();
}

void EventDrivenRenderer::run() {
    while (!threadShouldExit()) {
        // Block until notify() or signalThreadShouldExit. Wakes are coalesced
        // by JUCE's WaitableEvent — many notifies before the worker runs
        // produce a single wake.
        wait(-1);
        if (threadShouldExit()) {
            break;
        }

        // 120 Hz gate: enforce a minimum interval between consecutive render
        // starts. Use Thread::sleep (NOT wait) — wait() returns early on
        // notify, which lets a flurry of audio-thread / slider-drag notifies
        // defeat the gate and re-enter doRender() before the audio thread
        // has had time to rotate the triple buffer through the previous
        // render. That race produces clicks/pops because the audio thread
        // can read a worker-target buffer while the worker is still writing
        // to it. The gate is timestamped at render START so the cap matches
        // the old message-thread rate-limiter (~120 Hz on continuous wakes).
        const double now = juce::Time::getMillisecondCounterHiRes();
        if (now < nextAllowedRenderTimeMs_) {
            const int remainingMs = static_cast<int>(nextAllowedRenderTimeMs_ - now) + 1;
            juce::Thread::sleep(remainingMs);
            if (threadShouldExit()) {
                break;
            }
        }

        const double renderStartMs = juce::Time::getMillisecondCounterHiRes();
        renderIfNeeded();
        // Advance the gate UNCONDITIONALLY — even when renderIfNeeded() skips
        // (version unchanged, mid-crossfade, scan-mode amplitude). Without
        // this, mid-crossfade skips combined with audio-thread notifies on
        // every audio block (applyMorphWithEnvelope → setModulationEnvValue
        // → onVersionChanged) cause the worker to busy-spin at 100% CPU
        // during every 5 ms crossfade window. The Priority::high worker
        // starves the realtime audio thread → buffer underrun → click.
        nextAllowedRenderTimeMs_ = renderStartMs + RENDER_MIN_INTERVAL_MS;
    }
}

bool EventDrivenRenderer::renderIfNeeded() {
    // Serialize against forceRender from the message thread. Cheap when
    // uncontended; brief block (one render's worth) when forceRender wins.
    std::lock_guard<std::mutex> lk(renderMutex_);

    const uint64_t currentFullVersion = laneMixer.getVersion();
    if (currentFullVersion == lastRenderedFullVersion) {
        return false; // Nothing changed
    }

    const uint64_t currentMixVersion = laneMixer.getMixVersion();

    // Detect whether only mix-related changes occurred (cheap to re-render).
    const bool curveChanged =
        (currentFullVersion - currentMixVersion) != (lastRenderedFullVersion - lastRenderedMixVersion);

    // In scan mode, amplitude changes don't affect the output (computeScan
    // ignores amplitudes). Skip the render to avoid unnecessary crossfade
    // artifacts from stale buffer data.
    if (!curveChanged && laneMixer.getMixerMode() == LaneMixer::MixerMode::Scan) {
        const double currentScanPos = laneMixer.getScanPosition();
        if (currentScanPos == lastRenderedScanPosition) {
            lastRenderedFullVersion = currentFullVersion;
            lastRenderedMixVersion = currentMixVersion;
            return false;
        }
    }

    // Skip if the audio engine is mid-crossfade — the next notify (or the 5 Hz
    // safety timer) will retry. This mirrors the two-speed worker contract
    // documented on AudioEngine::isCrossfading().
    if (audioEngine.isCrossfading()) {
        return false;
    }

    // Skip if the audio thread hasn't consumed our previous render yet
    // (newLUTReady still true). Otherwise we'd race its checkForNewLUT
    // rotation: it would read the stale workerTargetIndex (still pointing
    // at our last buffer), rotate that buffer to PRIMARY, then read it
    // while we're still writing — manifesting as bursts of garbage when
    // audio blocks are delayed by CPU pressure (slow laptop) or jitter.
    // The flag is the producer-consumer signal: worker writes when false,
    // audio consumes by setting it false again after rotation.
    if (audioEngine.getNewLUTReadyFlag().load(std::memory_order_acquire)) {
        return false;
    }

    doRender();
    return true;
}

void EventDrivenRenderer::doRender() {
    // Compute directly into the audio engine's worker-target LUT buffer.
    // The audio thread only swaps to this buffer when newLUTReady is set
    // (release/acquire), so the in-progress writes here are not yet visible
    // to the audio thread.
    // Acquire-load to pair with the audio thread's release-store after
    // rotation in checkForNewLUT — ensures we see the post-rotation buffer
    // index, not a stale value that points at a buffer audio is now using
    // as PRIMARY. The newLUTReady gate in renderIfNeeded() is the primary
    // protection against the rotation race; this is a belt-and-suspenders
    // measure on weakly-ordered architectures (Apple Silicon).
    const int targetIdx = audioEngine.getWorkerTargetIndexReference().load(std::memory_order_acquire);
    LUTBuffer* outputBuffer = &audioEngine.getLUTBuffers()[targetIdx];

    const auto mode = laneMixer.getMixerMode();
    if (mode == LaneMixer::MixerMode::Scan) {
        laneMixer.computeScan(outputBuffer->data.data(), TABLE_SIZE);
    } else if (mode == LaneMixer::MixerMode::Series) {
        laneMixer.computeSeries(outputBuffer->data.data(), TABLE_SIZE);
    } else {
        laneMixer.computeSum(outputBuffer->data.data(), TABLE_SIZE);
    }

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

    // Snapshot the freshly-rendered sum for the visualizer dispatcher under
    // mutex. Done before signalling the audio thread so the dispatcher's
    // reader doesn't see a torn buffer if it races with the next render.
    {
        std::lock_guard<std::mutex> lk(lastRenderedSumMutex_);
        std::copy(outputBuffer->data.begin(), outputBuffer->data.end(), lastRenderedSum_.begin());
        hasLastRenderedSum_ = true;
    }

    // Release-store newLUTReady — synchronizes all the writes above with the
    // audio thread's acquire-load on the next block.
    audioEngine.getNewLUTReadyFlag().store(true, std::memory_order_release);

    lastRenderedFullVersion = laneMixer.getVersion();
    lastRenderedMixVersion = laneMixer.getMixVersion();
    lastRenderedScanPosition = laneMixer.getScanPosition();

    // Notify visualizer dispatcher so automation-driven amplitude changes
    // (which bypass onVersionChanged) still update the UI promptly.
    // triggerAsyncUpdate is cross-thread-safe by JUCE contract.
    if (visualizerDispatcher_ != nullptr) {
        visualizerDispatcher_->triggerAsyncUpdate();
    }
}

bool EventDrivenRenderer::copyLastRenderedSum(std::array<double, TABLE_SIZE>& dest) const {
    std::lock_guard<std::mutex> lk(lastRenderedSumMutex_);
    if (!hasLastRenderedSum_) {
        return false;
    }
    dest = lastRenderedSum_;
    return true;
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
        // LaneMixer recompute. The renderer's worker thread writes to that
        // buffer under a mutex; copyLastRenderedSum copies it under the
        // same mutex so we can downsample without holding the lock during
        // the loop. Fallback handles the (rare) case where the dispatcher
        // is triggered without a prior render — e.g., before init completes.
        std::array<double, LaneMixer::TABLE_SIZE> sumSnapshot{};
        bool haveSnapshot =
            (sourceRenderer_ != nullptr) ? sourceRenderer_->copyLastRenderedSum(sumSnapshot) : false;

        if (!haveSnapshot) {
            const auto mode = laneMixer.getMixerMode();
            if (mode == LaneMixer::MixerMode::Scan) {
                laneMixer.computeScan(sumSnapshot.data(), LaneMixer::TABLE_SIZE);
            } else if (mode == LaneMixer::MixerMode::Series) {
                laneMixer.computeSeries(sumSnapshot.data(), LaneMixer::TABLE_SIZE);
            } else {
                laneMixer.computeSum(sumSnapshot.data(), LaneMixer::TABLE_SIZE);
            }
        }

        // Downsample from TABLE_SIZE (16384) to VISUALIZER_LUT_SIZE (1024)
        const auto& sum = sumSnapshot;
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
