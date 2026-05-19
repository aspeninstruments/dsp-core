#include "BiasStage.h"

namespace dsp_core::audio_pipeline {

namespace {
constexpr double kBiasRampSeconds = 0.01; // 10 ms — matches GainStage / DryWetMix / Hysteresis makeup
} // namespace

void BiasStage::prepareToPlay(double sampleRate, int /*samplesPerBlock*/, int /*numChannels*/) {
    smoothedBias_.reset(sampleRate, kBiasRampSeconds);
    smoothedBias_.setCurrentAndTargetValue(bias_.load(std::memory_order_acquire));
}

void BiasStage::process(juce::AudioBuffer<double>& buffer) {
    const bool isEnabled = enabled_.load(std::memory_order_acquire);

    // Always ride the smoother toward the target — even when disabled, so
    // toggling DC blocking off mid-stream ramps the residual offset out of
    // the buffer instead of cutting it abruptly. Re-enabling later then
    // resumes from the actual current state without a jump.
    smoothedBias_.setTargetValue(isEnabled ? bias_.load(std::memory_order_acquire) : 0.0);

    const int numChannels = buffer.getNumChannels();
    const int numSamples = buffer.getNumSamples();

    // Hot path: if both current and target are at zero, the smoother has
    // nothing to do — skip the per-sample loop.
    if (!smoothedBias_.isSmoothing() && smoothedBias_.getCurrentValue() == 0.0) {
        return;
    }

    if (numChannels == 1) {
        auto* data = buffer.getWritePointer(0);
        for (int i = 0; i < numSamples; ++i) {
            data[i] += smoothedBias_.getNextValue();
        }
        return;
    }

    // Multi-channel: same per-sample bias for every channel. Stash each
    // sample's smoothed value so all channels read the same ramp point.
    for (int i = 0; i < numSamples; ++i) {
        const double b = smoothedBias_.getNextValue();
        for (int ch = 0; ch < numChannels; ++ch) {
            buffer.getWritePointer(ch)[i] += b;
        }
    }
}

void BiasStage::reset() {
    smoothedBias_.setCurrentAndTargetValue(bias_.load(std::memory_order_acquire));
}

} // namespace dsp_core::audio_pipeline
