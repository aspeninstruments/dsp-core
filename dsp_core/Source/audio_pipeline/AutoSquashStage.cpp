#include "AutoSquashStage.h"

#include <algorithm>
#include <cmath>

namespace dsp_core::audio_pipeline {

namespace {
// Match GainStage's 10 ms ramp so the squash target follows InputGain on the
// same trajectory the input gain stage itself uses.
constexpr double kTargetRampSeconds = 0.01;
} // namespace

void AutoSquashStage::prepareToPlay(double sampleRate, int samplesPerBlock) {
    state_.prepare(sampleRate, samplesPerBlock);
    smoothedTargetPeak_.reset(sampleRate, kTargetRampSeconds);
    smoothedTargetPeak_.setCurrentAndTargetValue(
        state_.targetPeakLinear.load(std::memory_order_acquire));
}

void AutoSquashStage::process(juce::AudioBuffer<double>& buffer) {
    const int numSamples = buffer.getNumSamples();
    const int numChannels = buffer.getNumChannels();
    if (numSamples == 0 || numChannels == 0) {
        return;
    }

    // Resize gainHistory if the host changed its block size on us. This
    // shouldn't happen mid-stream in a well-behaved host, but be defensive.
    if (state_.gainHistory.getNumSamples() < numSamples) {
        state_.gainHistory.setSize(1, numSamples, false, true, true);
    }

    auto* gainBuf = state_.gainHistory.getWritePointer(0);

    const auto& k = state_.k;
    const bool targetEnabled = state_.enabled.load(std::memory_order_acquire);
    smoothedTargetPeak_.setTargetValue(state_.targetPeakLinear.load(std::memory_order_acquire));
    const double enableStep = 1.0 / static_cast<double>(state_.enableFadeSamples);

    for (int n = 0; n < numSamples; ++n) {
        const double targetPeak = smoothedTargetPeak_.getNextValue();
        // Stereo-linked peak detection: max(|L|, |R|).
        double sampleAbs = 0.0;
        for (int ch = 0; ch < numChannels; ++ch) {
            const double s = buffer.getReadPointer(ch)[n];
            const double a = std::abs(s);
            sampleAbs = std::max(sampleAbs, a);
        }

        // Peak follower with smoothed attack: when sampleAbs exceeds the
        // current envelope, the envelope chases via attackAlpha rather than
        // snapping. This trades a brief overshoot above target on big level
        // jumps (the waveshaper saturates it — clipping for ~1-2 ms is fine)
        // for a non-instantaneous envelope, which collapses per-sample gain
        // steps and removes the audio-rate ring modulation. Hold (now ~40 ms)
        // refreshes on every attack iteration, so for sustained tones the
        // envelope rides at peak between cycles instead of releasing-and-re-
        // attacking each cycle.
        if (sampleAbs > state_.envelope) {
            state_.envelope += state_.attackAlpha * (sampleAbs - state_.envelope);
            state_.holdCounter = state_.holdSamples;
        } else if (state_.holdCounter > 0) {
            --state_.holdCounter;
        } else {
            state_.envelope += state_.releaseAlpha * (sampleAbs - state_.envelope);
        }

        // Derive target gain. Below the noise floor, slew gain toward 1.0 so
        // pure silence does not produce a wild gain.
        double rawGain = 1.0;
        if (state_.envelope > k.noiseFloorLinear) {
            rawGain = targetPeak / state_.envelope;
            rawGain = std::min(rawGain, k.maxGainLinear);
            // safety floor
            rawGain = std::max(rawGain, 1.0 / k.maxGainLinear);
        }

        // Enable crossfade: linearly mix between unity and the computed gain.
        const double targetMix = targetEnabled ? 1.0 : 0.0;
        if (state_.enableMix < targetMix) {
            state_.enableMix = std::min(targetMix, state_.enableMix + enableStep);
        } else if (state_.enableMix > targetMix) {
            state_.enableMix = std::max(targetMix, state_.enableMix - enableStep);
        }

        const double g = 1.0 + state_.enableMix * (rawGain - 1.0);

        // Asymmetric one-pole on the applied gain. When gain is dropping
        // (attack — input got louder), pass through instantly so the smoothed
        // envelope's reduction takes effect on the audio with no extra lag.
        // When gain is rising (release — input got quieter, gain ramps back
        // up), smooth across ~gainSmoothTauSeconds to spread out per-sample
        // steps that would otherwise become audible zipper through the
        // 1/envelope nonlinearity.
        if (g < state_.smoothedGain) {
            state_.smoothedGain = g;
        } else {
            state_.smoothedGain += state_.gainSmoothAlpha * (g - state_.smoothedGain);
        }
        const double appliedGain = state_.smoothedGain;

        gainBuf[n] = appliedGain;
        for (int ch = 0; ch < numChannels; ++ch) {
            buffer.getWritePointer(ch)[n] *= appliedGain;
        }
    }
}

void AutoSquashStage::reset() {
    state_.resetRuntime();
    smoothedTargetPeak_.setCurrentAndTargetValue(
        state_.targetPeakLinear.load(std::memory_order_acquire));
}

} // namespace dsp_core::audio_pipeline
