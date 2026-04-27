#include "AutoSquashStage.h"

#include <algorithm>
#include <cmath>

namespace dsp_core::audio_pipeline {

void AutoSquashStage::prepareToPlay(double sampleRate, int samplesPerBlock) {
    state_.prepare(sampleRate, samplesPerBlock);
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
    const double targetPeak = state_.targetPeakLinear.load(std::memory_order_acquire);
    const double enableStep = 1.0 / static_cast<double>(state_.enableFadeSamples);

    for (int n = 0; n < numSamples; ++n) {
        // Stereo-linked peak detection: max(|L|, |R|).
        double sampleAbs = 0.0;
        for (int ch = 0; ch < numChannels; ++ch) {
            const double s = buffer.getReadPointer(ch)[n];
            const double a = std::abs(s);
            sampleAbs = std::max(sampleAbs, a);
        }

        // Peak follower: instantaneous attack, smooth release after a hold
        // window. Any sample whose magnitude exceeds the current envelope
        // snaps the envelope up to it, which guarantees the internal signal
        // is bounded by target (no lookahead needed).
        if (sampleAbs > state_.envelope) {
            state_.envelope = sampleAbs;
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
        // (attack — input got louder, must reduce gain immediately to keep
        // the internal signal bounded by target), pass through instantly so
        // the no-overshoot guarantee is preserved. When gain is rising
        // (release — input got quieter, gain ramps back up), smooth across
        // ~gainSmoothTauSeconds to spread out per-sample steps that would
        // otherwise become audible zipper through the 1/envelope nonlinearity.
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
}

} // namespace dsp_core::audio_pipeline
