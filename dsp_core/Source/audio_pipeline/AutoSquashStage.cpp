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
    const double targetRms = state_.targetPeakLinear.load(std::memory_order_acquire);
    const double enableStep = 1.0 / static_cast<double>(state_.enableFadeSamples);

    for (int n = 0; n < numSamples; ++n) {
        // Stereo-linked power detection: max(L², R²). Using power (sample²)
        // rather than absolute value lets the one-pole below act as a true
        // RMS averager once we sqrt at the end.
        double samplePower = 0.0;
        for (int ch = 0; ch < numChannels; ++ch) {
            const double s = buffer.getReadPointer(ch)[n];
            samplePower = std::max(samplePower, s * s);
        }

        // Asymmetric one-pole: fast on attack (samplePower above current
        // mean — level is rising, catch it quickly), slow on release (level
        // dropping — coast smoothly to avoid pumping). The slow release is
        // what makes this *not* a peak follower: a single sine cycle's
        // power dip does not pull the mean down, so the gain doesn't track
        // the per-cycle envelope and produce audio-rate ring modulation.
        const double alpha = (samplePower > state_.meanSquare) ? state_.rmsAttackAlpha : state_.rmsReleaseAlpha;
        state_.meanSquare += alpha * (samplePower - state_.meanSquare);

        const double envRms = std::sqrt(state_.meanSquare);

        // Derive target gain. Below the noise floor, slew gain toward 1.0 so
        // pure silence does not produce a wild gain.
        double rawGain = 1.0;
        if (envRms > k.noiseFloorLinear) {
            rawGain = targetRms / envRms;
            rawGain = std::min(rawGain, k.maxGainLinear);
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

        // Asymmetric one-pole on the applied gain. Instant when gain drops
        // (so RMS-driven level reductions take effect immediately on the
        // audio); smoothed when gain rises (cleans up any residual ripple
        // from the 1/envelope nonlinearity near the noise floor).
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
