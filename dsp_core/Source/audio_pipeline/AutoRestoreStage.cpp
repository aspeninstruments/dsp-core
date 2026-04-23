#include "AutoRestoreStage.h"

namespace dsp_core::audio_pipeline {

void AutoRestoreStage::prepareToPlay(double sampleRate, int samplesPerBlock) {
    // State is shared with AutoSquashStage; recompute coefficients is
    // idempotent but harmless. Do NOT reset envelope/enableMix here — the
    // squash side owns those.
    state_.recomputeCoefficients(sampleRate);
    if (state_.gainHistory.getNumSamples() < samplesPerBlock)
        state_.gainHistory.setSize(1, samplesPerBlock, false, true, true);
}

void AutoRestoreStage::process(juce::AudioBuffer<double>& buffer) {
    const int numSamples = buffer.getNumSamples();
    const int numChannels = buffer.getNumChannels();
    if (numSamples == 0 || numChannels == 0)
        return;

    const auto* gainBuf = state_.gainHistory.getReadPointer(0);

    for (int n = 0; n < numSamples; ++n) {
        const double g = gainBuf[n];
        // Guard against degenerate cases. g is clamped to [1/max, max] by the
        // squash side, so this is a cheap safety net, not a hot path.
        const double inv = (g > 0.0) ? (1.0 / g) : 1.0;
        for (int ch = 0; ch < numChannels; ++ch)
            buffer.getWritePointer(ch)[n] *= inv;
    }
}

void AutoRestoreStage::reset() {
    // Restore does not own the envelope state; squash resets it.
}

} // namespace dsp_core::audio_pipeline
