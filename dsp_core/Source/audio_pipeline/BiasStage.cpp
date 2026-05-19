#include "BiasStage.h"

namespace dsp_core::audio_pipeline {

void BiasStage::prepareToPlay(double /*sampleRate*/, int /*samplesPerBlock*/, int /*numChannels*/) {
    // State-free: nothing to prepare.
}

void BiasStage::process(juce::AudioBuffer<double>& buffer) {
    if (!enabled_.load(std::memory_order_acquire)) {
        return;
    }

    const double biasSnap = bias_.load(std::memory_order_acquire);
    if (biasSnap == 0.0) {
        return;
    }

    const int numChannels = buffer.getNumChannels();
    const int numSamples = buffer.getNumSamples();

    for (int ch = 0; ch < numChannels; ++ch) {
        auto* data = buffer.getWritePointer(ch);
        for (int i = 0; i < numSamples; ++i) {
            data[i] += biasSnap;
        }
    }
}

void BiasStage::reset() {
    // No per-channel state to clear.
}

} // namespace dsp_core::audio_pipeline
