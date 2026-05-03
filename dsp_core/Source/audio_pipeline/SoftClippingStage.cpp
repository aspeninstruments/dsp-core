#include "SoftClippingStage.h"

namespace dsp_core::audio_pipeline {

void SoftClippingStage::prepareToPlay(double /*sampleRate*/, int /*samplesPerBlock*/, int /*numChannels*/) {
    // No preparation needed - solver constants are computed at construction
}

void SoftClippingStage::process(juce::AudioBuffer<double>& buffer) {
    // Early return if disabled
    if (!enabled_.load(std::memory_order_acquire)) {
        return;
    }

    const int numChannels = buffer.getNumChannels();
    const int numSamples = buffer.getNumSamples();

    for (int ch = 0; ch < numChannels; ++ch) {
        auto* data = buffer.getWritePointer(ch);

        for (int i = 0; i < numSamples; ++i) {
            data[i] = solver_.process(data[i]);
        }
    }
}

void SoftClippingStage::reset() {
    // No state to reset - solver is stateless
}

} // namespace dsp_core::audio_pipeline
