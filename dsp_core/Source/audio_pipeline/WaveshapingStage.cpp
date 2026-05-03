#include "WaveshapingStage.h"

namespace dsp_core::audio_pipeline {

WaveshapingStage::WaveshapingStage(dsp_core::SeamlessTransferFunction& tf) : seamlessTransferFunction_(&tf) {}

WaveshapingStage::WaveshapingStage(dsp_core::LayeredTransferFunction& ltf) : layeredTransferFunction_(&ltf) {}

void WaveshapingStage::prepareToPlay(double sampleRate, int samplesPerBlock, int /*numChannels*/) {
    // Forward to the shared transfer function at this (possibly oversampled) rate
    // so the LUT crossfade and surge-weight step both track wall time correctly.
    if (seamlessTransferFunction_ != nullptr) {
        seamlessTransferFunction_->prepareToPlay(sampleRate, samplesPerBlock);
    }
}

void WaveshapingStage::process(juce::AudioBuffer<double>& buffer) {
    if (seamlessTransferFunction_ != nullptr) {
        // Use new multi-channel processBuffer() API for correct crossfade handling
        seamlessTransferFunction_->processBuffer(buffer);
    } else if (layeredTransferFunction_ != nullptr) {
        // LayeredTransferFunction still uses per-channel processing
        const int numChannels = buffer.getNumChannels();
        const int numSamples = buffer.getNumSamples();

        for (int ch = 0; ch < numChannels; ++ch) {
            double* channelData = buffer.getWritePointer(ch);
            layeredTransferFunction_->processBlock(channelData, numSamples);
        }
    }
}

void WaveshapingStage::reset() {
    // Waveshaping is stateless
}

} // namespace dsp_core::audio_pipeline
