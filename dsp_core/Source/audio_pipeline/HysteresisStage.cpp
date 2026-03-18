#include "HysteresisStage.h"

namespace dsp_core::audio_pipeline {

HysteresisStage::HysteresisStage(const dsp_core::SeamlessTransferFunction& tf)
    : transferFunction_(&tf) {}

void HysteresisStage::prepareToPlay(double sampleRate, int /*samplesPerBlock*/) {
    for (int ch = 0; ch < 2; ++ch) {
        processors_[ch].prepareToPlay(sampleRate);
        processors_[ch].setNonlinearity([this](double x) {
            return transferFunction_->applyTransferFunction(x);
        });
    }
}

void HysteresisStage::process(juce::AudioBuffer<double>& buffer) {
    const int numChannels = buffer.getNumChannels();
    const int numSamples = buffer.getNumSamples();

    if (numSamples == 0)
        return;

    if (!hysteresisEnabled_.load(std::memory_order_acquire)) {
        // Memoryless waveshaping fallback (identical to WaveshapingStage)
        transferFunction_->processBuffer(buffer);
        return;
    }

    // Per-channel, per-sample hysteresis processing with makeup gain
    const int channelsToProcess = std::min(numChannels, 2);
    for (int ch = 0; ch < channelsToProcess; ++ch) {
        auto* data = buffer.getWritePointer(ch);
        for (int i = 0; i < numSamples; ++i) {
            data[i] = processors_[ch].process(data[i]) * makeupGain_;
        }
    }
}

void HysteresisStage::reset() {
    for (auto& proc : processors_) {
        proc.reset();
    }
}

void HysteresisStage::setHysteresisEnabled(bool enabled) {
    hysteresisEnabled_.store(enabled, std::memory_order_release);
}

void HysteresisStage::setDrive(double drive) {
    for (auto& proc : processors_) {
        proc.setDrive(drive);
    }
}

void HysteresisStage::setSaturation(double sat) {
    for (auto& proc : processors_) {
        proc.setSaturation(sat);
    }
}

void HysteresisStage::setWidth(double width) {
    for (auto& proc : processors_) {
        proc.setWidth(width);
    }
}

void HysteresisStage::setMakeupGain(double gain) {
    makeupGain_ = gain;
}

} // namespace dsp_core::audio_pipeline
