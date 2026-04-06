#include "HysteresisStage.h"

namespace dsp_core::audio_pipeline {

HysteresisStage::HysteresisStage(const dsp_core::SeamlessTransferFunction& tf)
    : transferFunction_(&tf) {}

void HysteresisStage::prepareToPlay(double sampleRate, int /*samplesPerBlock*/) {
    for (int ch = 0; ch < 2; ++ch) {
        processors_[ch].prepareToPlay(sampleRate);
        processors_[ch].setOperatingPoint(1.0); // Audio-range: LUT sees raw signal
        processors_[ch].setNonlinearity([this](double x) {
            return transferFunction_->applyTransferFunction(x);
        });
    }

    smoothedMakeupGain_.reset(sampleRate, 0.01); // 10ms ramp
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

    // Check for new LUT at block start (crossfade lifecycle)
    transferFunction_->beginBlock();

    // Per-sample, per-channel hysteresis processing with makeup gain
    // Sample-outer loop ensures crossfade advances once per sample (stereo consistency)
    const int channelsToProcess = std::min(numChannels, 2);
    for (int i = 0; i < numSamples; ++i) {
        const double gain = smoothedMakeupGain_.getNextValue();

        for (int ch = 0; ch < channelsToProcess; ++ch) {
            auto* data = buffer.getWritePointer(ch);
            data[i] = processors_[ch].process(data[i]) * gain;
        }
        transferFunction_->advanceCrossfadeSample();
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
    smoothedMakeupGain_.setTargetValue(gain);
}

void HysteresisStage::setOperatingPoint(double Ms) {
    for (auto& proc : processors_) {
        proc.setOperatingPoint(Ms);
    }
}

} // namespace dsp_core::audio_pipeline
