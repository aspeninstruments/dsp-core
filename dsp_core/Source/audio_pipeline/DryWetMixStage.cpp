#include "DryWetMixStage.h"
#include <juce_core/juce_core.h>

namespace dsp_core::audio_pipeline {

namespace {
constexpr int kMaxChannels = 8; // Support stereo, 5.1, 7.1
constexpr double kMixRampTimeSeconds = 0.01; // 10ms ramp, matches GainStage
} // namespace

DryWetMixStage::DryWetMixStage(std::unique_ptr<AudioPipeline> effectsPipeline)
    : effectsPipeline_(std::move(effectsPipeline)) {
    jassert(effectsPipeline_ != nullptr);
}

void DryWetMixStage::prepareToPlay(double sampleRate, int samplesPerBlock) {
    dryBuffer_.setSize(kMaxChannels, samplesPerBlock, false, true, true);
    effectsPipeline_->prepareToPlay(sampleRate, samplesPerBlock);

    mixSmoothed_.reset(sampleRate, kMixRampTimeSeconds);
    mixSmoothed_.setCurrentAndTargetValue(1.0); // 100% wet by default

    const int latencySamples = effectsPipeline_->getLatencySamples();
    if (latencySamples > 0) {
        const int delayBufferSize = latencySamples + samplesPerBlock;
        delayBuffer_.setSize(kMaxChannels, delayBufferSize, false, true, true);
        delayBufferWritePos_ = 0;
    } else {
        delayBuffer_.setSize(0, 0);
    }
}

void DryWetMixStage::process(juce::AudioBuffer<double>& buffer) {
    const int numChannels = buffer.getNumChannels();
    const int numSamples = buffer.getNumSamples();
    const int latencySamples = effectsPipeline_->getLatencySamples();

    if (latencySamples > 0 && delayBuffer_.getNumSamples() > 0) {
        for (int ch = 0; ch < numChannels; ++ch) {
            const double* inputData = buffer.getReadPointer(ch);

            for (int i = 0; i < numSamples; ++i) {
                const int writeIdx = (delayBufferWritePos_ + i) % delayBuffer_.getNumSamples();
                delayBuffer_.setSample(ch, writeIdx, inputData[i]);
            }
        }

        const int readPos =
            (delayBufferWritePos_ + delayBuffer_.getNumSamples() - latencySamples) % delayBuffer_.getNumSamples();
        for (int ch = 0; ch < numChannels; ++ch) {
            double* dryData = dryBuffer_.getWritePointer(ch);

            for (int i = 0; i < numSamples; ++i) {
                const int readIdx = (readPos + i) % delayBuffer_.getNumSamples();
                dryData[i] = delayBuffer_.getSample(ch, readIdx);
            }
        }

        delayBufferWritePos_ = (delayBufferWritePos_ + numSamples) % delayBuffer_.getNumSamples();
    } else {
        captureDrySignal(buffer);
    }

    effectsPipeline_->process(buffer);
    applyMix(buffer);
}

void DryWetMixStage::reset() {
    effectsPipeline_->reset();
    dryBuffer_.clear();
    delayBuffer_.clear();
    delayBufferWritePos_ = 0;
}

juce::String DryWetMixStage::getName() const {
    return "DryWetMix(" + effectsPipeline_->getName() + ")";
}

int DryWetMixStage::getLatencySamples() const {
    return effectsPipeline_->getLatencySamples();
}

AudioPipeline* DryWetMixStage::getEffectsPipeline() {
    return effectsPipeline_.get();
}

void DryWetMixStage::setMixAmount(double mix) {
    mixSmoothed_.setTargetValue(juce::jlimit(0.0, 1.0, mix));
}

void DryWetMixStage::captureDrySignal(const juce::AudioBuffer<double>& buffer) {
    const int numChannels = buffer.getNumChannels();
    const int numSamples = buffer.getNumSamples();

    jassert(numChannels <= dryBuffer_.getNumChannels());
    jassert(numSamples <= dryBuffer_.getNumSamples());

    for (int ch = 0; ch < numChannels; ++ch) {
        dryBuffer_.copyFrom(ch, 0, buffer, ch, 0, numSamples);
    }
}

void DryWetMixStage::applyMix(juce::AudioBuffer<double>& wetBuffer) {
    const int numChannels = wetBuffer.getNumChannels();
    const int numSamples = wetBuffer.getNumSamples();

    for (int i = 0; i < numSamples; ++i) {
        const double wetGain = mixSmoothed_.getNextValue();
        const double dryGain = 1.0 - wetGain;

        for (int ch = 0; ch < numChannels; ++ch) {
            double* wetData = wetBuffer.getWritePointer(ch);
            const double* dryData = dryBuffer_.getReadPointer(ch);
            wetData[i] = wetGain * wetData[i] + dryGain * dryData[i];
        }
    }
}

} // namespace dsp_core::audio_pipeline
