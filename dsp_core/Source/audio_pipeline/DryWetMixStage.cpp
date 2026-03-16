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
}

void DryWetMixStage::process(juce::AudioBuffer<double>& buffer) {
    captureDrySignal(buffer);
    effectsPipeline_->process(buffer);
    applyMix(buffer);
}

void DryWetMixStage::reset() {
    effectsPipeline_->reset();
    dryBuffer_.clear();
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
