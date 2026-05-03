#include "AudioPipeline.h"
#include <string>

namespace dsp_core::audio_pipeline {

void AudioPipeline::addStage(std::unique_ptr<AudioProcessingStage> stage, const std::string& tag) {
    jassert(stage != nullptr);

    std::string finalTag = tag;
    if (finalTag.empty()) {
        // Auto-generate tag
        finalTag = "stage_" + std::to_string(autoTagCounter_++);
    }

    // Store the stage index before moving
    const size_t index = stages_.size();
    tagToIndex_[finalTag] = index;
    stages_.push_back(std::move(stage));
}

void AudioPipeline::prepareToPlay(double sampleRate, int samplesPerBlock, int numChannels) {
    sampleRate_ = sampleRate;
    maxBlockSize_ = samplesPerBlock;

    for (auto& stage : stages_) {
        stage->prepareToPlay(sampleRate, samplesPerBlock, numChannels);
    }
}

void AudioPipeline::process(juce::AudioBuffer<double>& buffer) {
    for (auto& stage : stages_) {
        stage->process(buffer);
    }
}

void AudioPipeline::reset() {
    for (auto& stage : stages_) {
        stage->reset();
    }
}

juce::String AudioPipeline::getName() const {
    juce::String name = "Pipeline[";
    for (size_t i = 0; i < stages_.size(); ++i) {
        if (i > 0) {
            name += " -> ";
        }
        name += stages_[i]->getName();
    }
    name += "]";
    return name;
}

int AudioPipeline::getLatencySamples() const {
    int totalLatency = 0;
    for (const auto& stage : stages_) {
        totalLatency += stage->getLatencySamples();
    }
    return totalLatency;
}

void AudioPipeline::clear() {
    stages_.clear();
    tagToIndex_.clear();
    autoTagCounter_ = 0;
}

AudioProcessingStage* AudioPipeline::getStage(const std::string& tag) {
    auto it = tagToIndex_.find(tag);
    if (it == tagToIndex_.end()) {
        return nullptr;
    }

    const size_t index = it->second;
    jassert(index < stages_.size());
    return stages_[index].get();
}

void AudioPipeline::addStage(std::unique_ptr<AudioProcessingStage> stage, StageTag tag) {
    addStage(std::move(stage), stageTagToString(tag));
}

AudioProcessingStage* AudioPipeline::getStage(StageTag tag) {
    return getStage(stageTagToString(tag));
}

} // namespace dsp_core::audio_pipeline
