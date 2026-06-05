#include "AudioPipelineBuilder.h"
#include <juce_core/juce_core.h>

namespace dsp_core::audio_pipeline {

AudioPipelineBuilder::AudioPipelineBuilder() : pipeline_(std::make_unique<AudioPipeline>()) {}

AudioPipelineBuilder& AudioPipelineBuilder::withDryWetMix() {
    jassert(!built_ && "Cannot modify builder after build()");
    jassert(pipeline_->getNumStages() == 0 && "withDryWetMix() must be called before adding stages");
    useDryWetMix_ = true;
    return *this;
}

AudioPipelineBuilder& AudioPipelineBuilder::withOversampling(int initialOrder) {
    jassert(!built_ && "Cannot modify builder after build()");
    useOversampling_ = true;
    oversamplingOrder_ = initialOrder;
    return *this;
}

AudioPipelineBuilder& AudioPipelineBuilder::beginOversampledGroup(StageTag tag, int order) {
    jassert(!built_ && "Cannot modify builder after build()");
    jassert(!inOversampledGroup_ && "beginOversampledGroup() already open");
    jassert(!useOversampling_ && "Cannot mix withOversampling() with an interior oversampled group");

    inOversampledGroup_ = true;
    groupTag_ = tag;
    groupOrder_ = order;
    groupPipeline_ = std::make_unique<AudioPipeline>();
    return *this;
}

AudioPipelineBuilder& AudioPipelineBuilder::endOversampledGroup() {
    jassert(!built_ && "Cannot modify builder after build()");
    jassert(inOversampledGroup_ && "endOversampledGroup() without matching beginOversampledGroup()");

    auto wrapper = std::make_unique<OversamplingWrapper>(std::move(groupPipeline_), groupOrder_);
    auto* rawPtr = wrapper.get();

    pipeline_->addStage(std::move(wrapper), groupTag_);
    handles_.addHandle(groupTag_, rawPtr);
    handles_.setOversampling(rawPtr);

    inOversampledGroup_ = false;
    return *this;
}

AudioPipelineBuilder::BuildResult AudioPipelineBuilder::build() {
    jassert(!built_ && "Builder already consumed");
    jassert(!inOversampledGroup_ && "build() called with an unclosed oversampled group");
    built_ = true;

    std::unique_ptr<AudioProcessingStage> result;

    if (useDryWetMix_) {
        auto* innerPipeline = pipeline_.get();
        handles_.setEffectsPipeline(innerPipeline);

        auto dryWet = std::make_unique<DryWetMixStage>(std::move(pipeline_));
        handles_.setDryWetMix(dryWet.get());

        result = std::move(dryWet);
    } else {
        handles_.setEffectsPipeline(pipeline_.get());
        result = std::move(pipeline_);
    }

    if (useOversampling_) {
        auto oversampling = std::make_unique<OversamplingWrapper>(std::move(result), oversamplingOrder_);
        handles_.setOversampling(oversampling.get());
        return {std::move(oversampling), std::move(handles_)};
    }

    return {std::move(result), std::move(handles_)};
}

} // namespace dsp_core::audio_pipeline
