#pragma once

#include "AudioPipeline.h"
#include <unordered_map>
#include <string>

namespace dsp_core::audio_pipeline {

// Forward declarations
class DryWetMixStage;
class OversamplingWrapper;
class AudioPipelineBuilder;

/**
 * Type-safe container for stage pointers returned by AudioPipelineBuilder.
 *
 * Stores raw pointers to stages owned by the built pipeline.
 * Lifetime: Valid as long as the associated AudioPipeline exists.
 *
 * Usage:
 *   auto [pipeline, stages] = AudioPipelineBuilder()
 *       .addStage<GainStage>(StageTag::InputGain)
 *       .build();
 *   auto* gain = stages.get<GainStage>(StageTag::InputGain);
 */
class StageHandles {
  public:
    /**
     * Get a stage by tag with type checking.
     * @tparam StageType Expected stage type (must derive from AudioProcessingStage)
     * @param tag StageTag used when adding the stage
     * @return Pointer to stage, or nullptr if not found or type mismatch
     */
    template <typename StageType>
    StageType* get(StageTag tag) const {
        auto it = handles_.find(stageTagToString(tag));
        if (it == handles_.end()) {
            return nullptr;
        }
        return dynamic_cast<StageType*>(it->second);
    }

    /**
     * Check if a stage exists for the given tag.
     * @param tag StageTag to check
     * @return true if stage exists
     */
    bool has(StageTag tag) const {
        return handles_.find(stageTagToString(tag)) != handles_.end();
    }

    /**
     * Get the DryWetMixStage if pipeline was built with withDryWetMix().
     * @return Pointer to DryWetMixStage, or nullptr if not used
     */
    DryWetMixStage* getDryWetMix() const {
        return dryWetMix_;
    }

    AudioPipeline* getEffectsPipeline() const {
        return effectsPipeline_;
    }

    OversamplingWrapper* getOversampling() const {
        return oversampling_;
    }

  private:
    friend class AudioPipelineBuilder;

    std::unordered_map<std::string, AudioProcessingStage*> handles_;
    DryWetMixStage* dryWetMix_ = nullptr;
    OversamplingWrapper* oversampling_ = nullptr;
    AudioPipeline* effectsPipeline_ = nullptr;

    void addHandle(StageTag tag, AudioProcessingStage* stage) {
        handles_[stageTagToString(tag)] = stage;
    }

    void setDryWetMix(DryWetMixStage* stage) {
        dryWetMix_ = stage;
    }

    void setOversampling(OversamplingWrapper* stage) {
        oversampling_ = stage;
    }

    void setEffectsPipeline(AudioPipeline* pipeline) {
        effectsPipeline_ = pipeline;
    }
};

} // namespace dsp_core::audio_pipeline
