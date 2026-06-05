#pragma once

#include "AudioPipeline.h"
#include "StageHandles.h"
#include "DryWetMixStage.h"
#include "OversamplingWrapper.h"
#include <memory>
#include <tuple>

namespace dsp_core::audio_pipeline {

class AudioPipelineBuilder {
  public:
    /**
     * Result of build(): ownership of pipeline + cached stage handles.
     */
    using BuildResult = std::tuple<std::unique_ptr<AudioProcessingStage>, StageHandles>;

    AudioPipelineBuilder();

    /**
     * Wrap the pipeline in a DryWetMixStage.
     * Must be called before addStage() calls.
     * @return Reference to this builder for chaining
     */
    AudioPipelineBuilder& withDryWetMix();

    AudioPipelineBuilder& withOversampling(int initialOrder);

    /**
     * Begin an interior oversampled group.
     *
     * Stages added between beginOversampledGroup() and endOversampledGroup()
     * are collected into a sub-pipeline wrapped by a single OversamplingWrapper,
     * inserted at this position in the parent pipeline. Stages before and after
     * the group run at the base sample rate.
     *
     * Use this (instead of withOversampling(), which wraps the WHOLE pipeline)
     * when only a contiguous middle section needs to run oversampled — e.g. a
     * nonlinear core flanked by linear/tone stages that gain nothing from
     * oversampling.
     *
     * @param tag Tag for retrieving the OversamplingWrapper (also registered as
     *            the StageHandles oversampling handle)
     * @param order Oversampling order (0=1x ... 4=16x)
     * @return Reference to this builder for chaining
     */
    AudioPipelineBuilder& beginOversampledGroup(StageTag tag, int order);

    /**
     * Close the interior oversampled group opened by beginOversampledGroup().
     * @return Reference to this builder for chaining
     */
    AudioPipelineBuilder& endOversampledGroup();

    /**
     * Add a stage to the pipeline.
     *
     * @tparam StageType The stage class (must derive from AudioProcessingStage)
     * @tparam Args Constructor argument types
     * @param tag Tag for later retrieval via StageHandles
     * @param args Arguments forwarded to StageType constructor
     * @return Reference to this builder for chaining
     */
    template <typename StageType, typename... Args> AudioPipelineBuilder& addStage(StageTag tag, Args&&... args) {
        jassert(!built_ && "Cannot modify builder after build()");

        auto stage = std::make_unique<StageType>(std::forward<Args>(args)...);
        auto* rawPtr = stage.get();

        targetPipeline().addStage(std::move(stage), tag);
        handles_.addHandle(tag, rawPtr);

        return *this;
    }

    /**
     * Add a wrapped stage (outer stage containing inner stage).
     *
     * Handles the common pattern where a wrapper stage (like OversamplingWrapper)
     * contains an inner processing stage (like WaveshapingStage).
     *
     * This safely caches the outer stage pointer before the move operation.
     *
     * @tparam OuterType Wrapper stage type (e.g., OversamplingWrapper)
     * @tparam InnerType Wrapped stage type (e.g., WaveshapingStage)
     * @tparam InnerArgs Constructor argument types for InnerType
     * @param tag Tag for retrieving the OUTER stage
     * @param innerArgs Arguments forwarded to InnerType constructor
     * @return Reference to this builder for chaining
     *
     * Note: OuterType must have constructor: OuterType(unique_ptr<AudioProcessingStage>)
     */
    template <typename OuterType, typename InnerType, typename... InnerArgs>
    AudioPipelineBuilder& addWrapped(StageTag tag, InnerArgs&&... innerArgs) {
        jassert(!built_ && "Cannot modify builder after build()");

        auto inner = std::make_unique<InnerType>(std::forward<InnerArgs>(innerArgs)...);
        auto outer = std::make_unique<OuterType>(std::move(inner));
        auto* rawPtr = outer.get();

        targetPipeline().addStage(std::move(outer), tag);
        handles_.addHandle(tag, rawPtr);

        return *this;
    }

    /**
     * Add a wrapped stage with additional outer constructor arguments.
     *
     * Use when the outer stage needs constructor arguments beyond the inner stage.
     *
     * @tparam OuterType Wrapper stage type
     * @tparam InnerType Wrapped stage type
     * @tparam OuterArgs Additional constructor argument types for OuterType
     * @tparam InnerArgs Constructor argument types for InnerType
     * @param tag Tag for retrieving the OUTER stage
     * @param outerArgs Tuple of additional args for OuterType (after inner stage)
     * @param innerArgs Arguments forwarded to InnerType constructor
     * @return Reference to this builder for chaining
     */
    template <typename OuterType, typename InnerType, typename... OuterArgs, typename... InnerArgs>
    AudioPipelineBuilder& addWrappedWithOuterArgs(StageTag tag, std::tuple<OuterArgs...> outerArgs,
                                                  InnerArgs&&... innerArgs) {

        jassert(!built_ && "Cannot modify builder after build()");

        auto inner = std::make_unique<InnerType>(std::forward<InnerArgs>(innerArgs)...);

        // Apply tuple as additional constructor arguments after inner stage
        auto outer = std::apply(
            [&inner](auto&&... args) {
                return std::make_unique<OuterType>(std::move(inner), std::forward<decltype(args)>(args)...);
            },
            std::move(outerArgs));

        auto* rawPtr = outer.get();
        targetPipeline().addStage(std::move(outer), tag);
        handles_.addHandle(tag, rawPtr);

        return *this;
    }

    /**
     * Build the pipeline and return ownership + stage handles.
     *
     * The builder is consumed after this call - do not reuse.
     *
     * @return Tuple of (pipeline ownership, stage handles)
     */
    BuildResult build();

  private:
    // Route addStage()/addWrapped*() into the open oversampled group when one is
    // active, otherwise into the main pipeline.
    AudioPipeline& targetPipeline() {
        return inOversampledGroup_ ? *groupPipeline_ : *pipeline_;
    }

    std::unique_ptr<AudioPipeline> pipeline_;
    StageHandles handles_;
    bool useDryWetMix_ = false;
    bool useOversampling_ = false;
    int oversamplingOrder_ = 0;
    bool built_ = false;

    // Interior oversampled group state (see beginOversampledGroup()).
    std::unique_ptr<AudioPipeline> groupPipeline_;
    bool inOversampledGroup_ = false;
    StageTag groupTag_ = StageTag::Oversampling;
    int groupOrder_ = 0;
};

} // namespace dsp_core::audio_pipeline
