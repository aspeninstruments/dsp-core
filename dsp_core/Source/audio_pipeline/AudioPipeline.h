#pragma once

#include "AudioProcessingStage.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <string>

namespace dsp_core::audio_pipeline {

/**
 * StageTag - Type-safe identifiers for pipeline stage retrieval
 *
 * Use these tags when adding stages that need later retrieval for parameter updates.
 * Stages without tags can still be added (auto-generated internal tag).
 */
enum class StageTag {
    InputGain,
    AudioInputWriter,
    SoftClip,
    Waveshaper,
    Oversampling,
    DCBlock,
    OutputGain,
    DryWetMix,
    Hysteresis,
    Surge,
    AutoSquash,
    AutoRestore,
    EnvelopeFollower,
};

/**
 * Convert StageTag to string for internal storage
 */
inline std::string stageTagToString(StageTag tag) {
    switch (tag) {
        case StageTag::InputGain:
            return "inputGain";
        case StageTag::AudioInputWriter:
            return "audioInputWriter";
        case StageTag::SoftClip:
            return "softClip";
        case StageTag::Waveshaper:
            return "waveshaper";
        case StageTag::Oversampling:
            return "oversampling";
        case StageTag::DCBlock:
            return "dcBlock";
        case StageTag::OutputGain:
            return "outputGain";
        case StageTag::DryWetMix:
            return "dryWetMix";
        case StageTag::Hysteresis:
            return "hysteresis";
        case StageTag::Surge:
            return "surge";
        case StageTag::AutoSquash:
            return "autoSquash";
        case StageTag::AutoRestore:
            return "autoRestore";
        case StageTag::EnvelopeFollower:
            return "envelopeFollower";
    }
    return "unknown";
}

/**
 * Serial pipeline of audio processing stages.
 *
 * Stages are executed in the order they were added.
 * Example:
 *   pipeline.addStage(makeGainStage(), StageTag::InputGain);
 *   pipeline.addStage(makeWaveshaperStage(), StageTag::Waveshaper);
 *   pipeline.addStage(makeHighpassStage(), StageTag::DCBlock);
 *   pipeline.process(buffer);  // Gain → Waveshaper → Highpass
 *
 * THREADING CONTRACT
 * ==================
 * - Setup Phase: addStage(), clear() - MUST complete before prepareToPlay()
 * - Audio Thread: process(), getLatencySamples() - lock-free reads
 * - Parameter Thread: getStage() for stage pointer access - safe after setup
 *
 * LIFETIME
 * ========
 * - Stages owned by pipeline (unique_ptr)
 * - Stage pointers from getStage() valid until clear() or destruction
 */
class AudioPipeline : public AudioProcessingStage {
  public:
    AudioPipeline() = default;

    /**
     * Add stage to end of pipeline with optional tag.
     * MUST be called before prepareToPlay().
     * @param stage Stage to add
     * @param tag Optional tag for retrieval (auto-generated if empty)
     */
    void addStage(std::unique_ptr<AudioProcessingStage> stage, const std::string& tag = "");

    /**
     * Prepare all stages for playback.
     */
    void prepareToPlay(double sampleRate, int samplesPerBlock) override;

    /**
     * Process buffer through all stages in order.
     */
    void process(juce::AudioBuffer<double>& buffer) override;

    /**
     * Reset all stages.
     */
    void reset() override;

    /**
     * Get name of pipeline (lists all stages).
     */
    juce::String getName() const override;

    /**
     * Get total latency of all stages combined.
     */
    int getLatencySamples() const override;

    /**
     * Get number of stages in pipeline.
     */
    int getNumStages() const {
        return static_cast<int>(stages_.size());
    }

    /**
     * Clear all stages (for reconstruction).
     */
    void clear();

    /**
     * Get stage by tag (type-erased).
     * @param tag Tag used when adding the stage
     * @return Non-owning pointer to stage, or nullptr if not found
     */
    AudioProcessingStage* getStage(const std::string& tag);

    /**
     * Get stage by tag (typed).
     * @tparam StageType Expected stage type
     * @param tag Tag used when adding the stage
     * @return Non-owning pointer to stage, or nullptr if not found or type mismatch
     */
    template <typename StageType> StageType* getStage(const std::string& tag) {
        auto* stage = getStage(tag);
        return dynamic_cast<StageType*>(stage);
    }

    // ========================================================================
    // Type-safe StageTag overloads (preferred API)
    // ========================================================================

    /**
     * Add stage to end of pipeline with type-safe tag.
     * MUST be called before prepareToPlay().
     * @param stage Stage to add
     * @param tag StageTag enum for retrieval
     */
    void addStage(std::unique_ptr<AudioProcessingStage> stage, StageTag tag);

    /**
     * Get stage by type-safe tag (type-erased).
     * @param tag StageTag enum used when adding the stage
     * @return Non-owning pointer to stage, or nullptr if not found
     */
    AudioProcessingStage* getStage(StageTag tag);

    /**
     * Get stage by type-safe tag (typed).
     * @tparam StageType Expected stage type
     * @param tag StageTag enum used when adding the stage
     * @return Non-owning pointer to stage, or nullptr if not found or type mismatch
     */
    template <typename StageType> StageType* getStage(StageTag tag) {
        return getStage<StageType>(stageTagToString(tag));
    }

  private:
    std::vector<std::unique_ptr<AudioProcessingStage>> stages_;
    std::unordered_map<std::string, size_t> tagToIndex_; // Tag -> stages_ index
    int autoTagCounter_ = 0;
    double sampleRate_ = 44100.0;
    int maxBlockSize_ = 512;
};

} // namespace dsp_core::audio_pipeline
