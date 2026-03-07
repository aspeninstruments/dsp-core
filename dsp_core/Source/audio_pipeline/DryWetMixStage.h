#pragma once

#include "AudioProcessingStage.h"
#include "AudioPipeline.h"
#include <juce_audio_basics/juce_audio_basics.h>
#include <memory>

namespace dsp_core::audio_pipeline {

/**
 * Wrapper stage that applies dry/wet mixing to an effects pipeline with latency compensation.
 *
 * Wraps an AudioPipeline that typically contains input gain → effects → output gain.
 * Gain stages are only applied to the wet signal, so 0% mix = pure dry bypass.
 *
 * Latency compensation: The dry path is delayed by the same amount as the wet path
 * to prevent comb filtering when mixing the signals together.
 *
 * Usage:
 *   auto effectsPipeline = std::make_unique<AudioPipeline>();
 *   effectsPipeline->addStage(std::make_unique<GainStage>(), "inputGain");
 *   effectsPipeline->addStage(std::make_unique<WaveshapingStage>(ltf), "waveshaper");
 *   effectsPipeline->addStage(std::make_unique<GainStage>(), "outputGain");
 *
 *   auto mixed = std::make_unique<DryWetMixStage>(std::move(effectsPipeline));
 *   mixed->setMixAmount(0.5);  // 50% dry, 50% wet
 *
 *   // Access stages via tags
 *   auto* inputGain = mixed->getEffectsPipeline()->getStage<GainStage>("inputGain");
 *   inputGain->setGainDB(6.0);
 *
 * Signal flow: dry capture → delay buffer → mix with wet
 *             wet: effects pipeline → mix with dry
 */
class DryWetMixStage : public AudioProcessingStage {
  public:
    /**
     * Wrap an effects pipeline with dry/wet mixing.
     * @param effectsPipeline Pipeline containing the effects chain (e.g., input gain → waveshaper → output gain)
     */
    explicit DryWetMixStage(std::unique_ptr<AudioPipeline> effectsPipeline);

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void process(juce::AudioBuffer<double>& buffer) override;
    void reset() override;
    juce::String getName() const override;
    int getLatencySamples() const override;

    /**
     * Set mix amount.
     * @param mix 0.0 = 100% dry (bypass), 1.0 = 100% wet (full effect)
     */
    void setMixAmount(double mix);

    /**
     * Get current mix amount (target value, not the smoothed current value).
     */
    double getMixAmount() const {
        return mixSmoothed_.getTargetValue();
    }

    /**
     * Get the wrapped effects pipeline for tag-based stage access.
     * @return Non-owning pointer to effects pipeline
     */
    AudioPipeline* getEffectsPipeline();

  private:
    void captureDrySignal(const juce::AudioBuffer<double>& buffer);
    void applyMix(juce::AudioBuffer<double>& wetBuffer);

    std::unique_ptr<AudioPipeline> effectsPipeline_;
    juce::AudioBuffer<double> dryBuffer_;   // Current dry signal
    juce::AudioBuffer<double> delayBuffer_; // Circular buffer for latency compensation
    int delayBufferWritePos_ = 0;           // Write position in delay buffer
    juce::SmoothedValue<double, juce::ValueSmoothingTypes::Linear> mixSmoothed_; // 100% wet by default
};

} // namespace dsp_core::audio_pipeline
