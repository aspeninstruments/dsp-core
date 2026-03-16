#pragma once

#include "AudioProcessingStage.h"
#include "AudioPipeline.h"
#include <juce_audio_basics/juce_audio_basics.h>
#include <memory>

namespace dsp_core::audio_pipeline {

class DryWetMixStage : public AudioProcessingStage {
  public:
    explicit DryWetMixStage(std::unique_ptr<AudioPipeline> effectsPipeline);

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void process(juce::AudioBuffer<double>& buffer) override;
    void reset() override;
    juce::String getName() const override;
    int getLatencySamples() const override;

    void setMixAmount(double mix);

    double getMixAmount() const {
        return mixSmoothed_.getTargetValue();
    }

    AudioPipeline* getEffectsPipeline();

  private:
    void captureDrySignal(const juce::AudioBuffer<double>& buffer);
    void applyMix(juce::AudioBuffer<double>& wetBuffer);

    std::unique_ptr<AudioPipeline> effectsPipeline_;
    juce::AudioBuffer<double> dryBuffer_;
    juce::SmoothedValue<double, juce::ValueSmoothingTypes::Linear> mixSmoothed_;
};

} // namespace dsp_core::audio_pipeline
