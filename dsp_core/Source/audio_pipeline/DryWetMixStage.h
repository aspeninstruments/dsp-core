#pragma once

#include "AudioProcessingStage.h"
#include "AudioPipeline.h"
#include <juce_audio_basics/juce_audio_basics.h>
#include <atomic>
#include <memory>

namespace dsp_core::audio_pipeline {

class DryWetMixStage : public AudioProcessingStage {
  public:
    explicit DryWetMixStage(std::unique_ptr<AudioPipeline> effectsPipeline);

    void prepareToPlay(double sampleRate, int samplesPerBlock, int numChannels) override;
    void process(juce::AudioBuffer<double>& buffer) override;
    void reset() override;
    juce::String getName() const override;
    int getLatencySamples() const override;

    void setMixAmount(double mix);

    double getMixAmount() const {
        return mixSmoothed_.getTargetValue();
    }

    AudioPipeline* getEffectsPipeline();

    /**
     * Re-read the effects-pipeline latency and update the dry-path delay.
     *
     * The wet path runs through effectsPipeline_; if it carries latency (e.g. an
     * interior oversampler), the dry path is delayed by the same amount so the
     * parallel blend stays phase-aligned. prepareToPlay() does this once; call
     * this when the wet-path latency changes without a re-prepare (e.g. a runtime
     * oversampling-order change). Lock-free, no allocation.
     */
    void refreshLatencyCompensation();

  private:
    void captureDrySignal(const juce::AudioBuffer<double>& buffer);
    void applyMix(juce::AudioBuffer<double>& wetBuffer);

    std::unique_ptr<AudioPipeline> effectsPipeline_;
    juce::AudioBuffer<double> dryBuffer_;
    juce::SmoothedValue<double, juce::ValueSmoothingTypes::Linear> mixSmoothed_;

    // Dry-path latency-compensation delay line (per-channel ring). Sized to a
    // fixed power-of-two ceiling in prepareToPlay(); delaySamples_ == 0 takes a
    // direct-copy fast path so zero-latency wet chains are bit-identical.
    juce::AudioBuffer<double> delayLine_;
    int delayCapacity_ = 0;
    int delayWritePos_ = 0;
    std::atomic<int> delaySamples_{0};
};

} // namespace dsp_core::audio_pipeline
