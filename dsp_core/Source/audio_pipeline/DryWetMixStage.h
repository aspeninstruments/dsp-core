#pragma once

#include "AudioProcessingStage.h"
#include "AudioPipeline.h"
#include "OversamplingWrapper.h"
#include <juce_audio_basics/juce_audio_basics.h>
#include <array>
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
     * Match the dry path's oversampling order to the wet path's core.
     *
     * The dry signal is run through an identical half-band up/downsample so it
     * picks up the SAME frequency-dependent IIR phase as the wet path's interior
     * oversampler — a pure sample delay only matches group delay, not the IIR's
     * per-frequency phase, and would comb-filter the blend at HF. Call this with
     * the core's oversampling order whenever it changes (incl. at startup).
     * Also refreshes the residual latency compensation. Order 0 = passthrough.
     */
    void setDryOversamplingOrder(int order);

    /**
     * Re-read the residual latency mismatch (wet-path latency minus the dry
     * oversampler's own latency) and update the dry-path delay. The dry
     * oversampler already imparts the shared up/down latency; this only covers
     * any extra wet-path latency on top. Lock-free, no allocation.
     */
    void refreshLatencyCompensation();

  private:
    void captureDrySignal(const juce::AudioBuffer<double>& buffer);
    void applyResidualDelay(int numChannels, int numSamples);
    void applyMix(juce::AudioBuffer<double>& wetBuffer);

    std::unique_ptr<AudioPipeline> effectsPipeline_;
    juce::AudioBuffer<double> dryBuffer_;
    juce::SmoothedValue<double, juce::ValueSmoothingTypes::Linear> mixSmoothed_;

    // Dry path runs through its own half-band up/down (wrapping an identity
    // stage) so it shares the wet core's exact IIR phase. Order tracks the core.
    std::unique_ptr<OversamplingWrapper> dryOversampler_;
    std::array<double*, 8> dryChannelPtrs_{}; // view into dryBuffer_ for the oversampler

    // Residual dry-path delay (per-channel ring) for any wet-path latency beyond
    // the shared up/down. Sized to a fixed power-of-two ceiling in prepareToPlay;
    // delaySamples_ == 0 takes a direct fast path (the common case here).
    juce::AudioBuffer<double> delayLine_;
    int delayCapacity_ = 0;
    int delayWritePos_ = 0;
    std::atomic<int> delaySamples_{0};
};

} // namespace dsp_core::audio_pipeline
