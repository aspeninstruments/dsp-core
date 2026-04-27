#pragma once

#include "AudioProcessingStage.h"
#include "AudioPipeline.h"
#include <juce_dsp/juce_dsp.h>
#include <array>

namespace dsp_core::audio_pipeline {

/**
 * Wraps a pipeline or stage with oversampling.
 *
 * The wrapped pipeline processes at a higher sample rate,
 * then downsampled back to the original rate.
 *
 * Example:
 *   auto innerPipeline = std::make_unique<AudioPipeline>();
 *   innerPipeline->addStage(makeDryWet(makeWaveshaper()));
 *
 *   auto wrapped = std::make_unique<OversamplingWrapper>(
 *       std::move(innerPipeline),
 *       3  // 8x oversampling (2^3)
 *   );
 *
 *   mainPipeline.addStage(std::move(wrapped));
 */
class OversamplingWrapper : public AudioProcessingStage {
  public:
    /**
     * @param wrappedStage Stage to process at oversampled rate
     * @param oversamplingOrder 0=1x, 1=2x, 2=4x, 3=8x, 4=16x
     */
    OversamplingWrapper(std::unique_ptr<AudioProcessingStage> wrappedStage,
                        int oversamplingOrder = 3 // 8x default
    );

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void process(juce::AudioBuffer<double>& buffer) override;
    void reset() override;
    juce::String getName() const override;
    int getLatencySamples() const override;

    /**
     * Change oversampling factor at runtime.
     * Call from UI thread only, triggers prepareToPlay() on wrapper.
     */
    void setOversamplingOrder(int order);

    int getOversamplingOrder() const {
        return currentOrder_;
    }

    // Factory helpers for common oversampling factors
    static constexpr int orderForNone() { return 0; }
    static constexpr int orderFor2x() { return 1; }
    static constexpr int orderFor4x() { return 2; }
    static constexpr int orderFor8x() { return 3; }
    static constexpr int orderFor16x() { return 4; }

  private:
    std::unique_ptr<AudioProcessingStage> wrappedStage_;

    // Pre-allocated oversamplers (1x, 2x, 4x, 8x, 16x)
    std::array<std::unique_ptr<juce::dsp::Oversampling<double>>, 5> oversamplers_;

    // Pre-allocated channel pointers array (avoid std::vector allocation per process call)
    std::array<double*, 8> channelPointers_{}; // Max 8 channels (7.1 surround)

    int currentOrder_ = 0; // 1x default (no oversampling)
    double sampleRate_ = 44100.0;
    int maxBlockSize_ = 512;
};

} // namespace dsp_core::audio_pipeline
