#pragma once

#include "AudioProcessingStage.h"
#include <juce_dsp/juce_dsp.h>

namespace dsp_core::audio_pipeline {

/**
 * Applies gain with smoothing to prevent clicks.
 *
 * Features:
 * - Automatic channel count detection
 * - 10ms smoothing time for parameter changes
 * - dB-based gain control
 */
class GainStage : public AudioProcessingStage {
  public:
    GainStage() = default;

    void prepareToPlay(double sampleRate, int samplesPerBlock, int numChannels) override;
    void process(juce::AudioBuffer<double>& buffer) override;
    void reset() override;
    juce::String getName() const override {
        return "Gain";
    }

    /**
     * Set gain in decibels.
     * Smoothed over ~10ms to prevent clicks.
     */
    void setGainDB(double gainDB);

  private:
    juce::dsp::Gain<double> gainProcessor_;
    int numChannels_ = 2;
    bool isPrepared_ = false;
};

} // namespace dsp_core::audio_pipeline
