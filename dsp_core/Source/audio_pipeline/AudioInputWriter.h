#pragma once

#include "AudioProcessingStage.h"
#include "AudioInputBuffer.h"

namespace dsp_core::audio_pipeline {

/**
 * Passthrough stage that writes samples to a buffer for UI visualization.
 *
 * Insert after InputGainStage to capture post-gain, pre-waveshaping audio.
 * The audio thread simply writes samples - peak detection happens on UI thread.
 *
 * Thread safety:
 * - Audio thread writes samples via process()
 * - UI thread reads samples via AudioInputBuffer::readAndComputePeak()
 * - Lock-free coordination via juce::AbstractFifo
 */
class AudioInputWriter : public AudioProcessingStage {
  public:
    /**
     * @param buffer Reference to AudioInputBuffer owned by processor
     */
    explicit AudioInputWriter(AudioInputBuffer& buffer) : buffer_(buffer) {}

    void prepareToPlay(double sampleRate, int samplesPerBlock, int numChannels) override {
        juce::ignoreUnused(sampleRate, samplesPerBlock, numChannels);
        // Buffer prepare is called separately by processor
    }

    void process(juce::AudioBuffer<double>& audioBuffer) override {
        // Simply write samples to buffer - UI thread handles peak detection
        buffer_.writeSamples(audioBuffer);
    }

    void reset() override {
        buffer_.reset();
    }

    juce::String getName() const override {
        return "AudioInputWriter";
    }

  private:
    AudioInputBuffer& buffer_;
};

} // namespace dsp_core::audio_pipeline
