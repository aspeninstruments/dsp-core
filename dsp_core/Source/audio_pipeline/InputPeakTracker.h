#pragma once

#include "AudioProcessingStage.h"
#include <atomic>
#include <cmath>

namespace dsp_core::audio_pipeline {

/**
 * Passthrough stage that tracks peak amplitude for visualization.
 *
 * Insert after InputGainStage to capture post-gain, pre-waveshaping amplitude.
 * Stores the peak amplitude atomically for UI thread access.
 *
 * Thread safety:
 * - process() writes with memory_order_release (audio thread)
 * - UI thread reads with memory_order_acquire via the atomic reference
 */
class InputPeakTracker : public AudioProcessingStage {
  public:
    /**
     * @param peakStorage Reference to atomic owned by processor.
     *                    Audio thread writes, UI thread reads.
     */
    explicit InputPeakTracker(std::atomic<double>& peakStorage) : peakStorage_(peakStorage) {}

    void prepareToPlay(double sampleRate, int samplesPerBlock, int numChannels) override {
        juce::ignoreUnused(sampleRate, samplesPerBlock, numChannels);
    }

    void process(juce::AudioBuffer<double>& buffer) override {
        double currentPeak = 0.0;

        for (int channel = 0; channel < buffer.getNumChannels(); ++channel) {
            const auto* data = buffer.getReadPointer(channel);
            for (int i = 0; i < buffer.getNumSamples(); ++i) {
                currentPeak = std::max(currentPeak, std::abs(data[i]));
            }
        }

        peakStorage_.store(currentPeak, std::memory_order_release);
    }

    void reset() override {
        peakStorage_.store(0.0, std::memory_order_release);
    }

    juce::String getName() const override {
        return "InputPeakTracker";
    }

  private:
    std::atomic<double>& peakStorage_;
};

} // namespace dsp_core::audio_pipeline
