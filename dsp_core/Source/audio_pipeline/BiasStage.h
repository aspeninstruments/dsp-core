#pragma once

#include "AudioProcessingStage.h"
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include <atomic>

namespace dsp_core::audio_pipeline {

/**
 * Constant DC-offset stage that pushes the signal into one rail of the
 * downstream transfer function (asymmetric drive). Bias ∈ [-1, 1] is added
 * to every sample, ramped over ~10 ms to eliminate audible clicks on
 * parameter changes or fast automation.
 *
 * Pipeline position: between Surge and the LUT/Hysteresis consumer. The
 * downstream DC blocking filter cancels the resulting DC at the output, so
 * the audible effect is purely the asymmetric distortion the bias induces
 * in the nonlinear shaper. Callers should gate enabled_ on the DC-blocking
 * filter being on, otherwise bias would leak DC out of the plugin.
 *
 * Thread Safety:
 *   - enabled_, bias_: atomic (UI writes, audio reads)
 *   - smoothedBias_: audio thread only (target pulled from bias_ each block)
 *   - process: audio thread only
 *   - prepareToPlay / reset: UI thread only
 */
class BiasStage : public AudioProcessingStage {
  public:
    BiasStage() = default;

    void prepareToPlay(double sampleRate, int samplesPerBlock, int numChannels) override;
    void process(juce::AudioBuffer<double>& buffer) override;
    void reset() override;
    juce::String getName() const override {
        return "Bias";
    }

    void setEnabled(bool shouldBeEnabled) {
        enabled_.store(shouldBeEnabled, std::memory_order_release);
    }

    bool isEnabled() const {
        return enabled_.load(std::memory_order_acquire);
    }

    void setBias(double biasValue) {
        bias_.store(biasValue, std::memory_order_release);
    }

    double getBias() const {
        return bias_.load(std::memory_order_acquire);
    }

  private:
    std::atomic<bool> enabled_{false};
    std::atomic<double> bias_{0.0};
    juce::SmoothedValue<double, juce::ValueSmoothingTypes::Linear> smoothedBias_{0.0};
};

} // namespace dsp_core::audio_pipeline
