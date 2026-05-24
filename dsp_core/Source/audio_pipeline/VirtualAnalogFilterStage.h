#pragma once

#include "AudioProcessingStage.h"
#include "FilterStage.h"
#include "Tanh2xLUT.h"
#include <array>
#include <atomic>
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_dsp/juce_dsp.h>
#include <memory>
#include <vector>

namespace dsp_core::audio_pipeline {

/**
 * Wrapper that cascades up to 4 polymorphic FilterStages with RK4 +
 * tanh-saturated feedback from the last stage's output back to the input.
 *
 * Companion to LadderStage / OTAStage (templated, topology fixed at compile
 * time). This class allows heterogeneous cascades — e.g., a Ladder→OTA→Ladder
 * →OTA hybrid — at the cost of one vcall per stage per RK4 sub-step
 * (4 stages × 4 RK4 sub-steps = 16 vcalls/sample).
 *
 * Configuration (setStage, setUniform, setNumStages) is UI-thread only and
 * must complete before audio starts. Live topology swaps are not supported.
 *
 * Thread Safety:
 *   - enabled_, cutoffHz_, resonance_: atomic (UI writes, audio reads)
 *   - stages_, numStages_: set before audio thread starts, then read-only
 *   - channels_, smoothCutoff_, smoothResonance_: audio thread only
 */
class VirtualAnalogFilterStage : public AudioProcessingStage {
  public:
    static constexpr int kMaxStages = 4;

    enum class StageType { Ladder, OTA };

    VirtualAnalogFilterStage();

    void prepareToPlay(double sampleRate, int samplesPerBlock, int numChannels) override;
    void process(juce::AudioBuffer<double>& buffer) override;
    void reset() override;
    juce::String getName() const override {
        return "VirtualAnalog";
    }

    /** Replace stage at `idx`. UI thread only. `idx` must be in [0, kMaxStages). */
    void setStage(int idx, std::unique_ptr<FilterStage> stage);

    /** Convenience: fill all kMaxStages slots with the given topology. UI thread only. */
    void setUniform(StageType type);

    /** 1..kMaxStages. Upper-stage outputs short-circuit to numStages-1 for lower-order responses. */
    void setNumStages(int n);
    int getNumStages() const {
        return numStages_;
    }

    void setEnabled(bool shouldBeEnabled) {
        enabled_.store(shouldBeEnabled, std::memory_order_release);
    }
    bool isEnabled() const {
        return enabled_.load(std::memory_order_acquire);
    }

    void setCutoffFrequency(double frequencyHz);
    double getCutoffFrequency() const {
        return cutoffHz_.load(std::memory_order_acquire);
    }

    void setResonance(double r);
    double getResonance() const {
        return resonance_.load(std::memory_order_acquire);
    }

  private:
    struct ChannelState {
        std::array<double, kMaxStages> y{};
        double Vn_1 = 0.0;
        double g_n1 = 0.0;
    };

    inline std::array<double, kMaxStages> cascade(double V,
                                                  const std::array<double, kMaxStages>& s,
                                                  double gVal, double R) const noexcept;

    std::array<std::unique_ptr<FilterStage>, kMaxStages> stages_;
    int numStages_ = kMaxStages;

    std::atomic<bool> enabled_{true};
    std::atomic<double> cutoffHz_{20000.0};
    std::atomic<double> resonance_{0.0};

    double sampleRate_ = 48000.0;
    double T_ = 1.0 / 48000.0;
    double lastCutoffTarget_ = -1.0;
    double lastResonanceTarget_ = -1.0;

    juce::SmoothedValue<double, juce::ValueSmoothingTypes::Linear> smoothCutoff_;
    juce::SmoothedValue<double, juce::ValueSmoothingTypes::Linear> smoothResonance_;

    std::vector<ChannelState> channels_;
};

} // namespace dsp_core::audio_pipeline
