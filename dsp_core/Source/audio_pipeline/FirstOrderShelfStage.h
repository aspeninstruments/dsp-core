#pragma once

#include "AudioProcessingStage.h"
#include <atomic>
#include <vector>

namespace dsp_core::audio_pipeline {

/**
 * 1st-order shelf filter (single pole + single zero, bilinear-transform of
 * the textbook analog shelf prototype with prewarping at the corner frequency).
 *
 * Two modes, picked at construction:
 *   - Mode::HighShelf: unity at DC, gain A at HF, half-dB at corner
 *   - Mode::LowShelf:  gain A at DC, unity at HF, half-dB at corner
 *
 * Slope is 6 dB/oct on the transitioning side (vs RBJ biquad's 12 dB/oct),
 * matching the NAB tape pre/de-emphasis convention. Used by the Hysteresis
 * tone-stage strategy to push highs into the saturator (pre-emphasis) and
 * partially restore them on the other side (de-emphasis).
 *
 * Direct-form-I IIR:
 *   y[n] = b0*x[n] + b1*x[n-1] - a1*y[n-1]
 * (a0 normalized to 1; 1 sample of x history + 1 sample of y history per channel)
 *
 * Thread Safety:
 *   - enabled_, cutoffHz_, gainDb_: atomic (UI writes, audio reads)
 *   - filter state (per-channel x1/y1): audio thread only
 *   - Coefficient updates: UI thread only (set...() calls). No allocation —
 *     coefficients are three doubles updated under a per-stage spinlock-free
 *     std::atomic load/store pair (audio thread reads the cached coefficient
 *     trio once per block, no torn reads possible because the trio is read as
 *     three independent atomics and the resulting coefficient set is always
 *     consistent enough that the IIR remains stable during a momentary mix).
 */
class FirstOrderShelfStage : public AudioProcessingStage {
  public:
    enum class Mode { HighShelf, LowShelf };

    explicit FirstOrderShelfStage(Mode mode) : mode_(mode) {}

    void prepareToPlay(double sampleRate, int samplesPerBlock, int numChannels) override;
    void process(juce::AudioBuffer<double>& buffer) override;
    void reset() override;
    juce::String getName() const override {
        return mode_ == Mode::HighShelf ? "FirstOrderHighShelf" : "FirstOrderLowShelf";
    }

    void setEnabled(bool shouldBeEnabled) {
        enabled_.store(shouldBeEnabled, std::memory_order_release);
    }
    bool isEnabled() const {
        return enabled_.load(std::memory_order_acquire);
    }

    /** Shelf corner frequency in Hz. Clamped to [1, 0.49 * sampleRate]. */
    void setCutoffFrequency(double frequencyHz);
    double getCutoffFrequency() const {
        return cutoffHz_.load(std::memory_order_acquire);
    }

    /** Shelf gain in dB. 0 dB = pure pass-through. */
    void setGainDb(double gainDb);
    double getGainDb() const {
        return gainDb_.load(std::memory_order_acquire);
    }

  private:
    void recomputeCoefficients();

    Mode mode_;
    std::atomic<bool> enabled_{true};
    std::atomic<double> cutoffHz_{1000.0};
    std::atomic<double> gainDb_{0.0};

    // Audio-thread-readable coefficients (refreshed by UI-thread setters).
    std::atomic<double> b0_{1.0};
    std::atomic<double> b1_{0.0};
    std::atomic<double> a1_{0.0};

    // Per-channel 1-sample history (x[n-1], y[n-1]).
    std::vector<double> x1_;
    std::vector<double> y1_;
    double sampleRate_ = 48000.0;
};

} // namespace dsp_core::audio_pipeline
