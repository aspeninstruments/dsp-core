#pragma once

#include "AudioProcessingStage.h"
#include <juce_dsp/juce_dsp.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <atomic>
#include <vector>

namespace dsp_core::audio_pipeline {

/**
 * 2nd-order parametric peaking ("bell") filter (RBJ biquad).
 *
 * Boosts or cuts a band centred on a tunable centre frequency, with the
 * bandwidth controlled by Q. Outside the bell's skirt the response is flat.
 *
 * Topology: juce::dsp::IIR::Coefficients<double>::makePeakFilter with
 * user-controllable Q. Higher Q narrows the bandwidth; lower Q broadens it.
 *
 * Pipeline Position: TBD — used by BellStrategy inside ToneStage.
 *
 * Thread Safety:
 *   - enabled_, centreHz_, gainDb_, q_: atomic (UI writes, audio reads)
 *   - filters_: per-instance, audio thread only
 *   - Coefficient updates: UI thread only (set...() calls), because
 *     juce::dsp::IIR::Coefficients::makePeakFilter returns a ReferenceCounted
 *     Ptr (shared_ptr — allocates). Mirrors LowShelfStage / HighShelfStage.
 */
class BellStage : public AudioProcessingStage {
  public:
    BellStage() = default;

    // AudioProcessingStage interface
    void prepareToPlay(double sampleRate, int samplesPerBlock, int numChannels) override;
    void process(juce::AudioBuffer<double>& buffer) override;
    void reset() override;
    juce::String getName() const override {
        return "Bell";
    }

    // Control interface (thread-safe)
    void setEnabled(bool shouldBeEnabled) {
        enabled_.store(shouldBeEnabled, std::memory_order_release);
    }

    bool isEnabled() const {
        return enabled_.load(std::memory_order_acquire);
    }

    /**
     * Set bell centre frequency in Hz. Clamped to [20, 20000].
     * Thread-safe: stores the target; the audio thread smooths toward it.
     */
    void setCutoffFrequency(double frequencyHz);

    double getCutoffFrequency() const {
        return centreHz_.load(std::memory_order_acquire);
    }

    /**
     * Set bell peak gain in dB. Clamped to [-24, +24]. 0 dB = transparent.
     * Thread-safe: stores the target; the audio thread smooths toward it.
     */
    void setGainDb(double gainDb);

    double getGainDb() const {
        return gainDb_.load(std::memory_order_acquire);
    }

    /**
     * Set bell Q (bandwidth). Clamped to [0.1, 10]. Lower Q broadens the
     * bell; higher Q narrows it.
     * Thread-safe: stores the target; the audio thread smooths toward it.
     */
    void setQ(double q);

    double getQ() const {
        return q_.load(std::memory_order_acquire);
    }

    // Smoothing ramp time for centre/gain/Q. Matches the Ladder filters' 2 ms —
    // long enough that per-sample coefficient updates are click-free, short
    // enough not to smear cutoff/gain modulation. Public for tests.
    static constexpr double kSmoothingTimeSec = 0.002;

  private:
    std::atomic<bool> enabled_{true};
    std::atomic<double> centreHz_{1000.0}; // Musically central default
    std::atomic<double> gainDb_{0.0};      // Transparent on construction
    std::atomic<double> q_{1.0};           // Moderately broad bell (~1.4 oct at -3 dB)

    // Per-channel IIR filters (audio thread only)
    std::vector<juce::dsp::IIR::Filter<double>> filters_;
    double sampleRate_ = 48000.0;

    // Per-sample parameter smoothing (audio thread only). Centre frequency uses
    // Multiplicative (log/pitch) smoothing; gain (dB) and Q use Linear. The
    // smoothers chase the atomic targets so coefficients move continuously
    // instead of jumping each block — that discontinuity was the click source.
    juce::SmoothedValue<double, juce::ValueSmoothingTypes::Multiplicative> smoothCutoff_;
    juce::SmoothedValue<double, juce::ValueSmoothingTypes::Linear> smoothGainDb_;
    juce::SmoothedValue<double, juce::ValueSmoothingTypes::Linear> smoothQ_;
    double lastCutoffTarget_ = -1.0;
    double lastGainTarget_ = -1000.0;
    double lastQTarget_ = -1.0;

    // Allocating coefficient sizer — used only in prepareToPlay and the
    // channel-count-change path. Sizes each filter's coefficient array to a
    // 2nd-order biquad so writeCoefficients() can update it in place afterwards.
    void updateFilterCoefficients();

    // Recompute the peaking-biquad coefficients in place from explicit
    // (already-smoothed) parameters. No allocation — audio-thread safe. Matches
    // juce::dsp::IIR::Coefficients<double>::makePeakFilter exactly.
    void writeCoefficients(double centreHz, double gainDb, double q);
};

} // namespace dsp_core::audio_pipeline
