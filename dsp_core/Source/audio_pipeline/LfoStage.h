#pragma once

#include "AudioProcessingStage.h"
#include <atomic>
#include <cstdint>

namespace dsp_core::audio_pipeline {

/**
 * Low-frequency oscillator. Audio is passthrough — like EnvelopeFollowerStage,
 * the stage publishes a unipolar [0, 1] modulation value to a processor-owned
 * atomic. Phase advances per sample either free-running (Hz mode) or locked to
 * the host ppq position (BPM mode).
 *
 * Thread safety:
 *   - All UI-writable state is held in atomics; audio thread only reads them.
 *   - setHostTransport() is called once per block from the audio thread before
 *     process(), so it's audio-thread only.
 */
class LfoStage : public AudioProcessingStage {
  public:
    enum class Shape : int { Off = 0, Sin = 1, Tri = 2, Saw = 3, Sq = 4, Random = 5 };
    enum class Units : int { Hz = 0, BPM = 1 };
    enum class Division : int { Sixteenth = 0, Eighth = 1, Quarter = 2, Half = 3, Bar = 4, TwoBar = 5, FourBar = 6 };
    enum class Flavor : int { Straight = 0, Triplet = 1, Dotted = 2 };

    explicit LfoStage(std::atomic<double>& lfoStorage);

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void process(juce::AudioBuffer<double>& buffer) override;
    void reset() override;
    juce::String getName() const override {
        return "Lfo";
    }

    void setEnabled(bool shouldBeEnabled) {
        enabled_.store(shouldBeEnabled, std::memory_order_release);
    }
    bool isEnabled() const {
        return enabled_.load(std::memory_order_acquire);
    }

    void setShape(Shape s) {
        shape_.store(static_cast<int>(s), std::memory_order_release);
    }
    void setUnits(Units u) {
        units_.store(static_cast<int>(u), std::memory_order_release);
    }
    void setRateHz(double hz) {
        rateHz_.store(hz, std::memory_order_release);
    }
    void setDivision(Division d) {
        division_.store(static_cast<int>(d), std::memory_order_release);
    }
    void setFlavor(Flavor f) {
        flavor_.store(static_cast<int>(f), std::memory_order_release);
    }
    void setSeed(unsigned int seed) {
        seed_.store(seed, std::memory_order_release);
    }

    /// Phase offset in normalized cycles (degrees / 360). Applied before wave
    /// evaluation so it shifts where the cycle starts relative to the beat in
    /// BPM mode (and the free-run cycle in Hz mode).
    void setPhaseOffset(double normalized) {
        phaseOffset_.store(normalized, std::memory_order_release);
    }

    /// Set once per block from the audio thread before process(). bpm and
    /// ppqPosition come straight from juce::AudioPlayHead. isPlaying gates
    /// host-locked phase derivation in BPM mode.
    void setHostTransport(double bpm, double ppqPosition, bool isPlaying) {
        hostBpm_ = bpm;
        hostPpq_ = ppqPosition;
        hostIsPlaying_ = isPlaying;
    }

    /// Beats-per-cycle for the configured Division and Flavor. Exposed for
    /// testing and for UI display.
    static double periodInBeats(Division d, Flavor f);

  private:
    static double evaluateShape(Shape s, double phase, unsigned int seed);

    std::atomic<double>& lfoStorage_;

    std::atomic<bool> enabled_{false};
    std::atomic<int> shape_{static_cast<int>(Shape::Sin)};
    std::atomic<int> units_{static_cast<int>(Units::Hz)};
    std::atomic<double> rateHz_{1.0};
    std::atomic<int> division_{static_cast<int>(Division::Quarter)};
    std::atomic<int> flavor_{static_cast<int>(Flavor::Straight)};
    std::atomic<double> phaseOffset_{0.0};
    std::atomic<unsigned int> seed_{0xA5F37B12u};

    double sampleRate_{48000.0};

    // Audio-thread-only state.
    double phase_{0.0};
    double hostBpm_{120.0};
    double hostPpq_{0.0};
    bool hostIsPlaying_{false};
};

} // namespace dsp_core::audio_pipeline
