#pragma once

#include "AudioProcessingStage.h"
#include "LadderRk4Solver.h"
#include "LadderSolver.h"
#include <array>
#include <atomic>
#include <cstdint>
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_dsp/juce_dsp.h>
#include <memory>

namespace dsp_core {
class SeamlessTransferFunction;
}

namespace dsp_core::audio_pipeline {

/**
 * NonlinearLadderStage — a virtual-analog Moog ladder whose per-stage
 * feedforward saturations and the global feedback saturation are user-drawable
 * transfer functions (SeamlessTransferFunction LUTs) instead of the fixed tanh
 * of VirtualAnalogFilterStage.
 *
 * This is a GENERALIZATION of VirtualAnalogFilterStage: same RK4 skeleton, same
 * cutoff/resonance smoothing pattern, same atomic UI/audio split. The
 * integration math lives in a pluggable LadderSolver strategy (RK4 in v1; a ZDF
 * solver slot is reserved).
 *
 * Nonlinearity slots (non-owning, supplied by the host/plugin):
 *   slot 0    : feedback nonlinearity   Phi_fb
 *   slot 1..4 : feedforward nonlinearity Phi_i for stages 1..4
 *
 * Thread Safety (mirrors VirtualAnalogFilterStage):
 *   - enabled_, cutoffHz_, resonance_, topology_, solverIndex_: atomic
 *     (UI writes, audio reads, explicit memory order).
 *   - solvers_ / STF pointers: set before audio starts, then read-only.
 *   - smoothers: audio thread only.
 *
 * The 5 STFs are prepared (prepareToPlay) and pumped (beginBlock once/block,
 * advanceCrossfadeSample once/sample GLOBALLY) by this stage so their LUT
 * crossfades track wall time at the (possibly oversampled) rate.
 */
class NonlinearLadderStage : public AudioProcessingStage {
  public:
    static constexpr int kMaxStages = LadderRk4Solver::kMaxStages;
    static constexpr int kNumSlots = 5;

    /** 2 ms cutoff/resonance smoothing (vs VirtualAnalogFilterStage's 0.2 ms —
     *  this stage sits at the head of a signal chain, not inside oversampling,
     *  and a longer ramp is click-safer for live topology/cutoff moves). */
    static constexpr double kSmoothingTimeSec = 0.002;

    // Resonance knob (0..1) maps to loop feedback R = kResonanceScale * v. 4.2
    // puts the top of the knob just past the R=4 self-oscillation threshold of a
    // tanh 4-pole ladder — matches the VirtualAnalog family's musical intent.
    static constexpr double kResonanceScale = 4.2;

    // Cutoff upper bound as a fraction of fs (the possibly-oversampled rate).
    // 0.18*fs keeps the ladder's RK4 well inside its stable region for any curve.
    static constexpr double kCutoffNyquistFraction = 0.18;
    static constexpr double kMinCutoffHz = 20.0;

    /** @param nls slot 0 = feedback, slots 1..4 = feedforward for stages 1..4.
     *  Non-owning; the pointed-to STFs must outlive this stage. */
    explicit NonlinearLadderStage(const std::array<const SeamlessTransferFunction*, kNumSlots>& nls);

    void prepareToPlay(double sampleRate, int samplesPerBlock, int numChannels) override;
    void process(juce::AudioBuffer<double>& buffer) override;
    void reset() override;
    juce::String getName() const override {
        return "NonlinearLadder";
    }

    // ---- UI-thread setters / getters (mirror VirtualAnalogFilterStage) -------

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

    /** Resonance knob position in [0, 1]. */
    void setResonance(double r);
    double getResonance() const {
        return resonance_.load(std::memory_order_acquire);
    }

    /** Active stage count, clamped to [1, 4]. */
    void setStageCount(int n);
    int getStageCount() const;

    /** Set one stage's output mode (LP/HP). stageIdx in [0, 3]. */
    void setStageMode(int stageIdx, Mode mode);
    Mode getStageMode(int stageIdx) const;

    /** Atomic set of the whole topology at once. numStages in [1,4], modes[0..3]. */
    void setTopology(int numStages, const std::array<Mode, kMaxStages>& modes);

    /** Solver selection. 0 = RK4 (default). 1 (ZDF) is reserved / not yet built. */
    void setSolverIndex(int idx);
    int getSolverIndex() const {
        return solverIndex_.load(std::memory_order_acquire);
    }

    /** Test-only read of the cutoff smoother's current value. */
    double getCurrentSmoothedCutoff() const {
        return smoothCutoff_.getCurrentValue();
    }

    /** Test hook: force channel state to NaN to exercise the recovery guard. */
    void injectNaNForTest(int ch);

    // ---- Topology packing (public so tests can assert round-trips) -----------
    static uint32_t packTopology(int numStages, const std::array<Mode, kMaxStages>& modes);
    static int unpackNumStages(uint32_t packed);
    static LadderTopologySnapshot unpackTopology(uint32_t packed);

  private:
    double maxCutoffHz() const;

    // Non-owning nonlinearity views (slot 0 = FB, 1..4 = FF).
    std::array<const SeamlessTransferFunction*, kNumSlots> nls_{};

    // DISTINCT non-null STF pointers among nls_, built once in prepareToPlay.
    // The three global lifecycle calls (prepareToPlay-forward, beginBlock,
    // advanceCrossfadeSample) must fire EXACTLY ONCE per underlying STF per
    // block/sample — when the plugin aliases the same STF into multiple slots
    // (one linked curve driving the whole filter), iterating nls_ directly
    // would over-trigger them (e.g. advance the shared crossfade N× too fast).
    // The per-eval applyTransferFunction reads in the solver stay keyed by the
    // original 5-slot indices (aliased reads are correct); only the lifecycle
    // pumps are deduplicated. Harmless when the 5 slots are distinct.
    std::array<const SeamlessTransferFunction*, kNumSlots> uniqueSlots_{};
    int numUniqueSlots_ = 0;

    std::atomic<bool> enabled_{true};
    std::atomic<double> cutoffHz_{20000.0};
    std::atomic<double> resonance_{0.0};
    std::atomic<uint32_t> topology_{0}; // packed numStages + mode bits
    std::atomic<int> solverIndex_{0};

    double sampleRate_ = 48000.0;
    double lastCutoffTarget_ = -1.0;
    double lastResonanceTarget_ = -1.0;

    juce::SmoothedValue<double, juce::ValueSmoothingTypes::Multiplicative> smoothCutoff_;
    juce::SmoothedValue<double, juce::ValueSmoothingTypes::Linear> smoothResonance_;

    // Both solvers pre-allocated. Slot 0 = RK4 (built). Slot 1 = ZDF (reserved,
    // nullptr for now). No allocation on the audio thread ever.
    std::array<std::unique_ptr<LadderSolver>, 2> solvers_{};

    int numChannels_ = 2;
};

} // namespace dsp_core::audio_pipeline
