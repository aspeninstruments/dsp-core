#include "NonlinearLadderStage.h"
#include "../SeamlessTransferFunction.h"
#include <cmath>
#include <limits>
#include <juce_core/juce_core.h>

namespace dsp_core::audio_pipeline {

namespace {
// Topology packing layout in a uint32_t:
//   bits 0..2  : numStages (stored as n-1, i.e. 0..3 -> 1..4 stages)
//   bits 8..11 : per-stage mode, bit set == Highpass (stage i -> bit 8+i)
constexpr uint32_t kNumStagesMask = 0x7u;
constexpr uint32_t kModeShift = 8u;
} // namespace

uint32_t NonlinearLadderStage::packTopology(int numStages,
                                            const std::array<Mode, kMaxStages>& modes) {
    const int n = juce::jlimit(1, kMaxStages, numStages);
    uint32_t packed = static_cast<uint32_t>(n - 1) & kNumStagesMask;
    for (int i = 0; i < kMaxStages; ++i) {
        if (modes[static_cast<std::size_t>(i)] == Mode::Highpass) {
            packed |= (1u << (kModeShift + static_cast<uint32_t>(i)));
        }
    }
    return packed;
}

int NonlinearLadderStage::unpackNumStages(uint32_t packed) {
    return static_cast<int>(packed & kNumStagesMask) + 1;
}

LadderTopologySnapshot NonlinearLadderStage::unpackTopology(uint32_t packed) {
    LadderTopologySnapshot topo;
    topo.numStages = unpackNumStages(packed);
    for (int i = 0; i < kMaxStages; ++i) {
        const bool hp = (packed & (1u << (kModeShift + static_cast<uint32_t>(i)))) != 0u;
        topo.modes[static_cast<std::size_t>(i)] = hp ? Mode::Highpass : Mode::Lowpass;
    }
    return topo;
}

NonlinearLadderStage::NonlinearLadderStage(
    const std::array<const SeamlessTransferFunction*, kNumSlots>& nls)
    : nls_(nls) {
    // Default topology: 4 stages, all Lowpass (classic 24 dB/oct ladder LP).
    const std::array<Mode, kMaxStages> lp{Mode::Lowpass, Mode::Lowpass, Mode::Lowpass,
                                          Mode::Lowpass};
    topology_.store(packTopology(kMaxStages, lp), std::memory_order_release);

    // Pre-allocate the RK4 solver; ZDF slot reserved (nullptr).
    solvers_[0] = std::make_unique<LadderRk4Solver>();
    solvers_[1] = nullptr;
}

double NonlinearLadderStage::maxCutoffHz() const {
    return std::max(kMinCutoffHz, sampleRate_ * kCutoffNyquistFraction);
}

void NonlinearLadderStage::prepareToPlay(double sampleRate, int samplesPerBlock, int numChannels) {
    sampleRate_ = sampleRate;
    numChannels_ = std::max(1, numChannels);

    // Forward to all 5 STFs so their LUT crossfade tracks wall time at this
    // (possibly oversampled) rate. prepareToPlay mutates; our slots are const
    // views (the audio-thread API applyTransferFunction/begin/advance is const),
    // so const_cast here on the UI thread is safe and intentional.
    for (const auto* nl : nls_) {
        if (nl != nullptr) {
            const_cast<SeamlessTransferFunction*>(nl)->prepareToPlay(sampleRate, samplesPerBlock);
        }
    }

    for (auto& solver : solvers_) {
        if (solver != nullptr) {
            solver->prepare(sampleRate, numChannels_);
        }
    }

    const double clampedCutoff =
        juce::jlimit(kMinCutoffHz, maxCutoffHz(), cutoffHz_.load(std::memory_order_acquire));
    cutoffHz_.store(clampedCutoff, std::memory_order_release);

    smoothCutoff_.reset(sampleRate, kSmoothingTimeSec);
    smoothResonance_.reset(sampleRate, kSmoothingTimeSec);
    smoothCutoff_.setCurrentAndTargetValue(clampedCutoff);
    smoothResonance_.setCurrentAndTargetValue(resonance_.load(std::memory_order_acquire));
    lastCutoffTarget_ = clampedCutoff;
    lastResonanceTarget_ = resonance_.load(std::memory_order_acquire);
}

void NonlinearLadderStage::process(juce::AudioBuffer<double>& buffer) {
    // Flush denormals — decaying integrator states enter the denormal range and
    // cost 10-100x normal arithmetic without FTZ.
    const juce::ScopedNoDenormals noDenormals;

    if (!enabled_.load(std::memory_order_acquire)) {
        return;
    }

    const int solverIdx = solverIndex_.load(std::memory_order_acquire);
    LadderSolver* solver =
        (solverIdx >= 0 && solverIdx < static_cast<int>(solvers_.size())) ? solvers_[static_cast<std::size_t>(solverIdx)].get() : nullptr;
    if (solver == nullptr) {
        // Reserved/unbuilt solver selected — fall back to RK4 (always present).
        solver = solvers_[0].get();
    }

    const double cutTarget = cutoffHz_.load(std::memory_order_acquire);
    const double resTarget = resonance_.load(std::memory_order_acquire);
    if (cutTarget != lastCutoffTarget_) {
        smoothCutoff_.setTargetValue(cutTarget);
        lastCutoffTarget_ = cutTarget;
    }
    if (resTarget != lastResonanceTarget_) {
        smoothResonance_.setTargetValue(resTarget);
        lastResonanceTarget_ = resTarget;
    }

    // Unpack topology once per block.
    const LadderTopologySnapshot topo =
        unpackTopology(topology_.load(std::memory_order_acquire));

    // beginBlock() once per block on all 5 STFs (picks up any newly rendered LUT).
    for (const auto* nl : nls_) {
        if (nl != nullptr) {
            nl->beginBlock();
        }
    }

    const int numChannels = std::min(buffer.getNumChannels(), numChannels_);
    const int numSamples = buffer.getNumSamples();
    constexpr double kTwoPi = 2.0 * juce::MathConstants<double>::pi;

    for (int i = 0; i < numSamples; ++i) {
        const double cutoff = smoothCutoff_.getNextValue();
        const double v = smoothResonance_.getNextValue();
        const double gNow = kTwoPi * cutoff;
        const double R = kResonanceScale * v;

        for (int ch = 0; ch < numChannels; ++ch) {
            const double in = buffer.getWritePointer(ch)[i];
            buffer.getWritePointer(ch)[i] =
                solver->processSample(in, ch, gNow, R, topo, nls_);
        }

        // advanceCrossfadeSample() once per SAMPLE, GLOBALLY (after all channels),
        // on all 5 STFs — the crossfade position is shared across channels.
        for (const auto* nl : nls_) {
            if (nl != nullptr) {
                nl->advanceCrossfadeSample();
            }
        }
    }

    // Per-block NaN guard: if any active channel produced a non-finite output,
    // reset the whole solver (cheap; only fires on numerical blowup).
    bool anyBad = false;
    for (int ch = 0; ch < numChannels && !anyBad; ++ch) {
        const double* d = buffer.getReadPointer(ch);
        for (int i = 0; i < numSamples; ++i) {
            if (!std::isfinite(d[i])) {
                anyBad = true;
                break;
            }
        }
    }
    if (anyBad) {
        solver->reset();
        // Zero the block so downstream never sees NaN/Inf.
        for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
            juce::FloatVectorOperations::clear(buffer.getWritePointer(ch), numSamples);
        }
    }
}

void NonlinearLadderStage::reset() {
    for (auto& solver : solvers_) {
        if (solver != nullptr) {
            solver->reset();
        }
    }
}

void NonlinearLadderStage::setCutoffFrequency(double frequencyHz) {
    frequencyHz = juce::jlimit(kMinCutoffHz, maxCutoffHz(), frequencyHz);
    cutoffHz_.store(frequencyHz, std::memory_order_release);
}

void NonlinearLadderStage::setResonance(double r) {
    r = juce::jlimit(0.0, 1.0, r);
    resonance_.store(r, std::memory_order_release);
}

void NonlinearLadderStage::setStageCount(int n) {
    n = juce::jlimit(1, kMaxStages, n);
    uint32_t packed = topology_.load(std::memory_order_acquire);
    packed = (packed & ~kNumStagesMask) | (static_cast<uint32_t>(n - 1) & kNumStagesMask);
    topology_.store(packed, std::memory_order_release);
}

int NonlinearLadderStage::getStageCount() const {
    return unpackNumStages(topology_.load(std::memory_order_acquire));
}

void NonlinearLadderStage::setStageMode(int stageIdx, Mode mode) {
    if (stageIdx < 0 || stageIdx >= kMaxStages) {
        return;
    }
    uint32_t packed = topology_.load(std::memory_order_acquire);
    const uint32_t bit = (1u << (kModeShift + static_cast<uint32_t>(stageIdx)));
    if (mode == Mode::Highpass) {
        packed |= bit;
    } else {
        packed &= ~bit;
    }
    topology_.store(packed, std::memory_order_release);
}

Mode NonlinearLadderStage::getStageMode(int stageIdx) const {
    if (stageIdx < 0 || stageIdx >= kMaxStages) {
        return Mode::Lowpass;
    }
    const uint32_t packed = topology_.load(std::memory_order_acquire);
    const bool hp = (packed & (1u << (kModeShift + static_cast<uint32_t>(stageIdx)))) != 0u;
    return hp ? Mode::Highpass : Mode::Lowpass;
}

void NonlinearLadderStage::setTopology(int numStages,
                                       const std::array<Mode, kMaxStages>& modes) {
    topology_.store(packTopology(numStages, modes), std::memory_order_release);
}

void NonlinearLadderStage::setSolverIndex(int idx) {
    solverIndex_.store(idx, std::memory_order_release);
}

void NonlinearLadderStage::injectNaNForTest(int ch) {
    // Poke the RK4 solver's state directly (test-only path).
    if (auto* rk4 = dynamic_cast<LadderRk4Solver*>(solvers_[0].get())) {
        rk4->pokeStateForTest(ch, std::numeric_limits<double>::quiet_NaN());
    }
}

} // namespace dsp_core::audio_pipeline
