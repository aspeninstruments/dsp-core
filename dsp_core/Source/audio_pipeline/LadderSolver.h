#pragma once

#include <array>
#include <cstdint>

namespace dsp_core {
class SeamlessTransferFunction;
}

namespace dsp_core::audio_pipeline {

/**
 * Strategy interface for the nonlinear ladder integrator.
 *
 * A LadderSolver advances one input sample through an up-to-4-stage Moog-style
 * ladder whose per-stage feedforward saturations and the global feedback
 * saturation are arbitrary user-drawable transfer functions (LUTs) rather than
 * the fixed tanh of the classic prototype.
 *
 * Two solvers share this interface:
 *   - LadderRk4Solver (index 0): classic RK4 over the ODE cascade. Ships in v1.
 *   - A future ZDF/Newton solver (index 1): resolves the HP-final algebraic loop
 *     exactly. Reserved; not implemented yet.
 *
 * Contract mirrors VirtualAnalogFilterStage:
 *   - prepare()/reset() are UI-thread (before audio) or reset-safe.
 *   - processSample() is audio-thread, lock-free, no allocation.
 *   - The 5 nonlinearity pointers are non-owning views into SeamlessTransferFunctions
 *     owned upstream (the host stage / plugin). The solver only reads them.
 */

/** One-pole stage output mode. Lowpass taps the LP state; Highpass taps
 *  (input - LP state) — the complementary one-pole HP. */
enum class Mode { Lowpass, Highpass };

/**
 * Immutable per-block description of the cascade topology.
 *   numStages  : number of active stages, 1..4.
 *   modes      : per-stage output mode (index 0 = stage 1, ... index 3 = stage 4).
 * Unpacked once per block from the host stage's packed atomic and passed by
 * const-ref into processSample so the hot loop never touches an atomic.
 */
struct LadderTopologySnapshot {
    int numStages = 4;
    std::array<Mode, 4> modes{Mode::Lowpass, Mode::Lowpass, Mode::Lowpass, Mode::Lowpass};
};

/**
 * Non-owning view of the 5 nonlinearity LUTs for one ladder.
 *   slots[0]   : feedback nonlinearity   Phi_fb  (applied to R * y_fb)
 *   slots[1..4]: feedforward nonlinearity Phi_i for stages 1..4 (applied to x_i - s_i)
 * Any slot may be null; a null slot is treated as identity by the solver's caller
 * contract (the host stage always supplies all 5, so the solver assumes non-null).
 */
struct NlSlots {
    std::array<const SeamlessTransferFunction*, 5> slots{};
};

class LadderSolver {
  public:
    virtual ~LadderSolver() = default;

    /** Allocate/size per-channel state for the (possibly oversampled) rate. */
    virtual void prepare(double sampleRate, int numChannels) = 0;

    /** Zero all per-channel state. */
    virtual void reset() = 0;

    /**
     * Advance one input sample for one channel.
     *
     * @param x          input sample
     * @param ch         channel index (into the solver's per-channel state)
     * @param cutoffRad  2*pi*cutoffHz for this sample (the existing g convention)
     * @param R          resonance feedback amount (already mapped from the knob)
     * @param topo       active stage count + per-stage modes for this block
     * @param nls        the 5 nonlinearity LUTs (slot 0 = FB, 1..4 = FF)
     * @return           the ladder output sample o_M (M = topo.numStages)
     */
    virtual double processSample(double x, int ch, double cutoffRad, double R,
                                 const LadderTopologySnapshot& topo,
                                 const std::array<const SeamlessTransferFunction*, 5>& nls) = 0;
};

} // namespace dsp_core::audio_pipeline
