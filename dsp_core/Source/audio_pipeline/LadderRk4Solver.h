#pragma once

#include "LadderSolver.h"
#include <array>
#include <vector>

namespace dsp_core::audio_pipeline {

/**
 * LadderRk4Solver — generalized RK4 integrator for the nonlinear ladder.
 *
 * This is a direct generalization of VirtualAnalogFilterStage::cascade + its
 * classic RK4 loop. The fixed per-stage tanh(2(x-s)) and feedback tanh(R*y) are
 * replaced by user LUTs Phi_i and Phi_fb, while the RK4 skeleton, the T=1/fs
 * convention, the peek-ahead chaining, and the "unused upper stages mirror the
 * last active stage" short-circuit are preserved bit-for-bit.
 *
 * ODE (per stage i, 1-based math / 0-based code):
 *     ds_i/dt = omega * Phi_i(x_i - s_i)          (s_i is always the LP state)
 *   In code the derivative contribution folds T:
 *     d_i = T * g * Phi_i(x_i - s_i),  g = cutoffRad = 2*pi*cutoff
 *
 * Output tap of stage i:
 *     o_i = s_i                 if mode_i == Lowpass
 *     o_i = x_i - s_i           if mode_i == Highpass   (complementary 1-pole HP)
 *
 * Input chain:
 *     x_1     = V - Phi_fb(R * y_fb)
 *     x_{i+1} = o_i
 *     output  = o_M            (M = numStages)
 *
 * Peek-ahead (matches the prototype's s[i-1] + d[i-1]): inside one derivative
 * eval, stage i+1's input uses o_i formed from the *peeked* state sigma_i + d_i.
 *
 * Feedback tap y_fb:
 *     final stage LP -> y_fb = s_M          (bit-compatible with the prototype
 *                                            which taps s[numStages-1])
 *     final stage HP -> y_fb = yFbPrev      (last committed o_M; breaks the
 *                                            delay-free algebraic loop that a HP
 *                                            final tap would otherwise create.
 *                                            A future ZDF solver fixes this exactly.)
 *
 * Safety clamps (Linear/Mirror LUT extrapolation in a feedback loop is not
 * passivity-safe): the feedback-NL output is clamped to +/-kFbOutMax and every
 * committed stage state to +/-kStateMax. Inaudible in normal operation; they
 * guarantee boundedness for any drawable curve.
 */
class LadderRk4Solver final : public LadderSolver {
  public:
    static constexpr int kMaxStages = 4;

    // Clamp the feedback nonlinearity output. tanh feedback is bounded to 1, but
    // a user curve under Linear/Mirror extrapolation can grow without bound; a
    // hot resonant loop must not diverge. 4.0 is far above any musical feedback.
    static constexpr double kFbOutMax = 4.0;

    // Clamp each committed integrator state. The classic ladder settles to |s|<=1
    // for |V|<=1; 16 leaves >24 dB of headroom for hot input + exotic curves while
    // still bounding a runaway.
    static constexpr double kStateMax = 16.0;

    LadderRk4Solver() = default;

    void prepare(double sampleRate, int numChannels) override;
    void reset() override;
    double processSample(double x, int ch, double cutoffRad, double R,
                         const LadderTopologySnapshot& topo,
                         const std::array<const SeamlessTransferFunction*, 5>& nls) override;

    /** Test hook: number of prepared channels. */
    int getNumChannels() const {
        return static_cast<int>(channels_.size());
    }

    /** Test hook: force a channel's stored state to a value (used to exercise the
     *  host stage's NaN-recovery guard). ch is bounds-checked. */
    void pokeStateForTest(int ch, double value);

  private:
    struct ChannelState {
        std::array<double, kMaxStages> s{}; // one-pole LP states
        double Vn_1 = 0.0;                  // previous input (for RK4 midpoint)
        double g_n1 = 0.0;                  // previous g (for RK4 midpoint)
        double yFbPrev = 0.0;               // last committed o_M (HP-final feedback tap)
    };

    // One RK4 derivative evaluation of the whole cascade. Returns d[i] for all
    // stages (T folded in). yFb is the feedback tap value chosen by the caller
    // (s_M for LP-final, yFbPrev for HP-final) so this eval is delay-free-loop free.
    inline std::array<double, kMaxStages>
    cascade(double V, const std::array<double, kMaxStages>& s, double gVal, double R, double yFb,
            const LadderTopologySnapshot& topo,
            const std::array<const SeamlessTransferFunction*, 5>& nls, int ch) const noexcept;

    double sampleRate_ = 48000.0;
    double T_ = 1.0 / 48000.0;
    std::vector<ChannelState> channels_;
};

} // namespace dsp_core::audio_pipeline
