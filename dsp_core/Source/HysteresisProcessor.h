#pragma once

#include <juce_audio_basics/juce_audio_basics.h>
#include <cmath>
#include <functional>

namespace dsp_core {

/**
 * Jiles-Atherton hysteresis processor with custom nonlinearity injection.
 *
 * Implements the J-A magnetization model using RK4 integration. The standard
 * Langevin function can be replaced with an arbitrary nonlinearity (e.g.,
 * a user-drawn transfer function), which becomes the anhysteretic curve.
 *
 * Safety: All outputs are bounded, NaN/Inf inputs trigger state reset,
 * and an internal DC blocker prevents asymmetric NLs from causing runaway.
 *
 * Thread Safety:
 *   - process(): audio thread only (stateful, not reentrant)
 *   - setDrive/setSaturation/setWidth: UI thread only, before or between process calls
 *   - setNonlinearity(): not audio-thread safe (uses std::function); set during prepareToPlay
 *   - setNonlinearityDerivative(): same contract as setNonlinearity
 *   - setTransferFunction(): set once at construction, pointer must outlive processor
 */
class HysteresisProcessor {
  public:
    using NonlinearityFunc = std::function<double(double)>;
    using NonlinearityDerivFunc = std::function<double(double)>;

    HysteresisProcessor();

    void prepareToPlay(double sampleRate);
    void reset();
    double process(double inputH);

    // Inject arbitrary NL (not audio-thread safe — set during prepareToPlay)
    void setNonlinearity(NonlinearityFunc func);

    /**
     * Inject analytical NL derivative (not audio-thread safe).
     *
     * When set, langevinDeriv uses this instead of a central-difference estimate
     * on the custom NL. Essential when the NL has stateful side effects (e.g.,
     * Surge phase state) that make the central difference unreliable or
     * non-deterministic across repeated calls.
     *
     * The callback receives the same argument that NonlinearityFunc receives
     * (i.e. Q / scale_ after mapping).
     */
    void setNonlinearityDerivative(NonlinearityDerivFunc func);

    // Production: use SeamlessTransferFunction pointer (thread-safe via triple buffer)
    // void setTransferFunction(const SeamlessTransferFunction* tf);

    void setDrive(double drive);      // 0-1
    void setSaturation(double sat);   // 0-1
    void setWidth(double width);      // 0-1

    /**
     * Set fixed operating point for audio-range signals.
     * Ms controls the scaling between audio range and internal J-A dynamics:
     *   - a = Ms, so Q = H/Ms (normalize input to operating range)
     *   - LUT sees normalized signal directly (scale fixed at 1.0)
     *   - Output M tends toward Ms·L(Q), so Ms=1.0 keeps output in [-1,1]
     *
     * For audio plugins: call setOperatingPoint(1.0) — all scaling is identity.
     * Bypasses setDrive/setSaturation. Width (c) remains independently controllable.
     */
    void setOperatingPoint(double Ms);

    // Public for testing — these implement the core math
    double langevin(double Q) const;
    double langevinDeriv(double Q) const;

    /**
     * Compute dM/dt for the J-A model.
     * @param M      Current magnetization
     * @param H      Current field (may be RK4-interpolated)
     * @param dH_dt  Field derivative — constant across all RK4 stages (linear H interpolation)
     * @param delta   Branch direction (+1 ascending, -1 descending) — constant per sample
     */
    double dMdt(double M, double H, double dH_dt, double delta) const;

  private:
    // Jiles-Atherton state
    double M_n1_ = 0.0;      // previous magnetization
    double H_n1_ = 0.0;      // previous field
    double H_d_n1_ = 0.0;    // previous field derivative (legacy, kept for reset/tests)
    double lastDelta_ = 1.0;  // branch direction with deadband hold

    // Parameters (normalized, from ChowTape)
    double M_s_ = 1.0;
    double a_ = 0.25;
    double k_ = 0.47875;
    double c_ = 0.17;
    static constexpr double alpha_ = 1.6e-3;

    // Timing
    double T_ = 1.0 / (48000.0 * 16.0);
    double sampleRate_ = 48000.0 * 16.0;

    // H derivative (alpha-transform)
    static constexpr double dAlpha_ = 0.75;
    double Talpha_ = T_ / 1.9;

    // Input scaling: maps Q into transfer function range
    double scale_ = 1.0;

    // Custom nonlinearity (defaults to standard Langevin)
    NonlinearityFunc customNL_;
    bool useCustomNL_ = false;

    // Optional analytical derivative of the custom NL. When provided, replaces
    // the central-difference estimate used by langevinDeriv. Essential for NLs
    // with stateful side effects (e.g., Surge) since central difference would
    // advance that state twice per derivative evaluation.
    NonlinearityDerivFunc customNLDeriv_;
    bool useCustomNLDeriv_ = false;

    // Safety limits
    static constexpr double upperLimit_ = 20.0;
    static constexpr double outputLimit_ = 15.848931924611134;  // +24 dB, matches linear-extrapolation path
    static constexpr double inputLimit_ = 10.0;
    static constexpr double hdLimit_ = 1e6;
    int consecutiveResets_ = 0;
    int muteCountdown_ = 0;

    // Internal DC blocker
    double dcX_n1_ = 0.0;
    double dcY_n1_ = 0.0;
    double dcR_ = 0.995;  // Updated dynamically in prepareToPlay()
    double dcBlock(double x);

    // Operating point override (-1 = use auto scale a*4, >= 0 = fixed)
    double scaleOverride_ = -1.0;

    // Smoothed c parameter (prevents clicks on width changes)
    juce::SmoothedValue<double, juce::ValueSmoothingTypes::Linear> smoothedC_{0.17};

    // Derived parameter cache
    double M_s_oa_ = 0.0;      // M_s / a
    double M_s_oa_tc_ = 0.0;   // c * M_s / a

    void updateDerivedParams();
    double computeHDerivative(double H) const;

    // Standard Langevin (used when no custom NL is set)
    static double standardLangevin(double x);
    static double standardLangevinDeriv(double x);
};

} // namespace dsp_core
