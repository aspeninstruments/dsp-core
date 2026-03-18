#pragma once

#include "audio_pipeline/SoftClippingStage.h"
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
 *   - setNonlinearity(): test-only, not audio-thread safe (uses std::function)
 *   - setTransferFunction(): set once at construction, pointer must outlive processor
 */
class HysteresisProcessor {
  public:
    using NonlinearityFunc = std::function<double(double)>;

    HysteresisProcessor();

    void prepareToPlay(double sampleRate);
    void reset();
    double process(double inputH);

    // Test-only: inject arbitrary NL (not audio-thread safe)
    void setNonlinearity(NonlinearityFunc func);

    // Production: use SeamlessTransferFunction pointer (thread-safe via triple buffer)
    // void setTransferFunction(const SeamlessTransferFunction* tf);

    void setDrive(double drive);      // 0-1
    void setSaturation(double sat);   // 0-1
    void setWidth(double width);      // 0-1

    // Public for testing — these implement the core math
    double langevin(double Q) const;
    double langevinDeriv(double Q) const;
    double dMdt(double M, double H, double H_d) const;

  private:
    // Jiles-Atherton state
    double M_n1_ = 0.0;      // previous magnetization
    double H_n1_ = 0.0;      // previous field
    double H_d_n1_ = 0.0;    // previous field derivative

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

    // Input mapping: Q → [-1, 1]
    audio_pipeline::SoftClippingSolver inputMapper_{0.95};
    double scale_ = 1.0;

    // Custom nonlinearity (defaults to standard Langevin)
    NonlinearityFunc customNL_;
    bool useCustomNL_ = false;

    // Safety limits
    static constexpr double upperLimit_ = 20.0;
    static constexpr double outputLimit_ = 2.0;
    static constexpr double inputLimit_ = 10.0;
    static constexpr double hdLimit_ = 1e6;
    int consecutiveResets_ = 0;
    int muteCountdown_ = 0;

    // Internal DC blocker
    double dcX_n1_ = 0.0;
    double dcY_n1_ = 0.0;
    double dcR_ = 0.995;  // Updated dynamically in prepareToPlay()
    double dcBlock(double x);

    // Derived parameter cache
    double M_s_oa_ = 0.0;      // M_s / a
    double M_s_oa_tc_ = 0.0;   // c * M_s / a

    void updateDerivedParams();
    double computeHDerivative(double H);

    // Standard Langevin (used when no custom NL is set)
    static double standardLangevin(double x);
    static double standardLangevinDeriv(double x);
};

} // namespace dsp_core
