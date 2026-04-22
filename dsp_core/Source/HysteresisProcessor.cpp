#include "HysteresisProcessor.h"
#include <algorithm>

namespace dsp_core {

HysteresisProcessor::HysteresisProcessor() {
    updateDerivedParams();
}

void HysteresisProcessor::prepareToPlay(double sampleRate) {
    sampleRate_ = sampleRate;
    T_ = 1.0 / sampleRate;
    Talpha_ = T_ / 1.9;

    // 20Hz DC blocker cutoff — increase if DC accumulation issues arise
    constexpr double dcCutoffHz = 20.0;
    dcR_ = 1.0 - 2.0 * 3.14159265358979323846 * dcCutoffHz / sampleRate;

    smoothedC_.reset(sampleRate, 0.01); // 10ms ramp

    reset();
}

void HysteresisProcessor::reset() {
    M_n1_ = 0.0;
    H_n1_ = 0.0;
    H_d_n1_ = 0.0;
    dcX_n1_ = 0.0;
    dcY_n1_ = 0.0;
    consecutiveResets_ = 0;
    muteCountdown_ = 0;
}

double HysteresisProcessor::process(double inputH) {
    // Mute countdown (after consecutive resets)
    if (muteCountdown_ > 0) {
        --muteCountdown_;
        // Still update state so we don't get stuck
        H_n1_ = 0.0;
        H_d_n1_ = 0.0;
        M_n1_ = 0.0;
        return 0.0;
    }

    // Input safety: NaN/Inf check
    if (!std::isfinite(inputH)) {
        reset();
        consecutiveResets_++;
        if (consecutiveResets_ > 10)
            muteCountdown_ = static_cast<int>(sampleRate_ * 0.1);
        return 0.0;
    }

    // Smooth c parameter toward target (~10ms ramp)
    c_ = smoothedC_.getNextValue();
    M_s_oa_tc_ = c_ * M_s_oa_;

    // Input clamping
    double H = std::clamp(inputH, -inputLimit_, inputLimit_);

    // Field derivative: constant across the timestep (linear interpolation of H between samples)
    double dH_dt = (H - H_n1_) / T_;
    dH_dt = std::clamp(dH_dt, -hdLimit_, hdLimit_);

    // Branch direction with deadband (hold last direction when dH is negligible)
    double dH_raw = H - H_n1_;
    constexpr double eps = 1e-12;
    if (dH_raw > eps) lastDelta_ = 1.0;
    else if (dH_raw < -eps) lastDelta_ = -1.0;

    // RK4 integration — dH_dt and delta are constant across all stages
    double k1 = T_ * dMdt(M_n1_, H_n1_, dH_dt, lastDelta_);
    double k2 = T_ * dMdt(M_n1_ + k1 / 2.0, (H + H_n1_) / 2.0, dH_dt, lastDelta_);
    double k3 = T_ * dMdt(M_n1_ + k2 / 2.0, (H + H_n1_) / 2.0, dH_dt, lastDelta_);
    double k4 = T_ * dMdt(M_n1_ + k3, H, dH_dt, lastDelta_);

    double M = M_n1_ + k1 / 6.0 + k2 / 3.0 + k3 / 3.0 + k4 / 6.0;

    // Post-solver safety: NaN/Inf/runaway detection
    if (!std::isfinite(M) || std::abs(M) > upperLimit_) {
        reset();
        consecutiveResets_++;
        if (consecutiveResets_ > 10)
            muteCountdown_ = static_cast<int>(sampleRate_ * 0.1);
        return 0.0;
    }

    // Successful output — reset consecutive counter
    consecutiveResets_ = 0;

    // Update state
    M_n1_ = M;
    H_n1_ = H;

    // Output clamping
    return std::clamp(M, -outputLimit_, outputLimit_);
}

void HysteresisProcessor::setNonlinearity(NonlinearityFunc func) {
    customNL_ = std::move(func);
    useCustomNL_ = (customNL_ != nullptr);
}

void HysteresisProcessor::setNonlinearityDerivative(NonlinearityDerivFunc func) {
    customNLDeriv_ = std::move(func);
    useCustomNLDeriv_ = (customNLDeriv_ != nullptr);
}

void HysteresisProcessor::setDrive(double drive) {
    drive = std::clamp(drive, 0.0, 1.0);
    a_ = M_s_ / (0.01 + 6.0 * drive);
    updateDerivedParams();
}

void HysteresisProcessor::setSaturation(double sat) {
    sat = std::clamp(sat, 0.0, 1.0);
    M_s_ = 0.5 + 1.5 * (1.0 - sat);
    updateDerivedParams();
}

void HysteresisProcessor::setWidth(double width) {
    width = std::clamp(width, 0.0, 1.0);
    // Map width [0, 1] → c [0.999, 0.05]
    // Above c≈0.05 (old width=0.95) the loop becomes unusable,
    // and c=1.0 is a degenerate regime. This compresses the useful
    // range into the full knob travel.
    constexpr double cMax = 0.999;
    constexpr double cMin = 0.05;
    smoothedC_.setTargetValue(cMax - width * (cMax - cMin));
}

void HysteresisProcessor::setOperatingPoint(double Ms) {
    M_s_ = Ms;
    a_ = Ms;              // Q = H/Ms — normalizes input to operating range
    scaleOverride_ = 1.0; // LUT sees normalized signal directly

    // Coercivity set to match ChowTape's k/M_s ratio (~0.383) rather than its k/a.
    // Seamless-bypass requires a=M_s here, so we can't replicate ChowTape's M_s/a≈3.
    // Matching k/M_s instead keeps the coercive-vs-saturation ratio similar, which
    // is the dimensionless group that controls amplitude sensitivity of peak loss.
    k_ = 0.4 * Ms;

    updateDerivedParams();
}

// =============================================================================
// Core Math
// =============================================================================

double HysteresisProcessor::langevin(double Q) const {
    if (useCustomNL_) {
        // Scale Q into transfer function range; soft clipping (if enabled)
        // is handled inside the LUT evaluation (SeamlessTransferFunction)
        double mapped = Q / scale_;
        return customNL_(mapped);
    }
    return standardLangevin(Q);
}

double HysteresisProcessor::langevinDeriv(double Q) const {
    if (useCustomNLDeriv_) {
        // Analytical derivative — matches the Q → LUT-input mapping in langevin().
        return customNLDeriv_(Q / scale_);
    }
    if (useCustomNL_) {
        // Fallback: numerical central difference. Unreliable when the custom NL
        // has state with per-call side effects (e.g., Surge).
        constexpr double h = 1e-4;
        return (langevin(Q + h) - langevin(Q - h)) / (2.0 * h);
    }
    return standardLangevinDeriv(Q);
}

double HysteresisProcessor::dMdt(double M, double H, double dH_dt, double delta) const {
    double Q = (H + alpha_ * M) / a_;
    double M_diff = M_s_ * langevin(Q) - M;
    double delta_M = (std::signbit(delta) == std::signbit(M_diff)) ? 1.0 : 0.0;

    double L_prime = langevinDeriv(Q);
    double denominator = 1.0 - c_ * alpha_ * M_s_oa_ * L_prime;

    if (std::abs(denominator) < 1e-15)
        return 0.0;

    // Irreversible term
    double t1_den = (1.0 - c_) * delta * k_ - alpha_ * M_diff;
    if (std::abs(t1_den) < 1e-15)
        return (c_ * M_s_oa_ * dH_dt * L_prime) / denominator;

    double t1_num = (1.0 - c_) * delta_M * M_diff;
    double t1 = (t1_num / t1_den) * dH_dt;

    // Reversible term — same dH_dt as t1, no mixed-derivative artifacts
    double t2 = c_ * M_s_oa_ * dH_dt * L_prime;

    return (t1 + t2) / denominator;
}

// =============================================================================
// Private
// =============================================================================

double HysteresisProcessor::computeHDerivative(double H) {
    return ((1.0 + dAlpha_) / Talpha_) * (H - H_n1_) - dAlpha_ * H_d_n1_;
}

double HysteresisProcessor::dcBlock(double x) {
    double y = x - dcX_n1_ + dcR_ * dcY_n1_;
    dcX_n1_ = x;
    dcY_n1_ = y;
    return y;
}

void HysteresisProcessor::updateDerivedParams() {
    M_s_oa_ = M_s_ / a_;
    M_s_oa_tc_ = c_ * M_s_oa_;

    // Scale for input mapping: maps Q values into SoftClippingSolver range
    // When operating point is set, scale is fixed at 1.0 (LUT sees raw signal)
    // Otherwise, use ChowTape-style scaling (a*4) for standard Langevin
    scale_ = (scaleOverride_ >= 0.0) ? scaleOverride_ : a_ * 4.0;
}

double HysteresisProcessor::standardLangevin(double x) {
    if (std::abs(x) > 1e-4)
        return (1.0 / std::tanh(x)) - (1.0 / x);
    return x / 3.0;
}

double HysteresisProcessor::standardLangevinDeriv(double x) {
    if (std::abs(x) > 1e-4) {
        double cothx = 1.0 / std::tanh(x);
        return (1.0 / (x * x)) - cothx * cothx + 1.0;
    }
    return 1.0 / 3.0;
}

} // namespace dsp_core
