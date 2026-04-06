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

    // Input clamping
    double H = std::clamp(inputH, -inputLimit_, inputLimit_);

    // H derivative (alpha-transform)
    double H_d = computeHDerivative(H);

    // H_d clamping
    H_d = std::clamp(H_d, -hdLimit_, hdLimit_);

    // RK4 integration
    double k1 = T_ * dMdt(M_n1_, H_n1_, H_d_n1_);
    double k2 = T_ * dMdt(M_n1_ + k1 / 2.0, (H + H_n1_) / 2.0, (H_d + H_d_n1_) / 2.0);
    double k3 = T_ * dMdt(M_n1_ + k2 / 2.0, (H + H_n1_) / 2.0, (H_d + H_d_n1_) / 2.0);
    double k4 = T_ * dMdt(M_n1_ + k3, H, H_d);

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
    H_d_n1_ = H_d;

    // Output clamping
    return std::clamp(M, -outputLimit_, outputLimit_);
}

void HysteresisProcessor::setNonlinearity(NonlinearityFunc func) {
    customNL_ = std::move(func);
    useCustomNL_ = (customNL_ != nullptr);
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
    c_ = std::sqrt(1.0 - width) - 0.01;
    c_ = std::max(c_, 0.001);  // prevent negative/zero
    updateDerivedParams();
}

void HysteresisProcessor::setOperatingPoint(double Ms) {
    M_s_ = Ms;
    a_ = Ms;              // Q = H/Ms — normalizes input to operating range
    scaleOverride_ = 1.0; // LUT sees normalized signal directly
    updateDerivedParams();
}

// =============================================================================
// Core Math
// =============================================================================

double HysteresisProcessor::langevin(double Q) const {
    if (useCustomNL_) {
        // Map Q to [-1, 1] using SoftClippingSolver, then apply custom NL
        double mapped = inputMapper_.process(Q / scale_);
        double result = customNL_(mapped);
        return result;
    }
    return standardLangevin(Q);
}

double HysteresisProcessor::langevinDeriv(double Q) const {
    if (useCustomNL_) {
        // Numerical central difference
        constexpr double h = 1e-4;
        return (langevin(Q + h) - langevin(Q - h)) / (2.0 * h);
    }
    return standardLangevinDeriv(Q);
}

double HysteresisProcessor::dMdt(double M, double H, double H_d) const {
    double Q = (H + alpha_ * M) / a_;
    double M_diff = M_s_ * langevin(Q) - M;
    double delta = (H_d > 0.0) ? 1.0 : -1.0;
    double delta_M = (std::signbit(delta) == std::signbit(M_diff)) ? 0.0 : 1.0;
    // delta_M = 1 when sign(delta) == sign(M_diff)
    // But signbit returns true for negative, so equal signbits means same sign
    // Actually: sign(delta) == sign(M_diff) → delta_M = 1
    // signbit(a) == signbit(b) means same sign
    // So delta_M = 1 when signbit(delta) == signbit(M_diff)
    delta_M = (std::signbit(delta) == std::signbit(M_diff)) ? 1.0 : 0.0;

    double L_prime = langevinDeriv(Q);

    double denominator = 1.0 - c_ * alpha_ * M_s_oa_ * L_prime;

    // Prevent division by zero
    if (std::abs(denominator) < 1e-15)
        return 0.0;

    double t1_den = (1.0 - c_) * delta * k_ - alpha_ * M_diff;

    // Prevent division by zero in t1
    if (std::abs(t1_den) < 1e-15)
        return (c_ * M_s_oa_ * H_d * L_prime) / denominator;

    double t1_num = (1.0 - c_) * delta_M * M_diff;
    double t1 = (t1_num / t1_den) * H_d;

    double t2 = c_ * M_s_oa_ * H_d * L_prime;

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
