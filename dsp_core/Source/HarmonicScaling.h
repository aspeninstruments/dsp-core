#pragma once

#include <cmath>
#include <algorithm>

namespace dsp_core {

/**
 * Per-harmonic nonlinear slider scaling.
 *
 * Maps slider position [0,1] to coefficient [0,1] using a midpoint-anchored
 * two-exponent curve. The slider midpoint (0.5) maps to each harmonic's
 * "Goldilocks" default coefficient, giving more resolution in the musically
 * useful range while still reaching the full [0,1] interval at the extremes.
 *
 * Constraints:
 *   x = 0   -> y = 0
 *   x = 0.5 -> y = c_n  (Goldilocks coefficient for harmonic n)
 *   x = 1   -> y = 1
 */
struct HarmonicScaling
{
    // Shape exponents for the lower and upper halves of the curve.
    // Higher values = more resolution near the midpoint.
    static constexpr double lowerExponent = 2.5;
    static constexpr double upperExponent = 2.0;

    /**
     * Goldilocks default coefficient for harmonic n.
     *
     * Power-law decay anchored at H3 and H39 (from the original tanh-derived
     * formula). H5 is roughly half of H3, with smooth monotonic decay to H39.
     * All harmonics — odd and even — follow the same curve.
     *
     * Formula: c_n = c3 * (3/n)^alpha, where alpha = ln(c3/c39) / ln(39/3)
     */
    static double defaultCoefficientForHarmonic(int n)
    {
        if (n <= 0)
            return 0.5; // WT mix

        if (n == 1)
            return 0.5;

        // Anchor values from original tanh formula
        static const double c3    = 0.12549929;
        static const double alpha = 1.2131; // ln(c3/0.00558810) / ln(39/3)

        const double nd = static_cast<double>(n);

        if (n == 2)
        {
            // Log-interpolate between H1 (0.5) and H3
            const double t = (nd - 1.0) / 2.0;
            return std::exp(std::log(0.5) + t * (std::log(c3) - std::log(0.5)));
        }

        // H3+: power-law decay
        return c3 * std::pow(3.0 / nd, alpha);
    }

    /**
     * Map slider position x ∈ [0,1] to coefficient y ∈ [0,1].
     *
     * @param x  Slider position (0 = bottom, 1 = top)
     * @param c  Goldilocks midpoint coefficient for this harmonic
     * @param a  Lower-half exponent (default: lowerExponent)
     * @param b  Upper-half exponent (default: upperExponent)
     */
    static double sliderToCoefficient(double x, double c,
                                      double a = lowerExponent,
                                      double b = upperExponent)
    {
        x = std::clamp(x, 0.0, 1.0);

        // S-curve: both halves concentrate resolution near the midpoint.
        // Lower half rushes away from 0, lingers near c.
        // Upper half lingers near c, rushes toward 1.
        if (c <= 0.0)
            return std::pow(x, b); // no midpoint — simple power curve

        if (x <= 0.5)
        {
            const double t = x / 0.5; // normalise to [0,1]
            return c * (1.0 - std::pow(1.0 - t, a));
        }

        const double t = (x - 0.5) / 0.5; // normalise to [0,1]
        return c + (1.0 - c) * std::pow(t, b);
    }

    /**
     * Inverse: map coefficient y ∈ [0,1] back to slider position x ∈ [0,1].
     *
     * Used when restoring state from model → UI (preset loads, undo/redo).
     */
    static double coefficientToSlider(double y, double c,
                                      double a = lowerExponent,
                                      double b = upperExponent)
    {
        y = std::clamp(y, 0.0, 1.0);

        if (c <= 0.0)
            return std::pow(y, 1.0 / b); // inverse of pow(x, b)

        if (y <= c)
        {
            // Inverse of: c * (1 - pow(1-t, a))
            // y/c = 1 - pow(1-t, a)  =>  pow(1-t, a) = 1 - y/c  =>  t = 1 - pow(1 - y/c, 1/a)
            const double t = 1.0 - std::pow(1.0 - y / c, 1.0 / a);
            return 0.5 * t;
        }

        if (c >= 1.0)
            return 1.0;

        // Inverse of: c + (1-c) * pow(t, b)
        // (y-c)/(1-c) = pow(t, b)  =>  t = pow((y-c)/(1-c), 1/b)
        const double t = std::pow((y - c) / (1.0 - c), 1.0 / b);
        return 0.5 + 0.5 * t;
    }
};

} // namespace dsp_core
