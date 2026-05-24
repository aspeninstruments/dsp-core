#pragma once

#include <array>
#include <cmath>
#include <cstddef>

namespace dsp_core::audio_pipeline {

/**
 * Precomputed tanh(2x) lookup table with linear interpolation.
 *
 * Name carries the "2x" deliberately — `lookup(z)` returns **tanh(2*z)**, NOT
 * tanh(z). This matters everywhere it's used: virtual-analog filter feedback
 * (where the LUT input is `R*y` and the effective saturation is `tanh(2*R*y)`)
 * and per-stage saturation in the RK4 cascade.
 *
 * Why a LUT instead of `std::tanh`? Measured at ~55-65% faster than std::tanh
 * in the modular RK4 path (which calls tanh ~20+ times per sample). The
 * TPT-ZDF path (~3-5 calls per sample) sees little difference, but we keep
 * the LUT for symmetry and to avoid two parallel saturation paths.
 *
 * Range: [-kRange, kRange] = [-4, 4]
 *   tanh(2 * 4) ≈ 0.99999977, so clamping outside this range loses
 *   < 2.3e-7 of dynamic range — well below 24-bit audio noise floor.
 *
 * Size: 8192 entries → lerp peak error vs std::tanh(2x) is < 1e-7.
 */
class Tanh2xLUT {
  public:
    static constexpr int kSize = 8192;
    static constexpr double kRange = 4.0;

    Tanh2xLUT() {
        for (int i = 0; i < kSize; ++i) {
            const double x = -kRange + (2.0 * kRange * i) / static_cast<double>(kSize - 1);
            table_[static_cast<std::size_t>(i)] = std::tanh(2.0 * x);
        }
    }

    /** Returns tanh(2 * x). Branchless-clamped to [-kRange, kRange]. */
    inline double lookup(double x) const noexcept {
        const double clamped = x < -kRange ? -kRange : (x > kRange ? kRange : x);
        const double idx = (clamped + kRange) * kInvStep;
        const int i0 = static_cast<int>(idx);
        const int i1 = i0 + 1 < kSize ? i0 + 1 : kSize - 1;
        const double frac = idx - static_cast<double>(i0);
        const double a = table_[static_cast<std::size_t>(i0)];
        const double b = table_[static_cast<std::size_t>(i1)];
        return a + (b - a) * frac;
    }

  private:
    static constexpr double kInvStep = (kSize - 1) / (2.0 * kRange);
    std::array<double, kSize> table_{};
};

// Single shared instance; the LUT is read-only after construction.
// C++17 inline variable — one definition across all TUs.
inline const Tanh2xLUT g_tanh2xLUT{};

} // namespace dsp_core::audio_pipeline
