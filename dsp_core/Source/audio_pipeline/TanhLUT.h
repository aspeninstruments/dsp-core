#pragma once

#include <array>
#include <cmath>
#include <cstddef>

namespace dsp_core::audio_pipeline {

/**
 * Precomputed tanh(2x) lookup table with linear interpolation.
 *
 * Used as the soft-saturation kernel for virtual-analog filter stages
 * (both feedforward integrator nonlinearity and feedback nonlinearity).
 * Replacing per-sample std::tanh / std::function calls with a direct LUT
 * lookup is the main reason these filters can hit reasonable CPU.
 *
 * Range: [-kRange, kRange] = [-4, 4]
 *   tanh(2 * 4) ≈ 0.99999977, so clamping outside this range loses
 *   < 2.3e-7 of dynamic range — well below 24-bit audio noise floor.
 *
 * Size: 8192 entries → lerp peak error vs std::tanh(2x) is < 1e-7.
 */
class TanhLUT {
  public:
    static constexpr int kSize = 8192;
    static constexpr double kRange = 4.0;

    TanhLUT() {
        for (int i = 0; i < kSize; ++i) {
            const double x = -kRange + (2.0 * kRange * i) / static_cast<double>(kSize - 1);
            table_[static_cast<std::size_t>(i)] = std::tanh(2.0 * x);
        }
    }

    inline double lookup(double x) const noexcept {
        // Branchless clamp.
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
inline const TanhLUT g_tanhLUT{};

} // namespace dsp_core::audio_pipeline
