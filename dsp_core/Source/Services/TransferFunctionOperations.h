#pragma once
#include <vector>

namespace dsp_core::Services {

/**
 * TransferFunctionOperations - Pure service for transfer function transformations
 *
 * Provides stateless operations on raw curve data (std::vector<double>)
 * that can be used from any context.
 *
 * Service Pattern (5/5 score):
 *   - Pure static methods (no state)
 *   - Unit testable in isolation
 *   - Reusable across modules
 */
class TransferFunctionOperations {
  public:

    /** Invert curve vertically: f(x) → -f(x) */
    static void invert(std::vector<double>& curveData);

    /** Invert curve horizontally: reverse the table so f(x) → f(-x) */
    static void invertHorizontal(std::vector<double>& curveData);

    /** Remove instantaneous DC: subtract value at midpoint so f(0) = 0 */
    static void removeDCInstantaneous(std::vector<double>& curveData);

    /** Remove steady-state DC: subtract average of all values */
    static void removeDCSteadyState(std::vector<double>& curveData);

    /** Normalize to [-1, 1] range. No-op if max < 1e-10. */
    static void normalize(std::vector<double>& curveData);

    /** Smooth curve using box filter (sliding window average).
     *  Single-pass with boundary extrapolation. Clamps output to [-1, 1]. */
    static void smooth(std::vector<double>& curveData);

    /** Stronger smooth (3x window radius). */
    static void smoother(std::vector<double>& curveData);

  private:
    TransferFunctionOperations() = delete; // Pure static utility
};

} // namespace dsp_core::Services
