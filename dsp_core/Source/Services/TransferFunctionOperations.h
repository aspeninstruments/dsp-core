#pragma once
#include "../LayeredTransferFunction.h"
#include <vector>

namespace dsp_core::Services {

/**
 * TransferFunctionOperations - Pure service for transfer function transformations
 *
 * Provides stateless operations on LayeredTransferFunction or raw curve data
 * (std::vector<double>) that can be used from any context.
 *
 * Service Pattern (5/5 score):
 *   - Pure static methods (no state)
 *   - Unit testable in isolation
 *   - Reusable across modules
 */
class TransferFunctionOperations {
  public:
    // ========================================================================
    // LayeredTransferFunction overloads (legacy — used during migration)
    // ========================================================================

    static void invert(LayeredTransferFunction& ltf);
    static void removeDCInstantaneous(LayeredTransferFunction& ltf);
    static void removeDCSteadyState(LayeredTransferFunction& ltf);
    static void normalize(LayeredTransferFunction& ltf);

    // ========================================================================
    // CurveData overloads (lane-scoped — primary API post-Phase 10)
    // ========================================================================

    /** Invert curve: f(x) → -f(x) */
    static void invert(std::vector<double>& curveData);

    /** Remove instantaneous DC: subtract value at midpoint so f(0) = 0 */
    static void removeDCInstantaneous(std::vector<double>& curveData);

    /** Remove steady-state DC: subtract average of all values */
    static void removeDCSteadyState(std::vector<double>& curveData);

    /** Normalize to [-1, 1] range. No-op if max < 1e-10. */
    static void normalize(std::vector<double>& curveData);

  private:
    TransferFunctionOperations() = delete; // Pure static utility
};

} // namespace dsp_core::Services
