#pragma once
#include "../SplineTypes.h"
#include <map>
#include <set>
#include <vector>

namespace dsp_core::Services {

/**
 * AnchorConstraintService - Projects ideal anchor positions to constrained positions.
 *
 * Problem: When batch-dragging multiple anchors, we want them to:
 * - Maintain relative spacing when unconstrained
 * - Compress gracefully when hitting boundaries
 * - Restore original spacing when dragging away from boundaries
 *
 * Solution: Track "ideal" (unconstrained) positions separately from "actual" positions.
 * Use two-pass constraint propagation to project ideal → actual:
 *
 * PASS 1 (left→right): Enforce left neighbor constraints
 * PASS 2 (right→left): Enforce right neighbor constraints, propagate backward
 *
 * This service is stateless and pure - it only performs the projection math.
 *
 * Lives in dsp_core because it's used both by the in-plugin spline editor
 * (via CurveEditorController::morphLaneAnchors) AND by the static thumbnail
 * evaluator (LaneMixer::evaluateStaticThumbnail) that bakes morph snapshots
 * for the community-website preview.
 */
class AnchorConstraintService {
  public:
    /**
     * Plain (x, y) pair used for ideal positions. dsp-core deliberately doesn't
     * depend on juce_graphics; juce::Point<double> is interchangeable at the
     * call site (any struct with .x/.y).
     */
    struct Point {
        double x = 0.0;
        double y = 0.0;
    };

    struct ConstraintResult {
        std::map<int, double> actualX; // anchor index → constrained X position
        std::map<int, double> actualY; // anchor index → constrained Y position
    };

    static ConstraintResult projectIdealToActual(const std::map<int, Point>& idealPositions,
                                                 const std::vector<dsp_core::SplineAnchor>& currentAnchors,
                                                 const std::set<int>& selectedIndices, double minGap = 0.001);

    AnchorConstraintService() = delete;
};

} // namespace dsp_core::Services
