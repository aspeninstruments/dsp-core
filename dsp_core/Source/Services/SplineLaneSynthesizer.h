#pragma once

#include "../SplineTypes.h"
#include <vector>

namespace dsp_core::Services {

/**
 * Side-effect-free spline-lane LUT synthesis at a given morph value.
 *
 * Mirrors the live editor flow in CurveEditorController::morphLaneAnchors —
 * for each anchor with a recorded morphGesture, samples the gesture at
 * `morph` to compute an "ideal" position (home + dx/dy), then projects all
 * gesture-bearing anchors through AnchorConstraintService for Y-clamp +
 * min-gap enforcement. The constrained positions are written into a working
 * copy of the input anchors; tangents are recomputed; SplineEvaluator
 * evaluates the curve into outLut.
 *
 * Used by:
 *   - CurveEditorController::morphLaneAnchors (live, with anchor write-back)
 *   - LaneMixer::evaluateStaticThumbnail (preview-only, no write-back)
 *
 * Behavior:
 *   - If `anchors` is empty or `tableSize <= 0`: returns false; outAnchors /
 *     outLut are unchanged.
 *   - If no anchor carries a morphGesture: outAnchors is a verbatim copy of
 *     the input (positions unchanged), outLut is a static spline evaluation.
 *   - Otherwise: outAnchors carries the constrained morphed positions and
 *     outLut is the spline through them.
 *
 * Returns true on success.
 */
bool synthesizeSplineLaneLUT(const std::vector<SplineAnchor>& anchors, double morph, int tableSize,
                             std::vector<SplineAnchor>& outAnchors, std::vector<double>& outLut);

} // namespace dsp_core::Services
