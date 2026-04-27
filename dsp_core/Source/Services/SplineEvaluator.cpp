#include "SplineEvaluator.h"
#include <algorithm>
#include <cmath>
#include <juce_core/juce_core.h>

namespace dsp_core::Services {

double SplineEvaluator::evaluate(const std::vector<SplineAnchor>& anchors, double x) {

    if (anchors.empty()) {
        return 0.0;
    }
    if (anchors.size() == 1) {
        return anchors[0].y;
    }

    // Find segment containing x
    const int segIdx = findSegment(anchors, x);
    if (segIdx < 0) {
        return anchors.front().y; // Before first anchor
    }
    if (segIdx >= static_cast<int>(anchors.size()) - 1) {
        return anchors.back().y; // After last anchor
    }

    // Evaluate cubic Hermite on segment
    return evaluateSegment(anchors[segIdx], anchors[segIdx + 1], x);
}

void SplineEvaluator::evaluateBatch(const std::vector<SplineAnchor>& anchors, const double* xValues, double* yValues,
                                    int count) {

    if (count <= 0) {
        return;
    }

    // Degenerate cases
    if (anchors.empty()) {
        std::fill(yValues, yValues + count, 0.0);
        return;
    }
    if (anchors.size() == 1) {
        std::fill(yValues, yValues + count, anchors[0].y);
        return;
    }

    // Incremental segment search (assumes xValues are sorted)
    // This eliminates 256 binary searches, replacing with linear scan
    int currentSegment = 0;

    for (int i = 0; i < count; ++i) {
        const double x = xValues[i];

        // Advance segment while x is beyond current segment's end
        // This works because xValues are sorted (each x >= previous x)
        while (currentSegment < static_cast<int>(anchors.size()) - 1 && x > anchors[currentSegment + 1].x) {
            ++currentSegment;
        }

        // Handle out-of-bounds cases
        if (x < anchors.front().x) {
            yValues[i] = anchors.front().y;
            currentSegment = 0; // Reset for next iteration
        } else if (x > anchors.back().x) {
            yValues[i] = anchors.back().y;
            currentSegment = static_cast<int>(anchors.size()) - 2; // Stay at last segment
        } else {
            // Clamp to valid segment range (safety check)
            if (currentSegment >= static_cast<int>(anchors.size()) - 1) {
                currentSegment = static_cast<int>(anchors.size()) - 2;
            }
            currentSegment = std::max(currentSegment, 0);

            // Evaluate segment using cubic Hermite interpolation
            yValues[i] = evaluateSegment(anchors[currentSegment], anchors[currentSegment + 1], x);
        }
    }
}

double SplineEvaluator::evaluateDerivative(const std::vector<SplineAnchor>& anchors, double x) {

    if (anchors.empty()) {
        return 0.0;
    }
    if (anchors.size() == 1) {
        return 0.0;
    }

    // Find segment containing x
    const int segIdx = findSegment(anchors, x);
    if (segIdx < 0) {
        return anchors.front().tangent; // Before first anchor
    }
    if (segIdx >= static_cast<int>(anchors.size()) - 1) {
        return anchors.back().tangent; // After last anchor
    }

    // Evaluate cubic Hermite derivative on segment
    return evaluateSegmentDerivative(anchors[segIdx], anchors[segIdx + 1], x);
}

int SplineEvaluator::findSegment(const std::vector<SplineAnchor>& anchors, double x) {

    // Binary search for segment containing x
    // Find first anchor with anchor.x >= x
    auto it = std::lower_bound(anchors.begin(), anchors.end(), x,
                               [](const SplineAnchor& anchor, double val) { return anchor.x < val; });

    if (it == anchors.begin()) {
        return -1; // Before first anchor
    }
    if (it == anchors.end()) {
        return static_cast<int>(anchors.size()) - 1; // After last anchor
    }

    // Return index of segment start (the anchor before 'it')
    return static_cast<int>(std::distance(anchors.begin(), it)) - 1;
}

double SplineEvaluator::evaluateSegment(const SplineAnchor& p0, const SplineAnchor& p1, double x) {

    // Normalize to [0, 1] within segment
    const double dx = p1.x - p0.x;
    if (std::abs(dx) < 1e-12) {
        return p0.y; // Degenerate segment
    }

    double t = (x - p0.x) / dx;
    t = juce::jlimit(0.0, 1.0, t); // Clamp to segment bounds

    // Cubic Hermite basis functions
    // h00(t) = 2t³ - 3t² + 1 = (1 + 2t)(1-t)²
    // h10(t) = t³ - 2t² + t = t(1-t)²
    // h01(t) = -2t³ + 3t² = t²(3-2t)
    // h11(t) = t³ - t² = t²(t-1)

    const double t2 = t * t;
    const double omt = 1.0 - t;    // (1-t)
    const double omt2 = omt * omt; // (1-t)²

    const double h00 = (1.0 + 2.0 * t) * omt2; // 2t³ - 3t² + 1
    const double h10 = t * omt2;               // t³ - 2t² + t
    const double h01 = t2 * (3.0 - 2.0 * t);   // -2t³ + 3t²
    const double h11 = t2 * (t - 1.0);         // t³ - t²

    // Get tangent values (slopes at control points)
    const double m0 = p0.tangent;
    const double m1 = p1.tangent;

    // Hermite interpolation
    // H(t) = h00·y0 + h10·Δx·m0 + h01·y1 + h11·Δx·m1
    return h00 * p0.y + h10 * dx * m0 + h01 * p1.y + h11 * dx * m1;
}

double SplineEvaluator::evaluateSegmentDerivative(const SplineAnchor& p0, const SplineAnchor& p1, double x) {

    // Normalize to [0, 1] within segment
    const double dx = p1.x - p0.x;
    if (std::abs(dx) < 1e-12) {
        return 0.0; // Degenerate segment
    }

    double t = (x - p0.x) / dx;
    t = juce::jlimit(0.0, 1.0, t); // Clamp to segment bounds

    // Derivatives of cubic Hermite basis functions (with respect to t)
    // h00'(t) = 6t² - 6t = 6t(t-1)
    // h10'(t) = 3t² - 4t + 1
    // h01'(t) = -6t² + 6t = 6t(1-t)
    // h11'(t) = 3t² - 2t

    const double t2 = t * t;

    const double h00_dt = 6.0 * t * (t - 1.0);      // 6t² - 6t
    const double h10_dt = 3.0 * t2 - 4.0 * t + 1.0; // 3t² - 4t + 1
    const double h01_dt = 6.0 * t * (1.0 - t);      // -6t² + 6t
    const double h11_dt = 3.0 * t2 - 2.0 * t;       // 3t² - 2t

    // Get tangent values
    const double m0 = p0.tangent;
    const double m1 = p1.tangent;

    // H'(t) with respect to t
    const double dH_dt = h00_dt * p0.y + h10_dt * dx * m0 + h01_dt * p1.y + h11_dt * dx * m1;

    // Convert to derivative with respect to x: dH/dx = (dH/dt) / (dx/dt) = (dH/dt) / dx
    return dH_dt / dx;
}

} // namespace dsp_core::Services
