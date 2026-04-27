#include "AdaptiveToleranceCalculator.h"
#include <algorithm>
#include <cmath>

namespace dsp_core::Services {

double AdaptiveToleranceCalculator::computeTolerance(double verticalRange, int currentAnchors, int maxAnchors,
                                                     const Config& config) {
    // Sanity checks
    if (maxAnchors <= 0) {
        return config.relativeErrorTarget * verticalRange; // Fallback to baseline
    }

    currentAnchors = std::max(0, currentAnchors); // Clamp to zero

    // Compute baseline tolerance from vertical range
    const double baselineTolerance = verticalRange * config.relativeErrorTarget;

    // Compute anchor density ratio (0.0 = no anchors, 1.0 = at capacity)
    const double anchorRatio = std::min(1.0, static_cast<double>(currentAnchors) / static_cast<double>(maxAnchors));

    // Linear scaling: tolerance = baseline × (1 + anchorRatio × multiplier)
    // Relaxes tolerance as anchor count increases toward capacity
    const double adaptiveTolerance = baselineTolerance * (1.0 + anchorRatio * config.anchorDensityMultiplier);

    return adaptiveTolerance;
}

} // namespace dsp_core::Services
