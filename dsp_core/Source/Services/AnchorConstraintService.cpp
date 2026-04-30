#include "AnchorConstraintService.h"
#include <algorithm>
#include <juce_core/juce_core.h> // juce::jlimit

namespace dsp_core::Services {

AnchorConstraintService::ConstraintResult
AnchorConstraintService::projectIdealToActual(const std::map<int, Point>& idealPositions,
                                              const std::vector<dsp_core::SplineAnchor>& currentAnchors,
                                              const std::set<int>& selectedIndices, double minGap) {

    ConstraintResult result;

    std::vector<int> sortedIndices(selectedIndices.begin(), selectedIndices.end());
    std::sort(sortedIndices.begin(), sortedIndices.end());

    for (const int idx : sortedIndices) {
        auto it = idealPositions.find(idx);
        if (it != idealPositions.end()) {
            result.actualX[idx] = it->second.x;
            result.actualY[idx] = it->second.y;
        }
    }

    const int anchorCount = static_cast<int>(currentAnchors.size());

    // PASS 1: Left-to-right - enforce left neighbor constraints.
    for (const int idx : sortedIndices) {
        if (idx == 0) {
            result.actualX[idx] = -1.0;
            continue;
        }

        double leftNeighborX;
        if (result.actualX.count(idx - 1) != 0) {
            leftNeighborX = result.actualX[idx - 1];
        } else {
            leftNeighborX = currentAnchors[idx - 1].x;
        }

        const double minAllowedX = leftNeighborX + minGap;
        result.actualX[idx] = std::max(result.actualX[idx], minAllowedX);
    }

    // PASS 2: Right-to-left - enforce right neighbor constraints.
    for (int i = static_cast<int>(sortedIndices.size()) - 1; i >= 0; --i) {
        const int idx = sortedIndices[i];

        if (idx == anchorCount - 1) {
            result.actualX[idx] = 1.0;
        } else {
            double rightNeighborX;
            if (result.actualX.count(idx + 1) != 0) {
                rightNeighborX = result.actualX[idx + 1];
            } else {
                rightNeighborX = currentAnchors[idx + 1].x;
            }

            const double maxAllowedX = rightNeighborX - minGap;
            result.actualX[idx] = std::min(result.actualX[idx], maxAllowedX);
        }

        result.actualY[idx] = juce::jlimit(-1.0, 1.0, result.actualY[idx]);
    }

    return result;
}

} // namespace dsp_core::Services
