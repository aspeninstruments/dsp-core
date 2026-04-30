#include "SplineLaneSynthesizer.h"
#include "../LaneMixer.h"
#include "AnchorConstraintService.h"
#include "SplineEvaluator.h"
#include "SplineFitter.h"

#include <algorithm>
#include <juce_core/juce_core.h>

namespace dsp_core::Services {

bool synthesizeSplineLaneLUT(const std::vector<SplineAnchor>& anchors, double morph, int tableSize,
                             std::vector<SplineAnchor>& outAnchors, std::vector<double>& outLut) {
    if (anchors.empty() || tableSize <= 0) {
        return false;
    }

    outAnchors = anchors;

    const bool anyGesture =
        std::any_of(outAnchors.begin(), outAnchors.end(), [](const SplineAnchor& a) { return !a.morphGesture.empty(); });

    if (anyGesture) {
        // For each gesture-bearing anchor, compute home + sample(morph) as the
        // ideal target. Non-gesture anchors stay where they are.
        std::map<int, AnchorConstraintService::Point> idealPositions;
        std::set<int> gestureBearingIds;
        for (size_t i = 0; i < outAnchors.size(); ++i) {
            const auto& a = outAnchors[i];
            if (a.morphGesture.empty()) {
                continue;
            }
            const auto sample = a.morphGesture.sampleAt(morph);
            idealPositions[static_cast<int>(i)] = {a.homeX + sample.dx, a.homeY + sample.dy};
            gestureBearingIds.insert(static_cast<int>(i));
        }

        const auto constrained =
            AnchorConstraintService::projectIdealToActual(idealPositions, outAnchors, gestureBearingIds, /*minGap=*/0.001);

        for (const auto& [idx, x] : constrained.actualX) {
            outAnchors[static_cast<size_t>(idx)].x = juce::jlimit(-1.0, 1.0, x);
        }
        for (const auto& [idx, y] : constrained.actualY) {
            outAnchors[static_cast<size_t>(idx)].y = juce::jlimit(-1.0, 1.0, y);
        }
    }

    SplineFitter::computeTangents(outAnchors, SplineFitConfig::tight());

    std::vector<double> xValues(static_cast<size_t>(tableSize));
    for (int i = 0; i < tableSize; ++i) {
        xValues[static_cast<size_t>(i)] = LaneMixer::normalizeIndex(i);
    }
    outLut.assign(static_cast<size_t>(tableSize), 0.0);
    SplineEvaluator::evaluateBatch(outAnchors, xValues.data(), outLut.data(), tableSize);

    return true;
}

} // namespace dsp_core::Services
