#include "TransferFunctionOperations.h"
#include <algorithm>
#include <cmath>

namespace dsp_core::Services {

void TransferFunctionOperations::invert(LayeredTransferFunction& ltf) {
    const int tableSize = ltf.getTableSize();

    for (int i = 0; i < tableSize; ++i) {
        const double currentValue = ltf.getBaseLayerValue(i);
        ltf.setBaseLayerValue(i, -currentValue);
    }

    // Version counter incremented by setBaseLayerValue() - renderer will update at next poll
}

void TransferFunctionOperations::removeDCInstantaneous(LayeredTransferFunction& ltf) {
    const int tableSize = ltf.getTableSize();
    const int midIndex = tableSize / 2;
    const double dcOffset = ltf.getBaseLayerValue(midIndex);

    for (int i = 0; i < tableSize; ++i) {
        const double currentValue = ltf.getBaseLayerValue(i);
        ltf.setBaseLayerValue(i, currentValue - dcOffset);
    }

    // Version counter incremented by setBaseLayerValue() - renderer will update at next poll
}

void TransferFunctionOperations::removeDCSteadyState(LayeredTransferFunction& ltf) {
    const int tableSize = ltf.getTableSize();

    double sum = 0.0;
    for (int i = 0; i < tableSize; ++i) {
        sum += ltf.getBaseLayerValue(i);
    }
    const double average = sum / static_cast<double>(tableSize);

    for (int i = 0; i < tableSize; ++i) {
        const double currentValue = ltf.getBaseLayerValue(i);
        ltf.setBaseLayerValue(i, currentValue - average);
    }

    // Version counter incremented by setBaseLayerValue() - renderer will update at next poll
}

void TransferFunctionOperations::normalize(LayeredTransferFunction& ltf) {
    const int tableSize = ltf.getTableSize();

    double maxAbsValue = 0.0;
    for (int i = 0; i < tableSize; ++i) {
        const double absValue = std::abs(ltf.getBaseLayerValue(i));
        maxAbsValue = std::max(maxAbsValue, absValue);
    }

    constexpr double kMinNormalizeThreshold = 1e-10;
    if (maxAbsValue < kMinNormalizeThreshold) {
        return;
    }

    const double scaleFactor = 1.0 / maxAbsValue;
    for (int i = 0; i < tableSize; ++i) {
        const double currentValue = ltf.getBaseLayerValue(i);
        ltf.setBaseLayerValue(i, currentValue * scaleFactor);
    }

    // Version counter incremented by setBaseLayerValue() - renderer will update at next poll
}

// ============================================================================
// CurveData overloads (lane-scoped)
// ============================================================================

void TransferFunctionOperations::invert(std::vector<double>& curveData) {
    for (auto& v : curveData) {
        v = -v;
    }
}

void TransferFunctionOperations::invertHorizontal(std::vector<double>& curveData) {
    std::reverse(curveData.begin(), curveData.end());
}

void TransferFunctionOperations::removeDCInstantaneous(std::vector<double>& curveData) {
    if (curveData.empty()) {
        return;
    }
    const double dcOffset = curveData[curveData.size() / 2];
    for (auto& v : curveData) {
        v -= dcOffset;
    }
}

void TransferFunctionOperations::removeDCSteadyState(std::vector<double>& curveData) {
    if (curveData.empty()) {
        return;
    }
    double sum = 0.0;
    for (const auto v : curveData) {
        sum += v;
    }
    const double average = sum / static_cast<double>(curveData.size());
    for (auto& v : curveData) {
        v -= average;
    }
}

void TransferFunctionOperations::normalize(std::vector<double>& curveData) {
    double maxAbsValue = 0.0;
    for (const auto v : curveData) {
        maxAbsValue = std::max(maxAbsValue, std::abs(v));
    }

    constexpr double kMinNormalizeThreshold = 1e-10;
    if (maxAbsValue < kMinNormalizeThreshold) {
        return;
    }

    const double scaleFactor = 1.0 / maxAbsValue;
    for (auto& v : curveData) {
        v *= scaleFactor;
    }
}

namespace {
void applyBoxFilter(std::vector<double>& curveData, int requestedRadius) {
    const auto tableSize = static_cast<int>(curveData.size());
    if (tableSize <= 1) {
        return;
    }

    const int windowRadius = std::min(requestedRadius, (tableSize - 1) / 2);

    // Boundary extrapolation with clamped slope (matches roller brush strategy)
    constexpr double kMaxSlope = 0.1;
    auto getValueWithExtrapolation = [&](int k) -> double {
        if (k >= 0 && k < tableSize) {
            return curveData[static_cast<size_t>(k)];
        }
        if (k < 0) {
            double slope = curveData[1] - curveData[0];
            slope = std::clamp(slope, -kMaxSlope, kMaxSlope);
            return curveData[0] + slope * k;
        }
        double slope = curveData[static_cast<size_t>(tableSize - 1)] -
                        curveData[static_cast<size_t>(tableSize - 2)];
        slope = std::clamp(slope, -kMaxSlope, kMaxSlope);
        return curveData[static_cast<size_t>(tableSize - 1)] + slope * (k - (tableSize - 1));
    };

    // O(n) box filter using running sum
    const int windowSize = 2 * windowRadius + 1;
    std::vector<double> smoothed(static_cast<size_t>(tableSize));

    double runningSum = 0.0;
    for (int k = -windowRadius; k <= windowRadius; ++k) {
        runningSum += getValueWithExtrapolation(k);
    }
    smoothed[0] = std::clamp(runningSum / windowSize, -1.0, 1.0);

    for (int i = 1; i < tableSize; ++i) {
        runningSum -= getValueWithExtrapolation(i - windowRadius - 1);
        runningSum += getValueWithExtrapolation(i + windowRadius);
        smoothed[static_cast<size_t>(i)] = std::clamp(runningSum / windowSize, -1.0, 1.0);
    }

    curveData = std::move(smoothed);
}
} // namespace

void TransferFunctionOperations::smooth(std::vector<double>& curveData) {
    applyBoxFilter(curveData, 100);
}

void TransferFunctionOperations::smoother(std::vector<double>& curveData) {
    applyBoxFilter(curveData, 300);
}

} // namespace dsp_core::Services
