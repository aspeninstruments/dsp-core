#include "TransferFunctionOperations.h"
#include <algorithm>
#include <cmath>
#include <optional>

namespace dsp_core::Services {

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
        double slope = curveData[static_cast<size_t>(tableSize - 1)] - curveData[static_cast<size_t>(tableSize - 2)];
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

void TransferFunctionOperations::linearize(std::vector<double>& curveData) {
    if (curveData.empty()) {
        return;
    }
    const auto size = curveData.size();
    for (size_t i = 0; i < size; ++i) {
        const double linear = -1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(size - 1);
        curveData[i] = 0.5 * curveData[i] + 0.5 * linear;
    }
    normalize(curveData);
}

namespace {
// Piecewise-linear horizontal remap: the sample at original x=pivotFromX is
// moved to x=pivotToX. Left segment [-1, pivotFromX] stretches/compresses onto
// [-1, pivotToX], right segment [pivotFromX, 1] onto [pivotToX, 1]. Endpoints
// (-1 and 1) are fixed. Linear interpolation between samples.
void remapWithPivot(std::vector<double>& curveData, double pivotFromX, double pivotToX) {
    const int n = static_cast<int>(curveData.size());
    if (n < 2) {
        return;
    }

    constexpr double kMaxBound = 0.95;
    pivotFromX = std::clamp(pivotFromX, -kMaxBound, kMaxBound);
    pivotToX = std::clamp(pivotToX, -kMaxBound, kMaxBound);
    if (std::abs(pivotFromX - pivotToX) < 1e-12) {
        return;
    }

    std::vector<double> out(static_cast<size_t>(n));
    const double leftScale = (pivotFromX + 1.0) / (pivotToX + 1.0);  // new→orig on [-1, pivot]
    const double rightScale = (1.0 - pivotFromX) / (1.0 - pivotToX); // new→orig on [pivot, 1]
    const auto lastIdx = static_cast<double>(n - 1);

    for (int i = 0; i < n; ++i) {
        const double nx = -1.0 + 2.0 * static_cast<double>(i) / lastIdx;
        const double ox = (nx <= pivotToX) ? -1.0 + (nx + 1.0) * leftScale : pivotFromX + (nx - pivotToX) * rightScale;

        const double fIdx = std::clamp((ox + 1.0) * 0.5 * lastIdx, 0.0, lastIdx);
        const int i0 = static_cast<int>(std::floor(fIdx));
        const int i1 = std::min(i0 + 1, n - 1);
        const double f = fIdx - static_cast<double>(i0);
        out[static_cast<size_t>(i)] =
            curveData[static_cast<size_t>(i0)] * (1.0 - f) + curveData[static_cast<size_t>(i1)] * f;
    }
    curveData = std::move(out);
}
} // namespace

void TransferFunctionOperations::shiftHorizontal(std::vector<double>& curveData, double delta) {
    remapWithPivot(curveData, 0.0, delta);
}

void TransferFunctionOperations::shiftHorizontalLeft(std::vector<double>& curveData) {
    shiftHorizontal(curveData, -0.05);
}

void TransferFunctionOperations::shiftHorizontalRight(std::vector<double>& curveData) {
    shiftHorizontal(curveData, 0.05);
}

namespace {
// Tests whether segment [i, i+1] contains a zero crossing, returning its x-position if so.
std::optional<double> crossingInSegment(const std::vector<double>& c, int i, int n) {
    if (i < 0 || i + 1 >= n) {
        return std::nullopt;
    }
    const double a = c[static_cast<size_t>(i)];
    const double b = c[static_cast<size_t>(i) + 1];
    const auto lastIdx = static_cast<double>(n - 1);
    if (a == 0.0) {
        return -1.0 + 2.0 * static_cast<double>(i) / lastIdx;
    }
    const bool signsDiffer = (a < 0.0 && b > 0.0) || (a > 0.0 && b < 0.0) || b == 0.0;
    if (!signsDiffer) {
        return std::nullopt;
    }
    const double frac = std::abs(a) / (std::abs(a) + std::abs(b));
    return -1.0 + 2.0 * (static_cast<double>(i) + frac) / lastIdx;
}

std::optional<double> findNearestZeroCrossing(const std::vector<double>& curveData) {
    const int n = static_cast<int>(curveData.size());
    if (n < 2) {
        return std::nullopt;
    }
    const int mid = n / 2;
    const int maxOffset = std::max(mid, n - 1 - mid);

    // Scan outward from mid; within each offset return the nearer (to x=0) of left/right.
    for (int off = 0; off <= maxOffset; ++off) {
        auto leftCandidate = crossingInSegment(curveData, mid - off, n);
        auto rightCandidate = (off == 0) ? std::nullopt : crossingInSegment(curveData, mid + off - 1, n);
        if (leftCandidate && rightCandidate) {
            return std::abs(*leftCandidate) <= std::abs(*rightCandidate) ? leftCandidate : rightCandidate;
        }
        if (leftCandidate) {
            return leftCandidate;
        }
        if (rightCandidate) {
            return rightCandidate;
        }
    }
    return std::nullopt;
}
} // namespace

void TransferFunctionOperations::shiftToZeroCrossing(std::vector<double>& curveData) {
    auto crossing = findNearestZeroCrossing(curveData);
    if (!crossing.has_value()) {
        return;
    }
    remapWithPivot(curveData, *crossing, 0.0);
}

} // namespace dsp_core::Services
