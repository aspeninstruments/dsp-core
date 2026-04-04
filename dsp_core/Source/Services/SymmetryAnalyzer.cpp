#include "SymmetryAnalyzer.h"
#include "../LayeredTransferFunction.h"
#include <cmath>
#include <numeric>
#include <algorithm>

namespace dsp_core::Services {

SymmetryAnalyzer::Result SymmetryAnalyzer::analyzeOddSymmetry(const LayeredTransferFunction& ltf,
                                                              const Config& config) {

    Result result;
    result.centerX = 0.0; // Assume symmetry about origin

    // Sample curve at complementary points
    std::vector<double> fPositive, fNegative;
    fPositive.reserve(config.sampleCount);
    fNegative.reserve(config.sampleCount);

    const int tableSize = ltf.getTableSize();
    const int centerIdx = tableSize / 2;

    // Sample from center to right edge
    for (int i = 0; i < config.sampleCount; ++i) {
        // Map to [0, tableSize/2] range
        const double t = static_cast<double>(i) / (config.sampleCount - 1);
        const int positiveIdx = centerIdx + static_cast<int>(t * (tableSize - centerIdx - 1));
        const int negativeIdx = centerIdx - static_cast<int>(t * centerIdx);

        // Evaluate at the x positions (includes base + harmonics, ignores spline)
        const double xPositive = ltf.normalizeIndex(positiveIdx);
        const double xNegative = ltf.normalizeIndex(negativeIdx);

        const double yPositive = ltf.evaluateBaseAndHarmonics(xPositive);
        const double yNegative = ltf.evaluateBaseAndHarmonics(xNegative);

        fPositive.push_back(yPositive);
        fNegative.push_back(yNegative);
    }

    // CRITICAL: Check zero-crossing at origin for odd symmetry
    // For odd symmetry f(-x) = -f(x), we must have f(0) = 0
    // If |f(0)| > tolerance, the curve cannot be odd-symmetric
    const double yAtZero = ltf.evaluateBaseAndHarmonics(0.0);
    const double zeroCrossingTolerance = 0.1; // 10% tolerance

    if (std::abs(yAtZero) > zeroCrossingTolerance) {
        // Curve doesn't cross zero → NOT odd-symmetric
        result.score = 0.0;
        result.classification = Result::Classification::Asymmetric;
        return result;
    }

    // Compute symmetry score (correlation between f(x) and -f(-x))
    result.score = computeSymmetryScore(fPositive, fNegative);

    // Classify based on thresholds
    if (result.score >= config.perfectThreshold) {
        result.classification = Result::Classification::Perfect;
    } else if (result.score >= config.approximateThreshold) {
        result.classification = Result::Classification::Approximate;
    } else {
        result.classification = Result::Classification::Asymmetric;
    }

    return result;
}

double SymmetryAnalyzer::computeSymmetryScore(const std::vector<double>& fPositive,
                                              const std::vector<double>& fNegative) {

    if (fPositive.size() != fNegative.size() || fPositive.empty()) {
        return 0.0;
    }

    const int n = static_cast<int>(fPositive.size());

    // Compute means
    double meanPos = std::accumulate(fPositive.begin(), fPositive.end(), 0.0) / n;
    double meanNeg = std::accumulate(fNegative.begin(), fNegative.end(), 0.0) / n;

    // For odd symmetry: f(-x) = -f(x), so meanNeg should be ≈ -meanPos
    // Flip sign for comparison
    meanNeg = -meanNeg;

    // Compute Pearson correlation between f(x) and -f(-x)
    double numerator = 0.0;
    double denomPos = 0.0;
    double denomNeg = 0.0;

    for (int i = 0; i < n; ++i) {
        const double devPos = fPositive[i] - meanPos;
        const double devNeg = -fNegative[i] - meanNeg; // Flip sign

        numerator += devPos * devNeg;
        denomPos += devPos * devPos;
        denomNeg += devNeg * devNeg;
    }

    if (denomPos < 1e-12 || denomNeg < 1e-12) {
        // Degenerate case: flat curve or near-zero variance
        // Check if both sides are near zero (symmetric flatness)
        return (std::abs(meanPos) < 1e-6 && std::abs(meanNeg) < 1e-6) ? 1.0 : 0.0;
    }

    const double correlation = numerator / std::sqrt(denomPos * denomNeg);

    // Clamp to [0, 1] range (negative correlation = asymmetric)
    return std::max(0.0, std::min(1.0, correlation));
}

SymmetryAnalyzer::Result SymmetryAnalyzer::analyzeOddSymmetry(const LayeredTransferFunction& ltf) {
    return analyzeOddSymmetry(ltf, Config{});
}

SymmetryAnalyzer::Result SymmetryAnalyzer::analyzeOddSymmetry(const std::vector<double>& curveData,
                                                              const Config& config) {
    Result result;
    result.centerX = 0.0;

    const int tableSize = static_cast<int>(curveData.size());
    if (tableSize < 3) {
        result.score = 0.0;
        result.classification = Result::Classification::Asymmetric;
        return result;
    }

    const int centerIdx = tableSize / 2;

    // Check zero-crossing at origin
    const double zeroCrossingTolerance = 0.1;
    if (std::abs(curveData[static_cast<size_t>(centerIdx)]) > zeroCrossingTolerance) {
        result.score = 0.0;
        result.classification = Result::Classification::Asymmetric;
        return result;
    }

    // Sample complementary points for correlation
    std::vector<double> fPositive;
    std::vector<double> fNegative;
    fPositive.reserve(config.sampleCount);
    fNegative.reserve(config.sampleCount);

    for (int i = 0; i < config.sampleCount; ++i) {
        const double t = static_cast<double>(i) / (config.sampleCount - 1);
        const int positiveIdx = centerIdx + static_cast<int>(t * (tableSize - centerIdx - 1));
        const int negativeIdx = centerIdx - static_cast<int>(t * centerIdx);

        fPositive.push_back(curveData[static_cast<size_t>(positiveIdx)]);
        fNegative.push_back(curveData[static_cast<size_t>(negativeIdx)]);
    }

    result.score = computeSymmetryScore(fPositive, fNegative);

    if (result.score >= config.perfectThreshold) {
        result.classification = Result::Classification::Perfect;
    } else if (result.score >= config.approximateThreshold) {
        result.classification = Result::Classification::Approximate;
    } else {
        result.classification = Result::Classification::Asymmetric;
    }

    return result;
}

SymmetryAnalyzer::Result SymmetryAnalyzer::analyzeOddSymmetry(const std::vector<double>& curveData) {
    return analyzeOddSymmetry(curveData, Config{});
}

} // namespace dsp_core::Services
