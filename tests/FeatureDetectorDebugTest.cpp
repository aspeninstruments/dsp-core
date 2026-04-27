#include <dsp_core/dsp_core.h>
#include <gtest/gtest.h>
#include <iostream>
#include <iomanip>
#include <cmath>

namespace dsp_core_test {

/**
 * Debug test to understand why feature detector finds 0 extrema for H3/H5
 */
TEST(FeatureDetectorDebug, WhyNoExtremaForH3) {
    auto ltf = std::make_unique<dsp_core::LayeredTransferFunction>(16384, -1.0, 1.0);

    // Set H3 curve
    for (int i = 0; i < ltf->getTableSize(); ++i) {
        double x = ltf->normalizeIndex(i);
        x = std::clamp(x, -1.0, 1.0);
        double const y = std::sin(3.0 * std::asin(x));
        ltf->setBaseLayerValue(i, y);
    }

    std::cout << "\n=== H3 Curve Sampling ===" << '\n';

    // Check curve near expected extremum at x ≈ 0.866
    int const targetIdx = static_cast<int>(ltf->getTableSize() * (0.866 + 1.0) / 2.0);
    std::cout << "Expected extremum near x=0.866" << '\n';
    std::cout << "Checking indices around " << targetIdx << '\n';

    for (int offset = -5; offset <= 5; ++offset) {
        int const idx = targetIdx + offset;
        if (idx >= 0 && idx < ltf->getTableSize()) {
            double const x = ltf->normalizeIndex(idx);
            double const y = ltf->getBaseLayerValue(idx);
            std::cout << "  idx=" << idx << ", x=" << std::fixed << std::setprecision(6) << x << ", y=" << y << '\n';
        }
    }

    // Check vertical range
    double minY = ltf->getBaseLayerValue(0);
    double maxY = ltf->getBaseLayerValue(0);
    for (int i = 1; i < ltf->getTableSize(); ++i) {
        double const y = ltf->getBaseLayerValue(i);
        minY = std::min(minY, y);
        maxY = std::max(maxY, y);
    }
    double const verticalRange = maxY - minY;

    std::cout << "\nVertical range: [" << minY << ", " << maxY << "], span=" << verticalRange << '\n';

    // Extract curve data for service calls
    std::vector<double> curveData(ltf->getTableSize());
    for (int i = 0; i < ltf->getTableSize(); ++i) {
        curveData[i] = ltf->getBaseLayerValue(i);
    }

    // Now run feature detection with different thresholds
    std::cout << "\n=== Feature Detection with Default Thresholds ===" << '\n';
    {
        dsp_core::FeatureDetectionConfig const config;
        std::cout << "Config: derivativeThreshold=" << config.derivativeThreshold
                  << ", significanceThreshold=" << config.significanceThreshold << '\n';

        auto features = dsp_core::Services::CurveFeatureDetector::detectFeatures(
            curveData.data(), static_cast<int>(curveData.size()), ltf->getMinSignalValue(), ltf->getMaxSignalValue(),
            config);
        std::cout << "Result: " << features.localExtrema.size() << " extrema" << '\n';
    }

    // Try with NO thresholds
    std::cout << "\n=== Feature Detection with ZERO Thresholds ===" << '\n';
    {
        dsp_core::FeatureDetectionConfig config;
        config.derivativeThreshold = 0.0;
        config.significanceThreshold = 0.0;

        std::cout << "Config: All thresholds = 0.0" << '\n';

        auto features = dsp_core::Services::CurveFeatureDetector::detectFeatures(
            curveData.data(), static_cast<int>(curveData.size()), ltf->getMinSignalValue(), ltf->getMaxSignalValue(),
            config);
        std::cout << "Result: " << features.localExtrema.size() << " extrema" << '\n';

        for (size_t i = 0; i < std::min(size_t(10), features.localExtrema.size()); ++i) {
            int const idx = features.localExtrema[i];
            double const x = ltf->normalizeIndex(idx);
            double const y = ltf->getBaseLayerValue(idx);
            std::cout << "  Extremum " << i << ": idx=" << idx << ", x=" << x << ", y=" << y << '\n';
        }
    }

    // Manually check derivatives near expected extremum
    std::cout << "\n=== Manual Derivative Check at x=0.866 ===" << '\n';
    for (int offset = -2; offset <= 2; ++offset) {
        int const idx = targetIdx + offset;
        if (idx > 0 && idx < ltf->getTableSize() - 1) {
            double const x = ltf->normalizeIndex(idx);
            double const y = ltf->getBaseLayerValue(idx);

            // Manual central difference
            double const x0 = ltf->normalizeIndex(idx - 1);
            double const x1 = ltf->normalizeIndex(idx + 1);
            double const y0 = ltf->getBaseLayerValue(idx - 1);
            double const y1 = ltf->getBaseLayerValue(idx + 1);
            double const deriv = (y1 - y0) / (x1 - x0);

            std::cout << "  idx=" << idx << ", x=" << std::fixed << std::setprecision(6) << x << ", y=" << y
                      << ", dy/dx=" << std::setprecision(8) << deriv << '\n';
        }
    }

    // Check near the other extremum at x ≈ -0.866
    int const targetIdx2 = static_cast<int>(ltf->getTableSize() * (-0.866 + 1.0) / 2.0);
    std::cout << "\n=== Manual Derivative Check at x=-0.866 ===" << '\n';
    for (int offset = -2; offset <= 2; ++offset) {
        int const idx = targetIdx2 + offset;
        if (idx > 0 && idx < ltf->getTableSize() - 1) {
            double const x = ltf->normalizeIndex(idx);
            double const y = ltf->getBaseLayerValue(idx);

            double const x0 = ltf->normalizeIndex(idx - 1);
            double const x1 = ltf->normalizeIndex(idx + 1);
            double const y0 = ltf->getBaseLayerValue(idx - 1);
            double const y1 = ltf->getBaseLayerValue(idx + 1);
            double const deriv = (y1 - y0) / (x1 - x0);

            std::cout << "  idx=" << idx << ", x=" << std::fixed << std::setprecision(6) << x << ", y=" << y
                      << ", dy/dx=" << std::setprecision(8) << deriv << '\n';
        }
    }
}

} // namespace dsp_core_test
