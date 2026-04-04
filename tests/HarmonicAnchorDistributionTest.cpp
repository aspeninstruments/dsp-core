#include <dsp_core/dsp_core.h>
#include <gtest/gtest.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace dsp_core_test {

/**
 * Diagnostic test to understand anchor distribution for harmonics H3 and H5
 */
class HarmonicAnchorDistributionTest : public ::testing::Test {
  protected:
    void SetUp() override {
        ltf = std::make_unique<dsp_core::LayeredTransferFunction>(16384, -1.0, 1.0);
    }

    void setHarmonicCurve(int harmonicNumber) {
        for (int i = 0; i < ltf->getTableSize(); ++i) {
            double x = ltf->normalizeIndex(i);
            x = std::clamp(x, -1.0, 1.0);

            double y = 0.0;
            if (harmonicNumber % 2 == 0) {
                y = std::cos(harmonicNumber * std::acos(x));
            } else {
                y = std::sin(harmonicNumber * std::asin(x));
            }

            ltf->setBaseLayerValue(i, y);
        }
    }

    static void analyzeAnchorDistribution(const std::vector<dsp_core::SplineAnchor>& anchors, int harmonic) {
        std::cout << "\n=== Harmonic " << harmonic << " Anchor Distribution ===" << '\n';
        std::cout << "Total anchors: " << anchors.size() << '\n';

        // Count anchors in regions
        int nearMinusOne = 0; // [-1.0, -0.66]
        int middle = 0;       // (-0.66, 0.66)
        int nearPlusOne = 0;  // [0.66, 1.0]

        std::cout << "\nAnchors near x=-1 (x < -0.66):" << '\n';
        for (const auto& anchor : anchors) {
            if (anchor.x < -0.66) {
                nearMinusOne++;
                std::cout << "  x=" << std::fixed << std::setprecision(6) << anchor.x << ", y=" << anchor.y
                          << '\n';
            } else if (anchor.x < 0.66) {
                middle++;
            } else {
                nearPlusOne++;
            }
        }

        std::cout << "\nRegion distribution:" << '\n';
        std::cout << "  [-1.0, -0.66]: " << nearMinusOne << " anchors (" << (100.0 * static_cast<double>(nearMinusOne) / static_cast<double>(anchors.size()))
                  << "%)" << '\n';
        std::cout << "  (-0.66, 0.66): " << middle << " anchors (" << (100.0 * static_cast<double>(middle) / static_cast<double>(anchors.size())) << "%)"
                  << '\n';
        std::cout << "  [0.66, 1.0]:   " << nearPlusOne << " anchors (" << (100.0 * static_cast<double>(nearPlusOne) / static_cast<double>(anchors.size()))
                  << "%)" << '\n';
    }

    std::vector<double> extractCurveData() const {
        std::vector<double> data(ltf->getTableSize());
        for (int i = 0; i < ltf->getTableSize(); ++i) {
            data[i] = ltf->getBaseLayerValue(i);
        }
        return data;
    }

    std::unique_ptr<dsp_core::LayeredTransferFunction> ltf;
};

TEST_F(HarmonicAnchorDistributionTest, Harmonic3_FeatureDetection) {
    setHarmonicCurve(3);

    auto config = dsp_core::SplineFitConfig::tight();
    config.enableFeatureDetection = true;

    // Show what features are detected
    dsp_core::FeatureDetectionConfig featureConfig;
    featureConfig.maxFeatures = 100;
    auto curveData = extractCurveData();

    auto features = dsp_core::Services::CurveFeatureDetector::detectFeatures(
        curveData.data(), static_cast<int>(curveData.size()),
        ltf->getMinSignalValue(), ltf->getMaxSignalValue(), featureConfig);

    std::cout << "\n=== H3 Feature Detection ===" << '\n';
    std::cout << "Local extrema: " << features.localExtrema.size() << '\n';
    for (size_t i = 0; i < features.localExtrema.size(); ++i) {
        int const idx = features.localExtrema[i];
        double const x = ltf->normalizeIndex(idx);
        std::cout << "  Extremum " << i << ": x=" << x << '\n';
    }

    auto result = dsp_core::Services::SplineFitter::fitCurve(
        curveData.data(), static_cast<int>(curveData.size()),
        ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);
    ASSERT_TRUE(result.success);

    analyzeAnchorDistribution(result.anchors, 3);
}

TEST_F(HarmonicAnchorDistributionTest, Harmonic3_NoFeatureDetection) {
    setHarmonicCurve(3);

    // Test with NO feature detection at all
    auto config = dsp_core::SplineFitConfig::tight();
    config.enableFeatureDetection = false;

    auto curveData = extractCurveData();

    auto result = dsp_core::Services::SplineFitter::fitCurve(
        curveData.data(), static_cast<int>(curveData.size()),
        ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);
    ASSERT_TRUE(result.success);

    std::cout << "\n=== NO Feature Detection (Pure Greedy) ===" << '\n';
    analyzeAnchorDistribution(result.anchors, 3);
}

TEST_F(HarmonicAnchorDistributionTest, Harmonic5_FeatureDetection) {
    setHarmonicCurve(5);

    auto config = dsp_core::SplineFitConfig::tight();
    config.enableFeatureDetection = true;

    // Show what features are detected
    dsp_core::FeatureDetectionConfig featureConfig;
    featureConfig.maxFeatures = 100;
    auto curveData = extractCurveData();

    auto features = dsp_core::Services::CurveFeatureDetector::detectFeatures(
        curveData.data(), static_cast<int>(curveData.size()),
        ltf->getMinSignalValue(), ltf->getMaxSignalValue(), featureConfig);

    std::cout << "\n=== H5 Feature Detection ===" << '\n';
    std::cout << "Local extrema: " << features.localExtrema.size() << '\n';

    auto result = dsp_core::Services::SplineFitter::fitCurve(
        curveData.data(), static_cast<int>(curveData.size()),
        ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);
    ASSERT_TRUE(result.success);

    analyzeAnchorDistribution(result.anchors, 5);
}

// Regression test: Verify that boundary clustering bug is fixed
TEST_F(HarmonicAnchorDistributionTest, NoBoundaryClusteringRegression) {
    auto config = dsp_core::SplineFitConfig::tight();
    config.enableFeatureDetection = true;

    // Test H3
    {
        setHarmonicCurve(3);
        auto curveData = extractCurveData();

        auto result = dsp_core::Services::SplineFitter::fitCurve(
            curveData.data(), static_cast<int>(curveData.size()),
            ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);
        ASSERT_TRUE(result.success);

        // Count anchors near boundaries
        int nearMinusOne = 0;
        int nearPlusOne = 0;
        for (const auto& anchor : result.anchors) {
            if (anchor.x < -0.66) {
                nearMinusOne++;
}
            if (anchor.x > 0.66) {
                nearPlusOne++;
}
        }

        // Both sides should have similar anchor density
        double const leftRatio = static_cast<double>(nearMinusOne) / static_cast<double>(result.anchors.size());
        double const rightRatio = static_cast<double>(nearPlusOne) / static_cast<double>(result.anchors.size());

        // Neither boundary should have more than 50% of anchors clustered
        EXPECT_LT(leftRatio, 0.50) << "H3: Too many anchors clustered near x=-1";
        EXPECT_LT(rightRatio, 0.50) << "H3: Too many anchors clustered near x=+1";
    }

    // Test H5
    {
        setHarmonicCurve(5);
        auto curveData = extractCurveData();

        auto result = dsp_core::Services::SplineFitter::fitCurve(
            curveData.data(), static_cast<int>(curveData.size()),
            ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);
        ASSERT_TRUE(result.success);

        int nearMinusOne = 0;
        int nearPlusOne = 0;
        for (const auto& anchor : result.anchors) {
            if (anchor.x < -0.66) {
                nearMinusOne++;
}
            if (anchor.x > 0.66) {
                nearPlusOne++;
}
        }

        double const leftRatio = static_cast<double>(nearMinusOne) / static_cast<double>(result.anchors.size());
        double const rightRatio = static_cast<double>(nearPlusOne) / static_cast<double>(result.anchors.size());

        EXPECT_LT(leftRatio, 0.50) << "H5: Too many anchors clustered near x=-1";
        EXPECT_LT(rightRatio, 0.50) << "H5: Too many anchors clustered near x=+1";
    }
}

} // namespace dsp_core_test
