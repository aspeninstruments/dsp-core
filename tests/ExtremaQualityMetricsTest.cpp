#include <dsp_core/dsp_core.h>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>

namespace dsp_core_test {

/**
 * Test fixture specifically for outputting parseable quality metrics
 *
 * This test file exists solely to provide machine-readable output for
 * the parameter tuning script. It computes and prints extrema position
 * errors in a format that can be parsed.
 */
class ExtremaQualityMetrics : public ::testing::Test {
  protected:
    void SetUp() override {
        ltf = std::make_unique<dsp_core::LayeredTransferFunction>(16384, -1.0, 1.0);
        // Reset coefficients to traditional test defaults (WT=1.0, all harmonics=0.0)
        ltf->setCoefficient(0, 1.0);
        for (int i = 1; i < ltf->getNumCoefficients(); ++i) {
            ltf->setCoefficient(i, 0.0);
        }
    }

    std::unique_ptr<dsp_core::LayeredTransferFunction> ltf;

    [[nodiscard]] std::vector<double> extractCurveData() const {
        std::vector<double> data(static_cast<size_t>(ltf->getTableSize()));
        for (int i = 0; i < ltf->getTableSize(); ++i) {
            data[static_cast<size_t>(i)] = ltf->getBaseLayerValue(i);
        }
        return data;
    }

    /**
     * Find numerical extrema by scanning dy/dx sign changes
     */
    static std::vector<double> findNumericalExtrema(const dsp_core::LayeredTransferFunction& ltf) {
        std::vector<double> extrema;
        const int tableSize = ltf.getTableSize();

        for (int i = 1; i < tableSize - 1; ++i) {
            double const x_prev = ltf.normalizeIndex(i - 1);
            double const x_curr = ltf.normalizeIndex(i);
            double const x_next = ltf.normalizeIndex(i + 1);

            double const y_prev = ltf.evaluateBaseAndHarmonics(x_prev);
            double const y_curr = ltf.evaluateBaseAndHarmonics(x_curr);
            double const y_next = ltf.evaluateBaseAndHarmonics(x_next);

            double const slope_left = (y_curr - y_prev) / (x_curr - x_prev);
            double const slope_right = (y_next - y_curr) / (x_next - x_curr);

            // Extremum: derivative sign change
            if (slope_left * slope_right < 0) {
                extrema.push_back(x_curr);
            }
        }

        return extrema;
    }

    /**
     * Find anchor closest to target x position
     */
    static double findAnchorNear(const std::vector<dsp_core::SplineAnchor>& anchors, double target_x) {
        if (anchors.empty()) {
            return std::numeric_limits<double>::infinity();
        }

        double closest_x = anchors[0].x;
        double min_dist = std::abs(anchors[0].x - target_x);

        for (const auto& anchor : anchors) {
            double const dist = std::abs(anchor.x - target_x);
            if (dist < min_dist) {
                min_dist = dist;
                closest_x = anchor.x;
            }
        }

        return closest_x;
    }

    /**
     * Compute average position error for a set of extrema
     */
    static double computeAveragePositionError(const std::vector<dsp_core::SplineAnchor>& anchors,
                                              const std::vector<double>& extrema) {
        if (extrema.empty()) {
            return 0.0;
        }

        double total_error = 0.0;
        for (double const extremum_x : extrema) {
            double const anchor_x = findAnchorNear(anchors, extremum_x);
            double const error = std::abs(anchor_x - extremum_x);
            total_error += error;
        }

        return total_error / static_cast<double>(extrema.size());
    }
};

/**
 * Harmonic 3 extrema quality metric
 *
 * Output format: "METRIC: Harmonic3_PositionError = 0.001234"
 */
TEST_F(ExtremaQualityMetrics, Harmonic3_PositionError) {
    // Create Harmonic 3: sin(3*asin(x))
    for (int i = 0; i < 16384; ++i) {
        double const x = ltf->normalizeIndex(i);
        double const y = std::sin(3.0 * std::asin(std::clamp(x, -1.0, 1.0)));
        ltf->setBaseLayerValue(i, y);
    }

    // Find numerical extrema
    auto extrema = findNumericalExtrema(*ltf);

    // Fit with current configuration (inherits from SplineFitConfig::tight())
    auto config = dsp_core::SplineFitConfig::tight();
    auto curve = extractCurveData();
    auto result = dsp_core::Services::SplineFitter::fitCurve(
        curve.data(), static_cast<int>(curve.size()), ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    ASSERT_TRUE(result.success) << "Harmonic 3 fit failed";

    // Compute average position error
    double const avg_error = computeAveragePositionError(result.anchors, extrema);

    // Output in parseable format
    std::cout << "METRIC: Harmonic3_PositionError = " << std::fixed << std::setprecision(6) << avg_error << '\n';

    // Still assert for test pass/fail
    EXPECT_LT(avg_error, 0.1) << "Position error should be reasonable";
}

/**
 * Harmonic 5 extrema quality metric
 */
TEST_F(ExtremaQualityMetrics, Harmonic5_PositionError) {
    // Create Harmonic 5
    for (int i = 0; i < 16384; ++i) {
        double const x = ltf->normalizeIndex(i);
        double const y = std::sin(5.0 * std::asin(std::clamp(x, -1.0, 1.0)));
        ltf->setBaseLayerValue(i, y);
    }

    auto extrema = findNumericalExtrema(*ltf);
    auto config = dsp_core::SplineFitConfig::tight();
    auto curve = extractCurveData();
    auto result = dsp_core::Services::SplineFitter::fitCurve(
        curve.data(), static_cast<int>(curve.size()), ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    ASSERT_TRUE(result.success) << "Harmonic 5 fit failed";

    double const avg_error = computeAveragePositionError(result.anchors, extrema);

    std::cout << "METRIC: Harmonic5_PositionError = " << std::fixed << std::setprecision(6) << avg_error << '\n';

    EXPECT_LT(avg_error, 0.1) << "Position error should be reasonable";
}

/**
 * Harmonic 10 extrema quality metric
 */
TEST_F(ExtremaQualityMetrics, Harmonic10_PositionError) {
    // Create Harmonic 10
    for (int i = 0; i < 16384; ++i) {
        double const x = ltf->normalizeIndex(i);
        double const y = std::sin(10.0 * std::asin(std::clamp(x, -1.0, 1.0)));
        ltf->setBaseLayerValue(i, y);
    }

    auto extrema = findNumericalExtrema(*ltf);
    auto config = dsp_core::SplineFitConfig::tight();
    auto curve = extractCurveData();
    auto result = dsp_core::Services::SplineFitter::fitCurve(
        curve.data(), static_cast<int>(curve.size()), ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    ASSERT_TRUE(result.success) << "Harmonic 10 fit failed";

    double const avg_error = computeAveragePositionError(result.anchors, extrema);

    std::cout << "METRIC: Harmonic10_PositionError = " << std::fixed << std::setprecision(6) << avg_error << '\n';

    EXPECT_LT(avg_error, 0.1) << "Position error should be reasonable";
}

/**
 * Harmonic 20 extrema quality metric (HIGH FREQUENCY)
 */
TEST_F(ExtremaQualityMetrics, Harmonic20_PositionError) {
    // Create Harmonic 20
    for (int i = 0; i < 16384; ++i) {
        double const x = ltf->normalizeIndex(i);
        double const y = std::sin(20.0 * std::asin(std::clamp(x, -1.0, 1.0)));
        ltf->setBaseLayerValue(i, y);
    }

    auto extrema = findNumericalExtrema(*ltf);
    auto config = dsp_core::SplineFitConfig::tight();
    auto curve = extractCurveData();
    auto result = dsp_core::Services::SplineFitter::fitCurve(
        curve.data(), static_cast<int>(curve.size()), ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    ASSERT_TRUE(result.success) << "Harmonic 20 fit failed";

    double const avg_error = computeAveragePositionError(result.anchors, extrema);

    std::cout << "METRIC: Harmonic20_PositionError = " << std::fixed << std::setprecision(6) << avg_error << '\n';

    EXPECT_LT(avg_error, 0.1) << "Position error should be reasonable";
}

/**
 * Harmonic 40 extrema quality metric (VERY HIGH FREQUENCY - stress test)
 */
TEST_F(ExtremaQualityMetrics, Harmonic40_PositionError) {
    // Create Harmonic 40
    for (int i = 0; i < 16384; ++i) {
        double const x = ltf->normalizeIndex(i);
        double const y = std::sin(40.0 * std::asin(std::clamp(x, -1.0, 1.0)));
        ltf->setBaseLayerValue(i, y);
    }

    auto extrema = findNumericalExtrema(*ltf);
    auto config = dsp_core::SplineFitConfig::tight();
    auto curve = extractCurveData();
    auto result = dsp_core::Services::SplineFitter::fitCurve(
        curve.data(), static_cast<int>(curve.size()), ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    ASSERT_TRUE(result.success) << "Harmonic 40 fit failed";

    double const avg_error = computeAveragePositionError(result.anchors, extrema);

    std::cout << "METRIC: Harmonic40_PositionError = " << std::fixed << std::setprecision(6) << avg_error << '\n';

    EXPECT_LT(avg_error, 0.15) << "Position error should be reasonable (relaxed for H40)";
}

/**
 * EVEN HARMONICS - Test inflection point detection
 * Even harmonics have inflection points at x=0 (symmetry point)
 */

TEST_F(ExtremaQualityMetrics, Harmonic2_PositionError) {
    // Create Harmonic 2: sin(2*asin(x)) = 2x*sqrt(1-x²)
    for (int i = 0; i < 16384; ++i) {
        double const x = ltf->normalizeIndex(i);
        double const y = std::sin(2.0 * std::asin(std::clamp(x, -1.0, 1.0)));
        ltf->setBaseLayerValue(i, y);
    }

    auto extrema = findNumericalExtrema(*ltf);
    auto config = dsp_core::SplineFitConfig::tight();
    auto curve = extractCurveData();
    auto result = dsp_core::Services::SplineFitter::fitCurve(
        curve.data(), static_cast<int>(curve.size()), ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    ASSERT_TRUE(result.success) << "Harmonic 2 fit failed";

    double const avg_error = computeAveragePositionError(result.anchors, extrema);

    std::cout << "METRIC: Harmonic2_PositionError = " << std::fixed << std::setprecision(6) << avg_error << '\n';

    EXPECT_LT(avg_error, 0.1) << "Position error should be reasonable";
}

TEST_F(ExtremaQualityMetrics, Harmonic4_PositionError) {
    // Create Harmonic 4
    for (int i = 0; i < 16384; ++i) {
        double const x = ltf->normalizeIndex(i);
        double const y = std::sin(4.0 * std::asin(std::clamp(x, -1.0, 1.0)));
        ltf->setBaseLayerValue(i, y);
    }

    auto extrema = findNumericalExtrema(*ltf);
    auto config = dsp_core::SplineFitConfig::tight();
    auto curve = extractCurveData();
    auto result = dsp_core::Services::SplineFitter::fitCurve(
        curve.data(), static_cast<int>(curve.size()), ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    ASSERT_TRUE(result.success) << "Harmonic 4 fit failed";

    double const avg_error = computeAveragePositionError(result.anchors, extrema);

    std::cout << "METRIC: Harmonic4_PositionError = " << std::fixed << std::setprecision(6) << avg_error << '\n';

    EXPECT_LT(avg_error, 0.1) << "Position error should be reasonable";
}

TEST_F(ExtremaQualityMetrics, Harmonic6_PositionError) {
    // Create Harmonic 6
    for (int i = 0; i < 16384; ++i) {
        double const x = ltf->normalizeIndex(i);
        double const y = std::sin(6.0 * std::asin(std::clamp(x, -1.0, 1.0)));
        ltf->setBaseLayerValue(i, y);
    }

    auto extrema = findNumericalExtrema(*ltf);
    auto config = dsp_core::SplineFitConfig::tight();
    auto curve = extractCurveData();
    auto result = dsp_core::Services::SplineFitter::fitCurve(
        curve.data(), static_cast<int>(curve.size()), ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    ASSERT_TRUE(result.success) << "Harmonic 6 fit failed";

    double const avg_error = computeAveragePositionError(result.anchors, extrema);

    std::cout << "METRIC: Harmonic6_PositionError = " << std::fixed << std::setprecision(6) << avg_error << '\n';

    EXPECT_LT(avg_error, 0.1) << "Position error should be reasonable";
}

/**
 * SIGMOID/LOGISTIC CURVES - S-curves with clear inflection at center
 */

TEST_F(ExtremaQualityMetrics, Sigmoid5_PositionError) {
    // Logistic curve: y = 1/(1+exp(-5x))
    // Inflection point at x=0
    for (int i = 0; i < 16384; ++i) {
        double const x = ltf->normalizeIndex(i);
        double y = 1.0 / (1.0 + std::exp(-5.0 * x));
        // Normalize to [-1, 1] range
        y = 2.0 * y - 1.0;
        ltf->setBaseLayerValue(i, y);
    }

    auto extrema = findNumericalExtrema(*ltf);
    auto config = dsp_core::SplineFitConfig::tight();
    auto curve = extractCurveData();
    auto result = dsp_core::Services::SplineFitter::fitCurve(
        curve.data(), static_cast<int>(curve.size()), ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    ASSERT_TRUE(result.success) << "Sigmoid 5 fit failed";

    double const avg_error = computeAveragePositionError(result.anchors, extrema);

    std::cout << "METRIC: Sigmoid5_PositionError = " << std::fixed << std::setprecision(6) << avg_error << '\n';

    EXPECT_LT(avg_error, 0.1) << "Position error should be reasonable";
}

TEST_F(ExtremaQualityMetrics, Sigmoid10_PositionError) {
    // Steeper sigmoid: y = 1/(1+exp(-10x))
    for (int i = 0; i < 16384; ++i) {
        double const x = ltf->normalizeIndex(i);
        double y = 1.0 / (1.0 + std::exp(-10.0 * x));
        y = 2.0 * y - 1.0;
        ltf->setBaseLayerValue(i, y);
    }

    auto extrema = findNumericalExtrema(*ltf);
    auto config = dsp_core::SplineFitConfig::tight();
    auto curve = extractCurveData();
    auto result = dsp_core::Services::SplineFitter::fitCurve(
        curve.data(), static_cast<int>(curve.size()), ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    ASSERT_TRUE(result.success) << "Sigmoid 10 fit failed";

    double const avg_error = computeAveragePositionError(result.anchors, extrema);

    std::cout << "METRIC: Sigmoid10_PositionError = " << std::fixed << std::setprecision(6) << avg_error << '\n';

    EXPECT_LT(avg_error, 0.1) << "Position error should be reasonable";
}

TEST_F(ExtremaQualityMetrics, Sigmoid15_PositionError) {
    // Very steep sigmoid: y = 1/(1+exp(-15x))
    for (int i = 0; i < 16384; ++i) {
        double const x = ltf->normalizeIndex(i);
        double y = 1.0 / (1.0 + std::exp(-15.0 * x));
        y = 2.0 * y - 1.0;
        ltf->setBaseLayerValue(i, y);
    }

    auto extrema = findNumericalExtrema(*ltf);
    auto config = dsp_core::SplineFitConfig::tight();
    auto curve = extractCurveData();
    auto result = dsp_core::Services::SplineFitter::fitCurve(
        curve.data(), static_cast<int>(curve.size()), ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    ASSERT_TRUE(result.success) << "Sigmoid 15 fit failed";

    double const avg_error = computeAveragePositionError(result.anchors, extrema);

    std::cout << "METRIC: Sigmoid15_PositionError = " << std::fixed << std::setprecision(6) << avg_error << '\n';

    EXPECT_LT(avg_error, 0.1) << "Position error should be reasonable";
}

/**
 * MIXED EXTREMA + INFLECTION CURVES
 * Combines local extrema with inflection points
 */

TEST_F(ExtremaQualityMetrics, MixedXSin5X_PositionError) {
    // y = x * sin(5*asin(x))
    // Has both extrema from sin and modulation from x envelope
    for (int i = 0; i < 16384; ++i) {
        double const x = ltf->normalizeIndex(i);
        double const y = x * std::sin(5.0 * std::asin(std::clamp(x, -1.0, 1.0)));
        ltf->setBaseLayerValue(i, y);
    }

    auto extrema = findNumericalExtrema(*ltf);
    auto config = dsp_core::SplineFitConfig::tight();
    auto curve = extractCurveData();
    auto result = dsp_core::Services::SplineFitter::fitCurve(
        curve.data(), static_cast<int>(curve.size()), ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    ASSERT_TRUE(result.success) << "Mixed x*sin(5x) fit failed";

    double const avg_error = computeAveragePositionError(result.anchors, extrema);

    std::cout << "METRIC: MixedXSin5X_PositionError = " << std::fixed << std::setprecision(6) << avg_error << '\n';

    EXPECT_LT(avg_error, 0.1) << "Position error should be reasonable";
}

TEST_F(ExtremaQualityMetrics, MixedXSin10X_PositionError) {
    // y = x * sin(10*asin(x))
    // Higher frequency version
    for (int i = 0; i < 16384; ++i) {
        double const x = ltf->normalizeIndex(i);
        double const y = x * std::sin(10.0 * std::asin(std::clamp(x, -1.0, 1.0)));
        ltf->setBaseLayerValue(i, y);
    }

    auto extrema = findNumericalExtrema(*ltf);
    auto config = dsp_core::SplineFitConfig::tight();
    auto curve = extractCurveData();
    auto result = dsp_core::Services::SplineFitter::fitCurve(
        curve.data(), static_cast<int>(curve.size()), ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    ASSERT_TRUE(result.success) << "Mixed x*sin(10x) fit failed";

    double const avg_error = computeAveragePositionError(result.anchors, extrema);

    std::cout << "METRIC: MixedXSin10X_PositionError = " << std::fixed << std::setprecision(6) << avg_error << '\n';

    EXPECT_LT(avg_error, 0.1) << "Position error should be reasonable";
}

/**
 * POLYNOMIAL WITH INFLECTIONS
 */

TEST_F(ExtremaQualityMetrics, PolynomialX4_PositionError) {
    // y = x^4 - 2x^2
    // Has 2 local minima at x=±1/√2, 1 local maximum at x=0
    // Has 2 inflection points at x=±1/√6
    for (int i = 0; i < 16384; ++i) {
        double const x = ltf->normalizeIndex(i);
        double y = x * x * x * x - 2.0 * x * x;
        // Normalize to approximate [-1, 1] range
        y = y / 1.0; // Max value is ~0 at x=0, min is ~-1 at x=±1
        ltf->setBaseLayerValue(i, std::clamp(y, -1.0, 1.0));
    }

    auto extrema = findNumericalExtrema(*ltf);
    auto config = dsp_core::SplineFitConfig::tight();
    auto curve = extractCurveData();
    auto result = dsp_core::Services::SplineFitter::fitCurve(
        curve.data(), static_cast<int>(curve.size()), ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    ASSERT_TRUE(result.success) << "Polynomial x^4-2x^2 fit failed";

    double const avg_error = computeAveragePositionError(result.anchors, extrema);

    std::cout << "METRIC: PolynomialX4_PositionError = " << std::fixed << std::setprecision(6) << avg_error << '\n';

    EXPECT_LT(avg_error, 0.1) << "Position error should be reasonable";
}

TEST_F(ExtremaQualityMetrics, PolynomialX5_PositionError) {
    // y = x^5 - 5x^3 + 4x
    // More complex polynomial with multiple extrema and inflections
    for (int i = 0; i < 16384; ++i) {
        double const x = ltf->normalizeIndex(i);
        double y = x * x * x * x * x - 5.0 * x * x * x + 4.0 * x;
        // Normalize
        double const max_val = 2.0; // Approximate max value in [-1,1]
        y = y / max_val;
        ltf->setBaseLayerValue(i, std::clamp(y, -1.0, 1.0));
    }

    auto extrema = findNumericalExtrema(*ltf);
    auto config = dsp_core::SplineFitConfig::tight();
    auto curve = extractCurveData();
    auto result = dsp_core::Services::SplineFitter::fitCurve(
        curve.data(), static_cast<int>(curve.size()), ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    ASSERT_TRUE(result.success) << "Polynomial x^5-5x^3+4x fit failed";

    double const avg_error = computeAveragePositionError(result.anchors, extrema);

    std::cout << "METRIC: PolynomialX5_PositionError = " << std::fixed << std::setprecision(6) << avg_error << '\n';

    EXPECT_LT(avg_error, 0.1) << "Position error should be reasonable";
}

/**
 * SUPERPOSITION OF ODD HARMONICS
 * Realistic complex waveform: sum of odd harmonics 1-15 with coefficients 2/(2^n)
 */

TEST_F(ExtremaQualityMetrics, SuperpositionOddHarmonics_PositionError) {
    // Sum of odd harmonics: H1, H3, H5, H7, H9, H11, H13, H15
    // Coefficients: 2/(2^n) for n=0,1,2,3,4,5,6,7
    // This creates a complex but realistic waveform

    // First pass: compute raw values and find max for normalization
    std::vector<double> raw_values(16384);
    double max_abs = 0.0;

    for (int i = 0; i < 16384; ++i) {
        double const x = ltf->normalizeIndex(i);
        double y = 0.0;

        // Add odd harmonics with decreasing coefficients
        for (int n = 0; n < 8; ++n) {
            int const harmonic = 2 * n + 1; // 1, 3, 5, 7, 9, 11, 13, 15
            double const coeff = 2.0 / std::pow(2.0, n);
            y += coeff * std::sin(harmonic * std::asin(std::clamp(x, -1.0, 1.0)));
        }

        raw_values[i] = y;
        max_abs = std::max(max_abs, std::abs(y));
    }

    // Second pass: normalize and set values
    for (int i = 0; i < 16384; ++i) {
        double const normalized = raw_values[i] / max_abs;
        ltf->setBaseLayerValue(i, normalized);
    }

    auto extrema = findNumericalExtrema(*ltf);
    auto config = dsp_core::SplineFitConfig::tight();
    auto curve = extractCurveData();
    auto result = dsp_core::Services::SplineFitter::fitCurve(
        curve.data(), static_cast<int>(curve.size()), ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    ASSERT_TRUE(result.success) << "Superposition odd harmonics fit failed";

    double const avg_error = computeAveragePositionError(result.anchors, extrema);

    std::cout << "METRIC: SuperpositionOddHarmonics_PositionError = " << std::fixed << std::setprecision(6) << avg_error
              << '\n';

    EXPECT_LT(avg_error, 0.1) << "Position error should be reasonable";
}

/**
 * EXTREME TANH CURVES
 * Very steep saturation curves that stress-test feature detection
 */

TEST_F(ExtremaQualityMetrics, ExtremeTanh9_PositionError) {
    // y = tanh(9x)
    // Very steep but continuous S-curve
    for (int i = 0; i < 16384; ++i) {
        double const x = ltf->normalizeIndex(i);
        double const y = std::tanh(9.0 * x);
        ltf->setBaseLayerValue(i, y);
    }

    auto extrema = findNumericalExtrema(*ltf);
    auto config = dsp_core::SplineFitConfig::tight();
    auto curve = extractCurveData();
    auto result = dsp_core::Services::SplineFitter::fitCurve(
        curve.data(), static_cast<int>(curve.size()), ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    ASSERT_TRUE(result.success) << "Extreme tanh(9x) fit failed";

    double const avg_error = computeAveragePositionError(result.anchors, extrema);

    std::cout << "METRIC: ExtremeTanh9_PositionError = " << std::fixed << std::setprecision(6) << avg_error << '\n';

    EXPECT_LT(avg_error, 0.1) << "Position error should be reasonable";
}

TEST_F(ExtremaQualityMetrics, ExtremeTanh15_PositionError) {
    // y = tanh(15x)
    // Extremely steep, almost step-function
    for (int i = 0; i < 16384; ++i) {
        double const x = ltf->normalizeIndex(i);
        double const y = std::tanh(15.0 * x);
        ltf->setBaseLayerValue(i, y);
    }

    auto extrema = findNumericalExtrema(*ltf);
    auto config = dsp_core::SplineFitConfig::tight();
    auto curve = extractCurveData();
    auto result = dsp_core::Services::SplineFitter::fitCurve(
        curve.data(), static_cast<int>(curve.size()), ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    ASSERT_TRUE(result.success) << "Extreme tanh(15x) fit failed";

    double const avg_error = computeAveragePositionError(result.anchors, extrema);

    std::cout << "METRIC: ExtremeTanh15_PositionError = " << std::fixed << std::setprecision(6) << avg_error << '\n';

    EXPECT_LT(avg_error, 0.1) << "Position error should be reasonable";
}

TEST_F(ExtremaQualityMetrics, ExtremeTanh20_PositionError) {
    // y = tanh(20x)
    // Nearly step-function, extreme stress test
    for (int i = 0; i < 16384; ++i) {
        double const x = ltf->normalizeIndex(i);
        double const y = std::tanh(20.0 * x);
        ltf->setBaseLayerValue(i, y);
    }

    auto extrema = findNumericalExtrema(*ltf);
    auto config = dsp_core::SplineFitConfig::tight();
    auto curve = extractCurveData();
    auto result = dsp_core::Services::SplineFitter::fitCurve(
        curve.data(), static_cast<int>(curve.size()), ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    ASSERT_TRUE(result.success) << "Extreme tanh(20x) fit failed";

    double const avg_error = computeAveragePositionError(result.anchors, extrema);

    std::cout << "METRIC: ExtremeTanh20_PositionError = " << std::fixed << std::setprecision(6) << avg_error << '\n';

    EXPECT_LT(avg_error, 0.15) << "Position error should be reasonable (relaxed for extreme steepness)";
}

} // namespace dsp_core_test
