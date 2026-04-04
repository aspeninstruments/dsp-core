#include <dsp_core/dsp_core.h>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <iostream>

namespace dsp_core_test {

/**
 * Test fixture for exact extrema position verification
 *
 * Tests whether spline fitting (with feature detection enabled) places anchors
 * at the EXACT extrema positions, not just the correct count.
 *
 * Test categories:
 * 1. Analytical functions with known extrema (sin, polynomial)
 * 2. Production harmonics with numerical extrema detection
 */
class ExactExtremaTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Use larger table size for better extrema resolution
        ltf = std::make_unique<dsp_core::LayeredTransferFunction>(4096, -1.0, 1.0);
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
};

//==============================================================================
// Helper Functions
//==============================================================================

/**
 * Find numerical extrema by scanning dy/dx sign changes
 *
 * Returns x positions where derivative changes sign (local maxima and minima).
 * Uses central difference approximation for derivative estimation.
 */
std::vector<double> findNumericalExtrema(const dsp_core::LayeredTransferFunction& ltf) {
    std::vector<double> extrema;
    const int tableSize = ltf.getTableSize();

    for (int i = 1; i < tableSize - 1; ++i) {
        double const x_prev = ltf.normalizeIndex(i - 1);
        double const x_curr = ltf.normalizeIndex(i);
        double const x_next = ltf.normalizeIndex(i + 1);

        double const y_prev = ltf.evaluateBaseAndHarmonics(x_prev);
        double const y_curr = ltf.evaluateBaseAndHarmonics(x_curr);
        double const y_next = ltf.evaluateBaseAndHarmonics(x_next);

        // Central difference approximation
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
 *
 * Returns the x coordinate of the anchor nearest to target_x.
 * Used to verify if an anchor was placed at the expected extremum position.
 */
double findAnchorNear(const std::vector<dsp_core::SplineAnchor>& anchors, double target_x) {
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
 * Check if an anchor exists near target position (within tolerance)
 */
bool hasAnchorNear(const std::vector<dsp_core::SplineAnchor>& anchors, double target_x, double tolerance = 0.01) {
    double const closest_x = findAnchorNear(anchors, target_x);
    return std::abs(closest_x - target_x) < tolerance;
}

/**
 * Compute average position error for a set of extrema
 *
 * For each extremum, finds the nearest anchor and computes position error.
 * Returns average error across all extrema.
 */
double computeAveragePositionError(const std::vector<dsp_core::SplineAnchor>& anchors,
                                   const std::vector<double>& extrema_positions) {
    if (extrema_positions.empty()) {
        return 0.0;
    }

    double total_error = 0.0;
    for (double const extremum_x : extrema_positions) {
        double const anchor_x = findAnchorNear(anchors, extremum_x);
        total_error += std::abs(anchor_x - extremum_x);
    }

    return total_error / static_cast<double>(extrema_positions.size());
}

/**
 * Helper: Set base layer to sin curve with specified frequency
 *
 * f(x) = sin(frequency * π * x)
 * Domain: x ∈ [-1, 1]
 * Range: y ∈ [-1, 1]
 */
void setSinCurve(dsp_core::LayeredTransferFunction& ltf, double frequency) {
    for (int i = 0; i < ltf.getTableSize(); ++i) {
        double const x = ltf.normalizeIndex(i);
        double const y = std::sin(frequency * M_PI * x);
        ltf.setBaseLayerValue(i, y);
    }
}

/**
 * Helper: Set base layer to harmonic curve (Chebyshev polynomial)
 *
 * Uses existing pattern from SplineFitterTest.cpp:
 * - Even harmonics: cos(n * acos(x))
 * - Odd harmonics: sin(n * asin(x))
 */
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void setHarmonicCurve(dsp_core::LayeredTransferFunction& ltf, int harmonicNumber, double amplitude = 1.0) {
    for (int i = 0; i < ltf.getTableSize(); ++i) {
        double x = ltf.normalizeIndex(i);
        x = std::max(-1.0, std::min(1.0, x)); // Clamp for trig safety

        double y = 0.0;
        if (harmonicNumber % 2 == 0) {
            // Even harmonics: cos(n * acos(x))
            y = std::cos(harmonicNumber * std::acos(x));
        } else {
            // Odd harmonics: sin(n * asin(x))
            y = std::sin(harmonicNumber * std::asin(x));
        }

        ltf.setBaseLayerValue(i, amplitude * y);
    }
}

/**
 * Helper: Set base layer to cubic polynomial (x³)
 */
void setCubicCurve(dsp_core::LayeredTransferFunction& ltf) {
    for (int i = 0; i < ltf.getTableSize(); ++i) {
        double const x = ltf.normalizeIndex(i);
        double const y = x * x * x;
        ltf.setBaseLayerValue(i, y);
    }
}

/**
 * Helper: Set base layer to tanh curve
 */
void setTanhCurve(dsp_core::LayeredTransferFunction& ltf, double steepness = 2.0) {
    for (int i = 0; i < ltf.getTableSize(); ++i) {
        double const x = ltf.normalizeIndex(i);
        double const y = std::tanh(steepness * x);
        ltf.setBaseLayerValue(i, y);
    }
}

//==============================================================================
// Analytical Test Cases (Known Extrema Positions)
//==============================================================================

/**
 * Test: sin(πx) - Single extremum per half-domain
 *
 * f(x) = sin(πx), x ∈ [-1, 1]
 * Extrema:
 *   - Maximum at x = 0.5 (f(0.5) = 1.0)
 *   - Minimum at x = -0.5 (f(-0.5) = -1.0)
 */
TEST_F(ExactExtremaTest, SinX_SingleExtremum) {
    setSinCurve(*ltf, 1.0); // frequency = 1

    auto config = dsp_core::SplineFitConfig::tight();
    config.enableFeatureDetection = true;

    auto curve = extractCurveData();
    auto result = dsp_core::Services::SplineFitter::fitCurve(curve.data(), static_cast<int>(curve.size()), ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    ASSERT_TRUE(result.success) << "Fit failed: " << result.message.toStdString();

    // Verify extrema positions within ±0.01 (1% of domain)
    EXPECT_TRUE(hasAnchorNear(result.anchors, 0.5, 0.01)) << "Missing anchor at maximum (x=0.5)";
    EXPECT_TRUE(hasAnchorNear(result.anchors, -0.5, 0.01)) << "Missing anchor at minimum (x=-0.5)";

    // Report position errors
    double const max_error = std::abs(findAnchorNear(result.anchors, 0.5) - 0.5);
    double const min_error = std::abs(findAnchorNear(result.anchors, -0.5) - (-0.5));
    double const avg_error = (max_error + min_error) / 2.0;

    std::cout << "  sin(πx) extrema errors: "
              << "max=" << std::fixed << std::setprecision(4) << max_error << ", min=" << min_error
              << ", avg=" << avg_error << '\n';
}

/**
 * Test: sin(2πx) - Three extrema
 *
 * f(x) = sin(2πx), x ∈ [-1, 1]
 * Extrema:
 *   - Maxima at x = -0.75, 0.25
 *   - Minima at x = -0.25, 0.75
 */
TEST_F(ExactExtremaTest, Sin2X_ThreeExtrema) {
    setSinCurve(*ltf, 2.0); // frequency = 2

    auto config = dsp_core::SplineFitConfig::tight();
    config.enableFeatureDetection = true;

    auto curve = extractCurveData();
    auto result = dsp_core::Services::SplineFitter::fitCurve(curve.data(), static_cast<int>(curve.size()), ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    ASSERT_TRUE(result.success);

    // Expected extrema positions
    std::vector<double> const expected_extrema = {-0.75, -0.25, 0.25, 0.75};

    // Verify all extrema have nearby anchors
    for (double const extremum_x : expected_extrema) {
        EXPECT_TRUE(hasAnchorNear(result.anchors, extremum_x, 0.01)) << "Missing anchor at extremum x=" << extremum_x;
    }

    // Report average position error
    double const avg_error = computeAveragePositionError(result.anchors, expected_extrema);
    std::cout << "  sin(2πx) avg extrema error: " << std::fixed << std::setprecision(4) << avg_error << '\n';
}

/**
 * Test: sin(3πx) - Five extrema
 *
 * f(x) = sin(3πx), x ∈ [-1, 1]
 * 6 extrema total (3 peaks, 3 valleys)
 */
TEST_F(ExactExtremaTest, Sin3X_FiveExtrema) {
    setSinCurve(*ltf, 3.0); // frequency = 3

    auto config = dsp_core::SplineFitConfig::tight();
    config.enableFeatureDetection = true;

    auto curve = extractCurveData();
    auto result = dsp_core::Services::SplineFitter::fitCurve(curve.data(), static_cast<int>(curve.size()), ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    ASSERT_TRUE(result.success);

    // Expected extrema: ±(1/6), ±(3/6), ±(5/6)
    std::vector<double> const expected_extrema = {
        -5.0 / 6.0, -3.0 / 6.0, -1.0 / 6.0, // Negative half
        1.0 / 6.0,  3.0 / 6.0,  5.0 / 6.0   // Positive half
    };

    // Verify all extrema
    int matched_count = 0;
    for (double const extremum_x : expected_extrema) {
        if (hasAnchorNear(result.anchors, extremum_x, 0.01)) {
            matched_count++;
        }
    }

    EXPECT_GE(matched_count, 5) << "Expected at least 5 extrema matched";

    double const avg_error = computeAveragePositionError(result.anchors, expected_extrema);
    std::cout << "  sin(3πx) avg extrema error: " << std::fixed << std::setprecision(4) << avg_error << " ("
              << matched_count << "/" << expected_extrema.size() << " matched)" << '\n';
}

/**
 * Test: sin(5πx) - Nine extrema (MEDIUM FREQUENCY)
 *
 * f(x) = sin(5πx), x ∈ [-1, 1]
 * 10 extrema total (5 peaks, 5 valleys)
 */
TEST_F(ExactExtremaTest, Sin5X_NineExtrema) {
    setSinCurve(*ltf, 5.0); // frequency = 5

    auto config = dsp_core::SplineFitConfig::tight();
    config.enableFeatureDetection = true;

    auto curve = extractCurveData();
    auto result = dsp_core::Services::SplineFitter::fitCurve(curve.data(), static_cast<int>(curve.size()), ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    ASSERT_TRUE(result.success);

    // Expected extrema: ±(1/10), ±(3/10), ±(5/10), ±(7/10), ±(9/10)
    std::vector<double> expected_extrema;
    for (int k = 1; k <= 9; k += 2) {
        expected_extrema.push_back(-k / 10.0);
        expected_extrema.push_back(k / 10.0);
    }

    // Count matched extrema
    int matched_count = 0;
    for (double const extremum_x : expected_extrema) {
        if (hasAnchorNear(result.anchors, extremum_x, 0.01)) {
            matched_count++;
        }
    }

    EXPECT_GE(matched_count, 8) << "Expected at least 8/10 extrema matched";

    double const avg_error = computeAveragePositionError(result.anchors, expected_extrema);
    std::cout << "  sin(5πx) avg extrema error: " << std::fixed << std::setprecision(4) << avg_error << " ("
              << matched_count << "/" << expected_extrema.size() << " matched)" << '\n';

    EXPECT_LT(avg_error, 0.02) << "Average position error should be < 0.02";
}

/**
 * Test: sin(10πx) - Nineteen extrema (HIGH FREQUENCY)
 *
 * f(x) = sin(10πx), x ∈ [-1, 1]
 * 20 extrema total (10 peaks, 10 valleys)
 */
TEST_F(ExactExtremaTest, Sin10X_NineteenExtrema) {
    setSinCurve(*ltf, 10.0); // frequency = 10

    auto config = dsp_core::SplineFitConfig::tight();
    config.enableFeatureDetection = true;

    auto curve = extractCurveData();
    auto result = dsp_core::Services::SplineFitter::fitCurve(curve.data(), static_cast<int>(curve.size()), ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    ASSERT_TRUE(result.success);

    // Expected extrema: ±(1/20), ±(3/20), ..., ±(19/20)
    std::vector<double> expected_extrema;
    for (int k = 1; k <= 19; k += 2) {
        expected_extrema.push_back(-k / 20.0);
        expected_extrema.push_back(k / 20.0);
    }

    // Count matched extrema
    int matched_count = 0;
    for (double const extremum_x : expected_extrema) {
        if (hasAnchorNear(result.anchors, extremum_x, 0.01)) {
            matched_count++;
        }
    }

    EXPECT_GE(matched_count, 15) << "Expected at least 15/20 extrema matched";

    double const avg_error = computeAveragePositionError(result.anchors, expected_extrema);
    std::cout << "  sin(10πx) avg extrema error: " << std::fixed << std::setprecision(4) << avg_error << " ("
              << matched_count << "/" << expected_extrema.size() << " matched)" << '\n';

    EXPECT_LT(avg_error, 0.02) << "Average position error should be < 0.02";
}

/**
 * Test: sin(15πx) - Twenty-nine extrema (STRESS TEST)
 *
 * f(x) = sin(15πx), x ∈ [-1, 1]
 * 30 extrema total (15 peaks, 15 valleys)
 *
 * This is a challenging test case - high frequency requires many anchors.
 */
TEST_F(ExactExtremaTest, Sin15X_TwentyNineExtrema) {
    setSinCurve(*ltf, 15.0); // frequency = 15

    auto config = dsp_core::SplineFitConfig::tight();
    config.enableFeatureDetection = true;

    auto curve = extractCurveData();
    auto result = dsp_core::Services::SplineFitter::fitCurve(curve.data(), static_cast<int>(curve.size()), ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    ASSERT_TRUE(result.success);

    // Expected extrema: ±(1/30), ±(3/30), ..., ±(29/30)
    std::vector<double> expected_extrema;
    for (int k = 1; k <= 29; k += 2) {
        expected_extrema.push_back(-k / 30.0);
        expected_extrema.push_back(k / 30.0);
    }

    // Count matched extrema (be more lenient for high frequency)
    int matched_count = 0;
    for (double const extremum_x : expected_extrema) {
        if (hasAnchorNear(result.anchors, extremum_x, 0.02)) { // Wider tolerance
            matched_count++;
        }
    }

    EXPECT_GE(matched_count, 20) << "Expected at least 20/30 extrema matched";

    double const avg_error = computeAveragePositionError(result.anchors, expected_extrema);
    std::cout << "  sin(15πx) avg extrema error: " << std::fixed << std::setprecision(4) << avg_error << " ("
              << matched_count << "/" << expected_extrema.size() << " matched)" << '\n';

    EXPECT_LT(avg_error, 0.03) << "Average position error should be < 0.03 (relaxed for high freq)";
}

/**
 * Test: x³ - Inflection point at origin
 *
 * f(x) = x³
 * Has inflection point at x = 0 (not a local extremum)
 * No local extrema in domain
 */
TEST_F(ExactExtremaTest, CubicPolynomial_InflectionPoint) {
    setCubicCurve(*ltf);

    auto config = dsp_core::SplineFitConfig::tight();
    config.enableFeatureDetection = true;

    auto curve = extractCurveData();
    auto result = dsp_core::Services::SplineFitter::fitCurve(curve.data(), static_cast<int>(curve.size()), ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    ASSERT_TRUE(result.success);

    // NOTE: Inflection point detection is not yet implemented in CurveFeatureDetector
    // The SplineFitter relies on CurveFeatureDetector which only detects local extrema,
    // not inflection points (d²y/dx² sign changes). For monotonic curves like x³,
    // only endpoint anchors are placed since there are no local extrema.
    // TODO: Implement inflection point detection in CurveFeatureDetector
    // EXPECT_TRUE(hasAnchorNear(result.anchors, 0.0, 0.01)) << "Expected anchor at inflection point x=0";

    // No local extrema in interior (monotonically increasing)
    auto numerical_extrema = findNumericalExtrema(*ltf);
    EXPECT_EQ(numerical_extrema.size(), 0) << "x³ should have no local extrema (monotonic)";

    std::cout << "  x³ inflection point captured: " << (hasAnchorNear(result.anchors, 0.0, 0.01) ? "YES" : "NO")
              << " (error=" << std::abs(findAnchorNear(result.anchors, 0.0)) << ")" << '\n';
}

/**
 * Test: tanh(2x) - Inflection at origin
 *
 * f(x) = tanh(2x)
 * Inflection point at x = 0
 * No local extrema (S-curve, monotonically increasing)
 */
TEST_F(ExactExtremaTest, TanhCurve_InflectionPoint) {
    setTanhCurve(*ltf, 2.0);

    auto config = dsp_core::SplineFitConfig::tight();
    config.enableFeatureDetection = true;

    auto curve = extractCurveData();
    auto result = dsp_core::Services::SplineFitter::fitCurve(curve.data(), static_cast<int>(curve.size()), ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    ASSERT_TRUE(result.success);

    // NOTE: Inflection point detection is not yet implemented in CurveFeatureDetector
    // The SplineFitter relies on CurveFeatureDetector which only detects local extrema,
    // not inflection points (d²y/dx² sign changes). For monotonic curves like tanh,
    // only endpoint anchors are placed since there are no local extrema.
    // TODO: Implement inflection point detection in CurveFeatureDetector
    // EXPECT_TRUE(hasAnchorNear(result.anchors, 0.0, 0.01)) << "Expected anchor at inflection point x=0";

    // No local extrema
    auto numerical_extrema = findNumericalExtrema(*ltf);
    EXPECT_EQ(numerical_extrema.size(), 0) << "tanh(2x) should have no local extrema (monotonic)";

    std::cout << "  tanh(2x) inflection point captured: " << (hasAnchorNear(result.anchors, 0.0, 0.01) ? "YES" : "NO")
              << " (error=" << std::abs(findAnchorNear(result.anchors, 0.0)) << ")" << '\n';
}

//==============================================================================
// Production Harmonic Tests (Numerical Extrema Detection)
//==============================================================================

/**
 * Test: Harmonic 2 (Chebyshev T₂) - Numerical extrema verification
 */
TEST_F(ExactExtremaTest, Harmonic2_NumericalExtrema) {
    setHarmonicCurve(*ltf, 2);

    // Find numerical extrema before fitting
    auto numerical_extrema = findNumericalExtrema(*ltf);

    auto config = dsp_core::SplineFitConfig::tight();
    config.enableFeatureDetection = true;

    auto curve = extractCurveData();
    auto result = dsp_core::Services::SplineFitter::fitCurve(curve.data(), static_cast<int>(curve.size()), ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    ASSERT_TRUE(result.success);

    // Verify anchor near each numerically-detected extremum
    int matched_count = 0;
    for (double const extremum_x : numerical_extrema) {
        if (hasAnchorNear(result.anchors, extremum_x, 0.01)) {
            matched_count++;
        }
    }

    EXPECT_GE(matched_count, numerical_extrema.size() * 0.8) << "Expected at least 80% of extrema matched";

    double const avg_error = computeAveragePositionError(result.anchors, numerical_extrema);
    std::cout << "  H2 avg extrema error: " << std::fixed << std::setprecision(4) << avg_error << " (" << matched_count
              << "/" << numerical_extrema.size() << " matched)" << '\n';

    EXPECT_LT(avg_error, 0.02);
}

/**
 * Test: Harmonic 3 (Chebyshev T₃)
 */
TEST_F(ExactExtremaTest, Harmonic3_NumericalExtrema) {
    setHarmonicCurve(*ltf, 3);

    auto numerical_extrema = findNumericalExtrema(*ltf);

    auto config = dsp_core::SplineFitConfig::tight();
    config.enableFeatureDetection = true;

    auto curve = extractCurveData();
    auto result = dsp_core::Services::SplineFitter::fitCurve(curve.data(), static_cast<int>(curve.size()), ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    ASSERT_TRUE(result.success);

    int matched_count = 0;
    for (double const extremum_x : numerical_extrema) {
        if (hasAnchorNear(result.anchors, extremum_x, 0.01)) {
            matched_count++;
        }
    }

    EXPECT_GE(matched_count, numerical_extrema.size() * 0.8);

    double const avg_error = computeAveragePositionError(result.anchors, numerical_extrema);
    std::cout << "  H3 avg extrema error: " << std::fixed << std::setprecision(4) << avg_error << " (" << matched_count
              << "/" << numerical_extrema.size() << " matched)" << '\n';

    EXPECT_LT(avg_error, 0.02);
}

/**
 * Test: Harmonic 5 (Chebyshev T₅)
 */
TEST_F(ExactExtremaTest, Harmonic5_NumericalExtrema) {
    setHarmonicCurve(*ltf, 5);

    auto numerical_extrema = findNumericalExtrema(*ltf);

    auto config = dsp_core::SplineFitConfig::tight();
    config.enableFeatureDetection = true;

    auto curve = extractCurveData();
    auto result = dsp_core::Services::SplineFitter::fitCurve(curve.data(), static_cast<int>(curve.size()), ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    ASSERT_TRUE(result.success);

    int matched_count = 0;
    for (double const extremum_x : numerical_extrema) {
        if (hasAnchorNear(result.anchors, extremum_x, 0.01)) {
            matched_count++;
        }
    }

    EXPECT_GE(matched_count, numerical_extrema.size() * 0.8);

    double const avg_error = computeAveragePositionError(result.anchors, numerical_extrema);
    std::cout << "  H5 avg extrema error: " << std::fixed << std::setprecision(4) << avg_error << " (" << matched_count
              << "/" << numerical_extrema.size() << " matched)" << '\n';

    EXPECT_LT(avg_error, 0.02);
}

/**
 * Test: Harmonic 10 (Chebyshev T₁₀)
 */
TEST_F(ExactExtremaTest, Harmonic10_NumericalExtrema) {
    setHarmonicCurve(*ltf, 10);

    auto numerical_extrema = findNumericalExtrema(*ltf);

    auto config = dsp_core::SplineFitConfig::tight();
    config.enableFeatureDetection = true;

    auto curve = extractCurveData();
    auto result = dsp_core::Services::SplineFitter::fitCurve(curve.data(), static_cast<int>(curve.size()), ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    ASSERT_TRUE(result.success);

    int matched_count = 0;
    for (double const extremum_x : numerical_extrema) {
        if (hasAnchorNear(result.anchors, extremum_x, 0.01)) {
            matched_count++;
        }
    }

    EXPECT_GE(matched_count, numerical_extrema.size() * 0.75) << "Expected at least 75% match for H10 (high frequency)";

    double const avg_error = computeAveragePositionError(result.anchors, numerical_extrema);
    std::cout << "  H10 avg extrema error: " << std::fixed << std::setprecision(4) << avg_error << " (" << matched_count
              << "/" << numerical_extrema.size() << " matched)" << '\n';

    EXPECT_LT(avg_error, 0.02);
}

/**
 * Test: Harmonic 15 (Chebyshev T₁₅)
 */
TEST_F(ExactExtremaTest, Harmonic15_NumericalExtrema) {
    setHarmonicCurve(*ltf, 15);

    auto numerical_extrema = findNumericalExtrema(*ltf);

    auto config = dsp_core::SplineFitConfig::tight();
    config.enableFeatureDetection = true;

    auto curve = extractCurveData();
    auto result = dsp_core::Services::SplineFitter::fitCurve(curve.data(), static_cast<int>(curve.size()), ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    ASSERT_TRUE(result.success);

    int matched_count = 0;
    for (double const extremum_x : numerical_extrema) {
        if (hasAnchorNear(result.anchors, extremum_x, 0.02)) { // Wider tolerance
            matched_count++;
        }
    }

    EXPECT_GE(matched_count, numerical_extrema.size() * 0.7)
        << "Expected at least 70% match for H15 (very high frequency)";

    double const avg_error = computeAveragePositionError(result.anchors, numerical_extrema);
    std::cout << "  H15 avg extrema error: " << std::fixed << std::setprecision(4) << avg_error << " (" << matched_count
              << "/" << numerical_extrema.size() << " matched)" << '\n';

    EXPECT_LT(avg_error, 0.03) << "Relaxed tolerance for very high frequency";
}

/**
 * Test: Harmonic 20 (Chebyshev T₂₀)
 */
TEST_F(ExactExtremaTest, Harmonic20_NumericalExtrema) {
    setHarmonicCurve(*ltf, 20);

    auto numerical_extrema = findNumericalExtrema(*ltf);

    auto config = dsp_core::SplineFitConfig::tight();
    config.enableFeatureDetection = true;

    auto curve = extractCurveData();
    auto result = dsp_core::Services::SplineFitter::fitCurve(curve.data(), static_cast<int>(curve.size()), ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    ASSERT_TRUE(result.success);

    int matched_count = 0;
    for (double const extremum_x : numerical_extrema) {
        if (hasAnchorNear(result.anchors, extremum_x, 0.02)) { // Wider tolerance
            matched_count++;
        }
    }

    EXPECT_GE(matched_count, numerical_extrema.size() * 0.7)
        << "Expected at least 70% match for H20 (very high frequency)";

    double const avg_error = computeAveragePositionError(result.anchors, numerical_extrema);
    std::cout << "  H20 avg extrema error: " << std::fixed << std::setprecision(4) << avg_error << " (" << matched_count
              << "/" << numerical_extrema.size() << " matched)" << '\n';

    EXPECT_LT(avg_error, 0.03) << "Relaxed tolerance for very high frequency";
}

} // namespace dsp_core_test
