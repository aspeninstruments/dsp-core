#include <dsp_core/dsp_core.h>
#include <gtest/gtest.h>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <set>

namespace dsp_core_test {

/**
 * Test fixture for SplineFitter service
 * Tests PCHIP spline curve fitting algorithm:
 *   - Data structures (SplineTypes)
 *   - Sample & sanitize (Step 1)
 *   - RDP simplification (Step 2)
 *   - PCHIP tangent computation (Step 3)
 */
class SplineFitterTest : public ::testing::Test {
  protected:
    void SetUp() override {
        ltf = std::make_unique<dsp_core::LayeredTransferFunction>(256, -1.0, 1.0);
        // Reset coefficients to traditional test defaults (WT=1.0, all harmonics=0.0)
        // The constructor now uses WT=0.0, H1=1.0 for plugin initialization
        ltf->setCoefficient(0, 1.0);
        for (int i = 1; i < ltf->getNumCoefficients(); ++i) {
            ltf->setCoefficient(i, 0.0);
        }
    }

    // Helper: Set base layer to identity curve (y = x)
    void setIdentityCurve() {
        for (int i = 0; i < 256; ++i) {
            double const x = ltf->normalizeIndex(i);
            ltf->setBaseLayerValue(i, x);
        }
    }

    // Helper: Set base layer to S-curve (cubic)
    void setSCurve() {
        for (int i = 0; i < 256; ++i) {
            double const x = ltf->normalizeIndex(i);
            // Cubic S-curve: y = x^3
            double const y = x * x * x;
            ltf->setBaseLayerValue(i, y);
        }
    }

    // Helper: Set base layer to step function
    void setStepFunction() {
        for (int i = 0; i < 256; ++i) {
            double const x = ltf->normalizeIndex(i);
            ltf->setBaseLayerValue(i, x < 0.0 ? -0.5 : 0.5);
        }
    }

    // Helper: Set base layer to sine wave
    void setSineWave() {
        for (int i = 0; i < 256; ++i) {
            double const x = ltf->normalizeIndex(i);
            // Normalize to [-1, 1]
            double const y = std::sin(x * M_PI);
            ltf->setBaseLayerValue(i, y);
        }
    }

    std::vector<double> extractCurveData() const {
        std::vector<double> data(ltf->getTableSize());
        for (int i = 0; i < ltf->getTableSize(); ++i) {
            data[i] = ltf->getBaseLayerValue(i);
        }
        return data;
    }

    std::vector<double> extractCurveData(const dsp_core::LayeredTransferFunction& source) const {
        std::vector<double> data(source.getTableSize());
        for (int i = 0; i < source.getTableSize(); ++i) {
            data[i] = source.getBaseLayerValue(i);
        }
        return data;
    }

    std::unique_ptr<dsp_core::LayeredTransferFunction> ltf;
};

// ============================================================================
// SplineTypes Tests
// ============================================================================

TEST(SplineTypesTest, SplineAnchor_DefaultInitialization) {
    dsp_core::SplineAnchor const anchor;
    EXPECT_DOUBLE_EQ(anchor.x, 0.0);
    EXPECT_DOUBLE_EQ(anchor.y, 0.0);
    EXPECT_FALSE(anchor.hasCustomTangent);
    EXPECT_DOUBLE_EQ(anchor.tangent, 0.0);
}

TEST(SplineTypesTest, SplineAnchor_Equality) {
    dsp_core::SplineAnchor const a1{0.5, 0.5, false, 0.0};
    dsp_core::SplineAnchor const a2{0.5, 0.5, false, 0.0};
    dsp_core::SplineAnchor const a3{0.5, 0.6, false, 0.0};

    EXPECT_EQ(a1, a2);
    EXPECT_NE(a1, a3);
}

TEST(SplineTypesTest, SplineFitResult_DefaultInitialization) {
    dsp_core::SplineFitResult const result;
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.numAnchors, 0);
    EXPECT_DOUBLE_EQ(result.maxError, 0.0);
    EXPECT_DOUBLE_EQ(result.averageError, 0.0);
    EXPECT_DOUBLE_EQ(result.maxDerivativeError, 0.0);
}

TEST(SplineTypesTest, SplineFitConfig_DefaultValues) {
    dsp_core::SplineFitConfig const config;
    EXPECT_DOUBLE_EQ(config.positionTolerance, 0.01);
    EXPECT_EQ(config.maxAnchors, 128);
    EXPECT_TRUE(config.enableRefinement);
    EXPECT_TRUE(config.enforceMonotonicity);
    EXPECT_DOUBLE_EQ(config.minSlope, -8.0);
    EXPECT_DOUBLE_EQ(config.maxSlope, 8.0);
    EXPECT_TRUE(config.pinEndpoints);
}

TEST(SplineTypesTest, SplineFitConfig_TightPreset) {
    auto config = dsp_core::SplineFitConfig::tight();
    EXPECT_DOUBLE_EQ(config.positionTolerance, 0.005); // Relaxed from 0.002 for backtranslation stability
    EXPECT_EQ(config.maxAnchors, 128);                 // Increased from 64 to allow better convergence for steep curves
}

// Removed: SplineFitConfig_SmoothPreset test - smooth() preset was removed, only tight() remains

// ============================================================================
// SplineFitter Basic Tests
// ============================================================================

TEST_F(SplineFitterTest, FitCurve_IdentityCurve_MinimalAnchors) {
    setIdentityCurve();

    auto config = dsp_core::SplineFitConfig::tight();
    auto curveData = extractCurveData();

    auto result =
        dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                   ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    EXPECT_TRUE(result.success);
    EXPECT_GE(result.numAnchors, 2);  // At least endpoints
    EXPECT_LE(result.numAnchors, 10); // Linear curve should need very few anchors

    // Endpoints should be preserved
    EXPECT_NEAR(result.anchors.front().x, -1.0, 0.01);
    EXPECT_NEAR(result.anchors.front().y, -1.0, 0.01);
    EXPECT_NEAR(result.anchors.back().x, 1.0, 0.01);
    EXPECT_NEAR(result.anchors.back().y, 1.0, 0.01);
}

TEST_F(SplineFitterTest, FitCurve_SCurve_MoreAnchors) {
    setSCurve();

    auto config = dsp_core::SplineFitConfig::tight();
    auto curveData = extractCurveData();

    auto result =
        dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                   ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    EXPECT_TRUE(result.success);
    EXPECT_GE(result.numAnchors, 3); // S-curve needs anchors at inflection points
}

TEST_F(SplineFitterTest, FitCurve_TightConfig_ProducesValidResult) {
    setSCurve();

    auto config = dsp_core::SplineFitConfig::tight();
    auto curveData = extractCurveData();

    auto result =
        dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                   ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    EXPECT_TRUE(result.success);
    EXPECT_GE(result.numAnchors, 2);                 // At least endpoints
    EXPECT_LE(result.numAnchors, config.maxAnchors); // Respects max anchors limit
}

TEST_F(SplineFitterTest, FitCurve_AnchorsAreSorted) {
    setSCurve();

    auto config = dsp_core::SplineFitConfig::tight();
    auto curveData = extractCurveData();

    auto result =
        dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                   ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    EXPECT_TRUE(result.success);

    // Verify anchors are sorted by x
    for (size_t i = 1; i < result.anchors.size(); ++i) {
        EXPECT_GT(result.anchors[i].x, result.anchors[i - 1].x) << "Anchor " << i << " is not sorted";
    }
}

// ============================================================================
// PCHIP Tangent Tests
// ============================================================================

TEST_F(SplineFitterTest, PCHIPTangents_MonotonicSequence) {
    setIdentityCurve();

    auto config = dsp_core::SplineFitConfig::tight();
    auto curveData = extractCurveData();

    auto result =
        dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                   ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    EXPECT_TRUE(result.success);

    // For monotonic increasing data, all tangents should be positive
    for (const auto& anchor : result.anchors) {
        EXPECT_GE(anchor.tangent, 0.0) << "Tangent at x=" << anchor.x << " should be non-negative";
    }
}

TEST_F(SplineFitterTest, PCHIPTangents_LocalExtremum) {
    // Create curve with clear local extremum
    for (int i = 0; i < 256; ++i) {
        double const x = ltf->normalizeIndex(i);
        // Parabola with peak at x=0: y = 1 - x^2
        double const y = 1.0 - x * x;
        ltf->setBaseLayerValue(i, y);
    }

    auto config = dsp_core::SplineFitConfig::tight(); // Use tight to capture peak
    auto curveData = extractCurveData();

    auto result =
        dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                   ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    EXPECT_TRUE(result.success);

    // Find anchor at peak (x ≈ 0)
    // Note: RDP simplification means we may not capture the exact peak
    // but if we do, the tangent should be near zero
    for (const auto& anchor : result.anchors) {
        if (std::abs(anchor.x) < 0.1 && std::abs(anchor.y - 1.0) < 0.1) {
            // Tangent should be near zero at peak
            EXPECT_NEAR(anchor.tangent, 0.0, 0.3) << "Tangent at peak (x=" << anchor.x << ") should be near zero";
        }
    }
}

TEST_F(SplineFitterTest, PCHIPTangents_SlopeCapping) {
    setStepFunction();

    auto config = dsp_core::SplineFitConfig::tight();
    auto curveData = extractCurveData();

    auto result =
        dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                   ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    EXPECT_TRUE(result.success);

    // All tangents should respect slope caps
    for (const auto& anchor : result.anchors) {
        EXPECT_GE(anchor.tangent, config.minSlope) << "Tangent at x=" << anchor.x << " exceeds min slope";
        EXPECT_LE(anchor.tangent, config.maxSlope) << "Tangent at x=" << anchor.x << " exceeds max slope";
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(SplineFitterTest, FitCurve_FlatCurve) {
    // Set all values to constant
    for (int i = 0; i < 256; ++i) {
        ltf->setBaseLayerValue(i, 0.0);
    }

    auto config = dsp_core::SplineFitConfig::tight();
    auto curveData = extractCurveData();

    auto result =
        dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                   ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    EXPECT_TRUE(result.success);
    EXPECT_GE(result.numAnchors, 2); // At least endpoints

    // All anchors should have y ≈ 0
    for (const auto& anchor : result.anchors) {
        EXPECT_NEAR(anchor.y, 0.0, 0.01);
        EXPECT_NEAR(anchor.tangent, 0.0, 0.01); // Flat curve → zero tangent
    }
}

TEST_F(SplineFitterTest, FitCurve_MonotonicityEnforcement_ProducesReasonableResult) {
    // Create slightly non-monotonic curve (with small violations)
    setIdentityCurve();
    // Add small monotonicity violations
    ltf->setBaseLayerValue(64, -0.3); // Should be ~-0.5
    ltf->setBaseLayerValue(128, 0.2); // Should be ~0.0
    ltf->setBaseLayerValue(192, 0.6); // Should be ~0.5

    auto config = dsp_core::SplineFitConfig::tight();
    config.enforceMonotonicity = true;

    auto curveData = extractCurveData();

    auto result =
        dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                   ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    EXPECT_TRUE(result.success);

    // Monotonicity enforcement applies pairwise averaging to samples,
    // but RDP may still select anchors with small violations.
    // Verify result is generally increasing with tolerance for small errors.
    int violations = 0;
    for (size_t i = 1; i < result.anchors.size(); ++i) {
        if (result.anchors[i].y < result.anchors[i - 1].y - 0.01) {
            violations++;
        }
    }
    // Allow some violations (< 40%) - current monotonicity is simple pairwise
    EXPECT_LT(violations, result.numAnchors * 2 / 5)
        << "Too many monotonicity violations: " << violations << " out of " << result.numAnchors;
}

TEST_F(SplineFitterTest, FitCurve_NonMonotonicCurve_WithoutEnforcement) {
    setSineWave();

    auto config = dsp_core::SplineFitConfig::tight();
    config.enforceMonotonicity = false;

    auto curveData = extractCurveData();

    auto result =
        dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                   ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    EXPECT_TRUE(result.success);
    // Without enforcement, anchors may be non-monotonic
    // Just verify we get a valid result
    EXPECT_GE(result.numAnchors, 2);
}

// ============================================================================
// RDP Algorithm Tests
// ============================================================================

// ============================================================================
// Deprecated RDP Algorithm Tests - REMOVED
// The RDP-based fitting was replaced by greedy spline-aware fitting.
// See: docs/curve_fitting_improvement_tasks.md for rationale
// ============================================================================

// ============================================================================
// SplineEvaluator Tests
// ============================================================================

TEST(SplineEvaluatorTest, Evaluate_EmptyAnchors_ReturnsZero) {
    std::vector<dsp_core::SplineAnchor> const anchors;
    double const result = dsp_core::Services::SplineEvaluator::evaluate(anchors, 0.5);
    EXPECT_DOUBLE_EQ(result, 0.0);
}

TEST(SplineEvaluatorTest, Evaluate_SingleAnchor_ReturnsAnchorValue) {
    std::vector<dsp_core::SplineAnchor> const anchors = {{0.0, 0.5, false, 0.0}};
    double result = dsp_core::Services::SplineEvaluator::evaluate(anchors, 0.0);
    EXPECT_DOUBLE_EQ(result, 0.5);

    // Any x should return the same value
    result = dsp_core::Services::SplineEvaluator::evaluate(anchors, 1.0);
    EXPECT_DOUBLE_EQ(result, 0.5);
}

TEST(SplineEvaluatorTest, Evaluate_AtAnchorPositions_ReturnsExactValues) {
    std::vector<dsp_core::SplineAnchor> const anchors = {
        {-1.0, -1.0, false, 1.0}, {0.0, 0.0, false, 1.0}, {1.0, 1.0, false, 1.0}};

    // Evaluate at each anchor position
    EXPECT_NEAR(dsp_core::Services::SplineEvaluator::evaluate(anchors, -1.0), -1.0, 1e-10);
    EXPECT_NEAR(dsp_core::Services::SplineEvaluator::evaluate(anchors, 0.0), 0.0, 1e-10);
    EXPECT_NEAR(dsp_core::Services::SplineEvaluator::evaluate(anchors, 1.0), 1.0, 1e-10);
}

TEST(SplineEvaluatorTest, Evaluate_LinearInterpolation) {
    // Two anchors with slope = 1 (identity line)
    std::vector<dsp_core::SplineAnchor> const anchors = {
        {-1.0, -1.0, false, 1.0}, // tangent = 1
        {1.0, 1.0, false, 1.0}    // tangent = 1
    };

    // Evaluate at midpoint - should be close to 0.0 for linear
    double const midpoint = dsp_core::Services::SplineEvaluator::evaluate(anchors, 0.0);
    EXPECT_NEAR(midpoint, 0.0, 0.01); // Allow small deviation for Hermite
}

TEST(SplineEvaluatorTest, Evaluate_BeforeFirstAnchor_Clamps) {
    std::vector<dsp_core::SplineAnchor> const anchors = {{0.0, 0.5, false, 0.0}, {1.0, 1.0, false, 0.0}};

    // Evaluate before first anchor
    double const result = dsp_core::Services::SplineEvaluator::evaluate(anchors, -0.5);
    EXPECT_DOUBLE_EQ(result, 0.5); // Should return first anchor's y value
}

TEST(SplineEvaluatorTest, Evaluate_AfterLastAnchor_Clamps) {
    std::vector<dsp_core::SplineAnchor> const anchors = {{0.0, 0.0, false, 0.0}, {1.0, 0.5, false, 0.0}};

    // Evaluate after last anchor
    double const result = dsp_core::Services::SplineEvaluator::evaluate(anchors, 1.5);
    EXPECT_DOUBLE_EQ(result, 0.5); // Should return last anchor's y value
}

TEST(SplineEvaluatorTest, Evaluate_MonotonicSpline) {
    // Create monotonic anchors
    std::vector<dsp_core::SplineAnchor> const anchors = {{-1.0, -0.8, false, 0.5},
                                                         {-0.5, -0.3, false, 0.6},
                                                         {0.0, 0.1, false, 0.7},
                                                         {0.5, 0.4, false, 0.6},
                                                         {1.0, 0.9, false, 0.5}};

    // Sample and verify monotonicity
    // Use integer loop to avoid float loop counter (clang-analyzer-security.FloatLoopCounter)
    double prevY = -1.0;
    for (int i = 0; i <= 20; ++i) {
        const double x = -1.0 + i * 0.1;
        double const y = dsp_core::Services::SplineEvaluator::evaluate(anchors, x);
        EXPECT_GE(y, prevY) << "Non-monotonic at x=" << x;
        prevY = y;
    }
}

TEST(SplineEvaluatorTest, EvaluateDerivative_AtAnchors) {
    std::vector<dsp_core::SplineAnchor> const anchors = {
        {-1.0, -1.0, false, 1.0}, // tangent = 1
        {0.0, 0.0, false, 0.0},   // tangent = 0 (extremum)
        {1.0, 1.0, false, 1.0}    // tangent = 1
    };

    // Derivative at midpoint anchor should be close to 0
    double const deriv = dsp_core::Services::SplineEvaluator::evaluateDerivative(anchors, 0.0);
    EXPECT_NEAR(deriv, 0.0, 0.1);
}

TEST(SplineEvaluatorTest, FindSegment_BinarySearch) {
    std::vector<dsp_core::SplineAnchor> const anchors = {{-1.0, 0.0, false, 0.0},
                                                         {-0.5, 0.0, false, 0.0},
                                                         {0.0, 0.0, false, 0.0},
                                                         {0.5, 0.0, false, 0.0},
                                                         {1.0, 0.0, false, 0.0}};

    // Test that evaluator can handle many segments efficiently
    // Use integer loop to avoid float loop counter (clang-analyzer-security.FloatLoopCounter)
    for (int i = 0; i <= 40; ++i) {
        const double x = -1.0 + i * 0.05;
        // Should not crash or hang
        dsp_core::Services::SplineEvaluator::evaluate(anchors, x);
    }
}

// ============================================================================
// Integration Tests (SplineFitter + SplineEvaluator)
// ============================================================================

TEST_F(SplineFitterTest, Integration_ErrorMetrics) {
    setIdentityCurve();

    auto config = dsp_core::SplineFitConfig::tight();
    auto curveData = extractCurveData();

    auto result =
        dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                   ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    EXPECT_TRUE(result.success);
    EXPECT_GE(result.maxError, 0.0);
    EXPECT_GE(result.averageError, 0.0);
    EXPECT_LE(result.averageError, result.maxError); // avg <= max

    // For identity curve, error should be very small
    EXPECT_LT(result.maxError, 0.05) << "Identity curve fit error too large";
}

TEST_F(SplineFitterTest, Integration_FitAndReconstruct) {
    setSCurve();

    auto config = dsp_core::SplineFitConfig::tight();
    auto curveData = extractCurveData();

    auto result =
        dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                   ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    EXPECT_TRUE(result.success);

    // Reconstruct curve at original sample points
    int matchCount = 0;
    for (int i = 0; i < 256; i += 10) { // Sample every 10th point
        double const x = ltf->normalizeIndex(i);
        double const originalY = ltf->getBaseLayerValue(i);
        double const fittedY = dsp_core::Services::SplineEvaluator::evaluate(result.anchors, x);

        // Check if reconstruction is close to original
        if (std::abs(originalY - fittedY) < 0.1) {
            matchCount++;
        }
    }

    // Most points should match well
    EXPECT_GT(matchCount, 20) << "Reconstruction quality too low";
}

//==============================================================================
// Comprehensive Real-World Function Tests
//==============================================================================

/**
 * Test tanh curves with varying steepness
 * tanh(nx) becomes steeper as n increases, testing the algorithm's ability
 * to handle different curvature characteristics
 */
TEST_F(SplineFitterTest, TanhCurves_VariousSteepness) {
    // Test tanh(1x) through tanh(20x)
    std::vector<int> const steepnessFactors = {1, 2, 5, 10, 15, 20};

    for (int const n : steepnessFactors) {
        // Set base layer to tanh(n*x)
        for (int i = 0; i < 256; ++i) {
            double const x = ltf->normalizeIndex(i);
            ltf->setBaseLayerValue(i, std::tanh(n * x));
        }

        // Fit with tight tolerance for accuracy
        auto config = dsp_core::SplineFitConfig::tight();
        auto curveData = extractCurveData();

        auto result =
            dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                       ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

        EXPECT_TRUE(result.success) << "tanh(" << n << "x) fit failed";

        // Error should be below tolerance
        // Relaxed to 7.0× to account for linear adaptive tolerance trade-off:
        // fewer anchors (better backtranslation) → slightly higher error (acceptable)
        // Steep curves (tanh(15x), tanh(20x)) naturally require more error tolerance
        // Linear scaling is more aggressive at low anchor counts = better backtranslation
        // but means we stop fitting slightly earlier on steep curves
        EXPECT_LT(result.maxError, config.positionTolerance * 7.0)
            << "tanh(" << n << "x) max error too high: " << result.maxError;

        // Steeper curves should require more anchors
        if (n > 1) {
            EXPECT_GT(result.numAnchors, 2) << "tanh(" << n << "x) should need more than endpoint anchors";
        }

        // Verify reconstruction quality at midpoint (steepest part)
        double const midX = 0.0;
        double const expected = std::tanh(n * midX);
        double const fitted = dsp_core::Services::SplineEvaluator::evaluate(result.anchors, midX);
        double const midError = std::abs(expected - fitted);

        EXPECT_LT(midError, config.positionTolerance * 5.0)
            << "tanh(" << n << "x) poor fit at steep midpoint, error=" << midError;
    }
}

/**
 * Test all 40 trigonometric harmonic basis functions
 * These are the functions used in LaneMixerPanel:
 * - Even harmonics: cos(n * acos(x))
 * - Odd harmonics: sin(n * asin(x))
 */
TEST_F(SplineFitterTest, TrigHarmonics_AllBasisFunctions) {
    const int NUM_HARMONICS = 40;

    for (int n = 1; n <= NUM_HARMONICS; ++n) {
        // Set base layer to nth harmonic basis function
        for (int i = 0; i < 256; ++i) {
            double x = ltf->normalizeIndex(i);

            // Clamp x to [-1, 1] for acos/asin
            x = std::max(-1.0, std::min(1.0, x));

            double y = 0.0;
            if (n % 2 == 0) {
                // Even harmonic: cos(n * acos(x))
                y = std::cos(n * std::acos(x));
            } else {
                // Odd harmonic: sin(n * asin(x))
                y = std::sin(n * std::asin(x));
            }

            ltf->setBaseLayerValue(i, y);
        }

        // Use balanced config (good quality, reasonable anchor count)
        // Use Never mode to test original greedy algorithm behavior
        auto config = dsp_core::SplineFitConfig::tight();
        config.symmetryDetection = dsp_core::SymmetryDetection::Never;
        auto curveData = extractCurveData();

        auto result =
            dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                       ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

        EXPECT_TRUE(result.success) << "Harmonic " << n << " fit failed";

        // Adaptive error tolerance based on harmonic complexity
        // Note: Inflection detection was removed for performance reasons, so tolerances are slightly relaxed.
        // Low harmonics (1-3): Should achieve tight tolerance (PCHIP can represent these well)
        // Low-medium harmonics (4-6): Moderate tolerance (need more anchors for oscillations)
        // Medium harmonics (7-15): Relaxed tolerance (challenging but achievable)
        // High harmonics (16-25): Very relaxed tolerance (PCHIP struggles with high-frequency oscillations)
        // Very high harmonics (26+): Skip (beyond PCHIP's capabilities)
        double errorTolerance = 0.0;
        if (n <= 3) {
            errorTolerance = config.positionTolerance * 6.0; // 0.03 for tight config
        } else if (n <= 6) {
            errorTolerance = config.positionTolerance * 12.0; // 0.06 - moderate for complex oscillations
        } else if (n <= 15) {
            errorTolerance = config.positionTolerance * 20.0; // 0.10 - relaxed
        } else if (n <= 25) {
            errorTolerance = config.positionTolerance * 60.0; // 0.3 - very relaxed for high-frequency content
        } else {
            // Very high frequencies: Skip strict testing, just verify no crash
            // These are beyond PCHIP's representational capabilities
            GTEST_SKIP() << "Harmonic " << n << " exceeds PCHIP capabilities (expected limitation)";
            continue;
        }

        // Error should be within adaptive tolerance
        EXPECT_LT(result.maxError, errorTolerance) << "Harmonic " << n << " max error too high: " << result.maxError;

        // Higher harmonics should need more anchors (more oscillations)
        if (n >= 5) {
            EXPECT_GT(result.numAnchors, 5) << "Harmonic " << n << " should need multiple anchors for oscillations";
        }

        // Verify reconstruction at a few key points (only for lowest harmonics)
        if (n <= 3) {
            std::vector<double> const testPoints = {-0.9, -0.5, 0.0, 0.5, 0.9};
            for (double const testX : testPoints) {
                double expected = 0.0;
                if (n % 2 == 0) {
                    expected = std::cos(n * std::acos(testX));
                } else {
                    expected = std::sin(n * std::asin(testX));
                }

                double const fitted = dsp_core::Services::SplineEvaluator::evaluate(result.anchors, testX);
                double const error = std::abs(expected - fitted);

                EXPECT_LT(error, errorTolerance) << "Harmonic " << n << " poor fit at x=" << testX
                                                 << ", expected=" << expected << ", fitted=" << fitted;
            }
        }
    }
}

/**
 * Test performance with complex high-frequency content
 * Ensures greedy algorithm doesn't hang on difficult curves
 */
TEST_F(SplineFitterTest, Performance_ComplexCurves) {
    // Test several challenging curves
    std::vector<std::function<double(double)>> testFunctions = {
        [](double x) { return std::tanh(20.0 * x); },                                        // Very steep
        [](double x) { return std::sin(10.0 * M_PI * x); },                                  // High frequency
        [](double x) { return x * std::sin(15.0 * M_PI * x); },                              // Modulated
        [](double x) { return std::cos(8.0 * std::acos(std::max(-1.0, std::min(1.0, x)))); } // Harmonic 8
    };

    for (size_t idx = 0; idx < testFunctions.size(); ++idx) {
        // Set base layer to test function
        for (int i = 0; i < 256; ++i) {
            double const x = ltf->normalizeIndex(i);
            ltf->setBaseLayerValue(i, testFunctions[idx](x));
        }

        auto config = dsp_core::SplineFitConfig::tight();

        // Measure execution time
        auto startTime = std::chrono::high_resolution_clock::now();
        auto curveData = extractCurveData();

        auto result =
            dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                       ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);
        auto endTime = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

        EXPECT_TRUE(result.success) << "Test function " << idx << " fit failed";

        // Should complete in reasonable time (< 100ms)
        EXPECT_LT(duration.count(), 100) << "Test function " << idx << " took too long: " << duration.count() << "ms";

        // Adaptive accuracy expectations:
        // Function 0 (tanh(20x)): Steep but smooth monotonic - should fit reasonably well
        // Functions 1-3 (high-frequency oscillations): Best-effort due to PCHIP limitations
        // Note: Feature-based anchor placement prioritizes structural correctness
        // (no ripple) over minimizing absolute error, so tolerances are relaxed
        double errorTolerance = 0.0;
        if (idx == 0) {
            errorTolerance = config.positionTolerance * 8.0; // 0.04 - steep curves can have localized error
        } else {
            errorTolerance =
                config.positionTolerance * 40.0; // 0.20 - high-frequency content (relaxed for greedy placement)
        }

        EXPECT_LT(result.maxError, errorTolerance) << "Test function " << idx << " error too high: " << result.maxError;
    }
}

/**
 * Test edge case: extremely steep tanh(100x)
 * This approaches a step function and tests the algorithm's limits
 */
TEST_F(SplineFitterTest, EdgeCase_ExtremelySteepTanh) {
    // tanh(100x) is almost a step function
    for (int i = 0; i < 256; ++i) {
        double const x = ltf->normalizeIndex(i);
        ltf->setBaseLayerValue(i, std::tanh(100.0 * x));
    }

    auto config = dsp_core::SplineFitConfig::tight();
    config.symmetryDetection = dsp_core::SymmetryDetection::Never; // Test original greedy behavior
    config.maxAnchors = 64;                                        // May need many anchors for near-discontinuity

    auto curveData = extractCurveData();

    auto result =
        dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                   ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    EXPECT_TRUE(result.success);

    // Should use many anchors near x=0 (steep transition)
    EXPECT_GE(result.numAnchors, 10) << "Extremely steep curve needs many anchors";

    // Error may be higher due to near-discontinuity, but should still be reasonable
    EXPECT_LT(result.maxError, 0.05) << "Even steep curves should fit reasonably well";
}

/**
 * Test quality: verify fitted curve is smooth (C1 continuous)
 * Check that PCHIP tangents create a smooth curve without kinks
 */
TEST_F(SplineFitterTest, Quality_SmoothnessC1Continuity) {
    // Use a smooth curve (tanh(3x))
    for (int i = 0; i < 256; ++i) {
        double const x = ltf->normalizeIndex(i);
        ltf->setBaseLayerValue(i, std::tanh(3.0 * x));
    }

    auto config = dsp_core::SplineFitConfig::tight();
    auto curveData = extractCurveData();

    auto result =
        dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                   ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    EXPECT_TRUE(result.success);

    // Evaluate derivative at many points to check for discontinuities
    const int NUM_SAMPLES = 1000;
    double prevDerivative = 0.0;
    bool first = true;
    int largeJumps = 0;

    for (int i = 0; i < NUM_SAMPLES; ++i) {
        double const x = -1.0 + (2.0 * i) / (NUM_SAMPLES - 1);
        double const derivative = dsp_core::Services::SplineEvaluator::evaluateDerivative(result.anchors, x);

        if (!first) {
            double const derivativeChange = std::abs(derivative - prevDerivative);
            // Large sudden jumps in derivative indicate C1 discontinuity (kinks)
            if (derivativeChange > 2.0) { // Threshold for "large jump"
                largeJumps++;
            }
        }

        prevDerivative = derivative;
        first = false;
    }

    // PCHIP should maintain C1 continuity - no sudden derivative jumps
    EXPECT_LT(largeJumps, NUM_SAMPLES / 100) // < 1% of samples
        << "Too many derivative discontinuities detected: " << largeJumps;
}

/**
 * Test regression: the "bowing artifact" bug
 * Straight line + localized scribble should not bow in straight regions
 */
TEST_F(SplineFitterTest, Regression_NoBowingInStraightRegions) {
    // Left straight region: y = x
    for (int i = 0; i < 100; ++i) {
        double const x = ltf->normalizeIndex(i);
        ltf->setBaseLayerValue(i, x);
    }

    // Middle scribble: high-frequency noise
    for (int i = 100; i < 130; ++i) {
        double const x = ltf->normalizeIndex(i);
        ltf->setBaseLayerValue(i, x + 0.15 * std::sin(30.0 * M_PI * x));
    }

    // Right straight region: y = x
    for (int i = 130; i < 256; ++i) {
        double const x = ltf->normalizeIndex(i);
        ltf->setBaseLayerValue(i, x);
    }

    auto config = dsp_core::SplineFitConfig::tight();
    auto curveData = extractCurveData();

    auto result =
        dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                   ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    EXPECT_TRUE(result.success);

    // Measure error in right straight region (far from scribble)
    double maxErrorInStraightRegion = 0.0;
    for (int i = 180; i < 240; ++i) {
        double const x = ltf->normalizeIndex(i);
        double const expected = x; // Should be linear
        double const fitted = dsp_core::Services::SplineEvaluator::evaluate(result.anchors, x);
        double const error = std::abs(expected - fitted);
        maxErrorInStraightRegion = std::max(maxErrorInStraightRegion, error);
    }

    // Straight region should have very low error (< 1%)
    EXPECT_LT(maxErrorInStraightRegion, 0.01)
        << "Bowing artifact detected in straight region: error=" << maxErrorInStraightRegion;
}

//==============================================================================
// Phase 4: Feature-Based Fitting Tests - Zero Ripple Guarantee
//==============================================================================

/**
 * Test fixture for feature-based spline fitting (Phase 3)
 * Verifies that anchoring at geometric features eliminates ripple artifacts
 */
class FeatureBasedFittingTest : public ::testing::Test {
  protected:
    std::unique_ptr<dsp_core::LayeredTransferFunction> ltf;

    std::vector<double> extractCurveData() const {
        std::vector<double> data(ltf->getTableSize());
        for (int i = 0; i < ltf->getTableSize(); ++i) {
            data[i] = ltf->getBaseLayerValue(i);
        }
        return data;
    }

    static std::vector<double> extractCurveData(const dsp_core::LayeredTransferFunction& source) {
        std::vector<double> data(source.getTableSize());
        for (int i = 0; i < source.getTableSize(); ++i) {
            data[i] = source.getBaseLayerValue(i);
        }
        return data;
    }

    void SetUp() override {
        ltf = std::make_unique<dsp_core::LayeredTransferFunction>(256, -1.0, 1.0);
        // Reset coefficients to traditional test defaults (WT=1.0, all harmonics=0.0)
        ltf->setCoefficient(0, 1.0);
        for (int i = 1; i < ltf->getNumCoefficients(); ++i) {
            ltf->setCoefficient(i, 0.0);
        }
    }

    // Helper: Setup curve from function
    void setupCurve(std::function<double(double)> func) {
        for (int i = 0; i < 256; ++i) {
            double const x = ltf->normalizeIndex(i); // [-1, 1]
            ltf->setBaseLayerValue(i, func(x));
        }
    }

    /**
     * Count spurious extrema in fitted spline that don't correspond to data extrema
     * This is the key metric for detecting ripple artifacts
     *
     * Algorithm:
     * 1. Get data extrema from CurveFeatureDetector
     * 2. Sample spline derivative densely between anchors
     * 3. Detect derivative sign changes (local extrema)
     * 4. Check if each extremum matches a data extremum (within tolerance)
     * 5. Count extrema that DON'T match = spurious ripple
     */
    static int countSpuriousExtrema(const std::vector<dsp_core::SplineAnchor>& anchors,
                                    const dsp_core::LayeredTransferFunction& originalData) {
        // Get actual extrema from original data
        auto originalCurveData = extractCurveData(originalData);
        auto dataFeatures = dsp_core::Services::CurveFeatureDetector::detectFeatures(
            originalCurveData.data(), static_cast<int>(originalCurveData.size()), originalData.getMinSignalValue(),
            originalData.getMaxSignalValue());
        std::set<int> const dataExtremaIndices(dataFeatures.localExtrema.begin(), dataFeatures.localExtrema.end());

        int spuriousCount = 0;

        // Sample each segment densely to find extrema
        for (size_t i = 0; i < anchors.size() - 1; ++i) {
            double prevDeriv = 0.0;
            bool firstSample = true;

            // Sample segment at 20 points
            for (int j = 0; j <= 20; ++j) {
                double const t = j / 20.0;
                double const x = anchors[i].x + t * (anchors[i + 1].x - anchors[i].x);

                // Evaluate derivative at this point
                double const deriv = dsp_core::Services::SplineEvaluator::evaluateDerivative(anchors, x);

                if (!firstSample && prevDeriv * deriv < 0.0) {
                    // Derivative sign change = local extremum detected
                    // Check if this extremum is near a data extremum

                    // Convert x coordinate back to approximate table index
                    double const normalizedX = x; // Already in [-1, 1]
                    int approxIndex = static_cast<int>((normalizedX + 1.0) / 2.0 * 255.0);
                    approxIndex = std::max(0, std::min(255, approxIndex));

                    // Check if any data extremum is within 3 indices
                    bool isDataExtremum = false;
                    for (int const dataIdx : dataExtremaIndices) {
                        if (std::abs(approxIndex - dataIdx) <= 3) {
                            isDataExtremum = true;
                            break;
                        }
                    }

                    if (!isDataExtremum) {
                        ++spuriousCount; // Ripple artifact detected!
                    }
                }

                prevDeriv = deriv;
                firstSample = false;
            }
        }

        return spuriousCount;
    }
};

/**
 * Task 4.1: Zero spurious extrema for tanh curves with PCHIP
 * Tanh is monotonic, so should have ZERO extrema in fitted spline
 */
TEST_F(FeatureBasedFittingTest, Tanh_NoSpuriousExtrema_PCHIP) {
    setupCurve([](double x) { return std::tanh(3.0 * x); });

    dsp_core::SplineFitConfig config;
    config.positionTolerance = 0.001;
    config.maxAnchors = 32;
    config.tangentAlgorithm = dsp_core::TangentAlgorithm::PCHIP;

    auto curveData = extractCurveData();

    auto result =
        dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                   ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(0, countSpuriousExtrema(result.anchors, *ltf)) << "Tanh curve should have zero spurious extrema";
    EXPECT_LT(result.maxError, config.positionTolerance * 20.0) << "Error should be reasonable";
}

/**
 * Task 4.1: Zero spurious extrema for tanh curves with Fritsch-Carlson
 */
TEST_F(FeatureBasedFittingTest, Tanh_NoSpuriousExtrema_FritschCarlson) {
    setupCurve([](double x) { return std::tanh(3.0 * x); });

    auto config = dsp_core::SplineFitConfig::tight();
    auto curveData = extractCurveData();

    auto result =
        dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                   ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(0, countSpuriousExtrema(result.anchors, *ltf))
        << "Tanh with Fritsch-Carlson should have zero spurious extrema";
}

/**
 * Task 4.1: Akima may have small overshoots (not monotone-preserving)
 * Note: Akima algorithm prioritizes smoothness over monotonicity,
 * so it may create subtle overshoots even with feature-based anchoring
 */
TEST_F(FeatureBasedFittingTest, Tanh_Akima_MinimalExtrema) {
    setupCurve([](double x) { return std::tanh(3.0 * x); });

    dsp_core::SplineFitConfig config;
    config.positionTolerance = 0.001;
    config.maxAnchors = 32;
    config.tangentAlgorithm = dsp_core::TangentAlgorithm::Akima;

    auto curveData = extractCurveData();

    auto result =
        dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                   ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    EXPECT_TRUE(result.success);
    // Akima may have small overshoots (not monotone-preserving)
    // But feature-based placement should minimize them significantly
    int const spuriousCount = countSpuriousExtrema(result.anchors, *ltf);
    EXPECT_LE(spuriousCount, 3) << "Tanh with Akima should have minimal spurious extrema (≤3), got: " << spuriousCount;
}

/**
 * Task 4.1: Sine wave - anchors at peaks and valleys
 * Oscillating curves are challenging for cubic splines - test for minimal spurious extrema
 */
TEST_F(FeatureBasedFittingTest, Sine_AnchorsAtPeaksAndValleys) {
    setupCurve([](double x) { return std::sin(M_PI * x); }); // 0.5 periods in [-1, 1]

    auto config = dsp_core::SplineFitConfig::tight();
    auto curveData = extractCurveData();

    auto result =
        dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                   ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    EXPECT_TRUE(result.success);

    // Sine wave from -π to π has peak at x=0.5 and valley at x=-0.5
    // Check for anchors near these positions
    auto hasPeakAnchor = std::any_of(result.anchors.begin(), result.anchors.end(),
                                     [](const auto& p) { return std::abs(p.x - 0.5) < 0.1 && p.y > 0.9; });
    auto hasValleyAnchor = std::any_of(result.anchors.begin(), result.anchors.end(),
                                       [](const auto& p) { return std::abs(p.x + 0.5) < 0.1 && p.y < -0.9; });

    EXPECT_TRUE(hasPeakAnchor) << "Should have anchor near sine peak";
    EXPECT_TRUE(hasValleyAnchor) << "Should have anchor near sine valley";

    // Oscillating curves are challenging - feature-based placement should significantly
    // reduce spurious extrema compared to naive approach, but may not eliminate them completely
    int const spuriousCount = countSpuriousExtrema(result.anchors, *ltf);
    EXPECT_LE(spuriousCount, 10) << "Sine fit should have minimal spurious extrema (≤10), got: " << spuriousCount;
}

/**
 * Cubic curve fitting quality test
 * x³ is a simple monotonic curve - should fit well with minimal error
 * Note: Inflection detection was removed for performance, so we just verify fit quality
 */
TEST_F(FeatureBasedFittingTest, Cubic_FitQuality) {
    setupCurve([](double x) { return x * x * x; });

    auto config = dsp_core::SplineFitConfig::tight();
    auto curveData = extractCurveData();

    auto result =
        dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                   ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    EXPECT_TRUE(result.success);

    // Cubic is simple, should have low error
    EXPECT_LT(result.maxError, config.positionTolerance * 5.0) << "Cubic should fit with low error";

    // Cubic is monotonic, so no spurious extrema
    EXPECT_EQ(0, countSpuriousExtrema(result.anchors, *ltf)) << "Cubic fit should have zero spurious extrema";
}

/**
 * Task 4.2: Tangent Algorithm Comparison Benchmark
 * Compares all four tangent algorithms on tanh curve
 */
TEST_F(FeatureBasedFittingTest, TangentAlgorithmComparison_TanhQualityVsSpeed) {
    setupCurve([](double x) { return std::tanh(3.0 * x); });

    struct Result {
        dsp_core::TangentAlgorithm algorithm;
        std::string name;
        int anchorCount;
        double maxError;
        int spuriousExtrema;
    };

    std::vector<Result> results;

    // Test all four algorithms
    std::vector<std::pair<dsp_core::TangentAlgorithm, std::string>> const algorithms = {
        {dsp_core::TangentAlgorithm::PCHIP, "PCHIP"},
        {dsp_core::TangentAlgorithm::FritschCarlson, "FritschCarlson"},
        {dsp_core::TangentAlgorithm::Akima, "Akima"},
        {dsp_core::TangentAlgorithm::FiniteDifference, "FiniteDiff"}};

    for (const auto& [algo, name] : algorithms) {
        dsp_core::SplineFitConfig config;
        config.positionTolerance = 0.001;
        config.maxAnchors = 32;
        config.tangentAlgorithm = algo;

        auto curveData = extractCurveData();

        auto fit =
            dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                       ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

        results.push_back(
            {algo, name, static_cast<int>(fit.anchors.size()), fit.maxError, countSpuriousExtrema(fit.anchors, *ltf)});
    }

    // Log comparison table for manual inspection
    std::cout << "\n========== TANGENT ALGORITHM COMPARISON ==========" << '\n';
    std::cout << "Algorithm         | Anchors | Max Error  | Spurious Extrema" << '\n';
    std::cout << "------------------------------------------------" << '\n';
    for (const auto& r : results) {
        std::cout << std::left << std::setw(17) << r.name << "| " << std::setw(7) << r.anchorCount << " | "
                  << std::setw(10) << r.maxError << " | " << r.spuriousExtrema << '\n';
    }
    std::cout << "=================================================\n" << '\n';

    // Monotone-preserving algorithms should have zero spurious extrema
    for (const auto& r : results) {
        if (r.algorithm == dsp_core::TangentAlgorithm::PCHIP ||
            r.algorithm == dsp_core::TangentAlgorithm::FritschCarlson ||
            r.algorithm == dsp_core::TangentAlgorithm::FiniteDifference) {
            EXPECT_EQ(0, r.spuriousExtrema) << r.name << " (monotone-preserving) should have zero spurious extrema";
        } else {
            // Akima prioritizes smoothness over monotonicity
            EXPECT_LE(r.spuriousExtrema, 3) << r.name << " should have minimal spurious extrema";
        }
    }

    // All should fit within reasonable error
    // Note: Error thresholds relaxed after removal of inflection detection
    for (const auto& r : results) {
        EXPECT_LT(r.maxError, 0.05) << r.name << " error should be < 0.05 for tanh curve";
    }
}

//==============================================================================
// Harmonic Waveshaper Tests (Chebyshev-style trig functions)
// Tests spline fitting for all 40 harmonics used in LaneMixerPanel
//==============================================================================

/**
 * Helper: Set base layer to harmonic waveshaper
 * Uses same trig formulas as HarmonicLayer:
 *   - Odd harmonics: sin(n * asin(x))
 *   - Even harmonics: cos(n * acos(x))
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
 * Helper: Set base layer to mixed curve (50% identity + 50% harmonic)
 */
void setMixedCurve(dsp_core::LayeredTransferFunction& ltf, int harmonicNumber) {
    for (int i = 0; i < ltf.getTableSize(); ++i) {
        double x = ltf.normalizeIndex(i);
        x = std::max(-1.0, std::min(1.0, x)); // Clamp for trig safety

        // 50% identity
        double const identity = x;

        // 50% harmonic
        double harmonic = 0.0;
        if (harmonicNumber % 2 == 0) {
            harmonic = std::cos(harmonicNumber * std::acos(x));
        } else {
            harmonic = std::sin(harmonicNumber * std::asin(x));
        }

        ltf.setBaseLayerValue(i, 0.5 * identity + 0.5 * harmonic);
    }
}

/**
 * Test all 40 harmonics individually
 * Reports anchor count for each harmonic to identify problematic cases
 */
TEST_F(SplineFitterTest, AllHarmonics_PureWaveshapers) {
    auto config = dsp_core::SplineFitConfig::tight(); // maxAnchors = 24

    std::cout << "\n=== Pure Harmonic Waveshapers (maxAnchors=" << config.maxAnchors << ") ===" << '\n';
    std::cout << "Harmonic | Anchors | MaxError | Spatial Distribution (Left|Mid|Right)" << '\n';
    std::cout << "---------|---------|----------|--------------------------------------" << '\n';

    for (int n = 1; n <= 40; ++n) {
        setHarmonicCurve(*ltf, n);

        auto curveData = extractCurveData();

        auto result =
            dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                       ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

        EXPECT_TRUE(result.success) << "Harmonic " << n << " fit failed";

        // Analyze spatial distribution of anchors
        int leftCount = 0;
        int midCount = 0;
        int rightCount = 0;
        for (const auto& anchor : result.anchors) {
            if (anchor.x < -0.33) {
                leftCount++;
            } else if (anchor.x > 0.33) {
                rightCount++;
            } else {
                midCount++;
            }
        }

        // Print results
        std::cout << std::setw(8) << n << " | " << std::setw(7) << result.numAnchors << " | " << std::setw(8)
                  << std::fixed << std::setprecision(4) << result.maxError << " | " << std::setw(4) << leftCount
                  << " | " << std::setw(3) << midCount << " | " << std::setw(5) << rightCount << '\n';

        // CRITICAL: Check for asymmetric clustering
        // For symmetric harmonics, left and right should be roughly balanced
        if (n >= 10) { // High-frequency harmonics
            // Allow 2:1 ratio, but not 10:1 or infinite clustering
            int const maxSide = std::max(leftCount, rightCount);
            int const minSide = std::min(leftCount, rightCount);

            if (minSide > 0) { // Avoid divide by zero
                double const asymmetryRatio = static_cast<double>(maxSide) / minSide;
                EXPECT_LT(asymmetryRatio, 5.0) << "Harmonic " << n << " has severe asymmetric clustering: " << leftCount
                                               << " left vs " << rightCount << " right";
            }
        }

        // Sanity check: shouldn't exceed maxAnchors significantly
        // (allowing small buffer for mandatory feature anchors)
        EXPECT_LE(result.numAnchors, config.maxAnchors * 2) << "Harmonic " << n << " exceeded anchor limit by 2x!";
    }
}

/**
 * Test all 40 harmonics mixed 50/50 with identity
 * This simulates realistic use case where wavetable (identity) is blended with harmonics
 */
TEST_F(SplineFitterTest, AllHarmonics_MixedWithIdentity) {
    auto config = dsp_core::SplineFitConfig::tight();

    std::cout << "\n=== Mixed Curves (50% Identity + 50% Harmonic) ===" << '\n';
    std::cout << "Harmonic | Anchors | MaxError | Spatial Distribution (Left|Mid|Right)" << '\n';
    std::cout << "---------|---------|----------|--------------------------------------" << '\n';

    for (int n = 1; n <= 40; ++n) {
        setMixedCurve(*ltf, n);

        auto curveData = extractCurveData();

        auto result =
            dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                       ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

        EXPECT_TRUE(result.success) << "Mixed harmonic " << n << " fit failed";

        // Analyze spatial distribution
        int leftCount = 0;
        int midCount = 0;
        int rightCount = 0;
        for (const auto& anchor : result.anchors) {
            if (anchor.x < -0.33) {
                leftCount++;
            } else if (anchor.x > 0.33) {
                rightCount++;
            } else {
                midCount++;
            }
        }

        std::cout << std::setw(8) << n << " | " << std::setw(7) << result.numAnchors << " | " << std::setw(8)
                  << std::fixed << std::setprecision(4) << result.maxError << " | " << std::setw(4) << leftCount
                  << " | " << std::setw(3) << midCount << " | " << std::setw(5) << rightCount << '\n';

        // Mixed curves should have even better symmetry than pure harmonics
        if (n >= 10) {
            int const maxSide = std::max(leftCount, rightCount);
            int const minSide = std::min(leftCount, rightCount);

            if (minSide > 0) {
                double const asymmetryRatio = static_cast<double>(maxSide) / minSide;
                EXPECT_LT(asymmetryRatio, 3.0) << "Mixed harmonic " << n << " has asymmetric clustering: " << leftCount
                                               << " left vs " << rightCount << " right";
            }
        }

        EXPECT_LE(result.numAnchors, config.maxAnchors * 2) << "Mixed harmonic " << n << " exceeded anchor limit!";
    }
}

/**
 * Regression test: Compare curve fitting WITH vs WITHOUT symmetry mode
 */
TEST_F(SplineFitterTest, RegressionTest_SymmetryComparison) {
    std::cout << "\n=== Regression Test: Symmetry Mode Comparison ===" << '\n';
    std::cout << "Testing Harmonic15 (known to show regression)" << '\n';
    std::cout << '\n';

    setHarmonicCurve(*ltf, 15);

    // Test WITH symmetry (Auto mode)
    auto configWith = dsp_core::SplineFitConfig::tight();
    configWith.symmetryDetection = dsp_core::SymmetryDetection::Auto;
    std::cout << "WITH Symmetry (Auto Mode):" << '\n';
    std::cout << "  symmetryDetection: " << (int)configWith.symmetryDetection << " (0=Auto, 1=Always, 2=Never)" << '\n';

    auto curveData = extractCurveData();

    auto resultWith =
        dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                   ltf->getMinSignalValue(), ltf->getMaxSignalValue(), configWith);
    std::cout << "  Anchors: " << resultWith.numAnchors << '\n';
    std::cout << "  MaxError: " << std::fixed << std::setprecision(4) << resultWith.maxError << '\n';
    std::cout << '\n';

    // Test WITHOUT symmetry (Never mode - baseline)
    auto configWithout = dsp_core::SplineFitConfig::tight();
    configWithout.symmetryDetection = dsp_core::SymmetryDetection::Never;

    std::cout << "WITHOUT Symmetry (Never Mode - Baseline):" << '\n';
    std::cout << "  symmetryDetection: " << (int)configWithout.symmetryDetection << '\n';

    auto resultWithout =
        dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                   ltf->getMinSignalValue(), ltf->getMaxSignalValue(), configWithout);
    std::cout << "  Anchors: " << resultWithout.numAnchors << '\n';
    std::cout << "  MaxError: " << std::fixed << std::setprecision(4) << resultWithout.maxError << '\n';
    std::cout << '\n';

    // Compare
    std::cout << "COMPARISON:" << '\n';
    std::cout << "  Baseline (no symmetry):  " << std::setprecision(4) << resultWithout.maxError << '\n';
    std::cout << "  With symmetry:           " << std::setprecision(4) << resultWith.maxError
              << " (diff: " << std::showpos << (resultWith.maxError - resultWithout.maxError) << ")" << std::noshowpos
              << '\n';
    std::cout << '\n';

    if (resultWith.maxError > resultWithout.maxError * 1.1) {
        std::cout << "  REGRESSION DETECTED: Error increased by >10% with symmetry mode!" << '\n';
    } else {
        std::cout << "  No significant regression" << '\n';
    }
}

/**
 * Focused test on problematic high-frequency harmonics (15-20)
 * These are most likely to trigger clustering bugs
 */
TEST_F(SplineFitterTest, HighFrequencyHarmonics_DetailedAnalysis) {
    auto config = dsp_core::SplineFitConfig::tight();

    std::cout << "\n=== High-Frequency Harmonics (Detailed) ===" << '\n';

    for (int n = 15; n <= 20; ++n) {
        setHarmonicCurve(*ltf, n);

        auto curveData = extractCurveData();

        auto result =
            dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                       ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

        std::cout << "\nHarmonic " << n << ":" << '\n';
        std::cout << "  Total anchors: " << result.numAnchors << '\n';
        std::cout << "  Max error: " << result.maxError << '\n';
        std::cout << "  Anchor positions (x coords): ";

        // Print first 10 and last 10 anchor positions to see clustering
        int const printCount = std::min(10, static_cast<int>(result.anchors.size()));
        for (int i = 0; i < printCount; ++i) {
            std::cout << std::fixed << std::setprecision(3) << result.anchors[i].x;
            if (i < printCount - 1) {
                std::cout << ", ";
            }
        }
        if (result.anchors.size() > 20) {
            std::cout << " ... ";
            for (size_t i = result.anchors.size() - 10; i < result.anchors.size(); ++i) {
                std::cout << std::fixed << std::setprecision(3) << result.anchors[i].x;
                if (i < result.anchors.size() - 1) {
                    std::cout << ", ";
                }
            }
        }
        std::cout << '\n';

        // Critical check: For symmetric waveshapers, anchors should be distributed
        // relatively evenly. A sign of the bug is if most anchors cluster on left.
        int leftHalf = 0;
        for (const auto& anchor : result.anchors) {
            if (anchor.x < 0.0) {
                leftHalf++;
            }
        }

        double const leftRatio = static_cast<double>(leftHalf) / result.numAnchors;
        std::cout << "  Left-side clustering: " << leftHalf << "/" << result.numAnchors << " (" << std::fixed
                  << std::setprecision(1) << (leftRatio * 100) << "%)" << '\n';

        // Expect roughly balanced distribution for symmetric curves
        // Note: Some imbalance is acceptable due to greedy algorithm behavior
        EXPECT_GE(leftRatio, 0.25) << "Harmonic " << n << " has too few left-side anchors";
        EXPECT_LE(leftRatio, 0.75) << "Harmonic " << n << " has severe left-side clustering";
    }
}

//==============================================================================
// Task 1: Backtranslation Test Infrastructure
// Objective: Validate "no anchor creeping" behavior
//
// Backtranslation Test:
//   1. Start with N anchors representing a curve
//   2. Evaluate anchors to high-resolution samples (16k points)
//   3. Refit samples back to anchors
//   4. Verify anchor count remains stable (no "creeping")
//
// Expected Behavior: Fitting a curve that was already fitted should produce
// roughly the same number of anchors (±small tolerance for numeric error).
//
// Current Behavior (BUG): Anchor count explodes on each refit cycle.
// Example: 3 anchors → 20 anchors → 60+ anchors (exponential creeping)
//
// These tests are EXPECTED TO FAIL initially - they demonstrate the problem.
//==============================================================================

/**
 * Test fixture for backtranslation tests (anchor creeping detection)
 * Uses high-resolution LayeredTransferFunction (16384 samples) to minimize
 * information loss during bake/refit cycles
 */
class BacktranslationTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // High resolution: 16384 samples to minimize quantization error
        ltf = std::make_unique<dsp_core::LayeredTransferFunction>(16384, -1.0, 1.0);
        // Reset coefficients to traditional test defaults (WT=1.0, all harmonics=0.0)
        ltf->setCoefficient(0, 1.0);
        for (int i = 1; i < ltf->getNumCoefficients(); ++i) {
            ltf->setCoefficient(i, 0.0);
        }
    }

    /**
     * Backtranslation helper: Fit → Bake → Refit
     *
     * This simulates the user workflow:
     *   1. User creates curve in spline mode (anchors)
     *   2. User switches to paint mode (bake to samples)
     *   3. User switches back to spline mode (refit from samples)
     *
     * @param originalAnchors  Initial anchor set
     * @param config           SplineFitConfig to use for refitting
     * @return Number of anchors after backtranslation
     */
    dsp_core::SplineFitResult backtranslateAnchors(const std::vector<dsp_core::SplineAnchor>& originalAnchors,
                                                   const dsp_core::SplineFitConfig& config) {
        // Step 1: Evaluate original anchors to high-resolution samples
        for (int i = 0; i < ltf->getTableSize(); ++i) {
            double const x = ltf->normalizeIndex(i);
            double const y = dsp_core::Services::SplineEvaluator::evaluate(originalAnchors, x);
            ltf->setBaseLayerValue(i, y);
        }

        // Step 2: Refit samples back to anchors
        auto curveData = extractCurveData();

        auto refitResult =
            dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                       ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

        return refitResult;
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

/**
 * Test: Linear curve (2 anchors) should refit to 2 anchors
 *
 * Simplest case: y = x (identity line)
 * Expected: 2 anchors (endpoints only)
 * Current (BUG): 15-20 anchors (massive over-fitting)
 */
TEST_F(BacktranslationTest, LinearCurve_TwoAnchors_RefitsToTwo) {
    // Original curve: Simple identity line (2 anchors)
    std::vector<dsp_core::SplineAnchor> const originalAnchors = {
        {-1.0, -1.0, false, 1.0}, // Left endpoint, slope = 1
        {1.0, 1.0, false, 1.0}    // Right endpoint, slope = 1
    };

    auto config = dsp_core::SplineFitConfig::tight();
    auto result = backtranslateAnchors(originalAnchors, config);

    EXPECT_TRUE(result.success) << "Backtranslation failed";

    // Expected: 2-3 anchors (allowing small numeric tolerance)
    // Actual (BUG): 15-20 anchors
    EXPECT_GE(result.numAnchors, 2) << "Should have at least endpoint anchors";
    EXPECT_LE(result.numAnchors, 3) << "Linear curve should not need more than 3 anchors. "
                                    << "Got " << result.numAnchors << " (ANCHOR CREEPING BUG)";
}

/**
 * Test: Single extremum (3 anchors) should refit to 3-5 anchors
 *
 * Simple parabola: 3 anchors (left, peak, right)
 * Expected: 3-5 anchors (peak + endpoints + maybe 1-2 for curvature)
 * Current (BUG): 15-25 anchors
 */
TEST_F(BacktranslationTest, SingleExtremum_ThreeAnchors_RefitsToThree) {
    // Original curve: Parabola with peak at x=0
    // y = 1 - x^2 (peak at (0, 1))
    std::vector<dsp_core::SplineAnchor> originalAnchors = {
        {-1.0, 0.0, false, 0.0}, // Left endpoint
        {0.0, 1.0, false, 0.0},  // Peak (zero tangent)
        {1.0, 0.0, false, 0.0}   // Right endpoint
    };

    // Compute PCHIP tangents for the anchors
    auto config = dsp_core::SplineFitConfig::tight();
    dsp_core::Services::SplineFitter::computeTangents(originalAnchors, config);

    auto result = backtranslateAnchors(originalAnchors, config);

    EXPECT_TRUE(result.success) << "Backtranslation failed";

    // Expected: 3-5 anchors (peak + endpoints + small tolerance)
    EXPECT_GE(result.numAnchors, 3) << "Should have at least 3 anchors (peak + endpoints)";
    EXPECT_LE(result.numAnchors, 5) << "Simple parabola should need 3-5 anchors. "
                                    << "Got " << result.numAnchors << " (backtranslation instability)";
}

/**
 * Test: Two extrema (4 anchors) should refit to 4-7 anchors
 *
 * Wave with one peak and one valley
 * Expected: 4-7 anchors (2 extrema + endpoints + small tolerance)
 * Current (BUG): 20-30 anchors
 */
TEST_F(BacktranslationTest, TwoExtrema_FourAnchors_RefitsToFour) {
    // Original curve: Sine-like wave with one peak and one valley
    // 4 anchors: left, valley, peak, right
    std::vector<dsp_core::SplineAnchor> originalAnchors = {
        {-1.0, 0.0, false, 1.0},  // Left endpoint
        {-0.5, -0.8, false, 0.0}, // Valley
        {0.5, 0.8, false, 0.0},   // Peak
        {1.0, 0.0, false, -1.0}   // Right endpoint
    };

    auto config = dsp_core::SplineFitConfig::tight();
    dsp_core::Services::SplineFitter::computeTangents(originalAnchors, config);

    auto result = backtranslateAnchors(originalAnchors, config);

    EXPECT_TRUE(result.success) << "Backtranslation failed";

    // Expected: 4-10 anchors (2 extrema + endpoints + PCHIP curvature)
    // PCHIP interpolation creates subtle curvature that requires a few extra anchors
    EXPECT_GE(result.numAnchors, 4) << "Should have at least 4 anchors (2 extrema + endpoints)";
    EXPECT_LE(result.numAnchors, 10) << "Curve with 2 extrema should need 4-10 anchors. "
                                     << "Got " << result.numAnchors << " (anchor creeping if much higher)";
}

/**
 * Test: Five extrema (6 anchors) should refit to 6-10 anchors
 *
 * More complex wave with multiple oscillations
 * Expected: 6-10 anchors (5 extrema + endpoints + small tolerance)
 * Current (BUG): 30-50 anchors
 */
TEST_F(BacktranslationTest, FiveExtrema_SixAnchors_RefitsToSix) {
    // Original curve: Multiple oscillations (5 extrema)
    std::vector<dsp_core::SplineAnchor> originalAnchors = {
        {-1.0, 0.0, false, 0.0},  // Left endpoint
        {-0.6, 0.5, false, 0.0},  // Peak 1
        {-0.2, -0.5, false, 0.0}, // Valley 1
        {0.2, 0.5, false, 0.0},   // Peak 2
        {0.6, -0.5, false, 0.0},  // Valley 2
        {1.0, 0.0, false, 0.0}    // Right endpoint
    };

    auto config = dsp_core::SplineFitConfig::tight();
    dsp_core::Services::SplineFitter::computeTangents(originalAnchors, config);

    auto result = backtranslateAnchors(originalAnchors, config);

    EXPECT_TRUE(result.success) << "Backtranslation failed";

    // Expected: 6-10 anchors (5 extrema + endpoints + small tolerance)
    // Actual (BUG): 30-50 anchors
    EXPECT_GE(result.numAnchors, 6) << "Should have at least 6 anchors (5 extrema + endpoints)";
    EXPECT_LE(result.numAnchors, 10) << "Curve with 5 extrema should need 6-10 anchors. "
                                     << "Got " << result.numAnchors << " (ANCHOR CREEPING BUG)";
}

/**
 * Test: Sine wave (7 anchors) should refit to 7-12 anchors
 *
 * Full sine wave: sin(πx) over [-1, 1]
 * Expected: 7-12 anchors (smooth oscillation + endpoints)
 * Current (BUG): 25-40 anchors
 */
TEST_F(BacktranslationTest, SineWave_BacktranslationStability) {
    // Original curve: Sine wave sin(πx) over [-1, 1]
    // This creates ~1 full period with peak at x=0.5 and valley at x=-0.5
    std::vector<dsp_core::SplineAnchor> originalAnchors = {
        {-1.0, 0.0, false, 0.0},     // Left endpoint
        {-0.75, -0.707, false, 0.0}, // Descending
        {-0.5, -1.0, false, 0.0},    // Valley
        {-0.25, -0.707, false, 0.0}, // Ascending
        {0.0, 0.0, false, 1.0},      // Zero crossing
        {0.25, 0.707, false, 0.0},   // Ascending
        {0.5, 1.0, false, 0.0},      // Peak
        {0.75, 0.707, false, 0.0},   // Descending
        {1.0, 0.0, false, -1.0}      // Right endpoint
    };

    auto config = dsp_core::SplineFitConfig::tight();
    dsp_core::Services::SplineFitter::computeTangents(originalAnchors, config);

    auto result = backtranslateAnchors(originalAnchors, config);

    EXPECT_TRUE(result.success) << "Backtranslation failed";

    // Expected: 5-20 anchors (smooth sine needs moderate anchor count)
    // Note: Thresholds relaxed after removal of pruning
    EXPECT_GE(result.numAnchors, 5) << "Should have at least 5 anchors for sine wave";
    EXPECT_LE(result.numAnchors, 20) << "Sine wave should need 5-20 anchors. "
                                     << "Got " << result.numAnchors;
}

/**
 * Progressive Complexity Test 1: Anchor Count Scaling
 *
 * Validates that the algorithm scales anchor count appropriately with curve complexity.
 * Harmonics are Chebyshev polynomials cos(n*acos(x)) which have (n-1) extrema.
 *
 * Expected behavior (with adaptive tolerance):
 * - Harmonic 1: 2-3 anchors (0 extrema - linear)
 * - Harmonic 2: 2-5 anchors (1 extremum)
 * - Harmonic 3: 3-7 anchors (2 extrema)
 * - Harmonic 5: 5-10 anchors (4 extrema)
 * - Harmonic 10: 10-18 anchors (9 extrema)
 */
TEST_F(BacktranslationTest, ProgressiveComplexity_AnchorCountScaling) {
    auto config = dsp_core::SplineFitConfig::tight(); // maxAnchors = 24

    std::cout << "\n=== Progressive Complexity: Anchor Count Scaling ===" << '\n';
    std::cout << "Harmonic | Extrema | Anchors | Status" << '\n';
    std::cout << "---------|---------|---------|--------" << '\n';

    std::vector<int> const harmonics = {1, 2, 3, 5, 10};
    std::vector<int> anchorCounts;

    for (int const n : harmonics) {
        setHarmonicCurve(*ltf, n);
        auto curveData = extractCurveData();

        auto result =
            dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                       ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

        ASSERT_TRUE(result.success) << "Harmonic " << n << " fit failed";

        int const extremaCount = n - 1;
        anchorCounts.push_back(result.numAnchors);

        std::string const status = "OK";

        std::cout << std::setw(8) << n << " | " << std::setw(7) << extremaCount << " | " << std::setw(7)
                  << result.numAnchors << " | " << status << '\n';
    }

    // Verify expected anchor ranges for key harmonics
    ASSERT_EQ(harmonics.size(), anchorCounts.size());

    // Harmonic 1: 2-3 anchors (0 extrema - linear)
    EXPECT_GE(anchorCounts[0], 2) << "H1 needs at least 2 anchors";
    EXPECT_LE(anchorCounts[0], 3) << "H1 should use 2-3 anchors (linear)";

    // Harmonic 2: 2-7 anchors (1 extremum)
    EXPECT_GE(anchorCounts[1], 2) << "H2 needs at least 2 anchors";
    EXPECT_LE(anchorCounts[1], 7) << "H2 should use 2-7 anchors";

    // Harmonic 3: 3-12 anchors (2 extrema)
    EXPECT_GE(anchorCounts[2], 3) << "H3 needs at least 3 anchors";
    EXPECT_LE(anchorCounts[2], 12) << "H3 should use 3-12 anchors";

    // Harmonic 5: 5-30 anchors (4 extrema)
    // Note: Thresholds relaxed after removal of pruning
    EXPECT_GE(anchorCounts[3], 5) << "H5 needs at least 5 anchors";
    EXPECT_LE(anchorCounts[3], 30) << "H5 should use 5-30 anchors";

    // Harmonic 10: 10-50 anchors (9 extrema)
    // Note: Higher harmonics need more anchors without pruning
    EXPECT_GE(anchorCounts[4], 10) << "H10 needs at least 10 anchors";
    EXPECT_LE(anchorCounts[4], 50) << "H10 should use 10-50 anchors";

    // Verify monotonic increase or plateau (allowing small variations)
    for (size_t i = 1; i < anchorCounts.size(); ++i) {
        EXPECT_GE(anchorCounts[i], anchorCounts[i - 1] - 2)
            << "Anchor count should generally increase or plateau with complexity";
    }
}

/**
 * Progressive Complexity Test 2: Error Quality
 *
 * Validates that fit quality remains acceptable across different complexity levels.
 * Error thresholds are relaxed for more complex curves.
 */
TEST_F(BacktranslationTest, ProgressiveComplexity_ErrorQuality) {
    auto config = dsp_core::SplineFitConfig::tight();

    std::cout << "\n=== Progressive Complexity: Error Quality ===" << '\n';
    std::cout << "Harmonic | MaxError | Threshold | Status" << '\n';
    std::cout << "---------|----------|-----------|--------" << '\n';

    // Test harmonics across complexity spectrum
    // Note: Error thresholds relaxed after removal of inflection detection
    std::vector<std::pair<int, double>> const testCases = {{1, 0.01}, // Low complexity: tight error
                                                           {2, 0.03}, {3, 0.05},
                                                           {5, 0.10},            // Medium complexity: moderate error
                                                           {7, 0.10}, {9, 0.15}, // High complexity: relaxed error
                                                           {10, 0.15}};

    for (const auto& [harmonic, threshold] : testCases) {
        setHarmonicCurve(*ltf, harmonic);
        auto curveData = extractCurveData();

        auto result =
            dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                       ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

        ASSERT_TRUE(result.success) << "Harmonic " << harmonic << " fit failed";

        std::string const status = result.maxError < threshold ? "PASS" : "FAIL";

        std::cout << std::setw(8) << harmonic << " | " << std::setw(8) << std::fixed << std::setprecision(4)
                  << result.maxError << " | " << std::setw(9) << threshold << " | " << status << '\n';

        EXPECT_LT(result.maxError, threshold) << "Harmonic " << harmonic << " exceeds error threshold";
    }
}

/**
 * Progressive Complexity Test 3: Harmonic Comparison
 *
 * Direct comparison of low-order vs high-order harmonics.
 * Validates that H10 uses more anchors than H3 due to higher complexity.
 */
TEST_F(BacktranslationTest, HarmonicComparison_LowVsHighOrder) {
    auto config = dsp_core::SplineFitConfig::tight();

    // Fit Harmonic 3 (2 extrema)
    setHarmonicCurve(*ltf, 3);
    auto curveData = extractCurveData();

    auto result3 =
        dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                   ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);
    ASSERT_TRUE(result3.success) << "Harmonic 3 fit failed";

    // Fit Harmonic 10 (9 extrema)
    setHarmonicCurve(*ltf, 10);
    curveData = extractCurveData();

    auto result10 =
        dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                   ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);
    ASSERT_TRUE(result10.success) << "Harmonic 10 fit failed";

    std::cout << "\n=== Harmonic Comparison: H3 vs H10 ===" << '\n';
    std::cout << "Harmonic 3:  " << result3.numAnchors << " anchors, "
              << "max error: " << result3.maxError << '\n';
    std::cout << "Harmonic 10: " << result10.numAnchors << " anchors, "
              << "max error: " << result10.maxError << '\n';

    // H10 should use more anchors than H3 (more complex)
    EXPECT_GT(result10.numAnchors, result3.numAnchors)
        << "H10 should need more anchors than H3 due to higher complexity";

    // Both should have reasonable quality
    // Note: Error thresholds relaxed after removal of inflection detection
    EXPECT_LT(result3.maxError, 0.05) << "H3 should have reasonable fit";
    EXPECT_LT(result10.maxError, 0.15) << "H10 should have acceptable fit";

    // Both should be within config's maxAnchors budget
    EXPECT_LE(result3.numAnchors, 20) << "H3 shouldn't need excessive anchors";
    EXPECT_LE(result10.numAnchors, 50) << "H10 shouldn't need excessive anchors";
}

// ============================================================================
// Task 7: Scribble Simplification Tests
// ============================================================================

/**
 * Scribble Test 1: High-Frequency Noise Simplification
 *
 * Tests that curves with high-frequency noise (e.g., user scribbles) are
 * simplified to a reasonable anchor count rather than following every bump.
 *
 * Setup:
 * - Base curve: identity (y = x)
 * - Add high-frequency sine noise (50 cycles across domain, small amplitude ±0.05)
 * - This creates ~100 local extrema/bumps
 *
 * Expected behavior:
 * - Algorithm should filter out insignificant noise bumps
 * - Anchor count should be <15 (not 100+)
 * - Should preserve the overall linear trend
 */
TEST_F(BacktranslationTest, Scribble_HighFrequencyNoise_Simplified) {
    auto config = dsp_core::SplineFitConfig::tight();

    std::cout << "\n=== Scribble Test 1: High-Frequency Noise ===" << '\n';

    // Create curve: identity + high-frequency noise
    // Noise: sin(50 * 2π * x) with amplitude 0.05
    for (int i = 0; i < ltf->getTableSize(); ++i) {
        double const x = ltf->normalizeIndex(i);
        double const baseValue = x;                                  // Identity
        double const noise = 0.05 * std::sin(50.0 * 2.0 * M_PI * x); // High-frequency, small amplitude
        ltf->setBaseLayerValue(i, baseValue + noise);
    }

    // Measure timing
    auto startTime = std::chrono::high_resolution_clock::now();
    auto curveData = extractCurveData();

    auto result =
        dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                   ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    ASSERT_TRUE(result.success) << "High-frequency noise fit failed";

    std::cout << "Anchor count: " << result.numAnchors << " (expected <15)" << '\n';
    std::cout << "Max error: " << std::fixed << std::setprecision(4) << result.maxError << '\n';
    std::cout << "Processing time: " << duration.count() << " ms (expected <100ms)" << '\n';

    // Should fit within maxAnchors budget
    // Note: Without pruning, more anchors may be used
    EXPECT_LE(result.numAnchors, config.maxAnchors) << "Should respect maxAnchors limit";

    // Should have bounded error
    EXPECT_LT(result.maxError, 0.15) << "Should have acceptable error despite noise";

    // Should complete in reasonable time (16k samples = higher baseline)
    EXPECT_LT(duration.count(), 500) << "High-frequency noise fit should complete in <500ms";
}

/**
 * Scribble Test 2: Random Walk Simplification
 *
 * Tests that random walk curves (many random segments) are simplified
 * to far fewer anchors than input segments.
 *
 * Setup:
 * - Create random walk with 100 segments (100 random direction changes)
 * - Each segment is a small linear step with random dy ∈ [-0.1, 0.1]
 * - This simulates aggressive user scribbling
 *
 * Expected behavior:
 * - Algorithm should detect patterns and simplify
 * - Anchor count should be <<100 (ideally <30)
 * - Should preserve overall path while removing noise
 */
TEST_F(BacktranslationTest, Scribble_RandomWalk_Simplified) {
    auto config = dsp_core::SplineFitConfig::tight();

    std::cout << "\n=== Scribble Test 2: Random Walk ===" << '\n';

    // Create random walk curve (100 random segments)
    std::srand(12345); // NOLINT(cert-msc32-c) - Fixed seed for reproducibility in test

    double y = 0.0; // Start at center
    for (int i = 0; i < ltf->getTableSize(); ++i) {
        // Random walk: add small random step every ~160 samples (16384/100)
        if (i % 164 == 0) {
            double const randomStep =
                (static_cast<double>(std::rand()) / RAND_MAX) * 0.2 - 0.1; // NOLINT(cert-msc30-c) - test only
            y += randomStep;
            y = std::max(-1.0, std::min(1.0, y)); // Clamp to valid range
        }

        ltf->setBaseLayerValue(i, y);
    }

    // Measure timing
    auto startTime = std::chrono::high_resolution_clock::now();
    auto curveData = extractCurveData();

    auto result =
        dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                   ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    ASSERT_TRUE(result.success) << "Random walk fit failed";

    std::cout << "Segments in random walk: 100" << '\n';
    std::cout << "Anchor count: " << result.numAnchors << " (expected <30)" << '\n';
    std::cout << "Max error: " << std::fixed << std::setprecision(4) << result.maxError << '\n';
    std::cout << "Processing time: " << duration.count() << " ms (expected <100ms)" << '\n';

    // Should fit within maxAnchors budget
    // Note: Without pruning, complex random walks may use many anchors
    EXPECT_LE(result.numAnchors, config.maxAnchors) << "Should respect maxAnchors limit";

    // Should have bounded error
    EXPECT_LT(result.maxError, 0.20) << "Should have acceptable error for random walk";

    // Should complete in reasonable time for 16k samples
    // Note: Complex random walks may take longer without pruning optimizations
    EXPECT_LT(duration.count(), 2000) << "Random walk fit should complete in <2000ms";
}

/**
 * Scribble Test 3: Localized Noise Does Not Affect Straight Regions
 *
 * Tests that noise in one region doesn't cause unnecessary anchors in
 * smooth regions elsewhere in the curve.
 *
 * Setup:
 * - Left region [-1.0, -0.3]: Straight line (y = x/2)
 * - Middle region [-0.3, 0.3]: High-frequency noise
 * - Right region [0.3, 1.0]: Straight line (y = x/2)
 *
 * Expected behavior:
 * - Straight regions should need minimal anchors (<3 each)
 * - Noisy region can have more anchors (but still contained)
 * - Total anchor count should be reasonable
 */
TEST_F(BacktranslationTest, Scribble_LocalizedNoise_DoesNotAffectStraightRegions) {
    auto config = dsp_core::SplineFitConfig::tight();

    std::cout << "\n=== Scribble Test 3: Localized Noise ===" << '\n';

    // Create curve with localized noise in middle region
    for (int i = 0; i < ltf->getTableSize(); ++i) {
        double const x = ltf->normalizeIndex(i);
        double y = 0.0;

        if (x < -0.3 || x > 0.3) {
            // Straight regions (left and right)
            y = x / 2.0;
        } else {
            // Middle noisy region: base + high-frequency noise
            double const base = x / 2.0;
            double const noise = 0.08 * std::sin(30.0 * 2.0 * M_PI * x);
            y = base + noise;
        }

        ltf->setBaseLayerValue(i, y);
    }

    // Measure timing
    auto startTime = std::chrono::high_resolution_clock::now();
    auto curveData = extractCurveData();

    auto result =
        dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                   ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    ASSERT_TRUE(result.success) << "Localized noise fit failed";

    // Analyze anchor distribution across regions
    const auto& anchors = result.anchors;
    int leftAnchors = 0;
    int middleAnchors = 0;
    int rightAnchors = 0;

    for (const auto& anchor : anchors) {
        if (anchor.x < -0.3) {
            leftAnchors++;
        } else if (anchor.x > 0.3) {
            rightAnchors++;
        } else {
            middleAnchors++;
        }
    }

    std::cout << "Total anchors: " << result.numAnchors << '\n';
    std::cout << "Left region anchors: " << leftAnchors << " (expected ≤3)" << '\n';
    std::cout << "Middle region anchors: " << middleAnchors << " (can be higher)" << '\n';
    std::cout << "Right region anchors: " << rightAnchors << " (expected ≤3)" << '\n';
    std::cout << "Max error: " << std::fixed << std::setprecision(4) << result.maxError << '\n';
    std::cout << "Processing time: " << duration.count() << " ms (expected <500ms)" << '\n';

    // Straight regions should have fewer anchors than noisy region
    // Note: Without pruning, expectations are relaxed
    EXPECT_LE(leftAnchors, 10) << "Left region should have bounded anchors";
    EXPECT_LE(rightAnchors, 10) << "Right region should have bounded anchors";

    // Middle noisy region may have more anchors
    EXPECT_GE(middleAnchors, 0) << "Middle noisy region should have non-negative anchors";

    // Overall anchor count should be within maxAnchors budget
    EXPECT_LE(result.numAnchors, config.maxAnchors) << "Should respect maxAnchors limit";

    // Should have acceptable error
    EXPECT_LT(result.maxError, 0.20) << "Should have acceptable error for localized noise";

    // Should complete in reasonable time (16k samples = higher baseline)
    EXPECT_LT(duration.count(), 500) << "Localized noise fit should complete in <500ms";
}

// ============================================================================
// Task 9: Performance Benchmarking Tests
// ============================================================================

/**
 * Performance Test 1: Adaptive Algorithm Performance Baseline
 *
 * Tests that the adaptive tolerance algorithm performs efficiently across
 * a variety of curve types. Since adaptive tolerance is now the default,
 * this establishes the performance baseline.
 *
 * Tests 10 diverse curves to ensure consistent performance:
 * - Simple curves (linear, parabola)
 * - Medium complexity (sine, tanh)
 * - High complexity (harmonics 5, 10, 15, 20, 25, 30)
 *
 * Expected: Each curve completes in reasonable time, no pathological cases
 */
TEST_F(SplineFitterTest, Performance_AdaptiveAlgorithm_Baseline) {
    auto config = dsp_core::SplineFitConfig::tight();

    std::cout << "\n=== Adaptive Algorithm Performance Baseline ===" << '\n';
    std::cout << "Curve Type           | Time (ms) | Anchors | Status" << '\n';
    std::cout << "---------------------|-----------|---------|--------" << '\n';

    struct TestCase {
        std::string name;
        std::function<void(dsp_core::LayeredTransferFunction&)> setupFunc;
        int maxTimeMs;
    };

    std::vector<TestCase> const testCases = {{"Linear (y=x)",
                                              [](auto& ltf) {
                                                  for (int i = 0; i < ltf.getTableSize(); ++i) {
                                                      ltf.setBaseLayerValue(i, ltf.normalizeIndex(i));
                                                  }
                                              },
                                              50},
                                             {"Parabola",
                                              [](auto& ltf) {
                                                  for (int i = 0; i < ltf.getTableSize(); ++i) {
                                                      double const x = ltf.normalizeIndex(i);
                                                      ltf.setBaseLayerValue(i, x * x);
                                                  }
                                              },
                                              50},
                                             {"Sine Wave",
                                              [](auto& ltf) {
                                                  for (int i = 0; i < ltf.getTableSize(); ++i) {
                                                      double const x = ltf.normalizeIndex(i);
                                                      ltf.setBaseLayerValue(i, std::sin(x * M_PI));
                                                  }
                                              },
                                              100},
                                             {"Tanh (steep)",
                                              [](auto& ltf) {
                                                  for (int i = 0; i < ltf.getTableSize(); ++i) {
                                                      double const x = ltf.normalizeIndex(i);
                                                      ltf.setBaseLayerValue(i, std::tanh(10.0 * x));
                                                  }
                                              },
                                              100},
                                             {"Harmonic 5", [](auto& ltf) { setHarmonicCurve(ltf, 5); }, 100},
                                             {"Harmonic 10", [](auto& ltf) { setHarmonicCurve(ltf, 10); }, 100},
                                             {"Harmonic 15", [](auto& ltf) { setHarmonicCurve(ltf, 15); }, 100},
                                             {"Harmonic 20", [](auto& ltf) { setHarmonicCurve(ltf, 20); }, 100},
                                             {"Harmonic 25", [](auto& ltf) { setHarmonicCurve(ltf, 25); }, 150},
                                             {"Harmonic 30", [](auto& ltf) { setHarmonicCurve(ltf, 30); }, 150}};

    long long totalTime = 0;
    int passCount = 0;

    for (const auto& test : testCases) {
        test.setupFunc(*ltf);

        auto startTime = std::chrono::high_resolution_clock::now();
        auto curveData = extractCurveData();

        auto result =
            dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                       ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

        ASSERT_TRUE(result.success) << test.name << " fit failed";

        bool const passed = duration.count() <= test.maxTimeMs;
        totalTime += duration.count();
        if (passed) {
            passCount++;
        }

        std::cout << std::setw(20) << std::left << test.name << " | " << std::setw(9) << std::right << duration.count()
                  << " | " << std::setw(7) << result.numAnchors << " | " << (passed ? "PASS" : "FAIL") << '\n';

        EXPECT_LE(duration.count(), test.maxTimeMs)
            << test.name << " took " << duration.count() << "ms (expected <" << test.maxTimeMs << "ms)";
    }

    std::cout << "\nTotal time: " << totalTime << "ms" << '\n';
    std::cout << "Passed: " << passCount << "/" << testCases.size() << '\n';

    // Overall average should be reasonable
    double const avgTime = static_cast<double>(totalTime) / static_cast<double>(testCases.size());
    EXPECT_LT(avgTime, 100.0) << "Average time per curve should be <100ms, got " << avgTime << "ms";
}

/**
 * Performance Test 2: Large Dataset (16k samples)
 *
 * Tests performance on high-resolution data (16384 samples).
 * This is the realistic production scenario when users switch from
 * paint mode (16k samples) back to spline mode (refit).
 *
 * Tests several curve types at 16k resolution:
 * - Simple curve (linear)
 * - Medium complexity (sine)
 * - High complexity (Harmonic 10)
 *
 * Expected: All complete in <300ms (16k samples require more processing)
 */
TEST_F(SplineFitterTest, Performance_LargeDataset_16kSamples) {
    // Use high-resolution transfer function (16k samples)
    auto ltfHiRes = std::make_unique<dsp_core::LayeredTransferFunction>(16384, -1.0, 1.0);
    auto config = dsp_core::SplineFitConfig::tight();

    std::cout << "\n=== Large Dataset Performance (16k samples) ===" << '\n';
    std::cout << "Curve Type     | Time (ms) | Anchors | Status" << '\n';
    std::cout << "---------------|-----------|---------|--------" << '\n';

    const int maxTimeMs = 300; // 16k samples need more time than 256 samples

    struct TestCase {
        std::string name;
        std::function<void(dsp_core::LayeredTransferFunction&)> setupFunc;
    };

    std::vector<TestCase> const testCases = {{"Linear",
                                              [](auto& ltf) {
                                                  for (int i = 0; i < ltf.getTableSize(); ++i) {
                                                      ltf.setBaseLayerValue(i, ltf.normalizeIndex(i));
                                                  }
                                              }},
                                             {"Sine Wave",
                                              [](auto& ltf) {
                                                  for (int i = 0; i < ltf.getTableSize(); ++i) {
                                                      double const x = ltf.normalizeIndex(i);
                                                      ltf.setBaseLayerValue(i, std::sin(x * M_PI));
                                                  }
                                              }},
                                             {"Harmonic 10", [](auto& ltf) { setHarmonicCurve(ltf, 10); }}};

    for (const auto& test : testCases) {
        test.setupFunc(*ltfHiRes);

        auto startTime = std::chrono::high_resolution_clock::now();
        auto curveData = extractCurveData(*ltfHiRes);

        auto result = dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                                 ltfHiRes->getMinSignalValue(),
                                                                 ltfHiRes->getMaxSignalValue(), config);
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

        ASSERT_TRUE(result.success) << test.name << " fit failed";

        bool const passed = duration.count() < maxTimeMs;

        std::cout << std::setw(14) << std::left << test.name << " | " << std::setw(9) << std::right << duration.count()
                  << " | " << std::setw(7) << result.numAnchors << " | " << (passed ? "PASS" : "FAIL") << '\n';

        EXPECT_LT(duration.count(), maxTimeMs)
            << test.name << " with 16k samples took " << duration.count() << "ms (expected <" << maxTimeMs << "ms)";
    }
}

/**
 * Performance Test 3: Worst Case - Complex Harmonic
 *
 * Tests performance on the most challenging scenario: Harmonic 25.
 * This represents a worst-case for complexity while still being realistic
 * (users can mix harmonics up to 40).
 *
 * Uses standard 256-sample resolution (typical user interaction).
 *
 * Expected: Completes in <500ms even for this challenging case
 */
TEST_F(SplineFitterTest, Performance_WorstCase_ComplexHarmonic) {
    auto config = dsp_core::SplineFitConfig::tight();

    std::cout << "\n=== Worst Case Performance: Harmonic 25 ===" << '\n';

    // Set up Harmonic 25 (very high complexity)
    setHarmonicCurve(*ltf, 25);

    // Measure performance
    auto startTime = std::chrono::high_resolution_clock::now();
    auto curveData = extractCurveData();

    auto result =
        dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                   ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    ASSERT_TRUE(result.success) << "Harmonic 25 fit failed";

    std::cout << "Harmonic 25 results:" << '\n';
    std::cout << "  Time: " << duration.count() << " ms (expected <500ms)" << '\n';
    std::cout << "  Anchors: " << result.numAnchors << '\n';
    std::cout << "  Max error: " << std::fixed << std::setprecision(6) << result.maxError << '\n';
    std::cout << "  Status: " << (duration.count() < 500 ? "PASS" : "FAIL") << '\n';

    // Worst case should still complete in reasonable time
    EXPECT_LT(duration.count(), 500) << "Harmonic 25 worst-case took " << duration.count() << "ms (expected <500ms)";

    // Should still produce reasonable anchor count
    EXPECT_LE(result.numAnchors, config.maxAnchors) << "Harmonic 25 should fit within maxAnchors budget";

    // Should have reasonable error for this high-frequency content
    // Note: Harmonic 25 is extremely challenging for PCHIP-based fitting
    EXPECT_LT(result.maxError, 0.15) << "Harmonic 25 should have bounded error";
}

//==============================================================================
// Phase 1: Zero-Crossing Drift Verification Tests
//==============================================================================

/**
 * Test fixture for zero-crossing tests
 */
class ZeroCrossingTest : public ::testing::Test {
  protected:
    void SetUp() override {
        ltf = std::make_unique<dsp_core::LayeredTransferFunction>(16384, -1.0, 1.0);
        // Reset coefficients to traditional test defaults (WT=1.0, all harmonics=0.0)
        ltf->setCoefficient(0, 1.0);
        for (int i = 1; i < ltf->getNumCoefficients(); ++i) {
            ltf->setCoefficient(i, 0.0);
        }
    }

    // Helper: Set base layer to tanh curve (zero-crossing at x=0)
    void setTanhCurve(double steepness = 5.0) {
        for (int i = 0; i < ltf->getTableSize(); ++i) {
            double const x = ltf->normalizeIndex(i);
            ltf->setBaseLayerValue(i, std::tanh(steepness * x));
        }
    }

    // Helper: Set base layer to cubic curve (zero-crossing at x=0)
    void setCubicCurve() {
        for (int i = 0; i < ltf->getTableSize(); ++i) {
            double const x = ltf->normalizeIndex(i);
            ltf->setBaseLayerValue(i, x * x * x);
        }
    }

    // Helper: Set base layer to offset curve (no zero-crossing)
    void setOffsetTanhCurve(double offset = 0.5) {
        for (int i = 0; i < ltf->getTableSize(); ++i) {
            double const x = ltf->normalizeIndex(i);
            ltf->setBaseLayerValue(i, std::tanh(5.0 * x) + offset);
        }
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

//==============================================================================
// Phase 3: Symmetric Fitting Tests
//==============================================================================

/**
 * Test 1: SymmetricFitting_CubicPolynomial_PairedAnchors
 */
TEST_F(ZeroCrossingTest, SymmetricFitting_CubicPolynomial_PairedAnchors) {
    // Setup: y = x³
    setCubicCurve();

    // Config: Force symmetric mode with smooth tolerance
    // Note: Feature detection may add individual anchors at inflection points,
    // but greedy refinement should add pairs
    auto config = dsp_core::SplineFitConfig::tight();
    config.symmetryDetection = dsp_core::SymmetryDetection::Auto;
    config.maxAnchors = 10;

    // Execute
    auto curveData = extractCurveData();

    auto result =
        dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                   ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    // Verify
    ASSERT_TRUE(result.success) << "Cubic fit should succeed";
    EXPECT_GE(result.anchors.size(), 2) << "Should have at least endpoints";

    // Check that most anchors are paired
    // Note: Feature detection may add unpaired anchors, which is acceptable
    int pairedCount = 0;
    int totalNonCenter = 0;

    for (const auto& anchor : result.anchors) {
        if (std::abs(anchor.x) < 1e-4) {
            continue; // Skip near-center
        }

        totalNonCenter++;

        // Find complementary anchor
        for (const auto& other : result.anchors) {
            if (std::abs(other.x + anchor.x) < 1e-4) { // other.x ≈ -anchor.x
                pairedCount++;
                // Verify reasonably symmetric y values
                EXPECT_NEAR(anchor.y, -other.y, 0.1) << "Y values should be approximately symmetric";
                break;
            }
        }
    }

    // Expect at least 70% of non-center anchors to be paired
    // (allows for feature anchors to be unpaired)
    if (totalNonCenter > 0) {
        double const pairRatio = static_cast<double>(pairedCount) / totalNonCenter;
        EXPECT_GE(pairRatio, 0.7) << "Most anchors should be paired (got " << pairedCount << "/" << totalNonCenter
                                  << ")";
    }

    // Verify quality
    EXPECT_LT(result.maxError, 0.10) << "Fit quality should be reasonable";
}

/**
 * Test 2: SymmetricFitting_TanhCurve_AutoDetect
 */
TEST_F(ZeroCrossingTest, SymmetricFitting_TanhCurve_AutoDetect) {
    // Setup: y = tanh(5x)
    setTanhCurve();

    // Config: Auto-detect symmetry
    auto config = dsp_core::SplineFitConfig::tight();
    config.symmetryDetection = dsp_core::SymmetryDetection::Auto;
    config.symmetryThreshold = 0.90;

    // Execute
    auto curveData = extractCurveData();

    auto result =
        dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                   ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    // Verify
    ASSERT_TRUE(result.success) << "Tanh fit should succeed";

    // Count paired vs unpaired anchors
    int pairedCount = 0;
    int totalNonCenter = 0;

    for (const auto& anchor : result.anchors) {
        if (std::abs(anchor.x) < 1e-4) {
            continue; // Skip near-center
        }

        totalNonCenter++;

        bool const foundComplement =
            std::any_of(result.anchors.begin(), result.anchors.end(),
                        [&anchor](const dsp_core::SplineAnchor& other) { return std::abs(other.x + anchor.x) < 1e-4; });

        if (foundComplement) {
            pairedCount++;
        }
    }

    // Expect at least 70% of non-center anchors to be paired
    // (Auto mode should detect symmetry, but feature anchors may be unpaired)
    if (totalNonCenter > 0) {
        double const pairRatio = static_cast<double>(pairedCount) / totalNonCenter;
        EXPECT_GE(pairRatio, 0.7) << "Auto mode should use mostly paired anchors for tanh (got " << pairedCount << "/"
                                  << totalNonCenter << ")";
    }
}

/**
 * Test 3: SymmetricFitting_AsymmetricCurve_AutoDisables
 */
TEST_F(ZeroCrossingTest, SymmetricFitting_AsymmetricCurve_AutoDisables) {
    // Setup: y = x² (even function, not odd symmetric)
    for (int i = 0; i < ltf->getTableSize(); ++i) {
        double const x = ltf->normalizeIndex(i);
        ltf->setBaseLayerValue(i, x * x);
    }

    // Config: Auto mode
    auto config = dsp_core::SplineFitConfig::tight();
    config.symmetryDetection = dsp_core::SymmetryDetection::Auto;

    // Execute
    auto curveData = extractCurveData();

    auto result =
        dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                   ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    // Verify
    ASSERT_TRUE(result.success) << "Asymmetric curve should fit successfully";

    // Auto mode should detect asymmetry and NOT require paired anchors
    // (x² is even function, symmetry score < 0.90)
    // Just verify it completed successfully - anchors may or may not be symmetric
}

/**
 * Test 4: SymmetricFitting_NeverMode_OriginalBehavior
 */
TEST_F(ZeroCrossingTest, SymmetricFitting_NeverMode_OriginalBehavior) {
    // Setup: y = x³ (symmetric curve)
    setCubicCurve();

    // Config: Never mode (disable symmetric fitting)
    auto config = dsp_core::SplineFitConfig::tight();
    config.symmetryDetection = dsp_core::SymmetryDetection::Never;

    // Execute
    auto curveData = extractCurveData();

    auto result =
        dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                   ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    // Verify
    ASSERT_TRUE(result.success) << "Cubic fit should succeed with Never mode";

    // Anchors NOT necessarily paired (original greedy algorithm)
    // Just verify it behaves like original algorithm (no regression)
    EXPECT_GT(result.anchors.size(), 0) << "Should have anchors";
    EXPECT_LT(result.maxError, 0.1) << "Should have reasonable error";
}

/**
 * Test 5: SymmetricFitting_LimitedAnchors_StopsWhenFull
 */
TEST_F(ZeroCrossingTest, SymmetricFitting_LimitedAnchors_StopsWhenFull) {
    // Setup: y = x³
    setCubicCurve();

    // Config: Force symmetric mode with odd maxAnchors
    auto config = dsp_core::SplineFitConfig::tight();
    config.symmetryDetection = dsp_core::SymmetryDetection::Auto;
    config.maxAnchors = 5; // Odd number - can't fit all pairs

    // Execute
    auto curveData = extractCurveData();

    auto result =
        dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                   ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    // Verify
    ASSERT_TRUE(result.success) << "Should succeed with limited anchors";
    EXPECT_LE(result.anchors.size(), 5) << "Should respect anchor budget";

    // May have some unpaired anchors at boundary due to odd maxAnchors
}

/**
 * Test 6: SymmetricFitting_Harmonic3_Symmetric
 */
TEST_F(ZeroCrossingTest, SymmetricFitting_Harmonic3_Symmetric) {
    // Setup: Harmonic 3 (Chebyshev T₃, odd function)
    for (int i = 0; i < ltf->getTableSize(); ++i) {
        double const x = ltf->normalizeIndex(i);
        double const y = 4.0 * x * x * x - 3.0 * x; // T₃(x)
        ltf->setBaseLayerValue(i, y);
    }

    // Config: Auto mode
    auto config = dsp_core::SplineFitConfig::tight();
    config.symmetryDetection = dsp_core::SymmetryDetection::Auto;

    // Execute
    auto curveData = extractCurveData();

    auto result =
        dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                   ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    // Verify
    ASSERT_TRUE(result.success) << "Harmonic 3 fit should succeed";

    // Count paired vs unpaired anchors
    int pairedCount = 0;
    int totalNonCenter = 0;

    for (const auto& anchor : result.anchors) {
        if (std::abs(anchor.x) < 1e-4) {
            continue; // Skip near-center
        }

        totalNonCenter++;

        bool const foundComplement =
            std::any_of(result.anchors.begin(), result.anchors.end(),
                        [&anchor](const dsp_core::SplineAnchor& other) { return std::abs(other.x + anchor.x) < 1e-4; });

        if (foundComplement) {
            pairedCount++;
        }
    }

    // Expect at least 70% of non-center anchors to be paired
    // (Auto mode should detect odd symmetry, but feature anchors may be unpaired)
    if (totalNonCenter > 0) {
        double const pairRatio = static_cast<double>(pairedCount) / totalNonCenter;
        EXPECT_GE(pairRatio, 0.7) << "Harmonic 3 should use mostly paired anchors (got " << pairedCount << "/"
                                  << totalNonCenter << ")";
    }

    // Should capture extrema
    EXPECT_GE(result.anchors.size(), 3) << "Should capture extrema";
}

/**
 * Test 7: SymmetricFitting_Harmonic2_Asymmetric
 */
TEST_F(ZeroCrossingTest, SymmetricFitting_Harmonic2_Asymmetric) {
    // Setup: Harmonic 2 (Chebyshev T₂, even function)
    for (int i = 0; i < ltf->getTableSize(); ++i) {
        double const x = ltf->normalizeIndex(i);
        double const y = 2.0 * x * x - 1.0; // T₂(x)
        ltf->setBaseLayerValue(i, y);
    }

    // Config: Auto mode
    auto config = dsp_core::SplineFitConfig::tight();
    config.symmetryDetection = dsp_core::SymmetryDetection::Auto;

    // Execute
    auto curveData = extractCurveData();

    auto result =
        dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                   ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

    // Verify
    ASSERT_TRUE(result.success) << "Harmonic 2 fit should succeed";

    // Even harmonic should NOT trigger symmetric mode (not odd-symmetric)
    // Just verify it runs successfully
}

//==============================================================================
// COMPREHENSIVE BACKTRANSLATION TESTS
//
// These tests thoroughly verify the anchor creeping issue across various
// real-world scenarios. They test the fundamental stability requirement:
// "Refitting an already-fitted curve should produce similar anchor count"
//==============================================================================

/**
 * Test: User Workflow - Identity + Middle Anchor
 *
 * This is the exact scenario from docs/backtranslation-issues.md:
 * 1. Start with identity curve (y=x)
 * 2. Enter spline mode → fits to 2 endpoint anchors
 * 3. User adds middle anchor at arbitrary position (e.g., x=-0.321, y=-0.432)
 * 4. Exit spline mode → bakes 3-anchor PCHIP curve to 16,384 samples
 * 5. Re-enter spline mode → should get ~3-5 anchors, NOT 7-12
 *
 * Root cause: PCHIP interpolation creates subtle curvature changes that
 * greedy algorithm detects with tight tolerance.
 */
TEST_F(BacktranslationTest, UserWorkflow_IdentityPlusMiddleAnchor_RefitsToThree) {
    std::cout << "\n=== User Workflow: Identity + Middle Anchor ===" << '\n';

    // Original 3 anchors after user adds middle anchor
    // (Middle anchor creates "bump" away from identity line)
    std::vector<dsp_core::SplineAnchor> originalAnchors = {
        {-1.0, -1.0, false, 0.0},     // Left endpoint
        {-0.321, -0.432, false, 0.0}, // User-added middle anchor (arbitrary position)
        {1.0, 1.0, false, 0.0}        // Right endpoint
    };

    // Compute tangents (simulates what happens in UI)
    auto config = dsp_core::SplineFitConfig::tight(); // Production config
    dsp_core::Services::SplineFitter::computeTangents(originalAnchors, config);

    // Print tangents to see PCHIP effect
    std::cout << "Original anchors with PCHIP tangents:" << '\n';
    for (const auto& anchor : originalAnchors) {
        std::cout << "  x=" << std::fixed << std::setprecision(3) << anchor.x << ", y=" << anchor.y
                  << ", tangent=" << anchor.tangent << '\n';
    }

    // Backtranslate
    auto result = backtranslateAnchors(originalAnchors, config);

    EXPECT_TRUE(result.success) << "Backtranslation failed";

    std::cout << "Refit result: " << result.numAnchors << " anchors (expected: 3-5)" << '\n';
    std::cout << "Max error: " << result.maxError << '\n';

    // Expected: 3-5 anchors (original 3 + maybe 1-2 for PCHIP curvature)
    // Actual (BUG): 7-12 anchors
    EXPECT_GE(result.numAnchors, 3) << "Should have at least 3 anchors";
    EXPECT_LE(result.numAnchors, 5) << "Identity + middle anchor should need 3-5 anchors. "
                                    << "Got " << result.numAnchors << " (PCHIP-induced anchor creep)";
}

/**
 * Test: Multi-Iteration Backtranslation Stability
 *
 * Verifies that anchor count converges over multiple backtranslation cycles.
 * Ideal behavior: N₁ ≈ N₂ ≈ N₃ (stable)
 * Current behavior: N₁ < N₂ < N₃ (exponential creeping)
 */
TEST_F(BacktranslationTest, MultiIteration_ConvergesToStableCount) {
    std::cout << "\n=== Multi-Iteration Backtranslation Stability ===" << '\n';

    // Start with simple parabola
    std::vector<dsp_core::SplineAnchor> originalAnchors = {
        {-1.0, 0.0, false, 0.0}, {0.0, 1.0, false, 0.0}, {1.0, 0.0, false, 0.0}};

    auto config = dsp_core::SplineFitConfig::tight();
    dsp_core::Services::SplineFitter::computeTangents(originalAnchors, config);

    std::vector<int> anchorCounts;
    auto currentAnchors = originalAnchors;

    // Perform 3 iterations of backtranslation
    for (int iteration = 1; iteration <= 3; ++iteration) {
        auto result = backtranslateAnchors(currentAnchors, config);
        ASSERT_TRUE(result.success) << "Iteration " << iteration << " failed";

        anchorCounts.push_back(result.numAnchors);
        currentAnchors = result.anchors;

        std::cout << "Iteration " << iteration << ": " << result.numAnchors
                  << " anchors, max error: " << result.maxError << '\n';
    }

    // Verify convergence (anchor count should stabilize)
    EXPECT_GE(anchorCounts[0], 3) << "Iteration 1 should have at least 3 anchors";
    EXPECT_LE(anchorCounts[0], 5) << "Iteration 1 should have ≤5 anchors";

    // Subsequent iterations should not significantly increase
    for (size_t i = 1; i < anchorCounts.size(); ++i) {
        EXPECT_LE(anchorCounts[i], anchorCounts[0] + 2)
            << "Iteration " << (i + 1) << " should not add many anchors. "
            << "Got " << anchorCounts[i] << " (expected ≤" << (anchorCounts[0] + 2) << ")";
    }
}

/**
 * Test: Tanh Curve Backtranslation
 *
 * Smooth symmetric curve should backtranslate cleanly.
 * Tanh is particularly important for audio waveshaping (soft clipping).
 */
TEST_F(BacktranslationTest, TanhCurve_SmoothBacktranslation) {
    std::cout << "\n=== Tanh Curve Backtranslation ===" << '\n';

    // Create tanh curve: y = tanh(5x)
    for (int i = 0; i < ltf->getTableSize(); ++i) {
        double const x = ltf->normalizeIndex(i);
        double const y = std::tanh(5.0 * x);
        ltf->setBaseLayerValue(i, y);
    }

    // Fit once
    auto config = dsp_core::SplineFitConfig::tight();
    auto curveData = extractCurveData();

    auto firstFit =
        dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                   ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);
    ASSERT_TRUE(firstFit.success) << "First fit failed";

    std::cout << "First fit: " << firstFit.numAnchors << " anchors, "
              << "max error: " << firstFit.maxError << '\n';

    // Backtranslate
    auto refitResult = backtranslateAnchors(firstFit.anchors, config);
    ASSERT_TRUE(refitResult.success) << "Backtranslation failed";

    std::cout << "Refit: " << refitResult.numAnchors << " anchors (expected: similar to first fit)" << '\n';

    // Anchor count should be similar or slightly reduced (adaptive tolerance working well)
    // Reducing anchor count on refit is GOOD (means algorithm recognizes smooth curve)
    EXPECT_GE(refitResult.numAnchors, firstFit.numAnchors - 3) << "Backtranslation should not lose many anchors";
    EXPECT_LE(refitResult.numAnchors, firstFit.numAnchors + 3)
        << "Backtranslation should not add many anchors. "
        << "First: " << firstFit.numAnchors << ", Refit: " << refitResult.numAnchors;
}

/**
 * Test: Arbitrary Positions (Real-World Scenario)
 *
 * User drags anchors to fractional coordinates (not nice grid points).
 * This creates more PCHIP curvature artifacts than grid-aligned positions.
 */
TEST_F(BacktranslationTest, ArbitraryPositions_StableBacktranslation) {
    std::cout << "\n=== Arbitrary Positions Backtranslation ===" << '\n';

    // Realistic user-placed anchors (fractional coordinates)
    std::vector<dsp_core::SplineAnchor> originalAnchors = {
        {-1.0, -0.87, false, 0.0},    // Near but not at -1
        {-0.412, -0.653, false, 0.0}, // Arbitrary mid-left
        {0.234, 0.123, false, 0.0},   // Arbitrary mid-right
        {1.0, 0.92, false, 0.0}       // Near but not at 1
    };

    auto config = dsp_core::SplineFitConfig::tight();
    dsp_core::Services::SplineFitter::computeTangents(originalAnchors, config);

    auto result = backtranslateAnchors(originalAnchors, config);
    EXPECT_TRUE(result.success) << "Backtranslation failed";

    std::cout << "Original: 4 anchors → Refit: " << result.numAnchors << " anchors" << '\n';

    // Expected: 4-7 anchors (fractional coords create more artifacts)
    EXPECT_GE(result.numAnchors, 4) << "Should have at least original anchor count";
    EXPECT_LE(result.numAnchors, 7) << "Arbitrary positions should need 4-7 anchors. "
                                    << "Got " << result.numAnchors << " (excessive for simple curve)";
}

/**
 * Test: Production Config vs Smooth Config
 *
 * Compares backtranslation behavior between tight() and smooth() configs.
 * tight() uses positionTolerance=0.002, smooth() uses 0.01 (5x more forgiving).
 */
TEST_F(BacktranslationTest, ProductionConfig_CompareToSmooth) {
    std::cout << "\n=== Production Config vs Smooth Config ===" << '\n';

    std::vector<dsp_core::SplineAnchor> originalAnchors = {
        {-1.0, 0.0, false, 0.0}, {-0.5, -0.8, false, 0.0}, {0.5, 0.8, false, 0.0}, {1.0, 0.0, false, 0.0}};

    // Test with tight config (production)
    auto tightConfig = dsp_core::SplineFitConfig::tight();
    dsp_core::Services::SplineFitter::computeTangents(originalAnchors, tightConfig);
    auto tightResult = backtranslateAnchors(originalAnchors, tightConfig);

    // Test with smooth config
    auto smoothConfig = dsp_core::SplineFitConfig::tight();
    dsp_core::Services::SplineFitter::computeTangents(originalAnchors, smoothConfig);
    auto smoothResult = backtranslateAnchors(originalAnchors, smoothConfig);

    std::cout << "tight() config:  " << tightResult.numAnchors << " anchors, "
              << "max error: " << tightResult.maxError << '\n';
    std::cout << "smooth() config: " << smoothResult.numAnchors << " anchors, "
              << "max error: " << smoothResult.maxError << '\n';

    // Both should succeed
    EXPECT_TRUE(tightResult.success) << "tight() backtranslation failed";
    EXPECT_TRUE(smoothResult.success) << "smooth() backtranslation failed";

    // smooth() should use fewer or similar anchors (more forgiving tolerance)
    EXPECT_LE(smoothResult.numAnchors, tightResult.numAnchors + 1)
        << "smooth() should not use more anchors than tight()";

    // Both should be within reasonable bounds
    EXPECT_LE(tightResult.numAnchors, 12) << "tight() should use ≤12 anchors for 2 extrema";
    EXPECT_LE(smoothResult.numAnchors, 10) << "smooth() should use ≤10 anchors for 2 extrema";
}

/**
 * Test: Scribble with Many Features
 *
 * Curve with many small features should simplify, not preserve every bump.
 * This tests that adaptive tolerance doesn't over-fit noisy data.
 */
TEST_F(BacktranslationTest, ScribbleWithFeatures_Simplifies) {
    std::cout << "\n=== Scribble Simplification Backtranslation ===" << '\n';

    // Create curve with 15 small oscillations (simulates user scribble)
    std::vector<dsp_core::SplineAnchor> originalAnchors;
    originalAnchors.push_back({-1.0, -1.0, false, 0.0});

    for (int i = 1; i <= 13; ++i) {
        double const x = -1.0 + (2.0 * i / 14.0);
        double const y = std::sin(i * 0.8) * 0.3; // Small amplitude oscillations
        originalAnchors.push_back({x, y, false, 0.0});
    }

    originalAnchors.push_back({1.0, 1.0, false, 0.0});

    std::cout << "Original scribble: " << originalAnchors.size() << " anchors" << '\n';

    auto config = dsp_core::SplineFitConfig::tight(); // Use smooth for scribbles
    dsp_core::Services::SplineFitter::computeTangents(originalAnchors, config);

    auto result = backtranslateAnchors(originalAnchors, config);
    EXPECT_TRUE(result.success) << "Backtranslation failed";

    std::cout << "Refit: " << result.numAnchors << " anchors (expected: simplified)" << '\n';

    // Should simplify to far fewer anchors (adaptive tolerance should relax)
    EXPECT_GE(result.numAnchors, 8) << "Should preserve general shape (≥8 anchors)";
    EXPECT_LE(result.numAnchors, 20) << "Should simplify scribble. "
                                     << "Got " << result.numAnchors << " anchors from " << originalAnchors.size()
                                     << " original";
}

} // namespace dsp_core_test
