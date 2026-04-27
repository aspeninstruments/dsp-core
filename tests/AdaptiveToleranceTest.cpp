#include <dsp_core/dsp_core.h>
#include <gtest/gtest.h>
#include <cmath>

namespace dsp_core_test {

/**
 * Test fixture for AdaptiveToleranceCalculator service
 * Tests adaptive tolerance calculation that scales with anchor density
 */
class AdaptiveToleranceTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Configure test defaults explicitly (independent of production defaults)
        // These values match the original test expectations
        testConfig.relativeErrorTarget = 0.01;    // 1% error target
        testConfig.anchorDensityMultiplier = 2.0; // Original default multiplier
    }

    // Explicit test config (not relying on production defaults)
    dsp_core::Services::AdaptiveToleranceCalculator::Config testConfig;
};

// ============================================================================
// Task 3: Adaptive Tolerance Calculator Tests
// ============================================================================

/**
 * Test: Zero anchors should return baseline tolerance
 * Expected: No scaling applied, tolerance = verticalRange × relativeErrorTarget
 */
TEST_F(AdaptiveToleranceTest, ComputeTolerance_ZeroAnchors_ReturnsBaseline) {
    double const verticalRange = 2.0; // Typical range [-1, 1]
    int const currentAnchors = 0;
    int const maxAnchors = 64;

    double const tolerance = dsp_core::Services::AdaptiveToleranceCalculator::computeTolerance(
        verticalRange, currentAnchors, maxAnchors, testConfig);

    // Expected: 2.0 × 0.01 = 0.02
    double const expectedBaseline = 2.0 * 0.01;
    EXPECT_DOUBLE_EQ(tolerance, expectedBaseline) << "Zero anchors should return baseline tolerance";
}

/**
 * Test: Half capacity should return doubled tolerance
 * Expected: tolerance = baseline × (1 + 0.5 × 2.0) = 2× baseline
 */
TEST_F(AdaptiveToleranceTest, ComputeTolerance_HalfCapacity_ReturnsDoubled) {
    double const verticalRange = 2.0;
    int const currentAnchors = 32; // 50% of maxAnchors
    int const maxAnchors = 64;

    double const tolerance = dsp_core::Services::AdaptiveToleranceCalculator::computeTolerance(
        verticalRange, currentAnchors, maxAnchors, testConfig);

    // Expected: baseline × (1 + 0.5 × 2.0) = baseline × 2.0
    double const expectedBaseline = 2.0 * 0.01;
    double const expectedTolerance = expectedBaseline * 2.0;

    EXPECT_DOUBLE_EQ(tolerance, expectedTolerance) << "Half capacity should double the tolerance";
}

/**
 * Test: Full capacity should return tripled tolerance
 * Expected: tolerance = baseline × (1 + 1.0 × 2.0) = 3× baseline
 */
TEST_F(AdaptiveToleranceTest, ComputeTolerance_FullCapacity_ReturnsTripled) {
    double const verticalRange = 2.0;
    int const currentAnchors = 64; // 100% of maxAnchors
    int const maxAnchors = 64;

    double const tolerance = dsp_core::Services::AdaptiveToleranceCalculator::computeTolerance(
        verticalRange, currentAnchors, maxAnchors, testConfig);

    // Expected: baseline × (1 + 1.0 × 2.0) = baseline × 3.0
    double const expectedBaseline = 2.0 * 0.01;
    double const expectedTolerance = expectedBaseline * 3.0;

    EXPECT_DOUBLE_EQ(tolerance, expectedTolerance) << "Full capacity should triple the tolerance";
}

/**
 * Test: Tolerance scales linearly with vertical range
 * Expected: Larger range → proportionally larger tolerance
 */
TEST_F(AdaptiveToleranceTest, ComputeTolerance_LargeVerticalRange_ScalesWithRange) {
    double const smallRange = 1.0;
    double const largeRange = 10.0; // 10× larger
    int const currentAnchors = 32;
    int const maxAnchors = 64;

    double const smallTolerance = dsp_core::Services::AdaptiveToleranceCalculator::computeTolerance(
        smallRange, currentAnchors, maxAnchors, testConfig);

    double const largeTolerance = dsp_core::Services::AdaptiveToleranceCalculator::computeTolerance(
        largeRange, currentAnchors, maxAnchors, testConfig);

    // Large tolerance should be 10× the small tolerance
    EXPECT_DOUBLE_EQ(largeTolerance, smallTolerance * 10.0) << "Tolerance should scale linearly with vertical range";
}

/**
 * Test: Small vertical range produces small tolerance
 * Expected: Smaller range → proportionally smaller tolerance
 */
TEST_F(AdaptiveToleranceTest, ComputeTolerance_SmallVerticalRange_ScalesWithRange) {
    double const tinyRange = 0.1; // Very small range
    int const currentAnchors = 0;
    int const maxAnchors = 64;

    double const tolerance = dsp_core::Services::AdaptiveToleranceCalculator::computeTolerance(
        tinyRange, currentAnchors, maxAnchors, testConfig);

    // Expected: 0.1 × 0.01 = 0.001
    double const expectedTolerance = 0.1 * 0.01;
    EXPECT_DOUBLE_EQ(tolerance, expectedTolerance) << "Small range should produce small tolerance";
}

/**
 * Test: Edge case - negative anchors (should clamp to zero)
 */
TEST_F(AdaptiveToleranceTest, ComputeTolerance_NegativeAnchors_ClampsToZero) {
    double const verticalRange = 2.0;
    int const currentAnchors = -10; // Invalid negative value
    int const maxAnchors = 64;

    double const tolerance = dsp_core::Services::AdaptiveToleranceCalculator::computeTolerance(
        verticalRange, currentAnchors, maxAnchors, testConfig);

    // Should treat as zero anchors
    double const expectedBaseline = 2.0 * 0.01;
    EXPECT_DOUBLE_EQ(tolerance, expectedBaseline) << "Negative anchors should be clamped to zero";
}

/**
 * Test: Edge case - anchors exceed maximum (should clamp to 100%)
 */
TEST_F(AdaptiveToleranceTest, ComputeTolerance_ExcessAnchors_ClampsToMax) {
    double const verticalRange = 2.0;
    int const currentAnchors = 100; // Exceeds maxAnchors
    int const maxAnchors = 64;

    double const tolerance = dsp_core::Services::AdaptiveToleranceCalculator::computeTolerance(
        verticalRange, currentAnchors, maxAnchors, testConfig);

    // Should treat as 100% capacity (not 156%)
    double const expectedBaseline = 2.0 * 0.01;
    double const expectedTolerance = expectedBaseline * 3.0; // Same as 100% capacity

    EXPECT_DOUBLE_EQ(tolerance, expectedTolerance) << "Anchors exceeding maximum should clamp to 100% capacity";
}

/**
 * Test: Edge case - zero maxAnchors (should return baseline)
 */
TEST_F(AdaptiveToleranceTest, ComputeTolerance_ZeroMaxAnchors_ReturnsBaseline) {
    double const verticalRange = 2.0;
    int const currentAnchors = 10;
    int const maxAnchors = 0; // Invalid

    double const tolerance = dsp_core::Services::AdaptiveToleranceCalculator::computeTolerance(
        verticalRange, currentAnchors, maxAnchors, testConfig);

    // Should fallback to baseline tolerance
    double const expectedBaseline = 2.0 * 0.01;
    EXPECT_DOUBLE_EQ(tolerance, expectedBaseline) << "Zero maxAnchors should return baseline tolerance";
}

/**
 * Test: Custom config with different multiplier
 */
TEST_F(AdaptiveToleranceTest, ComputeTolerance_CustomMultiplier_AppliesCorrectly) {
    dsp_core::Services::AdaptiveToleranceCalculator::Config customConfig;
    customConfig.relativeErrorTarget = 0.01;
    customConfig.anchorDensityMultiplier = 4.0; // More aggressive scaling

    double const verticalRange = 2.0;
    int const currentAnchors = 32; // 50% capacity
    int const maxAnchors = 64;

    double const tolerance = dsp_core::Services::AdaptiveToleranceCalculator::computeTolerance(
        verticalRange, currentAnchors, maxAnchors, customConfig);

    // Expected: baseline × (1 + 0.5 × 4.0) = baseline × 3.0
    double const expectedBaseline = 2.0 * 0.01;
    double const expectedTolerance = expectedBaseline * 3.0;

    EXPECT_DOUBLE_EQ(tolerance, expectedTolerance) << "Custom multiplier should apply correctly";
}

/**
 * Test: Custom config with different relative error target
 */
TEST_F(AdaptiveToleranceTest, ComputeTolerance_CustomErrorTarget_AppliesCorrectly) {
    dsp_core::Services::AdaptiveToleranceCalculator::Config customConfig;
    customConfig.relativeErrorTarget = 0.05; // 5% instead of 1%
    customConfig.anchorDensityMultiplier = 2.0;

    double const verticalRange = 2.0;
    int const currentAnchors = 0;
    int const maxAnchors = 64;

    double const tolerance = dsp_core::Services::AdaptiveToleranceCalculator::computeTolerance(
        verticalRange, currentAnchors, maxAnchors, customConfig);

    // Expected: 2.0 × 0.05 = 0.1
    double const expectedTolerance = 2.0 * 0.05;

    EXPECT_DOUBLE_EQ(tolerance, expectedTolerance) << "Custom error target should apply correctly";
}

} // namespace dsp_core_test
