#include "../dsp_core/Source/Services/AnchorConstraintService.h"
#include <dsp_core/dsp_core.h>
#include <gtest/gtest.h>

namespace dsp_core_test {

using namespace dsp_core::Services;

/**
 * Test fixture for AnchorConstraintService
 * Tests the two-pass constraint propagation algorithm
 */
class AnchorConstraintServiceTest : public ::testing::Test {
  protected:
    // Create a simple anchor set for testing
    static std::vector<dsp_core::SplineAnchor> createAnchors(const std::vector<double>& xPositions) {
        std::vector<dsp_core::SplineAnchor> anchors;
        for (double const x : xPositions) {
            dsp_core::SplineAnchor anchor;
            anchor.x = x;
            anchor.y = 0.0; // Y position doesn't affect X constraints
            anchors.push_back(anchor);
        }
        return anchors;
    }
};

// ============================================================================
// Basic Constraint Tests
// ============================================================================

TEST_F(AnchorConstraintServiceTest, SingleAnchorIsEdgeLocked) {
    // Given: One anchor at x=0.0 (it's both first AND last)
    auto anchors = createAnchors({0.0});
    std::set<int> const selected = {0};
    std::map<int, AnchorConstraintService::Point> const ideal = {{0, {0.5, 0.0}}};

    // When: Project ideal to actual
    auto result = AnchorConstraintService::projectIdealToActual(ideal, anchors, selected);

    // Then: Since it's the last (and only) anchor, it gets locked at +1.0
    // (Pass 2 wins over Pass 1's -1.0)
    EXPECT_NEAR(result.actualX.at(0), 1.0, 1e-9);
}

TEST_F(AnchorConstraintServiceTest, LeftEdgeAnchorLocked) {
    // Given: First anchor (index 0) is selected
    auto anchors = createAnchors({-1.0, -0.5, 0.0, 0.5, 1.0});
    std::set<int> const selected = {0};
    std::map<int, AnchorConstraintService::Point> const ideal = {{0, {0.5, 0.0}}}; // Try to move to 0.5

    // When: Project
    auto result = AnchorConstraintService::projectIdealToActual(ideal, anchors, selected);

    // Then: Left edge anchor should be locked at -1.0
    EXPECT_NEAR(result.actualX.at(0), -1.0, 1e-9);
}

TEST_F(AnchorConstraintServiceTest, RightEdgeAnchorLocked) {
    // Given: Last anchor is selected
    auto anchors = createAnchors({-1.0, -0.5, 0.0, 0.5, 1.0});
    std::set<int> const selected = {4};
    std::map<int, AnchorConstraintService::Point> const ideal = {{4, {0.5, 0.0}}}; // Try to move to 0.5

    // When: Project
    auto result = AnchorConstraintService::projectIdealToActual(ideal, anchors, selected);

    // Then: Right edge anchor should be locked at +1.0
    EXPECT_NEAR(result.actualX.at(4), 1.0, 1e-9);
}

TEST_F(AnchorConstraintServiceTest, AnchorCannotCrossLeftNeighbor) {
    // Given: Middle anchor trying to cross left neighbor
    auto anchors = createAnchors({-1.0, -0.5, 0.0, 0.5, 1.0});
    std::set<int> const selected = {2};
    std::map<int, AnchorConstraintService::Point> const ideal = {{2, {-0.8, 0.0}}}; // Try to go left of index 1

    // When: Project
    auto result = AnchorConstraintService::projectIdealToActual(ideal, anchors, selected);

    // Then: Should stop at left neighbor + minGap
    EXPECT_GT(result.actualX.at(2), anchors[1].x);
    EXPECT_NEAR(result.actualX.at(2), -0.5 + 0.001, 1e-9);
}

TEST_F(AnchorConstraintServiceTest, AnchorCannotCrossRightNeighbor) {
    // Given: Middle anchor trying to cross right neighbor
    auto anchors = createAnchors({-1.0, -0.5, 0.0, 0.5, 1.0});
    std::set<int> const selected = {2};
    std::map<int, AnchorConstraintService::Point> const ideal = {{2, {0.8, 0.0}}}; // Try to go right of index 3

    // When: Project
    auto result = AnchorConstraintService::projectIdealToActual(ideal, anchors, selected);

    // Then: Should stop at right neighbor - minGap
    EXPECT_LT(result.actualX.at(2), anchors[3].x);
    EXPECT_NEAR(result.actualX.at(2), 0.5 - 0.001, 1e-9);
}

// ============================================================================
// Batch Drag Tests
// ============================================================================

TEST_F(AnchorConstraintServiceTest, BatchDragMaintainsSpacingWhenUnconstrained) {
    // Given: Two adjacent anchors with 0.5 spacing
    auto anchors = createAnchors({-1.0, -0.3, 0.2, 1.0});
    std::set<int> const selected = {1, 2}; // Select anchors at -0.3 and 0.2
    std::map<int, AnchorConstraintService::Point> const ideal = {
        {1, {-0.1, 0.0}}, // Move right by 0.2
        {2, {0.4, 0.0}}   // Move right by 0.2
    };

    // When: Project
    auto result = AnchorConstraintService::projectIdealToActual(ideal, anchors, selected);

    // Then: Both should move freely, maintaining spacing
    EXPECT_NEAR(result.actualX.at(1), -0.1, 1e-9);
    EXPECT_NEAR(result.actualX.at(2), 0.4, 1e-9);
    EXPECT_NEAR(result.actualX.at(2) - result.actualX.at(1), 0.5, 1e-9); // Spacing preserved
}

TEST_F(AnchorConstraintServiceTest, BatchDragCompressesWhenHittingWall) {
    // Given: Two anchors being pushed against right boundary
    auto anchors = createAnchors({-1.0, 0.0, 0.5, 1.0});
    std::set<int> const selected = {1, 2}; // Select middle anchors
    std::map<int, AnchorConstraintService::Point> const ideal = {
        {1, {1.05, 0.0}}, // Try to push past right edge
        {2, {1.10, 0.0}}  // Try to push past right edge
    };

    // When: Project
    auto result = AnchorConstraintService::projectIdealToActual(ideal, anchors, selected);

    // Then: Anchors should compress against right neighbor (1.0)
    EXPECT_LT(result.actualX.at(2), 1.0);
    EXPECT_LT(result.actualX.at(1), result.actualX.at(2));
    // Anchor 2 should be at right neighbor - minGap
    EXPECT_NEAR(result.actualX.at(2), 1.0 - 0.001, 1e-9);
    // Anchor 1 should be at anchor 2 - minGap (two-pass constraint propagation)
    EXPECT_NEAR(result.actualX.at(1), 1.0 - 0.002, 1e-9);
}

TEST_F(AnchorConstraintServiceTest, BatchDragCompressesAgainstNonSelectedNeighbor) {
    // Given: Anchors with non-selected neighbor in the way
    auto anchors = createAnchors({-1.0, 0.0, 0.4, 0.6, 1.0});
    std::set<int> const selected = {1, 2};                              // Select 0.0 and 0.4, but not 0.6
    std::map<int, AnchorConstraintService::Point> const ideal = {{1, {0.55, 0.0}}, // Try to push past anchor 3 at 0.6
                                                      {2, {0.58, 0.0}}};

    // When: Project
    auto result = AnchorConstraintService::projectIdealToActual(ideal, anchors, selected);

    // Then: Both should stop before non-selected anchor at 0.6
    EXPECT_LT(result.actualX.at(2), 0.6);
    EXPECT_LT(result.actualX.at(1), result.actualX.at(2));
}

// ============================================================================
// Y Coordinate Tests
// ============================================================================

TEST_F(AnchorConstraintServiceTest, YCoordinateClamped) {
    // Given: Anchor with Y outside [-1, 1]
    auto anchors = createAnchors({0.0});
    std::set<int> const selected = {0};
    std::map<int, AnchorConstraintService::Point> const ideal = {{0, {0.0, 1.5}}}; // Y > 1.0

    // When: Project
    auto result = AnchorConstraintService::projectIdealToActual(ideal, anchors, selected);

    // Then: Y should be clamped to 1.0
    EXPECT_NEAR(result.actualY.at(0), 1.0, 1e-9);
}

TEST_F(AnchorConstraintServiceTest, YCoordinateClampedNegative) {
    // Given: Anchor with Y < -1
    auto anchors = createAnchors({0.0});
    std::set<int> const selected = {0};
    std::map<int, AnchorConstraintService::Point> const ideal = {{0, {0.0, -1.5}}}; // Y < -1.0

    // When: Project
    auto result = AnchorConstraintService::projectIdealToActual(ideal, anchors, selected);

    // Then: Y should be clamped to -1.0
    EXPECT_NEAR(result.actualY.at(0), -1.0, 1e-9);
}

// ============================================================================
// Custom MinGap Tests
// ============================================================================

TEST_F(AnchorConstraintServiceTest, CustomMinGapRespected) {
    // Given: Anchor that would violate left neighbor with custom minGap
    auto anchors = createAnchors({-1.0, -0.5, 0.0, 0.5, 1.0});
    std::set<int> const selected = {2};
    // Position -0.46 is within customMinGap of left neighbor at -0.5
    std::map<int, AnchorConstraintService::Point> const ideal = {{2, {-0.46, 0.0}}};
    const double customMinGap = 0.05;

    // When: Project with custom gap
    auto result = AnchorConstraintService::projectIdealToActual(ideal, anchors, selected, customMinGap);

    // Then: Should be pushed to at least left neighbor + customMinGap
    EXPECT_GE(result.actualX.at(2), -0.5 + customMinGap - 1e-9);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(AnchorConstraintServiceTest, EmptySelectionReturnsEmpty) {
    // Given: No anchors selected
    auto anchors = createAnchors({-1.0, 0.0, 1.0});
    std::set<int> const selected = {};
    std::map<int, AnchorConstraintService::Point> const ideal = {};

    // When: Project
    auto result = AnchorConstraintService::projectIdealToActual(ideal, anchors, selected);

    // Then: Should return empty result
    EXPECT_TRUE(result.actualX.empty());
    EXPECT_TRUE(result.actualY.empty());
}

TEST_F(AnchorConstraintServiceTest, AllAnchorsSelectedCanStillMove) {
    // Given: All anchors selected
    auto anchors = createAnchors({-1.0, 0.0, 1.0});
    std::set<int> const selected = {0, 1, 2};
    std::map<int, AnchorConstraintService::Point> const ideal = {
        {0, {-0.9, 0.0}}, // Try to move right
        {1, {0.1, 0.0}},
        {2, {1.1, 0.0}} // Try to move right (will be clamped)
    };

    // When: Project
    auto result = AnchorConstraintService::projectIdealToActual(ideal, anchors, selected);

    // Then: Edge anchors locked, middle can move
    EXPECT_NEAR(result.actualX.at(0), -1.0, 1e-9); // Left edge locked
    EXPECT_NEAR(result.actualX.at(2), 1.0, 1e-9);  // Right edge locked
    // Middle should be somewhere in between
    EXPECT_GT(result.actualX.at(1), result.actualX.at(0));
    EXPECT_LT(result.actualX.at(1), result.actualX.at(2));
}

} // namespace dsp_core_test
