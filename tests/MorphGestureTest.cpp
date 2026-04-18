#include <gtest/gtest.h>
#include "dsp_core/Source/MorphGesture.h"
#include "dsp_core/Source/Services/GestureSmoother.h"
#include "dsp_core/Source/SplineTypes.h"
#include "dsp_core/Source/LaneMixer.h"
#include <cmath>

using namespace dsp_core;

namespace {
constexpr double kTolerance = 1e-9;

std::vector<GestureSample> makeLinePath() {
    std::vector<GestureSample> raw;
    raw.reserve(10);
    for (int i = 0; i < 10; ++i) {
        raw.push_back({0.1 * i, 0.05 * i});
    }
    return raw;
}

} // namespace

// =============================================================================
// AnchorMorphGesture - sampleAt
// =============================================================================

TEST(AnchorMorphGestureTest, EmptyGestureReportsEmpty) {
    AnchorMorphGesture g;
    EXPECT_TRUE(g.empty());
    const auto s = g.sampleAt(0.5);
    EXPECT_DOUBLE_EQ(0.0, s.dx);
    EXPECT_DOUBLE_EQ(0.0, s.dy);
}

TEST(AnchorMorphGestureTest, SampleAtEndpoints) {
    AnchorMorphGesture g;
    g.deltas = {{0.0, 0.0}, {0.5, 0.5}, {1.0, 0.0}};

    const auto first = g.sampleAt(0.0);
    EXPECT_NEAR(first.dx, 0.0, kTolerance);
    EXPECT_NEAR(first.dy, 0.0, kTolerance);

    const auto last = g.sampleAt(1.0);
    EXPECT_NEAR(last.dx, 1.0, kTolerance);
    EXPECT_NEAR(last.dy, 0.0, kTolerance);
}

TEST(AnchorMorphGestureTest, SampleAtClampsTOutsideRange) {
    AnchorMorphGesture g;
    g.deltas = {{0.0, 0.0}, {1.0, 1.0}};
    EXPECT_EQ(g.sampleAt(-0.5).dx, g.sampleAt(0.0).dx);
    EXPECT_EQ(g.sampleAt(2.0).dx, g.sampleAt(1.0).dx);
}

TEST(AnchorMorphGestureTest, MirrorRoundTrip) {
    AnchorMorphGesture g;
    g.deltas = {{0.0, 0.0}, {0.3, -0.4}, {0.7, 0.2}};
    EXPECT_EQ(g, g.mirrored().mirrored());
}

TEST(AnchorMorphGestureTest, MirrorNegatesAllDeltas) {
    AnchorMorphGesture g;
    g.deltas = {{0.0, 0.0}, {0.5, -0.3}};
    auto m = g.mirrored();
    EXPECT_DOUBLE_EQ(m.deltas[0].dx, 0.0);
    EXPECT_DOUBLE_EQ(m.deltas[0].dy, 0.0);
    EXPECT_DOUBLE_EQ(m.deltas[1].dx, -0.5);
    EXPECT_DOUBLE_EQ(m.deltas[1].dy, 0.3);
}

// =============================================================================
// GestureSmoother
// =============================================================================

TEST(GestureSmootherTest, EmptyInputProducesEmptyGesture) {
    auto g = GestureSmoother::smooth({});
    EXPECT_TRUE(g.empty());
}

TEST(GestureSmootherTest, SinglePointProducesEmptyGesture) {
    auto g = GestureSmoother::smooth({{0.1, 0.2}});
    EXPECT_TRUE(g.empty());
}

TEST(GestureSmootherTest, OutputCountMatchesRequest) {
    auto raw = makeLinePath();
    auto g = GestureSmoother::smooth(raw, 32);
    EXPECT_EQ(g.deltas.size(), 32u);
}

TEST(GestureSmootherTest, FirstSampleIsExactlyZero) {
    auto raw = makeLinePath();
    auto g = GestureSmoother::smooth(raw);
    EXPECT_DOUBLE_EQ(g.deltas.front().dx, 0.0);
    EXPECT_DOUBLE_EQ(g.deltas.front().dy, 0.0);
}

TEST(GestureSmootherTest, LinePathStaysOnLine) {
    // Raw samples on a line; smoothed samples should also be on (close to) the line.
    auto raw = makeLinePath();
    auto g = GestureSmoother::smooth(raw, 16);
    ASSERT_FALSE(g.empty());
    // Slope should be ~0.5 (dy/dx) per the input.
    for (const auto& s : g.deltas) {
        if (std::abs(s.dx) > 1e-6) {
            EXPECT_NEAR(s.dy / s.dx, 0.5, 0.05);
        }
    }
}

TEST(GestureSmootherTest, IgnoresConsecutiveDuplicates) {
    // All identical samples -> no meaningful path -> empty result.
    std::vector<GestureSample> raw(50, {0.5, 0.5});
    auto g = GestureSmoother::smooth(raw);
    EXPECT_TRUE(g.empty());
}

// =============================================================================
// ValueTree round-trip
// =============================================================================

TEST(MorphGestureSerializationTest, RoundTripPreservesDeltas) {
    AnchorMorphGesture g;
    g.deltas = {{0.0, 0.0}, {0.1, -0.2}, {0.3, 0.4}, {-0.5, 0.6}};
    const auto vt = g.toValueTree();
    const auto restored = AnchorMorphGesture::fromValueTree(vt);
    ASSERT_EQ(restored.deltas.size(), g.deltas.size());
    for (size_t i = 0; i < g.deltas.size(); ++i) {
        EXPECT_DOUBLE_EQ(restored.deltas[i].dx, g.deltas[i].dx);
        EXPECT_DOUBLE_EQ(restored.deltas[i].dy, g.deltas[i].dy);
    }
}

TEST(MorphGestureSerializationTest, EmptyGestureRoundTripsToEmpty) {
    AnchorMorphGesture g;
    const auto vt = g.toValueTree();
    const auto restored = AnchorMorphGesture::fromValueTree(vt);
    EXPECT_TRUE(restored.empty());
}

TEST(MorphGestureSerializationTest, InvalidValueTreeProducesEmpty) {
    juce::ValueTree wrong("NotAGesture");
    const auto restored = AnchorMorphGesture::fromValueTree(wrong);
    EXPECT_TRUE(restored.empty());
}

// =============================================================================
// LaneMixer round-trip with gestures
// =============================================================================

TEST(LaneMixerGestureRoundTripTest, GestureBearingAnchorSurvivesRoundTrip) {
    LaneMixer mixer;
    auto& lane = mixer.getMutableLane(0);
    lane.contentType = LaneContentType::Spline;
    lane.splineAnchors.clear();

    SplineAnchor a;
    a.x = 0.5;
    a.y = 0.3;
    a.homeX = 0.4;
    a.homeY = 0.2;
    a.morphGesture.deltas = {{0.0, 0.0}, {0.1, 0.1}, {0.2, -0.05}};
    lane.splineAnchors.push_back(a);

    SplineAnchor b;
    b.x = -0.5;
    b.y = -0.3;
    lane.splineAnchors.push_back(b);

    const auto vt = mixer.toValueTree();

    LaneMixer restored;
    restored.fromValueTree(vt);
    const auto& restoredLane = restored.getLane(0);
    ASSERT_EQ(restoredLane.splineAnchors.size(), 2u);
    const auto& ra = restoredLane.splineAnchors[0];
    EXPECT_DOUBLE_EQ(ra.x, 0.5);
    EXPECT_DOUBLE_EQ(ra.y, 0.3);
    EXPECT_DOUBLE_EQ(ra.homeX, 0.4);
    EXPECT_DOUBLE_EQ(ra.homeY, 0.2);
    ASSERT_EQ(ra.morphGesture.deltas.size(), 3u);
    EXPECT_DOUBLE_EQ(ra.morphGesture.deltas[1].dx, 0.1);
    EXPECT_DOUBLE_EQ(ra.morphGesture.deltas[2].dy, -0.05);

    // Anchor without gesture: morphGesture stays empty, homeX/Y unset (default 0).
    const auto& rb = restoredLane.splineAnchors[1];
    EXPECT_TRUE(rb.morphGesture.empty());
    EXPECT_DOUBLE_EQ(rb.homeX, 0.0);
}
