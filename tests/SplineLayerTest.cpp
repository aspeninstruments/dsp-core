#include <dsp_core/dsp_core.h>
#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <thread>

using namespace dsp_core;
using namespace dsp_core::Services;

//==============================================================================
// Test Fixtures
//==============================================================================

class SplineLayerTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Create simple test anchors
        linearAnchors = {{-1.0, -1.0, false, 0.0}, {1.0, 1.0, false, 0.0}};

        // Three-point curve
        threePtAnchors = {{-1.0, -1.0, false, 0.0}, {0.0, 0.5, false, 0.0}, {1.0, 1.0, false, 0.0}};

        // Compute tangents using Akima (default)
        SplineFitter::computeTangents(linearAnchors, SplineFitConfig::tight());
        SplineFitter::computeTangents(threePtAnchors, SplineFitConfig::tight());
    }

    std::vector<SplineAnchor> linearAnchors;
    std::vector<SplineAnchor> threePtAnchors;

    static constexpr double kTolerance = 1e-6;
};

//==============================================================================
// Basic Tests
//==============================================================================

TEST_F(SplineLayerTest, InitialState) {
    SplineLayer const layer;
    EXPECT_EQ(layer.getAnchors().size(), 0);
}

TEST_F(SplineLayerTest, SetAnchorsUpdatesEvaluation) {
    SplineLayer layer;

    layer.setAnchors(threePtAnchors);

    EXPECT_NEAR(layer.evaluate(-1.0), -1.0, 1e-9);
    EXPECT_NEAR(layer.evaluate(0.0), 0.5, kTolerance);
    EXPECT_NEAR(layer.evaluate(1.0), 1.0, 1e-9);
}

TEST_F(SplineLayerTest, LinearInterpolation) {
    SplineLayer layer;

    // Use 3 points to force linearity
    std::vector<SplineAnchor> linear3Pt = {{-1.0, -1.0, false, 0.0}, {0.0, 0.0, false, 0.0}, {1.0, 1.0, false, 0.0}};
    SplineFitter::computeTangents(linear3Pt, SplineFitConfig::tight());
    layer.setAnchors(linear3Pt);

    // Test at anchor points
    EXPECT_NEAR(layer.evaluate(-1.0), -1.0, 1e-9);
    EXPECT_NEAR(layer.evaluate(0.0), 0.0, 1e-9);
    EXPECT_NEAR(layer.evaluate(1.0), 1.0, 1e-9);

    // Test between anchors (should be close to linear: y = x)
    EXPECT_NEAR(layer.evaluate(-0.5), -0.5, 0.01); // Allow small deviation for cubic spline
    EXPECT_NEAR(layer.evaluate(0.5), 0.5, 0.01);
}

TEST_F(SplineLayerTest, GetAnchorsReturnsSetValue) {
    SplineLayer layer;
    layer.setAnchors(threePtAnchors);

    auto retrieved = layer.getAnchors();
    EXPECT_EQ(retrieved.size(), threePtAnchors.size());
    EXPECT_NEAR(retrieved[0].x, -1.0, 1e-9);
    EXPECT_NEAR(retrieved[1].x, 0.0, 1e-9);
    EXPECT_NEAR(retrieved[2].x, 1.0, 1e-9);
}

//==============================================================================
// Thread Safety Tests
//==============================================================================

TEST_F(SplineLayerTest, ThreadSafetyStressTest) {
    SplineLayer layer;

    // Initial anchors
    layer.setAnchors(linearAnchors);

    std::atomic<bool> stopFlag{false};
    std::atomic<int> errorCount{0};

    // Writer thread (simulates UI)
    std::thread writer([&]() {
        auto anchors = linearAnchors;
        for (int i = 0; i < 1000; ++i) {
            anchors[1].y = std::sin(i * 0.01);
            SplineFitter::computeTangents(anchors, SplineFitConfig::tight());
            layer.setAnchors(anchors);
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
        stopFlag.store(true);
    });

    // Reader thread (simulates audio thread)
    std::thread reader([&]() {
        int iterations = 0;
        while (!stopFlag.load()) {
            double const result = layer.evaluate(0.5);
            // Result should be reasonable (between -2 and 2)
            if (result < -2.0 || result > 2.0) {
                errorCount++;
            }
            iterations++;
        }
    });

    writer.join();
    reader.join();

    EXPECT_EQ(errorCount.load(), 0);
}

//==============================================================================
// Serialization Tests
//==============================================================================

TEST_F(SplineLayerTest, SerializationRoundTrip) {
    SplineLayer layer1;

    std::vector<SplineAnchor> const anchors = {{-1.0, -1.0, true, 0.5}, {0.0, 0.5, true, 1.0}, {1.0, 1.0, true, -0.5}};
    layer1.setAnchors(anchors);

    juce::ValueTree const vt = layer1.toValueTree();

    SplineLayer layer2;
    layer2.fromValueTree(vt);

    EXPECT_EQ(layer2.getAnchors().size(), 3);
    EXPECT_NEAR(layer2.evaluate(0.0), layer1.evaluate(0.0), kTolerance);
    EXPECT_NEAR(layer2.evaluate(0.5), layer1.evaluate(0.5), kTolerance);
}

TEST_F(SplineLayerTest, SerializationPreservesAnchors) {
    SplineLayer layer1;
    layer1.setAnchors(threePtAnchors);

    juce::ValueTree const vt = layer1.toValueTree();

    SplineLayer layer2;
    layer2.fromValueTree(vt);

    auto anchors1 = layer1.getAnchors();
    auto anchors2 = layer2.getAnchors();

    ASSERT_EQ(anchors1.size(), anchors2.size());
    for (size_t i = 0; i < anchors1.size(); ++i) {
        EXPECT_NEAR(anchors1[i].x, anchors2[i].x, 1e-9);
        EXPECT_NEAR(anchors1[i].y, anchors2[i].y, 1e-9);
        EXPECT_NEAR(anchors1[i].tangent, anchors2[i].tangent, 1e-9);
        EXPECT_EQ(anchors1[i].hasCustomTangent, anchors2[i].hasCustomTangent);
    }
}

//==============================================================================
// Edge Cases
//==============================================================================

TEST_F(SplineLayerTest, EmptyAnchorsReturnsZero) {
    SplineLayer layer;
    std::vector<SplineAnchor> const empty;
    layer.setAnchors(empty);

    EXPECT_NEAR(layer.evaluate(0.0), 0.0, 1e-9);
    EXPECT_NEAR(layer.evaluate(-1.0), 0.0, 1e-9);
    EXPECT_NEAR(layer.evaluate(1.0), 0.0, 1e-9);
}

TEST_F(SplineLayerTest, SingleAnchorReturnsConstant) {
    SplineLayer layer;
    std::vector<SplineAnchor> const single = {{0.0, 0.5, false, 0.0}};
    layer.setAnchors(single);

    EXPECT_NEAR(layer.evaluate(-1.0), 0.5, 1e-9);
    EXPECT_NEAR(layer.evaluate(0.0), 0.5, 1e-9);
    EXPECT_NEAR(layer.evaluate(1.0), 0.5, 1e-9);
}
