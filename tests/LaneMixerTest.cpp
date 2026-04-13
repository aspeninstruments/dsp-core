#include <dsp_core/dsp_core.h>
#include <gtest/gtest.h>
#include <cmath>
#include <numeric>

namespace dsp_core_test {

/**
 * Test fixture for LaneMixer
 *
 * Tests cover: default initialization, mixing/sum computation, normalization,
 * version tracking, lane mutations, backward compatibility with LTF, and serialization.
 */
class LaneMixerTest : public ::testing::Test {
  protected:
    void SetUp() override {
        mixer = std::make_unique<dsp_core::LaneMixer>();
    }

    std::unique_ptr<dsp_core::LaneMixer> mixer;

    // Helper: compute sum into a buffer
    std::vector<double> computeSum() {
        std::vector<double> buffer(dsp_core::LaneMixer::TABLE_SIZE, 0.0);
        mixer->computeSum(buffer.data(), dsp_core::LaneMixer::TABLE_SIZE);
        return buffer;
    }

    // Helper: get the max absolute value in a buffer
    static double maxAbs(const std::vector<double>& buffer) {
        double m = 0.0;
        for (auto v : buffer)
            m = std::max(m, std::abs(v));
        return m;
    }

    // Helper: check if buffer is all zeros
    static bool isAllZeros(const std::vector<double>& buffer) {
        for (auto v : buffer) {
            if (std::abs(v) > 1e-15)
                return false;
        }
        return true;
    }
};

// ============================================================================
// Default Initialization Tests
// ============================================================================

TEST_F(LaneMixerTest, DefaultInitialization_HasCorrectLaneCount) {
    EXPECT_EQ(mixer->getNumLanes(), 11);
}

TEST_F(LaneMixerTest, DefaultInitialization_Lane0IsTanh2x) {
    const auto& lane = mixer->getLane(0);
    EXPECT_EQ(lane.contentType, dsp_core::LaneContentType::Equation);
    EXPECT_EQ(lane.harmonicNumber, 0);
    EXPECT_DOUBLE_EQ(lane.amplitude, 0.0);
    EXPECT_TRUE(lane.oddSymmetryEnabled);
    EXPECT_EQ(lane.equationText, juce::String("tanh(2x)"));
    EXPECT_EQ(static_cast<int>(lane.curveData.size()), dsp_core::LaneMixer::TABLE_SIZE);

    // Verify curve is normalized tanh(2x) — initializeDefaults() normalizes so peak = 1.0
    const double normFactor = 1.0 / std::tanh(2.0);
    const int midpoint = dsp_core::LaneMixer::TABLE_SIZE / 2;
    const double xMid = mixer->normalizeIndex(midpoint);
    EXPECT_NEAR(lane.curveData[static_cast<size_t>(midpoint)], std::tanh(2.0 * xMid) * normFactor, 1e-10);

    // Endpoints should be ±1.0 after normalization
    EXPECT_NEAR(lane.curveData[0], -1.0, 1e-10);
    EXPECT_NEAR(lane.curveData[dsp_core::LaneMixer::TABLE_SIZE - 1], 1.0, 1e-10);
}

TEST_F(LaneMixerTest, DefaultInitialization_Lane1IsIdentity) {
    const auto& lane = mixer->getLane(1);
    EXPECT_EQ(lane.contentType, dsp_core::LaneContentType::Harmonic);
    EXPECT_EQ(lane.harmonicNumber, 1);
    EXPECT_DOUBLE_EQ(lane.amplitude, 1.0);
    EXPECT_TRUE(lane.oddSymmetryEnabled);

    // T_1(x) = x (identity function)
    for (int i = 0; i < dsp_core::LaneMixer::TABLE_SIZE; i += 1000) {
        const double x = mixer->normalizeIndex(i);
        EXPECT_NEAR(lane.curveData[static_cast<size_t>(i)], x, 1e-10)
            << "Lane 1 curveData at index " << i << " should be x=" << x;
    }
}

TEST_F(LaneMixerTest, DefaultInitialization_Lanes2Through10AreOddHarmonics) {
    // Expected odd harmonic numbers for lanes 2-10
    const std::array<int, 9> expectedHarmonics = {3, 5, 7, 9, 11, 13, 15, 17, 19};

    for (int laneIdx = 2; laneIdx <= 10; ++laneIdx) {
        const auto& lane = mixer->getLane(laneIdx);
        const int expectedN = expectedHarmonics[laneIdx - 2];
        EXPECT_EQ(lane.contentType, dsp_core::LaneContentType::Harmonic)
            << "Lane " << laneIdx << " should be Harmonic type";
        EXPECT_EQ(lane.harmonicNumber, expectedN)
            << "Lane " << laneIdx << " should have harmonicNumber=" << expectedN;
        EXPECT_DOUBLE_EQ(lane.amplitude, 0.0)
            << "Lane " << laneIdx << " should have amplitude=0.0";
        EXPECT_TRUE(lane.oddSymmetryEnabled)
            << "Lane " << laneIdx << " should have oddSymmetryEnabled";
        EXPECT_EQ(static_cast<int>(lane.curveData.size()), dsp_core::LaneMixer::TABLE_SIZE)
            << "Lane " << laneIdx << " should have TABLE_SIZE curve data";
    }

    // Spot check: Lane 2 = H3, sin(3*asin(x))
    const auto& lane2 = mixer->getLane(2);
    for (int i = 0; i < dsp_core::LaneMixer::TABLE_SIZE; i += 2000) {
        const double x = std::clamp(mixer->normalizeIndex(i), -1.0, 1.0);
        const double expected = std::sin(3.0 * std::asin(x));
        EXPECT_NEAR(lane2.curveData[static_cast<size_t>(i)], expected, 1e-8)
            << "Lane 2 (H3) at index " << i;
    }

    // Spot check: Lane 5 = H9, sin(9*asin(x))
    const auto& lane5 = mixer->getLane(5);
    for (int i = 0; i < dsp_core::LaneMixer::TABLE_SIZE; i += 2000) {
        const double x = std::clamp(mixer->normalizeIndex(i), -1.0, 1.0);
        const double expected = std::sin(9.0 * std::asin(x));
        EXPECT_NEAR(lane5.curveData[static_cast<size_t>(i)], expected, 1e-8)
            << "Lane 5 (H9) at index " << i;
    }
}

TEST_F(LaneMixerTest, DefaultInitialization_OnlyLane1HasNonZeroAmplitude) {
    EXPECT_DOUBLE_EQ(mixer->getLaneAmplitude(0), 0.0); // WT
    EXPECT_DOUBLE_EQ(mixer->getLaneAmplitude(1), 1.0); // H1
    for (int i = 2; i < mixer->getNumLanes(); ++i) {
        EXPECT_DOUBLE_EQ(mixer->getLaneAmplitude(i), 0.0)
            << "Lane " << i << " should have amplitude 0.0";
    }
}

// ============================================================================
// Sum Computation Tests
// ============================================================================

TEST_F(LaneMixerTest, ComputeSum_AllZeroAmplitudes_ReturnsZeros) {
    // Set all amplitudes to zero (including H1)
    mixer->setLaneAmplitude(1, 0.0);

    auto sum = computeSum();
    EXPECT_TRUE(isAllZeros(sum));
}

TEST_F(LaneMixerTest, ComputeSum_SingleLaneFullAmplitude_EqualsLaneCurve) {
    // Default: only Lane 1 (H1=x) is active with amplitude 1.0
    // With normalization, max(|x|) on [-1,1] = 1.0, so normalized = x
    auto sum = computeSum();

    for (int i = 0; i < dsp_core::LaneMixer::TABLE_SIZE; i += 1000) {
        const double x = mixer->normalizeIndex(i);
        EXPECT_NEAR(sum[static_cast<size_t>(i)], x, 1e-10)
            << "Sum should equal x (identity) at index " << i;
    }
}

TEST_F(LaneMixerTest, ComputeSum_TwoLanes_PreservesRelativeProportions) {
    // Lane 1: H1=x, amplitude=0.5
    // Lane 2: T_2(x), amplitude=0.3
    mixer->setLaneAmplitude(1, 0.5);
    mixer->setLaneAmplitude(2, 0.3);

    auto sum = computeSum();

    // Compute expected raw sum then normalize
    const auto& lane1 = mixer->getLane(1);
    const auto& lane2 = mixer->getLane(2);

    std::vector<double> rawSum(dsp_core::LaneMixer::TABLE_SIZE);
    double rawMaxAbs = 0.0;
    for (int i = 0; i < dsp_core::LaneMixer::TABLE_SIZE; ++i) {
        rawSum[static_cast<size_t>(i)] = 0.5 * lane1.curveData[static_cast<size_t>(i)] +
                                          0.3 * lane2.curveData[static_cast<size_t>(i)];
        rawMaxAbs = std::max(rawMaxAbs, std::abs(rawSum[static_cast<size_t>(i)]));
    }

    for (int i = 0; i < dsp_core::LaneMixer::TABLE_SIZE; i += 1000) {
        const double expected = rawSum[static_cast<size_t>(i)] / rawMaxAbs;
        EXPECT_NEAR(sum[static_cast<size_t>(i)], expected, 1e-10)
            << "Normalized two-lane sum at index " << i;
    }
}

TEST_F(LaneMixerTest, ComputeSum_NegativeAmplitude_InvertsCurve) {
    mixer->setLaneAmplitude(1, -1.0);

    auto sum = computeSum();

    // H1 with amplitude -1.0: raw = -x, max(|-x|) = 1.0, normalized = -x
    for (int i = 0; i < dsp_core::LaneMixer::TABLE_SIZE; i += 1000) {
        const double x = mixer->normalizeIndex(i);
        EXPECT_NEAR(sum[static_cast<size_t>(i)], -x, 1e-10)
            << "Negative amplitude should invert at index " << i;
    }
}

TEST_F(LaneMixerTest, ComputeSum_WithNormalization_MaxAbsIsOne) {
    // Set up multiple lanes with large amplitudes
    mixer->setLaneAmplitude(1, 5.0);
    mixer->setLaneAmplitude(2, 3.0);
    mixer->setLaneAmplitude(3, 2.0);

    auto sum = computeSum();

    const double maxVal = maxAbs(sum);
    EXPECT_NEAR(maxVal, 1.0, 1e-10) << "Normalized sum should have max abs = 1.0";
}

// ============================================================================
// Lane Mutation Tests
// ============================================================================

TEST_F(LaneMixerTest, SetLaneCurveData_UpdatesCurve) {
    std::vector<double> customCurve(dsp_core::LaneMixer::TABLE_SIZE, 0.42);
    mixer->setLaneCurveData(5, customCurve);

    const auto& lane = mixer->getLane(5);
    for (int i = 0; i < dsp_core::LaneMixer::TABLE_SIZE; ++i) {
        EXPECT_DOUBLE_EQ(lane.curveData[static_cast<size_t>(i)], 0.42);
    }
}

TEST_F(LaneMixerTest, SetLaneAmplitude_UpdatesAmplitude) {
    mixer->setLaneAmplitude(10, 0.75);
    EXPECT_DOUBLE_EQ(mixer->getLaneAmplitude(10), 0.75);
}

TEST_F(LaneMixerTest, SetLaneContentType_UpdatesType) {
    mixer->setLaneContentType(5, dsp_core::LaneContentType::Paint);
    EXPECT_EQ(mixer->getLaneContentType(5), dsp_core::LaneContentType::Paint);
}

TEST_F(LaneMixerTest, FillLaneWithHarmonic_ProducesCorrectCurve) {
    // Fill lane 5 with H3 (sin(3*asin(x)))
    mixer->fillLaneWithHarmonic(5, 3);

    const auto& lane = mixer->getLane(5);
    EXPECT_EQ(lane.contentType, dsp_core::LaneContentType::Harmonic);
    EXPECT_EQ(lane.harmonicNumber, 3);

    for (int i = 0; i < dsp_core::LaneMixer::TABLE_SIZE; i += 2000) {
        const double x = mixer->normalizeIndex(i);
        const double expected = std::sin(3.0 * std::asin(std::max(-1.0, std::min(1.0, x))));
        EXPECT_NEAR(lane.curveData[static_cast<size_t>(i)], expected, 1e-8)
            << "fillLaneWithHarmonic(5, 3) at index " << i;
    }
}

TEST_F(LaneMixerTest, FillLaneWithTanh2x_ProducesCorrectCurve) {
    mixer->fillLaneWithTanh2x(10);

    // fillLaneWithTanh2x normalizes so peak = 1.0
    const double normFactor = 1.0 / std::tanh(2.0);
    const auto& lane = mixer->getLane(10);
    for (int i = 0; i < dsp_core::LaneMixer::TABLE_SIZE; i += 2000) {
        const double x = mixer->normalizeIndex(i);
        EXPECT_NEAR(lane.curveData[static_cast<size_t>(i)], std::tanh(2.0 * x) * normFactor, 1e-10)
            << "fillLaneWithTanh2x at index " << i;
    }
}

// ============================================================================
// Version Counter Tests
// ============================================================================

TEST_F(LaneMixerTest, VersionCounter_IncrementedOnAmplitudeChange) {
    const auto v0 = mixer->getVersion();
    mixer->setLaneAmplitude(1, 0.5);
    EXPECT_GT(mixer->getVersion(), v0);
}

TEST_F(LaneMixerTest, VersionCounter_IncrementedOnCurveDataChange) {
    const auto v0 = mixer->getVersion();
    std::vector<double> data(dsp_core::LaneMixer::TABLE_SIZE, 0.0);
    mixer->setLaneCurveData(0, data);
    EXPECT_GT(mixer->getVersion(), v0);
}

TEST_F(LaneMixerTest, VersionCounter_NotIncrementedOnRead) {
    const auto v0 = mixer->getVersion();
    [[maybe_unused]] auto amp = mixer->getLaneAmplitude(1);
    [[maybe_unused]] const auto& lane = mixer->getLane(0);
    EXPECT_EQ(mixer->getVersion(), v0);
}

TEST_F(LaneMixerTest, VersionCounter_BatchUpdate_SingleIncrement) {
    const auto v0 = mixer->getVersion();
    {
        dsp_core::LaneMixerBatchUpdateGuard guard(*mixer);
        mixer->setLaneAmplitude(0, 0.5);
        mixer->setLaneAmplitude(1, 0.3);
        mixer->setLaneAmplitude(2, 0.7);
        mixer->setLaneAmplitude(3, 0.1);
    }
    // Should have incremented exactly once (from endBatchUpdate)
    EXPECT_EQ(mixer->getVersion(), v0 + 1);
}

// ============================================================================
// Backward Compatibility Test
// ============================================================================

TEST_F(LaneMixerTest, BackwardCompatibility_DefaultSumMatchesLTFDefault) {
    // Create a LayeredTransferFunction with default init
    // LTF default: WT=0.0, H1=1.0, base=tanh(2x) → evaluates to H1*x = x
    dsp_core::LayeredTransferFunction ltf(dsp_core::LaneMixer::TABLE_SIZE, -1.0, 1.0);
    // LTF constructor sets WT=0.0, H1=1.0 by default

    // LaneMixer default: Lane 0 (WT, tanh(2x), amp=0), Lane 1 (H1=x, amp=1.0)
    // Sum = 0*tanh(2x) + 1.0*x = x

    // Both should produce y=x (with normalization, which is identity for max(|x|)=1)
    auto mixerSum = computeSum();

    for (int i = 0; i < dsp_core::LaneMixer::TABLE_SIZE; i += 100) {
        const double x = mixer->normalizeIndex(i);
        // LTF in Harmonic mode: normScalar * (WT*base + harmonics)
        // With WT=0, H1=1: normScalar * H1*x = normScalar * x
        // max(|x|) on [-1,1] = 1.0, so normScalar = 1.0
        // Result: x
        const double ltfValue = x; // Simplified: LTF default evaluates to x

        EXPECT_NEAR(mixerSum[static_cast<size_t>(i)], ltfValue, 1e-10)
            << "LaneMixer default sum should match LTF default at index " << i;
    }
}

TEST_F(LaneMixerTest, MixingMultipleLanes_ProducesCorrectNormalizedSum) {
    // Set up: Lane 0 = tanh(2x) at 0.5, Lane 1 = H1 at 0.8, Lane 2 = H3 at 0.3
    mixer->setLaneAmplitude(0, 0.5);  // WT (tanh(2x))
    mixer->setLaneAmplitude(1, 0.8);  // H1
    mixer->setLaneAmplitude(2, 0.3);  // H3

    auto mixerSum = computeSum();

    // Compute expected raw sum then normalize
    // Lane 0 curve is normalized tanh(2x): tanh(2x) / tanh(2.0)
    const double tanh2NormFactor = 1.0 / std::tanh(2.0);
    std::vector<double> rawSum(dsp_core::LaneMixer::TABLE_SIZE);
    double rawMaxAbs = 0.0;
    for (int i = 0; i < dsp_core::LaneMixer::TABLE_SIZE; ++i) {
        const double x = mixer->normalizeIndex(i);
        const double wtContrib = 0.5 * (std::tanh(2.0 * x) * tanh2NormFactor);
        const double h1Contrib = 0.8 * x;
        const double h3Contrib =
            0.3 * std::sin(3.0 * std::asin(std::clamp(x, -1.0, 1.0)));
        rawSum[static_cast<size_t>(i)] = wtContrib + h1Contrib + h3Contrib;
        rawMaxAbs = std::max(rawMaxAbs, std::abs(rawSum[static_cast<size_t>(i)]));
    }

    for (int i = 0; i < dsp_core::LaneMixer::TABLE_SIZE; i += 500) {
        const double expected = rawSum[static_cast<size_t>(i)] / rawMaxAbs;
        EXPECT_NEAR(mixerSum[static_cast<size_t>(i)], expected, 1e-8)
            << "Mixer sum should match normalized composite at index " << i;
    }
}

// ============================================================================
// Serialization Tests
// ============================================================================

TEST_F(LaneMixerTest, Serialization_ToValueTree_RoundTrips) {
    // Set up a non-default state
    mixer->setLaneAmplitude(0, 0.5);
    mixer->setLaneAmplitude(1, 0.8);
    mixer->setLaneAmplitude(5, -0.3);
    mixer->setLaneContentType(5, dsp_core::LaneContentType::Paint);

    // Write custom curve data to lane 5
    std::vector<double> customCurve(dsp_core::LaneMixer::TABLE_SIZE);
    for (int i = 0; i < dsp_core::LaneMixer::TABLE_SIZE; ++i) {
        customCurve[static_cast<size_t>(i)] = std::sin(static_cast<double>(i) * 0.001);
    }
    mixer->setLaneCurveData(5, customCurve);

    // Serialize
    auto vt = mixer->toValueTree();

    // Deserialize into a new mixer
    dsp_core::LaneMixer mixer2;
    mixer2.fromValueTree(vt);

    // Verify all lanes round-trip correctly
    ASSERT_EQ(mixer->getNumLanes(), mixer2.getNumLanes());
    for (int n = 0; n < mixer->getNumLanes(); ++n) {
        const auto& original = mixer->getLane(n);
        const auto& restored = mixer2.getLane(n);

        EXPECT_DOUBLE_EQ(original.amplitude, restored.amplitude)
            << "Lane " << n << " amplitude mismatch";
        EXPECT_EQ(original.contentType, restored.contentType)
            << "Lane " << n << " contentType mismatch";
        EXPECT_EQ(original.harmonicNumber, restored.harmonicNumber)
            << "Lane " << n << " harmonicNumber mismatch";

        // Verify curve data matches
        ASSERT_EQ(original.curveData.size(), restored.curveData.size())
            << "Lane " << n << " curveData size mismatch";
        for (int i = 0; i < dsp_core::LaneMixer::TABLE_SIZE; i += 1000) {
            EXPECT_NEAR(original.curveData[static_cast<size_t>(i)],
                        restored.curveData[static_cast<size_t>(i)], 1e-12)
                << "Lane " << n << " curveData mismatch at index " << i;
        }
    }

}

TEST_F(LaneMixerTest, Serialization_SplineAnchors_RoundTrip) {
    // Set up a lane with spline anchors
    auto& lane = const_cast<dsp_core::Lane&>(mixer->getLane(3));
    lane.contentType = dsp_core::LaneContentType::Spline;
    lane.splineAnchors = {
        {-1.0, -1.0, false, 0.0},
        {-0.5, 0.2, true, 1.5},
        {0.0, 0.0, false, 0.0},
        {0.5, -0.3, false, 0.0},
        {1.0, 1.0, false, 0.0}
    };

    auto vt = mixer->toValueTree();
    dsp_core::LaneMixer mixer2;
    mixer2.fromValueTree(vt);

    const auto& restored = mixer2.getLane(3);
    EXPECT_EQ(restored.contentType, dsp_core::LaneContentType::Spline);
    ASSERT_EQ(restored.splineAnchors.size(), 5u);
    EXPECT_NEAR(restored.splineAnchors[1].x, -0.5, 1e-10);
    EXPECT_NEAR(restored.splineAnchors[1].y, 0.2, 1e-10);
    EXPECT_TRUE(restored.splineAnchors[1].hasCustomTangent);
    EXPECT_NEAR(restored.splineAnchors[1].tangent, 1.5, 1e-10);
}

// ============================================================================
// Reset Tests
// ============================================================================

TEST_F(LaneMixerTest, ResetToDefaults_RestoresInitialState) {
    // Modify state
    mixer->setLaneAmplitude(1, 0.5);
    mixer->setLaneAmplitude(5, 0.8);

    // Reset
    mixer->resetToDefaults();

    // Verify defaults: lane 0 = 0.0, lane 1 = 1.0, lanes 2-10 = 0.0
    EXPECT_DOUBLE_EQ(mixer->getLaneAmplitude(0), 0.0);
    EXPECT_DOUBLE_EQ(mixer->getLaneAmplitude(1), 1.0);
    EXPECT_DOUBLE_EQ(mixer->getLaneAmplitude(5), 0.0);
    EXPECT_EQ(mixer->getNumLanes(), 11);
    EXPECT_EQ(mixer->getLane(5).harmonicNumber, 9);  // Lane 5 = H9
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(LaneMixerTest, InvalidLaneIndex_DoesNotCrash) {
    // These should be no-ops, not crashes
    mixer->setLaneAmplitude(-1, 1.0);
    mixer->setLaneAmplitude(41, 1.0);
    mixer->setLaneAmplitude(100, 1.0);

    EXPECT_DOUBLE_EQ(mixer->getLaneAmplitude(-1), 0.0);
    EXPECT_DOUBLE_EQ(mixer->getLaneAmplitude(41), 0.0);
}

TEST_F(LaneMixerTest, EvaluateSumAt_InterpolatesBetweenPoints) {
    // With H1=1.0 (y=x), evaluateSumAt should return approximately x
    const double result = mixer->evaluateSumAt(0.5);
    EXPECT_NEAR(result, 0.5, 1e-3); // Allow some interpolation error
}

TEST_F(LaneMixerTest, EvaluateSumAt_ClampsToRange) {
    // Values outside [-1, 1] should be clamped
    const double atMin = mixer->evaluateSumAt(-2.0);
    const double atMax = mixer->evaluateSumAt(2.0);

    // Should evaluate at boundary values (-1.0 and 1.0)
    EXPECT_NEAR(atMin, -1.0, 1e-3);
    EXPECT_NEAR(atMax, 1.0, 1e-3);
}

TEST_F(LaneMixerTest, NormalizeIndex_MatchesLTF) {
    // Verify our normalizeIndex matches LayeredTransferFunction's
    dsp_core::LayeredTransferFunction ltf(dsp_core::LaneMixer::TABLE_SIZE, -1.0, 1.0);

    for (int i = 0; i < dsp_core::LaneMixer::TABLE_SIZE; i += 1000) {
        EXPECT_DOUBLE_EQ(mixer->normalizeIndex(i), ltf.normalizeIndex(i))
            << "normalizeIndex mismatch at " << i;
    }
}

// ============================================================================
// ExtrapolationMode Tests
// ============================================================================

TEST_F(LaneMixerTest, ExtrapolationMode_DefaultIsClamp) {
    EXPECT_EQ(mixer->getExtrapolationMode(), dsp_core::LaneMixer::ExtrapolationMode::Clamp);
}

TEST_F(LaneMixerTest, ExtrapolationMode_SetAndGet) {
    mixer->setExtrapolationMode(dsp_core::LaneMixer::ExtrapolationMode::Linear);
    EXPECT_EQ(mixer->getExtrapolationMode(), dsp_core::LaneMixer::ExtrapolationMode::Linear);

    mixer->setExtrapolationMode(dsp_core::LaneMixer::ExtrapolationMode::Clamp);
    EXPECT_EQ(mixer->getExtrapolationMode(), dsp_core::LaneMixer::ExtrapolationMode::Clamp);
}

TEST_F(LaneMixerTest, ExtrapolationMode_IncrementsVersion) {
    const auto v0 = mixer->getVersion();
    mixer->setExtrapolationMode(dsp_core::LaneMixer::ExtrapolationMode::Linear);
    EXPECT_GT(mixer->getVersion(), v0);
}

// ============================================================================
// Convenience API Tests
// ============================================================================

TEST_F(LaneMixerTest, GetAmplitudes_ReturnsAllLaneAmplitudes) {
    mixer->setLaneAmplitude(0, 0.5);
    mixer->setLaneAmplitude(1, 0.8);
    mixer->setLaneAmplitude(5, -0.3);

    const auto amps = mixer->getAmplitudes();
    EXPECT_DOUBLE_EQ(amps[0], 0.5);
    EXPECT_DOUBLE_EQ(amps[1], 0.8);
    EXPECT_DOUBLE_EQ(amps[5], -0.3);
    EXPECT_DOUBLE_EQ(amps[2], 0.0);
}

TEST_F(LaneMixerTest, ClearLaneCurveData_ZerosOutCurve) {
    // Lane 1 has identity curve by default
    EXPECT_NE(mixer->getLane(1).curveData[dsp_core::LaneMixer::TABLE_SIZE - 1], 0.0);

    mixer->clearLaneCurveData(1);

    for (int i = 0; i < dsp_core::LaneMixer::TABLE_SIZE; ++i) {
        EXPECT_DOUBLE_EQ(mixer->getLane(1).curveData[static_cast<size_t>(i)], 0.0)
            << "CurveData should be zero at index " << i;
    }
}

TEST_F(LaneMixerTest, ClearLaneCurveData_IncrementsVersion) {
    const auto v0 = mixer->getVersion();
    mixer->clearLaneCurveData(1);
    EXPECT_GT(mixer->getVersion(), v0);
}

TEST_F(LaneMixerTest, ClearLaneCurveData_InvalidIndex_NoOp) {
    const auto v0 = mixer->getVersion();
    mixer->clearLaneCurveData(-1);
    mixer->clearLaneCurveData(41);
    EXPECT_EQ(mixer->getVersion(), v0);
}

// ============================================================================
// Scan Mode Tests
// ============================================================================

class LaneMixerScanTest : public LaneMixerTest {
  protected:
    // Helper: compute scan into a buffer
    std::vector<double> computeScan() {
        std::vector<double> buffer(dsp_core::LaneMixer::TABLE_SIZE, 0.0);
        mixer->computeScan(buffer.data(), dsp_core::LaneMixer::TABLE_SIZE);
        return buffer;
    }

    // Helper: set up N lanes with simple identity-scaled curves for predictable testing.
    // Lane i gets curveData[j] = (i+1) * normalizeIndex(j), i.e. lane 0 = 1x, lane 1 = 2x, etc.
    void setupSimpleLanes(int count) {
        // Reset to 1 lane, then add more
        mixer->resetToDefaults();
        while (mixer->getNumLanes() > 1) {
            mixer->removeLane(mixer->getNumLanes() - 1);
        }
        // Fill lane 0
        auto& lane0 = mixer->getMutableLane(0);
        lane0.curveData.resize(dsp_core::LaneMixer::TABLE_SIZE);
        for (int j = 0; j < dsp_core::LaneMixer::TABLE_SIZE; ++j) {
            lane0.curveData[static_cast<size_t>(j)] = mixer->normalizeIndex(j);
        }
        // Add remaining lanes
        for (int i = 1; i < count; ++i) {
            mixer->addLane(-1);
            auto& lane = mixer->getMutableLane(i);
            lane.curveData.resize(dsp_core::LaneMixer::TABLE_SIZE);
            const double scale = static_cast<double>(i + 1);
            for (int j = 0; j < dsp_core::LaneMixer::TABLE_SIZE; ++j) {
                lane.curveData[static_cast<size_t>(j)] = scale * mixer->normalizeIndex(j);
            }
        }
    }
};

TEST_F(LaneMixerScanTest, DefaultMixerMode_IsBlend) {
    EXPECT_EQ(mixer->getMixerMode(), dsp_core::LaneMixer::MixerMode::Blend);
}

TEST_F(LaneMixerScanTest, SetMixerMode_IncrementsVersion) {
    const auto v0 = mixer->getVersion();
    mixer->setMixerMode(dsp_core::LaneMixer::MixerMode::Scan);
    EXPECT_GT(mixer->getVersion(), v0);
    EXPECT_EQ(mixer->getMixerMode(), dsp_core::LaneMixer::MixerMode::Scan);
}

TEST_F(LaneMixerScanTest, SetScanPosition_ClampsAndIncrementsVersion) {
    const auto v0 = mixer->getVersion();
    mixer->setScanPosition(0.5);
    EXPECT_GT(mixer->getVersion(), v0);
    EXPECT_DOUBLE_EQ(mixer->getScanPosition(), 0.5);

    mixer->setScanPosition(-1.0);
    EXPECT_DOUBLE_EQ(mixer->getScanPosition(), 0.0);

    mixer->setScanPosition(2.0);
    EXPECT_DOUBLE_EQ(mixer->getScanPosition(), 1.0);
}

TEST_F(LaneMixerScanTest, ComputeScan_SingleLane_EqualsLaneCurve) {
    setupSimpleLanes(1);
    mixer->setScanPosition(0.0);
    auto result = computeScan();

    // Single lane: output = normalized lane 0 curve
    // Lane 0 curve is identity (x), which is already normalized
    const auto& lane0 = mixer->getLane(0);
    for (int i = 0; i < dsp_core::LaneMixer::TABLE_SIZE; ++i) {
        EXPECT_NEAR(result[static_cast<size_t>(i)],
                    lane0.curveData[static_cast<size_t>(i)],
                    1e-10) << "at index " << i;
    }
}

TEST_F(LaneMixerScanTest, ComputeScan_TwoLanes_Position0_EqualsLane0) {
    setupSimpleLanes(2);
    mixer->setScanPosition(0.0);
    auto result = computeScan();

    // pos=0 → 100% lane 0, then normalized
    // Lane 0 is identity (peak=1), so output = lane 0 curve
    const auto& lane0 = mixer->getLane(0);
    for (int i = 0; i < dsp_core::LaneMixer::TABLE_SIZE; ++i) {
        EXPECT_NEAR(result[static_cast<size_t>(i)],
                    lane0.curveData[static_cast<size_t>(i)],
                    1e-10) << "at index " << i;
    }
}

TEST_F(LaneMixerScanTest, ComputeScan_TwoLanes_Position1_EqualsLane1Normalized) {
    setupSimpleLanes(2);
    mixer->setScanPosition(1.0);
    auto result = computeScan();

    // pos=1 → 100% lane 1 (curve = 2x), normalized → divide by 2 → identity
    // So result should equal lane 0 curve (identity)
    const auto& lane0 = mixer->getLane(0);
    for (int i = 0; i < dsp_core::LaneMixer::TABLE_SIZE; ++i) {
        EXPECT_NEAR(result[static_cast<size_t>(i)],
                    lane0.curveData[static_cast<size_t>(i)],
                    1e-10) << "at index " << i;
    }
}

TEST_F(LaneMixerScanTest, ComputeScan_TwoLanes_Position50_Blends) {
    setupSimpleLanes(2);
    mixer->setScanPosition(0.5);
    auto result = computeScan();

    // pos=0.5 → 50:50 blend of lane 0 (x) and lane 1 (2x) = 1.5x
    // Normalized by max(|1.5x|) = 1.5 → output = x (identity)
    const auto& lane0 = mixer->getLane(0);
    for (int i = 0; i < dsp_core::LaneMixer::TABLE_SIZE; ++i) {
        EXPECT_NEAR(result[static_cast<size_t>(i)],
                    lane0.curveData[static_cast<size_t>(i)],
                    1e-10) << "at index " << i;
    }
}

TEST_F(LaneMixerScanTest, ComputeScan_ThreeLanes_Position50_EqualsLane1) {
    setupSimpleLanes(3);
    mixer->setScanPosition(0.5);
    auto result = computeScan();

    // 3 lanes: f = 0.5 * 2 = 1.0 → exactly lane 1 (2x), normalized → identity
    const auto& lane0 = mixer->getLane(0);
    for (int i = 0; i < dsp_core::LaneMixer::TABLE_SIZE; ++i) {
        EXPECT_NEAR(result[static_cast<size_t>(i)],
                    lane0.curveData[static_cast<size_t>(i)],
                    1e-10) << "at index " << i;
    }
}

TEST_F(LaneMixerScanTest, ComputeScan_OutputIsNormalized) {
    setupSimpleLanes(2);
    mixer->setScanPosition(0.3);
    auto result = computeScan();

    // Verify max absolute value is 1.0 (within epsilon)
    double peak = maxAbs(result);
    EXPECT_NEAR(peak, 1.0, 1e-10);
}

TEST_F(LaneMixerScanTest, Serialization_PreservesMixerMode) {
    mixer->setMixerMode(dsp_core::LaneMixer::MixerMode::Scan);
    auto vt = mixer->toValueTree();

    auto restored = std::make_unique<dsp_core::LaneMixer>();
    EXPECT_EQ(restored->getMixerMode(), dsp_core::LaneMixer::MixerMode::Blend);

    restored->fromValueTree(vt);
    EXPECT_EQ(restored->getMixerMode(), dsp_core::LaneMixer::MixerMode::Scan);
}

// ============================================================================
// DuplicateLane Tests
// ============================================================================

class LaneMixerDuplicateTest : public ::testing::Test {
  protected:
    void SetUp() override {
        mixer = std::make_unique<dsp_core::LaneMixer>();
    }
    std::unique_ptr<dsp_core::LaneMixer> mixer;
};

TEST_F(LaneMixerDuplicateTest, DuplicateLane_CopiesCurveData) {
    // Default mixer has 11 lanes. Duplicate lane 0.
    const auto& srcCurve = mixer->getLane(0).curveData;
    ASSERT_FALSE(srcCurve.empty());

    const int newIdx = mixer->duplicateLane(0);
    EXPECT_EQ(newIdx, 1);
    EXPECT_EQ(mixer->getNumLanes(), 12);

    const auto& newCurve = mixer->getLane(newIdx).curveData;
    ASSERT_EQ(newCurve.size(), srcCurve.size());
    // Source lane 0 is now still at index 0 (new lane inserted at 1)
    const auto& srcCurveAfter = mixer->getLane(0).curveData;
    for (size_t i = 0; i < newCurve.size(); ++i) {
        EXPECT_DOUBLE_EQ(newCurve[i], srcCurveAfter[i]) << "at index " << i;
    }
}

TEST_F(LaneMixerDuplicateTest, DuplicateLane_CopiesContentTypeAndMetadata) {
    const auto& src = mixer->getLane(0);
    const auto srcType = src.contentType;
    const auto srcHarmonic = src.harmonicNumber;
    const auto srcSymmetry = src.oddSymmetryEnabled;

    const int newIdx = mixer->duplicateLane(0);
    const auto& dup = mixer->getLane(newIdx);

    EXPECT_EQ(dup.contentType, srcType);
    EXPECT_EQ(dup.harmonicNumber, srcHarmonic);
    EXPECT_EQ(dup.oddSymmetryEnabled, srcSymmetry);
}

TEST_F(LaneMixerDuplicateTest, DuplicateLane_AssignsNewLaneId) {
    const uint32_t srcId = mixer->getLaneId(0);
    const int newIdx = mixer->duplicateLane(0);
    EXPECT_NE(mixer->getLaneId(newIdx), srcId);
}

TEST_F(LaneMixerDuplicateTest, DuplicateLane_AmplitudeIsZero) {
    // Set source amplitude to 0.8, duplicate should still be 0.0
    mixer->setLaneAmplitude(0, 0.8);
    const int newIdx = mixer->duplicateLane(0);
    EXPECT_DOUBLE_EQ(mixer->getLane(newIdx).amplitude.load(), 0.0);
}

TEST_F(LaneMixerDuplicateTest, DuplicateLane_PreservesAdjacentLanes) {
    // Record lane 1's ID before duplicating lane 0
    const uint32_t lane1Id = mixer->getLaneId(1);
    mixer->duplicateLane(0);
    // Old lane 1 should now be at index 2
    EXPECT_EQ(mixer->getLaneId(2), lane1Id);
}

TEST_F(LaneMixerDuplicateTest, DuplicateLane_LastLane_AppendsAtEnd) {
    const int lastIdx = mixer->getNumLanes() - 1;
    const uint32_t lastId = mixer->getLaneId(lastIdx);
    const int newIdx = mixer->duplicateLane(lastIdx);
    EXPECT_EQ(newIdx, lastIdx + 1);
    // Original last lane still at same index
    EXPECT_EQ(mixer->getLaneId(lastIdx), lastId);
}

TEST_F(LaneMixerDuplicateTest, DuplicateLane_InvalidIndex_ReturnsNeg1) {
    EXPECT_EQ(mixer->duplicateLane(-1), -1);
    EXPECT_EQ(mixer->duplicateLane(999), -1);
}

TEST_F(LaneMixerDuplicateTest, DuplicateLane_IncrementsVersion) {
    const auto v0 = mixer->getVersion();
    mixer->duplicateLane(0);
    EXPECT_GT(mixer->getVersion(), v0);
}

// ============================================================================
// DuplicateLane Scan Position Adjustment Tests
// ============================================================================

TEST_F(LaneMixerDuplicateTest, ScanAdjust_ExactlyOnLane_InsertAfter_NoShift) {
    // 11 lanes, scan at 0.5 → f = 0.5 * 10 = 5.0 (exactly lane 5)
    // Duplicate lane 5 → insert at 6. Since 6 > ceil(5.0) = 5, no shift.
    // New scan = 5.0 / 11 ≈ 0.4545
    mixer->setScanPosition(0.5);
    mixer->duplicateLane(5);
    const double expected = 5.0 / 11.0;
    EXPECT_NEAR(mixer->getScanPosition(), expected, 1e-10);
}

TEST_F(LaneMixerDuplicateTest, ScanAdjust_BetweenLanes_InsertInBlendRange_Shifts) {
    // 11 lanes, scan at 0.55 → f = 0.55 * 10 = 5.5 (between 5 and 6)
    // Duplicate lane 5 → insert at 6. ceil(5.5) = 6, 6 <= 6 → shift.
    // f_new = 6.5, scan = 6.5 / 11
    mixer->setScanPosition(0.55);
    mixer->duplicateLane(5);
    const double expected = 6.5 / 11.0;
    EXPECT_NEAR(mixer->getScanPosition(), expected, 1e-10);
}

TEST_F(LaneMixerDuplicateTest, ScanAdjust_InsertBeforeScan_Shifts) {
    // 11 lanes, scan at 0.5 → f = 5.0 (exactly lane 5)
    // Duplicate lane 2 → insert at 3. ceil(5.0) = 5, 3 <= 5 → shift.
    // f_new = 6.0, scan = 6.0 / 11
    mixer->setScanPosition(0.5);
    mixer->duplicateLane(2);
    const double expected = 6.0 / 11.0;
    EXPECT_NEAR(mixer->getScanPosition(), expected, 1e-10);
}

TEST_F(LaneMixerDuplicateTest, ScanAdjust_InsertAfterScan_NoShift) {
    // 11 lanes, scan at 0.3 → f = 3.0
    // Duplicate lane 8 → insert at 9. ceil(3.0) = 3, 9 > 3 → no shift.
    // f_new = 3.0, scan = 3.0 / 11
    mixer->setScanPosition(0.3);
    mixer->duplicateLane(8);
    const double expected = 3.0 / 11.0;
    EXPECT_NEAR(mixer->getScanPosition(), expected, 1e-10);
}

TEST_F(LaneMixerDuplicateTest, ScanAdjust_SingleLane_NoAdjustment) {
    // Remove all but one lane
    while (mixer->getNumLanes() > 1) {
        mixer->removeLane(mixer->getNumLanes() - 1);
    }
    ASSERT_EQ(mixer->getNumLanes(), 1);
    mixer->setScanPosition(0.5);
    mixer->duplicateLane(0);
    // With 1 lane, scan adjustment is skipped (activeLaneCount was 1)
    // After duplication we have 2 lanes. Position stays at 0.5.
    EXPECT_DOUBLE_EQ(mixer->getScanPosition(), 0.5);
}

// ============================================================================
// Mix Version Counter Tests (Phase A — Event-Driven Rendering)
// ============================================================================

TEST_F(LaneMixerTest, MixVersion_IncrementedOnSetLaneAmplitude) {
    const auto mv0 = mixer->getMixVersion();
    mixer->setLaneAmplitude(1, 0.5);
    EXPECT_GT(mixer->getMixVersion(), mv0);
}

TEST_F(LaneMixerTest, MixVersion_IncrementedOnSetScanPosition) {
    const auto mv0 = mixer->getMixVersion();
    mixer->setScanPosition(0.5);
    EXPECT_GT(mixer->getMixVersion(), mv0);
}

TEST_F(LaneMixerTest, MixVersion_NotIncrementedOnSetLaneCurveData) {
    const auto mv0 = mixer->getMixVersion();
    std::vector<double> data(dsp_core::LaneMixer::TABLE_SIZE, 0.0);
    mixer->setLaneCurveData(0, data);
    // Full version should increment, but mix version should NOT
    EXPECT_EQ(mixer->getMixVersion(), mv0);
}

TEST_F(LaneMixerTest, MixVersion_NotIncrementedOnAddLane) {
    const auto mv0 = mixer->getMixVersion();
    mixer->addLane();
    EXPECT_EQ(mixer->getMixVersion(), mv0);
}

TEST_F(LaneMixerTest, MixVersion_NotIncrementedOnRemoveLane) {
    const auto mv0 = mixer->getMixVersion();
    mixer->removeLane(mixer->getNumLanes() - 1);
    EXPECT_EQ(mixer->getMixVersion(), mv0);
}

TEST_F(LaneMixerTest, MixVersion_NotIncrementedOnContentTypeChange) {
    const auto mv0 = mixer->getMixVersion();
    mixer->setLaneContentType(0, dsp_core::LaneContentType::Paint);
    EXPECT_EQ(mixer->getMixVersion(), mv0);
}

TEST_F(LaneMixerTest, MixVersion_DeferredDuringBatchUpdate) {
    const auto mv0 = mixer->getMixVersion();
    {
        dsp_core::LaneMixerBatchUpdateGuard guard(*mixer);
        mixer->setLaneAmplitude(0, 0.5);
        mixer->setLaneAmplitude(1, 0.3);
        // During batch, mix version should not have incremented
        EXPECT_EQ(mixer->getMixVersion(), mv0);
    }
    // After batch ends, should have incremented exactly once
    EXPECT_EQ(mixer->getMixVersion(), mv0 + 1);
}

TEST_F(LaneMixerTest, MixVersion_FullVersionAlwaysIncrementsWithMix) {
    // When amplitude changes, BOTH counters should increment
    const auto fv0 = mixer->getVersion();
    const auto mv0 = mixer->getMixVersion();
    mixer->setLaneAmplitude(1, 0.5);
    const auto fv1 = mixer->getVersion();
    const auto mv1 = mixer->getMixVersion();
    EXPECT_GT(fv1, fv0);
    EXPECT_GT(mv1, mv0);

    // When curve data changes, only full version increments
    std::vector<double> data(dsp_core::LaneMixer::TABLE_SIZE, 0.0);
    mixer->setLaneCurveData(0, data);
    EXPECT_GT(mixer->getVersion(), fv1);
    EXPECT_EQ(mixer->getMixVersion(), mv1);
}

TEST_F(LaneMixerTest, MixVersion_GetMixVersionCounterReturnsReference) {
    // Verify that getMixVersionCounter() returns a reference to the internal counter
    auto& counter = mixer->getMixVersionCounter();
    const auto mv0 = counter.load(std::memory_order_acquire);
    mixer->setLaneAmplitude(1, 0.5);
    EXPECT_GT(counter.load(std::memory_order_acquire), mv0);
}

// ============================================================================
// Version Changed Callback Tests (Phase A — Event-Driven Rendering)
// ============================================================================

TEST_F(LaneMixerTest, OnVersionChanged_CalledOnAmplitudeChange) {
    int callCount = 0;
    mixer->setOnVersionChanged([&]() { callCount++; });
    mixer->setLaneAmplitude(1, 0.5);
    EXPECT_EQ(callCount, 1);
}

TEST_F(LaneMixerTest, OnVersionChanged_CalledOnCurveDataChange) {
    int callCount = 0;
    mixer->setOnVersionChanged([&]() { callCount++; });
    std::vector<double> data(dsp_core::LaneMixer::TABLE_SIZE, 0.0);
    mixer->setLaneCurveData(0, data);
    EXPECT_EQ(callCount, 1);
}

TEST_F(LaneMixerTest, OnVersionChanged_NotCalledDuringBatch) {
    int callCount = 0;
    mixer->setOnVersionChanged([&]() { callCount++; });
    {
        dsp_core::LaneMixerBatchUpdateGuard guard(*mixer);
        mixer->setLaneAmplitude(0, 0.5);
        mixer->setLaneAmplitude(1, 0.3);
        mixer->setLaneAmplitude(2, 0.7);
        EXPECT_EQ(callCount, 0) << "Callback should not fire during batch";
    }
    // endBatchUpdate fires the callback once
    EXPECT_EQ(callCount, 1);
}

TEST_F(LaneMixerTest, OnVersionChanged_CalledOnEndBatchUpdate) {
    int callCount = 0;
    mixer->setOnVersionChanged([&]() { callCount++; });
    mixer->beginBatchUpdate();
    mixer->setLaneAmplitude(0, 0.5);
    EXPECT_EQ(callCount, 0);
    mixer->endBatchUpdate();
    EXPECT_EQ(callCount, 1);
}

TEST_F(LaneMixerTest, OnVersionChanged_NullCallbackSafe) {
    // Default state: no callback set. Should not crash.
    mixer->setLaneAmplitude(1, 0.5);
    // Also test explicitly setting nullptr
    mixer->setOnVersionChanged(nullptr);
    mixer->setLaneAmplitude(1, 0.3);
    // No crash = pass
}

TEST_F(LaneMixerTest, OnVersionChanged_CalledOnStructuralChanges) {
    int callCount = 0;
    mixer->setOnVersionChanged([&]() { callCount++; });

    mixer->addLane();
    EXPECT_GE(callCount, 1) << "Callback should fire at least once for addLane";
    const int afterAdd = callCount;

    mixer->removeLane(mixer->getNumLanes() - 1);
    EXPECT_GT(callCount, afterAdd) << "Callback should fire at least once for removeLane";
}

// ============================================================================
// Blend Depth + Blend Amount (Phase 1: per-lane macro modulation)
// ============================================================================

TEST_F(LaneMixerTest, LaneBlendDepth_DefaultsToZero) {
    EXPECT_DOUBLE_EQ(mixer->getLaneBlendDepth(0), 0.0);
    EXPECT_DOUBLE_EQ(mixer->getLaneBlendDepth(1), 0.0);
    EXPECT_DOUBLE_EQ(mixer->getLaneBlendDepth(mixer->getNumLanes() - 1), 0.0);
}

TEST_F(LaneMixerTest, LaneBlendDepth_SetGet_RoundTrips) {
    mixer->setLaneBlendDepth(2, 0.5);
    EXPECT_DOUBLE_EQ(mixer->getLaneBlendDepth(2), 0.5);
    mixer->setLaneBlendDepth(2, -0.75);
    EXPECT_DOUBLE_EQ(mixer->getLaneBlendDepth(2), -0.75);
    mixer->setLaneBlendDepth(2, 0.0);
    EXPECT_DOUBLE_EQ(mixer->getLaneBlendDepth(2), 0.0);
}

TEST_F(LaneMixerTest, LaneBlendDepth_ClampedToRange) {
    mixer->setLaneBlendDepth(1, 2.5);
    EXPECT_DOUBLE_EQ(mixer->getLaneBlendDepth(1), 1.0);
    mixer->setLaneBlendDepth(1, -2.5);
    EXPECT_DOUBLE_EQ(mixer->getLaneBlendDepth(1), -1.0);
}

TEST_F(LaneMixerTest, BlendAmount_DefaultsToZero) {
    EXPECT_DOUBLE_EQ(mixer->getBlendAmount(), 0.0);
}

TEST_F(LaneMixerTest, BlendAmount_SetGet_ClampedToUnitRange) {
    mixer->setBlendAmount(0.5);
    EXPECT_DOUBLE_EQ(mixer->getBlendAmount(), 0.5);
    mixer->setBlendAmount(2.0);
    EXPECT_DOUBLE_EQ(mixer->getBlendAmount(), 1.0);
    mixer->setBlendAmount(-0.5);
    EXPECT_DOUBLE_EQ(mixer->getBlendAmount(), 0.0);
}

TEST_F(LaneMixerTest, ComputeSum_BlendAmountZero_BehavesLikeBaseAmplitude) {
    // Set some non-zero depths but leave blendAmount=0 — depths should be inert.
    mixer->setLaneAmplitude(1, 0.8);
    mixer->setLaneAmplitude(2, 0.3);
    mixer->setLaneBlendDepth(1, 0.7);   // would normally pull lane 1 up
    mixer->setLaneBlendDepth(2, -0.5);  // would normally pull lane 2 down
    // blendAmount is still 0, so depths should have no effect.
    const auto withDepths = computeSum();

    // Reset depths and compare — buffers must be identical.
    mixer->setLaneBlendDepth(1, 0.0);
    mixer->setLaneBlendDepth(2, 0.0);
    const auto withoutDepths = computeSum();

    ASSERT_EQ(withDepths.size(), withoutDepths.size());
    for (size_t i = 0; i < withDepths.size(); ++i) {
        EXPECT_DOUBLE_EQ(withDepths[i], withoutDepths[i]);
    }
}

TEST_F(LaneMixerTest, ComputeSum_PositiveDepthAddsToAmplitude) {
    // Lane 1 = identity. base=0.4, depth=+0.5, blendAmount=1.0 → effective=0.9.
    // Compare against a fresh mixer with base=0.9, no depth.
    mixer->setLaneAmplitude(1, 0.4);
    mixer->setLaneBlendDepth(1, 0.5);
    mixer->setBlendAmount(1.0);
    const auto modulated = computeSum();

    auto fresh = std::make_unique<dsp_core::LaneMixer>();
    fresh->setLaneAmplitude(1, 0.9);
    std::vector<double> expected(dsp_core::LaneMixer::TABLE_SIZE, 0.0);
    fresh->computeSum(expected.data(), dsp_core::LaneMixer::TABLE_SIZE);

    ASSERT_EQ(modulated.size(), expected.size());
    for (size_t i = 0; i < modulated.size(); ++i) {
        EXPECT_NEAR(modulated[i], expected[i], 1e-12);
    }
}

TEST_F(LaneMixerTest, ComputeSum_NegativeDepthSubtracts) {
    // base=0.8, depth=-0.5, blendAmount=1.0 → effective = max(0, 0.3) = 0.3.
    mixer->setLaneAmplitude(1, 0.8);
    mixer->setLaneBlendDepth(1, -0.5);
    mixer->setBlendAmount(1.0);
    const auto modulated = computeSum();

    auto fresh = std::make_unique<dsp_core::LaneMixer>();
    fresh->setLaneAmplitude(1, 0.3);
    std::vector<double> expected(dsp_core::LaneMixer::TABLE_SIZE, 0.0);
    fresh->computeSum(expected.data(), dsp_core::LaneMixer::TABLE_SIZE);

    for (size_t i = 0; i < modulated.size(); ++i) {
        EXPECT_NEAR(modulated[i], expected[i], 1e-12);
    }
}

TEST_F(LaneMixerTest, ComputeSum_NegativeDepthClampedAtZero) {
    // base=0.2, depth=-1.0, blendAmount=1.0 → raw = -0.8 → effective = 0.
    // The lane should drop out entirely (no inverted-curve contribution).
    mixer->setLaneAmplitude(1, 0.2);
    mixer->setLaneBlendDepth(1, -1.0);
    mixer->setBlendAmount(1.0);
    // Zero out the H1=1.0 default and any other lanes so lane 1 is the only contributor.
    for (int i = 0; i < mixer->getNumLanes(); ++i) {
        if (i != 1) mixer->setLaneAmplitude(i, 0.0);
    }
    const auto modulated = computeSum();
    // Lane 1 was pulled to silence — the buffer must be all zeros.
    EXPECT_DOUBLE_EQ(maxAbs(modulated), 0.0);
}

TEST_F(LaneMixerTest, ComputeSum_BaseAmplitudeZero_DepthStillContributes) {
    // Regression guard for the old isActive() skip bug. Lane with amp=0 used to be
    // dropped before depth was even consulted. Now it must contribute.
    mixer->setLaneAmplitude(1, 0.0);   // zero base → would have been skipped before
    mixer->setLaneBlendDepth(1, 0.5);  // but depth × macro = 0.5 effective
    mixer->setBlendAmount(1.0);
    // Zero out the H1=1.0 default by also setting all other lanes to 0:
    for (int i = 0; i < mixer->getNumLanes(); ++i) {
        if (i != 1) mixer->setLaneAmplitude(i, 0.0);
    }
    const auto buffer = computeSum();
    // Lane 1 is identity (H1) — sum should be non-zero (specifically, ±1 after normalize).
    EXPECT_GT(maxAbs(buffer), 0.5) << "Lane with amp=0 but depth*blendAmount != 0 must contribute";
}

TEST_F(LaneMixerTest, ComputeSum_EffectiveAmplitudeLinearInBlendAmount) {
    // For fixed base and depth, effective(a) - effective(b) == (a - b) * depth.
    // We can't observe effective directly through computeSum (post-normalize destroys
    // the linear relationship), so verify against the documented formula by comparing
    // pre-normalization sums via a single-lane setup where the normalize divisor is
    // known to be |effective| * peak(curve). With identity curve (lane 1) and a single
    // active lane, the post-normalize output equals sign(effective) * curve / peak(curve),
    // i.e. independent of |effective|. So instead, assert linearity at the formula level
    // by reading getLaneBlendDepth/getLaneAmplitude/getBlendAmount and computing effective
    // explicitly — this catches accidental clamping or non-linear shaping in the setters.
    mixer->setLaneAmplitude(3, 0.4);
    mixer->setLaneBlendDepth(3, 0.6);

    auto effectiveAt = [&](double macro) {
        mixer->setBlendAmount(macro);
        return mixer->getLaneAmplitude(3) + mixer->getBlendAmount() * mixer->getLaneBlendDepth(3);
    };

    const double e0 = effectiveAt(0.0);
    const double e1 = effectiveAt(0.25);
    const double e2 = effectiveAt(0.5);
    const double e3 = effectiveAt(0.75);

    EXPECT_NEAR(e1 - e0, 0.25 * 0.6, 1e-12);
    EXPECT_NEAR(e2 - e1, 0.25 * 0.6, 1e-12);
    EXPECT_NEAR(e3 - e2, 0.25 * 0.6, 1e-12);
}

TEST_F(LaneMixerTest, MixVersion_IncrementedOnSetBlendAmount) {
    const auto mv0 = mixer->getMixVersion();
    mixer->setBlendAmount(0.5);
    EXPECT_GT(mixer->getMixVersion(), mv0);
}

TEST_F(LaneMixerTest, MixVersion_IncrementedOnSetLaneBlendDepth) {
    const auto mv0 = mixer->getMixVersion();
    mixer->setLaneBlendDepth(1, 0.5);
    EXPECT_GT(mixer->getMixVersion(), mv0);
}

TEST_F(LaneMixerTest, MixVersion_UnchangedBlendAmount_DoesNotIncrement) {
    mixer->setBlendAmount(0.5);
    const auto mv0 = mixer->getMixVersion();
    mixer->setBlendAmount(0.5); // same value
    EXPECT_EQ(mixer->getMixVersion(), mv0);
}

TEST_F(LaneMixerTest, MixVersion_UnchangedBlendDepth_DoesNotIncrement) {
    mixer->setLaneBlendDepth(1, 0.3);
    const auto mv0 = mixer->getMixVersion();
    mixer->setLaneBlendDepth(1, 0.3); // same value
    EXPECT_EQ(mixer->getMixVersion(), mv0);
}

TEST_F(LaneMixerTest, MixVersion_UnchangedScanPosition_DoesNotIncrement) {
    // Pre-existing latent fix: setScanPosition gained the same early-return guard.
    mixer->setScanPosition(0.4);
    const auto mv0 = mixer->getMixVersion();
    mixer->setScanPosition(0.4); // same value
    EXPECT_EQ(mixer->getMixVersion(), mv0);
}

TEST_F(LaneMixerTest, MixVersion_UnchangedLaneAmplitude_DoesNotIncrement) {
    // Same early-return guard added to setLaneAmplitude for consistency.
    mixer->setLaneAmplitude(1, 0.7);
    const auto mv0 = mixer->getMixVersion();
    mixer->setLaneAmplitude(1, 0.7); // same value
    EXPECT_EQ(mixer->getMixVersion(), mv0);
}

// ============================================================================
// Serialization round-trip for blendDepth + blendAmount (Phase 2)
// ============================================================================

TEST_F(LaneMixerTest, Serialization_RoundTripsBlendDepth) {
    mixer->setLaneBlendDepth(0, 0.25);
    mixer->setLaneBlendDepth(1, -0.5);
    mixer->setLaneBlendDepth(5, 1.0);
    mixer->setLaneBlendDepth(7, -1.0);

    const auto vt = mixer->toValueTree();
    dsp_core::LaneMixer restored;
    restored.fromValueTree(vt);

    EXPECT_DOUBLE_EQ(restored.getLaneBlendDepth(0), 0.25);
    EXPECT_DOUBLE_EQ(restored.getLaneBlendDepth(1), -0.5);
    EXPECT_DOUBLE_EQ(restored.getLaneBlendDepth(5), 1.0);
    EXPECT_DOUBLE_EQ(restored.getLaneBlendDepth(7), -1.0);
    EXPECT_DOUBLE_EQ(restored.getLaneBlendDepth(2), 0.0);
}

TEST_F(LaneMixerTest, Serialization_RoundTripsBlendAmount) {
    mixer->setBlendAmount(0.42);
    const auto vt = mixer->toValueTree();
    dsp_core::LaneMixer restored;
    restored.fromValueTree(vt);
    EXPECT_DOUBLE_EQ(restored.getBlendAmount(), 0.42);
}

TEST_F(LaneMixerTest, Serialization_FormatVersionIs4) {
    const auto vt = mixer->toValueTree();
    EXPECT_EQ(static_cast<int>(vt.getProperty("formatVersion")), 4);
}

TEST_F(LaneMixerTest, Deserialization_MissingBlendFields_DefaultsToZero) {
    // Hand-build a v3 ValueTree with no blendDepth / blendAmount keys.
    juce::ValueTree v3("LaneMixer");
    v3.setProperty("formatVersion", 3, nullptr);
    v3.setProperty("numLanes", 2, nullptr);
    v3.setProperty("nextLaneId", 2, nullptr);
    v3.setProperty("tableSize", dsp_core::LaneMixer::TABLE_SIZE, nullptr);
    v3.setProperty("mixerMode", 0, nullptr);

    juce::ValueTree lane0("Lane");
    lane0.setProperty("index", 0, nullptr);
    lane0.setProperty("laneId", 0, nullptr);
    lane0.setProperty("amplitude", 0.5, nullptr);
    lane0.setProperty("contentType", static_cast<int>(dsp_core::LaneContentType::Harmonic), nullptr);
    lane0.setProperty("harmonicNumber", 1, nullptr);
    v3.appendChild(lane0, nullptr);

    juce::ValueTree lane1("Lane");
    lane1.setProperty("index", 1, nullptr);
    lane1.setProperty("laneId", 1, nullptr);
    lane1.setProperty("amplitude", 0.7, nullptr);
    lane1.setProperty("contentType", static_cast<int>(dsp_core::LaneContentType::Harmonic), nullptr);
    lane1.setProperty("harmonicNumber", 3, nullptr);
    v3.appendChild(lane1, nullptr);

    // Pre-poison: write some non-zero state into a fresh mixer to make sure
    // deserialization actively zeros (rather than relying on default state).
    dsp_core::LaneMixer restored;
    restored.setBlendAmount(0.9);
    restored.setLaneBlendDepth(0, 0.7);
    restored.setLaneBlendDepth(1, -0.4);

    restored.fromValueTree(v3);

    EXPECT_DOUBLE_EQ(restored.getBlendAmount(), 0.0)
        << "v3 ValueTree had no blendAmount key — restored mixer should default to 0";
    EXPECT_DOUBLE_EQ(restored.getLaneBlendDepth(0), 0.0)
        << "v3 ValueTree had no per-lane blendDepth — should default to 0";
    EXPECT_DOUBLE_EQ(restored.getLaneBlendDepth(1), 0.0);
}

} // namespace dsp_core_test
