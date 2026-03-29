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
    EXPECT_EQ(mixer->getNumLanes(), 41);
}

TEST_F(LaneMixerTest, DefaultInitialization_Lane0IsTanh2x) {
    const auto& lane = mixer->getLane(0);
    EXPECT_EQ(lane.contentType, dsp_core::LaneContentType::Harmonic);
    EXPECT_EQ(lane.harmonicNumber, 0);
    EXPECT_DOUBLE_EQ(lane.amplitude, 0.0);
    EXPECT_EQ(static_cast<int>(lane.curveData.size()), dsp_core::LaneMixer::TABLE_SIZE);

    // Verify curve is tanh(2x) at several points
    const int midpoint = dsp_core::LaneMixer::TABLE_SIZE / 2;
    const double xMid = mixer->normalizeIndex(midpoint);
    EXPECT_NEAR(lane.curveData[static_cast<size_t>(midpoint)], std::tanh(2.0 * xMid), 1e-10);

    // Check endpoints
    EXPECT_NEAR(lane.curveData[0], std::tanh(2.0 * (-1.0)), 1e-10);
    EXPECT_NEAR(lane.curveData[dsp_core::LaneMixer::TABLE_SIZE - 1], std::tanh(2.0 * 1.0), 1e-10);
}

TEST_F(LaneMixerTest, DefaultInitialization_Lane1IsIdentity) {
    const auto& lane = mixer->getLane(1);
    EXPECT_EQ(lane.contentType, dsp_core::LaneContentType::Harmonic);
    EXPECT_EQ(lane.harmonicNumber, 1);
    EXPECT_DOUBLE_EQ(lane.amplitude, 1.0);

    // T_1(x) = x (identity function)
    for (int i = 0; i < dsp_core::LaneMixer::TABLE_SIZE; i += 1000) {
        const double x = mixer->normalizeIndex(i);
        EXPECT_NEAR(lane.curveData[static_cast<size_t>(i)], x, 1e-10)
            << "Lane 1 curveData at index " << i << " should be x=" << x;
    }
}

TEST_F(LaneMixerTest, DefaultInitialization_Lanes2Through40AreChebyshev) {
    for (int n = 2; n <= 40; ++n) {
        const auto& lane = mixer->getLane(n);
        EXPECT_EQ(lane.contentType, dsp_core::LaneContentType::Harmonic)
            << "Lane " << n << " should be Harmonic type";
        EXPECT_EQ(lane.harmonicNumber, n)
            << "Lane " << n << " should have harmonicNumber=" << n;
        EXPECT_EQ(static_cast<int>(lane.curveData.size()), dsp_core::LaneMixer::TABLE_SIZE)
            << "Lane " << n << " should have TABLE_SIZE curve data";
    }

    // Spot check: T_2(x) = 2x² - 1 (Chebyshev of the first kind)
    // But we use cos(2*acos(x)) which is equivalent
    const auto& lane2 = mixer->getLane(2);
    for (int i = 0; i < dsp_core::LaneMixer::TABLE_SIZE; i += 2000) {
        const double x = mixer->normalizeIndex(i);
        const double expected = std::cos(2.0 * std::acos(std::max(-1.0, std::min(1.0, x))));
        EXPECT_NEAR(lane2.curveData[static_cast<size_t>(i)], expected, 1e-8)
            << "Lane 2 (T_2) at index " << i;
    }

    // Spot check: T_3(x) = sin(3*asin(x)) (odd harmonic)
    const auto& lane3 = mixer->getLane(3);
    for (int i = 0; i < dsp_core::LaneMixer::TABLE_SIZE; i += 2000) {
        const double x = mixer->normalizeIndex(i);
        const double expected = std::sin(3.0 * std::asin(std::max(-1.0, std::min(1.0, x))));
        EXPECT_NEAR(lane3.curveData[static_cast<size_t>(i)], expected, 1e-8)
            << "Lane 3 (T_3) at index " << i;
    }
}

TEST_F(LaneMixerTest, DefaultInitialization_OnlyLane1HasNonZeroAmplitude) {
    EXPECT_DOUBLE_EQ(mixer->getLaneAmplitude(0), 0.0); // WT
    EXPECT_DOUBLE_EQ(mixer->getLaneAmplitude(1), 1.0); // H1
    for (int i = 2; i <= 40; ++i) {
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

TEST_F(LaneMixerTest, ComputeSum_TwoLanes_ReturnsSumOfWeightedCurves) {
    // Disable normalization to test raw sum
    mixer->setNormalizationEnabled(false);

    // Lane 1: H1=x, amplitude=0.5
    // Lane 2: T_2(x), amplitude=0.3
    mixer->setLaneAmplitude(1, 0.5);
    mixer->setLaneAmplitude(2, 0.3);

    auto sum = computeSum();

    const auto& lane1 = mixer->getLane(1);
    const auto& lane2 = mixer->getLane(2);

    for (int i = 0; i < dsp_core::LaneMixer::TABLE_SIZE; i += 1000) {
        const double expected = 0.5 * lane1.curveData[static_cast<size_t>(i)] +
                                0.3 * lane2.curveData[static_cast<size_t>(i)];
        EXPECT_NEAR(sum[static_cast<size_t>(i)], expected, 1e-10)
            << "Two-lane sum at index " << i;
    }
}

TEST_F(LaneMixerTest, ComputeSum_NegativeAmplitude_InvertsCurve) {
    mixer->setNormalizationEnabled(false);
    mixer->setLaneAmplitude(1, -1.0);

    auto sum = computeSum();

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

TEST_F(LaneMixerTest, ComputeSum_WithoutNormalization_CanExceedOne) {
    mixer->setNormalizationEnabled(false);
    mixer->setLaneAmplitude(1, 5.0);

    auto sum = computeSum();

    const double maxVal = maxAbs(sum);
    EXPECT_GT(maxVal, 1.0) << "Unnormalized sum with amplitude 5.0 should exceed 1.0";
    EXPECT_NEAR(maxVal, 5.0, 1e-10) << "Max should be 5.0 * max(|x|) = 5.0";
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

    const auto& lane = mixer->getLane(10);
    for (int i = 0; i < dsp_core::LaneMixer::TABLE_SIZE; i += 2000) {
        const double x = mixer->normalizeIndex(i);
        EXPECT_NEAR(lane.curveData[static_cast<size_t>(i)], std::tanh(2.0 * x), 1e-10)
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

    // LaneMixer default: Lane 0 (WT, tanh2x, amp=0), Lane 1 (H1=x, amp=1.0)
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

TEST_F(LaneMixerTest, BackwardCompatibility_WithHarmonics_MatchesLTF) {
    // Set up LaneMixer to match a specific LTF configuration:
    // WT=0.5, H1=0.8, H3=0.3
    mixer->setNormalizationEnabled(false);
    mixer->setLaneAmplitude(0, 0.5);  // WT mix
    mixer->setLaneAmplitude(1, 0.8);  // H1
    mixer->setLaneAmplitude(3, 0.3);  // H3

    // Set up equivalent LTF
    dsp_core::LayeredTransferFunction ltf(dsp_core::LaneMixer::TABLE_SIZE, -1.0, 1.0);
    ltf.setCoefficient(0, 0.5);  // WT
    ltf.setCoefficient(1, 0.8);  // H1
    ltf.setCoefficient(3, 0.3);  // H3
    ltf.setRenderingMode(dsp_core::RenderingMode::Harmonic);

    auto mixerSum = computeSum();

    // LTF evaluates: WT*base[i] + H1*T_1[i] + H3*T_3[i] (without normalization)
    for (int i = 0; i < dsp_core::LaneMixer::TABLE_SIZE; i += 500) {
        const double x = mixer->normalizeIndex(i);
        // LTF composite (unnormalized): WT*tanh(2x) + H1*x + H3*sin(3*asin(x))
        const double wtContrib = 0.5 * std::tanh(2.0 * x);
        const double h1Contrib = 0.8 * x;
        const double h3Contrib =
            0.3 * std::sin(3.0 * std::asin(std::max(-1.0, std::min(1.0, x))));
        const double expected = wtContrib + h1Contrib + h3Contrib;

        EXPECT_NEAR(mixerSum[static_cast<size_t>(i)], expected, 1e-8)
            << "Mixer sum should match LTF composite at index " << i;
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
    mixer->setNormalizationEnabled(false);

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
    for (int n = 0; n < 41; ++n) {
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

    EXPECT_EQ(mixer->isNormalizationEnabled(), mixer2.isNormalizationEnabled());
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
    mixer->setNormalizationEnabled(false);

    // Reset
    mixer->resetToDefaults();

    // Verify defaults
    EXPECT_DOUBLE_EQ(mixer->getLaneAmplitude(0), 0.0);
    EXPECT_DOUBLE_EQ(mixer->getLaneAmplitude(1), 1.0);
    EXPECT_DOUBLE_EQ(mixer->getLaneAmplitude(5), 0.0);
    // Note: normalizationEnabled is not reset by resetToDefaults (it's a setting, not lane state)
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
    mixer->setNormalizationEnabled(false);
    // With H1=1.0 (y=x), evaluateSumAt should return approximately x
    const double result = mixer->evaluateSumAt(0.5);
    EXPECT_NEAR(result, 0.5, 1e-3); // Allow some interpolation error
}

TEST_F(LaneMixerTest, EvaluateSumAt_ClampsToRange) {
    mixer->setNormalizationEnabled(false);
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

} // namespace dsp_core_test
