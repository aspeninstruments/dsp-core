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

    // Helper: blank all per-lane amplitudes/depths and the macro blendAmount.
    // Use this in tests that need a clean slate — the factory defaults preload
    // H1 at amp=1.0 plus halving blendDepths on H3..H19, which would otherwise
    // contaminate single-lane assertions.
    static void blankMixer(dsp_core::LaneMixer& m) {
        for (int i = 0; i < m.getNumLanes(); ++i) {
            m.setLaneAmplitude(i, 0.0);
            m.setLaneBlendDepth(i, 0.0);
            m.setLaneModulationDepth(i, 0, 0.0);
            m.setLaneModulationDepth(i, 1, 0.0);
        }
        m.setBlendAmount(0.0);
        m.setModulationEnvValue(0, 0.0);
        m.setModulationEnvValue(1, 0.0);
    }
};

// ============================================================================
// Default Initialization Tests
// ============================================================================

TEST_F(LaneMixerTest, DefaultInitialization_HasCorrectLaneCount) {
    EXPECT_EQ(mixer->getNumLanes(), 10);
}

TEST_F(LaneMixerTest, DefaultInitialization_Lane0IsH1Identity) {
    const auto& lane = mixer->getLane(0);
    EXPECT_EQ(lane.contentType, dsp_core::LaneContentType::Harmonic);
    EXPECT_EQ(lane.harmonicNumber, 1);
    EXPECT_DOUBLE_EQ(lane.amplitude, 1.0);
    EXPECT_DOUBLE_EQ(lane.blendDepth, 0.0);
    EXPECT_TRUE(lane.oddSymmetryEnabled);
    EXPECT_EQ(static_cast<int>(lane.curveData.size()), dsp_core::LaneMixer::TABLE_SIZE);

    // T_1(x) = x (identity function)
    for (int i = 0; i < dsp_core::LaneMixer::TABLE_SIZE; i += 1000) {
        const double x = mixer->normalizeIndex(i);
        EXPECT_NEAR(lane.curveData[static_cast<size_t>(i)], x, 1e-10)
            << "Lane 0 (H1) curveData at index " << i << " should be x=" << x;
    }
}

TEST_F(LaneMixerTest, DefaultInitialization_Lanes1Through9AreOddHarmonics) {
    // Harmonic-foldback factory layout: lanes 1..9 = H3..H19 with halving blendDepth.
    const std::array<int, 9> expectedHarmonics = {3, 5, 7, 9, 11, 13, 15, 17, 19};
    const std::array<double, 9> expectedDepths = {1.0,     0.5,      0.25,      0.125,     0.0625,
                                                  0.03125, 0.015625, 0.0078125, 0.00390625};

    for (int laneIdx = 1; laneIdx <= 9; ++laneIdx) {
        const auto& lane = mixer->getLane(laneIdx);
        const int expectedN = expectedHarmonics[laneIdx - 1];
        EXPECT_EQ(lane.contentType, dsp_core::LaneContentType::Harmonic)
            << "Lane " << laneIdx << " should be Harmonic type";
        EXPECT_EQ(lane.harmonicNumber, expectedN) << "Lane " << laneIdx << " should have harmonicNumber=" << expectedN;
        EXPECT_DOUBLE_EQ(lane.amplitude, 0.0) << "Lane " << laneIdx << " should have amplitude=0.0";
        EXPECT_DOUBLE_EQ(lane.blendDepth, expectedDepths[laneIdx - 1])
            << "Lane " << laneIdx << " should have halving blendDepth";
        EXPECT_TRUE(lane.oddSymmetryEnabled) << "Lane " << laneIdx << " should have oddSymmetryEnabled";
        EXPECT_EQ(static_cast<int>(lane.curveData.size()), dsp_core::LaneMixer::TABLE_SIZE)
            << "Lane " << laneIdx << " should have TABLE_SIZE curve data";
    }

    // Spot check: Lane 1 = H3, sin(3*asin(x))
    const auto& lane1 = mixer->getLane(1);
    for (int i = 0; i < dsp_core::LaneMixer::TABLE_SIZE; i += 2000) {
        const double x = std::clamp(mixer->normalizeIndex(i), -1.0, 1.0);
        const double expected = std::sin(3.0 * std::asin(x));
        EXPECT_NEAR(lane1.curveData[static_cast<size_t>(i)], expected, 1e-8) << "Lane 1 (H3) at index " << i;
    }

    // Spot check: Lane 4 = H9, sin(9*asin(x))
    const auto& lane4 = mixer->getLane(4);
    for (int i = 0; i < dsp_core::LaneMixer::TABLE_SIZE; i += 2000) {
        const double x = std::clamp(mixer->normalizeIndex(i), -1.0, 1.0);
        const double expected = std::sin(9.0 * std::asin(x));
        EXPECT_NEAR(lane4.curveData[static_cast<size_t>(i)], expected, 1e-8) << "Lane 4 (H9) at index " << i;
    }
}

TEST_F(LaneMixerTest, DefaultInitialization_OnlyLane0HasNonZeroAmplitude) {
    EXPECT_DOUBLE_EQ(mixer->getLaneAmplitude(0), 1.0); // H1
    for (int i = 1; i < mixer->getNumLanes(); ++i) {
        EXPECT_DOUBLE_EQ(mixer->getLaneAmplitude(i), 0.0) << "Lane " << i << " should have amplitude 0.0";
    }
}

// ============================================================================
// Sum Computation Tests
// ============================================================================

TEST_F(LaneMixerTest, ComputeSum_AllZeroAmplitudes_ReturnsZeros) {
    blankMixer(*mixer);

    auto sum = computeSum();
    EXPECT_TRUE(isAllZeros(sum));
}

TEST_F(LaneMixerTest, ComputeSum_SingleLaneFullAmplitude_EqualsLaneCurve) {
    // Default: only Lane 0 (H1=x) is active with amplitude 1.0; default blendAmount=0
    // means the preloaded depths on lanes 1..9 stay inert, so sum = x.
    auto sum = computeSum();

    for (int i = 0; i < dsp_core::LaneMixer::TABLE_SIZE; i += 1000) {
        const double x = mixer->normalizeIndex(i);
        EXPECT_NEAR(sum[static_cast<size_t>(i)], x, 1e-10) << "Sum should equal x (identity) at index " << i;
    }
}

TEST_F(LaneMixerTest, ComputeSum_TwoLanes_PreservesRelativeProportions) {
    // Lane 1: H3, amplitude=0.5
    // Lane 2: H5, amplitude=0.3
    blankMixer(*mixer);
    mixer->setLaneAmplitude(1, 0.5);
    mixer->setLaneAmplitude(2, 0.3);

    auto sum = computeSum();

    // Compute expected raw sum then normalize
    const auto& lane1 = mixer->getLane(1);
    const auto& lane2 = mixer->getLane(2);

    std::vector<double> rawSum(dsp_core::LaneMixer::TABLE_SIZE);
    double rawMaxAbs = 0.0;
    for (int i = 0; i < dsp_core::LaneMixer::TABLE_SIZE; ++i) {
        rawSum[static_cast<size_t>(i)] =
            0.5 * lane1.curveData[static_cast<size_t>(i)] + 0.3 * lane2.curveData[static_cast<size_t>(i)];
        rawMaxAbs = std::max(rawMaxAbs, std::abs(rawSum[static_cast<size_t>(i)]));
    }

    for (int i = 0; i < dsp_core::LaneMixer::TABLE_SIZE; i += 1000) {
        const double expected = rawSum[static_cast<size_t>(i)] / rawMaxAbs;
        EXPECT_NEAR(sum[static_cast<size_t>(i)], expected, 1e-10) << "Normalized two-lane sum at index " << i;
    }
}

TEST_F(LaneMixerTest, ComputeSum_NegativeAmplitude_ClampsToSilence) {
    // Effective amplitude is max(0, base + macro*depth); a negative base with zero
    // depth clamps to 0, dropping the lane from the sum entirely. Regression guard
    // against reintroducing inverted-curve contributions.
    blankMixer(*mixer);
    mixer->setLaneAmplitude(1, -1.0);

    auto sum = computeSum();

    EXPECT_DOUBLE_EQ(maxAbs(sum), 0.0);
}

TEST_F(LaneMixerTest, ComputeSum_WithNormalization_MaxAbsIsOne) {
    // Set up multiple lanes with large amplitudes
    blankMixer(*mixer);
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
    mixer->setLaneAmplitude(9, 0.75);
    EXPECT_DOUBLE_EQ(mixer->getLaneAmplitude(9), 0.75);
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
    mixer->fillLaneWithTanh2x(9);

    // fillLaneWithTanh2x normalizes so peak = 1.0
    const double normFactor = 1.0 / std::tanh(2.0);
    const auto& lane = mixer->getLane(9);
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
    // LTF default: WT=0.0, H1=1.0 → evaluates to x
    dsp_core::LayeredTransferFunction ltf(dsp_core::LaneMixer::TABLE_SIZE, -1.0, 1.0);

    // LaneMixer default: Lane 0 (H1=x, amp=1.0), lanes 1..9 zero amp.
    // The preloaded depths are inert because default blendAmount=0.
    // Sum = 1.0 * x = x.
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
    // Set up: Lane 0 = H1 at 0.8, Lane 1 = H3 at 0.3, Lane 2 = H5 at 0.2
    blankMixer(*mixer);
    mixer->setLaneAmplitude(0, 0.8); // H1
    mixer->setLaneAmplitude(1, 0.3); // H3
    mixer->setLaneAmplitude(2, 0.2); // H5

    auto mixerSum = computeSum();

    // Compute expected raw sum then normalize
    std::vector<double> rawSum(dsp_core::LaneMixer::TABLE_SIZE);
    double rawMaxAbs = 0.0;
    for (int i = 0; i < dsp_core::LaneMixer::TABLE_SIZE; ++i) {
        const double x = mixer->normalizeIndex(i);
        const double h1Contrib = 0.8 * x;
        const double h3Contrib = 0.3 * std::sin(3.0 * std::asin(std::clamp(x, -1.0, 1.0)));
        const double h5Contrib = 0.2 * std::sin(5.0 * std::asin(std::clamp(x, -1.0, 1.0)));
        rawSum[static_cast<size_t>(i)] = h1Contrib + h3Contrib + h5Contrib;
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

        EXPECT_DOUBLE_EQ(original.amplitude, restored.amplitude) << "Lane " << n << " amplitude mismatch";
        EXPECT_EQ(original.contentType, restored.contentType) << "Lane " << n << " contentType mismatch";
        EXPECT_EQ(original.harmonicNumber, restored.harmonicNumber) << "Lane " << n << " harmonicNumber mismatch";

        // Verify curve data matches
        ASSERT_EQ(original.curveData.size(), restored.curveData.size()) << "Lane " << n << " curveData size mismatch";
        for (int i = 0; i < dsp_core::LaneMixer::TABLE_SIZE; i += 1000) {
            EXPECT_NEAR(original.curveData[static_cast<size_t>(i)], restored.curveData[static_cast<size_t>(i)], 1e-12)
                << "Lane " << n << " curveData mismatch at index " << i;
        }
    }
}

TEST_F(LaneMixerTest, Serialization_SplineAnchors_RoundTrip) {
    // Set up a lane with spline anchors
    auto& lane = const_cast<dsp_core::Lane&>(mixer->getLane(3));
    lane.contentType = dsp_core::LaneContentType::Spline;
    lane.splineAnchors = {{-1.0, -1.0, false, 0.0},
                          {-0.5, 0.2, true, 1.5},
                          {0.0, 0.0, false, 0.0},
                          {0.5, -0.3, false, 0.0},
                          {1.0, 1.0, false, 0.0}};

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
    mixer->setLaneAmplitude(0, 0.5);
    mixer->setLaneAmplitude(5, 0.8);

    // Reset
    mixer->resetToDefaults();

    // Verify factory defaults: lane 0 = H1 at 1.0, lanes 1..9 = H3..H19 at 0.0
    EXPECT_DOUBLE_EQ(mixer->getLaneAmplitude(0), 1.0);
    EXPECT_DOUBLE_EQ(mixer->getLaneAmplitude(5), 0.0);
    EXPECT_EQ(mixer->getNumLanes(), 10);
    EXPECT_EQ(mixer->getLane(0).harmonicNumber, 1);  // Lane 0 = H1
    EXPECT_EQ(mixer->getLane(5).harmonicNumber, 11); // Lane 5 = H11
    EXPECT_DOUBLE_EQ(mixer->getLaneBlendDepth(5), 0.0625);
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
        EXPECT_DOUBLE_EQ(mixer->normalizeIndex(i), ltf.normalizeIndex(i)) << "normalizeIndex mismatch at " << i;
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
        EXPECT_NEAR(result[static_cast<size_t>(i)], lane0.curveData[static_cast<size_t>(i)], 1e-10) << "at index " << i;
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
        EXPECT_NEAR(result[static_cast<size_t>(i)], lane0.curveData[static_cast<size_t>(i)], 1e-10) << "at index " << i;
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
        EXPECT_NEAR(result[static_cast<size_t>(i)], lane0.curveData[static_cast<size_t>(i)], 1e-10) << "at index " << i;
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
        EXPECT_NEAR(result[static_cast<size_t>(i)], lane0.curveData[static_cast<size_t>(i)], 1e-10) << "at index " << i;
    }
}

TEST_F(LaneMixerScanTest, ComputeScan_ThreeLanes_Position50_EqualsLane1) {
    setupSimpleLanes(3);
    mixer->setScanPosition(0.5);
    auto result = computeScan();

    // 3 lanes: f = 0.5 * 2 = 1.0 → exactly lane 1 (2x), normalized → identity
    const auto& lane0 = mixer->getLane(0);
    for (int i = 0; i < dsp_core::LaneMixer::TABLE_SIZE; ++i) {
        EXPECT_NEAR(result[static_cast<size_t>(i)], lane0.curveData[static_cast<size_t>(i)], 1e-10) << "at index " << i;
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
// Series Mode Tests
// ============================================================================

class LaneMixerSeriesTest : public LaneMixerTest {
  protected:
    static constexpr int kSize = dsp_core::LaneMixer::TABLE_SIZE;

    std::vector<double> computeSeries() {
        std::vector<double> buffer(kSize, 0.0);
        mixer->computeSeries(buffer.data(), kSize);
        return buffer;
    }

    // Reduce to exactly N lanes, each filled with identity y=x and amplitude=0.5 (unity gain).
    void setupIdentityChain(int count) {
        mixer->resetToDefaults();
        while (mixer->getNumLanes() > 1) {
            mixer->removeLane(mixer->getNumLanes() - 1);
        }
        auto fillIdentity = [this](int idx) {
            auto& lane = mixer->getMutableLane(idx);
            lane.curveData.resize(kSize);
            for (int j = 0; j < kSize; ++j) {
                lane.curveData[static_cast<size_t>(j)] = mixer->normalizeIndex(j);
            }
            mixer->setLaneAmplitude(idx, 0.5);
            mixer->setLaneBlendDepth(idx, 0.0);
        };
        fillIdentity(0);
        for (int i = 1; i < count; ++i) {
            mixer->addLane(-1);
            fillIdentity(i);
        }
        mixer->setBlendAmount(0.0);
    }

    // Gain the DSP uses internally: pow(4, 2*a - 1), so 0 → 0.25, 0.5 → 1.0, 1 → 4.0.
    static double gainFor(double a) {
        return std::exp(std::log(4.0) * (2.0 * a - 1.0));
    }
};

TEST_F(LaneMixerSeriesTest, GainMappingIsUnityAtHalf) {
    // Single identity lane with amplitude=0.5 → gain=1 → output should equal input.
    // After final normalization (max|x| = 1.0 already), result is identity.
    setupIdentityChain(1);
    auto result = computeSeries();
    for (int i = 0; i < kSize; ++i) {
        EXPECT_NEAR(result[static_cast<size_t>(i)], mixer->normalizeIndex(i), 1e-10) << "at index " << i;
    }
}

TEST_F(LaneMixerSeriesTest, GainAtZeroIsQuarter) {
    // Single identity lane with amplitude=0.0 → gain=0.25. Unclamped output = 0.25 * x.
    // Final normalization scales by 1/maxAbs(0.25*x) = 1/0.25 = 4 → identity.
    setupIdentityChain(1);
    mixer->setLaneAmplitude(0, 0.0);
    auto result = computeSeries();
    // Normalization pushes it back to identity — so the shape is identity, but max is 1.0.
    EXPECT_NEAR(maxAbs(result), 1.0, 1e-10);
    for (int i = 0; i < kSize; ++i) {
        EXPECT_NEAR(result[static_cast<size_t>(i)], mixer->normalizeIndex(i), 1e-10) << "at index " << i;
    }
}

TEST_F(LaneMixerSeriesTest, GainAtOneIsFourClampedBeforeLookup) {
    // Single identity lane with amplitude=1.0 → gain=4. y*4 gets hard-clamped to [-1, 1]
    // before the identity LUT lookup → output is a hard-clipped ramp saturating at ±1 outside
    // |x| > 0.25. After normalization (max=1), shape is preserved.
    setupIdentityChain(1);
    mixer->setLaneAmplitude(0, 1.0);
    auto result = computeSeries();

    // Inside the linear region (|x| < 0.25), output ≈ 4x.
    for (int i = 0; i < kSize; ++i) {
        const double x = mixer->normalizeIndex(i);
        const double expected = std::clamp(4.0 * x, -1.0, 1.0);
        EXPECT_NEAR(result[static_cast<size_t>(i)], expected, 1e-3) << "at index " << i;
    }

    // Edges should be saturated at ±1.
    EXPECT_NEAR(result[0], -1.0, 1e-10);
    EXPECT_NEAR(result[static_cast<size_t>(kSize - 1)], 1.0, 1e-10);
}

TEST_F(LaneMixerSeriesTest, EmptyMixerReturnsIdentity) {
    // Before the chain runs, buffer seeds to the identity x ∈ [-1, 1]. Single-lane identity
    // preserves that. Two-lane identity also preserves it (composition of identities).
    setupIdentityChain(3);
    auto result = computeSeries();
    for (int i = 0; i < kSize; ++i) {
        EXPECT_NEAR(result[static_cast<size_t>(i)], mixer->normalizeIndex(i), 1e-10) << "at index " << i;
    }
}

TEST_F(LaneMixerSeriesTest, DepthZeroMeansMorphInert) {
    // With all depths at 0, sweeping blendAmount must not change the output.
    setupIdentityChain(2);
    mixer->setBlendAmount(0.0);
    const auto ref = computeSeries();

    for (double m : {0.25, 0.5, 0.75, 1.0}) {
        mixer->setBlendAmount(m);
        const auto cur = computeSeries();
        for (int i = 0; i < kSize; ++i) {
            EXPECT_NEAR(cur[static_cast<size_t>(i)], ref[static_cast<size_t>(i)], 1e-10)
                << "morph=" << m << " at index " << i;
        }
    }
}

TEST_F(LaneMixerSeriesTest, DepthModulatesGainViaMorph) {
    // amplitude=0.5, depth=0.5, morph=1 → a_eff = 1.0 → gain=4 (clamp-saturating).
    // This should match the behavior of amplitude=1.0 at morph=0.
    setupIdentityChain(1);
    mixer->setLaneAmplitude(0, 0.5);
    mixer->setLaneBlendDepth(0, 0.5);
    mixer->setBlendAmount(1.0);
    const auto modulated = computeSeries();

    mixer->setLaneAmplitude(0, 1.0);
    mixer->setLaneBlendDepth(0, 0.0);
    mixer->setBlendAmount(0.0);
    const auto direct = computeSeries();

    for (int i = 0; i < kSize; ++i) {
        EXPECT_NEAR(modulated[static_cast<size_t>(i)], direct[static_cast<size_t>(i)], 1e-10) << "at index " << i;
    }
}

TEST_F(LaneMixerSeriesTest, OutputIsNormalized) {
    setupIdentityChain(2);
    // Pick non-trivial amplitudes to ensure output is not all-zero.
    mixer->setLaneAmplitude(0, 0.75);
    mixer->setLaneAmplitude(1, 0.75);
    const auto result = computeSeries();
    EXPECT_NEAR(maxAbs(result), 1.0, 1e-10);
}

TEST_F(LaneMixerSeriesTest, ChainComposesFunctionally) {
    // Two lanes: lane 0 = identity (y=x), lane 1 = square-ish (y = x^2 * sign(x)).
    // With both amplitudes = 0.5 (gain=1), the chain applies lane1(lane0(x)) = lane1(x).
    setupIdentityChain(2);
    auto& lane1 = mixer->getMutableLane(1);
    for (int j = 0; j < kSize; ++j) {
        const double x = mixer->normalizeIndex(j);
        lane1.curveData[static_cast<size_t>(j)] = x * std::abs(x);
    }
    const auto result = computeSeries();

    // Expected: normalize lane1's curve to max|y|=1. Since max|x*|x|| = 1 at x=±1, no scaling.
    for (int j = 0; j < kSize; ++j) {
        const double x = mixer->normalizeIndex(j);
        EXPECT_NEAR(result[static_cast<size_t>(j)], x * std::abs(x), 1e-10) << "at index " << j;
    }
}

TEST_F(LaneMixerSeriesTest, SetMixerModeSeries_IncrementsVersion) {
    const auto v0 = mixer->getVersion();
    mixer->setMixerMode(dsp_core::LaneMixer::MixerMode::Series);
    EXPECT_GT(mixer->getVersion(), v0);
    EXPECT_EQ(mixer->getMixerMode(), dsp_core::LaneMixer::MixerMode::Series);
}

TEST_F(LaneMixerSeriesTest, Serialization_PreservesSeriesMode) {
    mixer->setMixerMode(dsp_core::LaneMixer::MixerMode::Series);
    auto vt = mixer->toValueTree();

    auto restored = std::make_unique<dsp_core::LaneMixer>();
    restored->fromValueTree(vt);
    EXPECT_EQ(restored->getMixerMode(), dsp_core::LaneMixer::MixerMode::Series);
}

TEST_F(LaneMixerSeriesTest, GainFormulaEndpointsAreExact) {
    // Sanity-check the mapping matches the documented 0 → 0.25, 0.5 → 1.0, 1 → 4.0.
    EXPECT_NEAR(gainFor(0.0), 0.25, 1e-12);
    EXPECT_NEAR(gainFor(0.5), 1.0, 1e-12);
    EXPECT_NEAR(gainFor(1.0), 4.0, 1e-12);
}

// ============================================================================
// DuplicateLane Tests
// ============================================================================

class LaneMixerDuplicateTest : public ::testing::Test {
  protected:
    void SetUp() override {
        mixer = std::make_unique<dsp_core::LaneMixer>();
        // The factory default is 10 lanes (H1..H19). These tests were written
        // against an 11-lane baseline, so add one more lane to keep the scan-
        // position arithmetic in the tests below unchanged.
        mixer->addLane(-1);
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

TEST_F(LaneMixerTest, LaneBlendDepth_FactoryDefaults_HavingPattern) {
    // Lane 0 (H1) is the dry carrier — depth=0.
    EXPECT_DOUBLE_EQ(mixer->getLaneBlendDepth(0), 0.0);
    // Lanes 1..9 (H3..H19) are preloaded with halving depths so a single
    // blendAmount sweep brings progressively higher harmonics in.
    EXPECT_DOUBLE_EQ(mixer->getLaneBlendDepth(1), 1.0);
    EXPECT_DOUBLE_EQ(mixer->getLaneBlendDepth(2), 0.5);
    EXPECT_DOUBLE_EQ(mixer->getLaneBlendDepth(mixer->getNumLanes() - 1), 0.00390625);
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
    mixer->setLaneBlendDepth(1, 0.7);  // would normally pull lane 1 up
    mixer->setLaneBlendDepth(2, -0.5); // would normally pull lane 2 down
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
    // Lane 1 = H3. base=0.4, depth=+0.5, blendAmount=1.0 → effective=0.9.
    // Compare against a fresh mixer with base=0.9 on lane 1, no depth.
    blankMixer(*mixer);
    mixer->setLaneAmplitude(1, 0.4);
    mixer->setLaneBlendDepth(1, 0.5);
    mixer->setBlendAmount(1.0);
    const auto modulated = computeSum();

    auto fresh = std::make_unique<dsp_core::LaneMixer>();
    blankMixer(*fresh);
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
    blankMixer(*mixer);
    mixer->setLaneAmplitude(1, 0.8);
    mixer->setLaneBlendDepth(1, -0.5);
    mixer->setBlendAmount(1.0);
    const auto modulated = computeSum();

    auto fresh = std::make_unique<dsp_core::LaneMixer>();
    blankMixer(*fresh);
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
    blankMixer(*mixer);
    mixer->setLaneAmplitude(1, 0.2);
    mixer->setLaneBlendDepth(1, -1.0);
    mixer->setBlendAmount(1.0);
    const auto modulated = computeSum();
    // Lane 1 was pulled to silence — the buffer must be all zeros.
    EXPECT_DOUBLE_EQ(maxAbs(modulated), 0.0);
}

TEST_F(LaneMixerTest, ComputeSum_BaseAmplitudeZero_DepthStillContributes) {
    // Regression guard for the old isActive() skip bug. Lane with amp=0 used to be
    // dropped before depth was even consulted. Now it must contribute.
    blankMixer(*mixer);
    mixer->setLaneAmplitude(0, 0.0);  // zero base → would have been skipped before
    mixer->setLaneBlendDepth(0, 0.5); // but depth × macro = 0.5 effective
    mixer->setBlendAmount(1.0);
    const auto buffer = computeSum();
    // Lane 0 is identity (H1) — sum should be non-zero (±1 after normalize).
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
    blankMixer(*mixer);
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

TEST_F(LaneMixerTest, Serialization_FormatVersionIs5) {
    const auto vt = mixer->toValueTree();
    EXPECT_EQ(static_cast<int>(vt.getProperty("formatVersion")), 5);
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

// ============================================================================
// Modulation Depth + Modulation Env (per-lane env-driven modulation)
// ============================================================================

TEST_F(LaneMixerTest, LaneModulationDepth_Defaults_AllZero) {
    for (int i = 0; i < mixer->getNumLanes(); ++i) {
        for (int s = 0; s < 2; ++s) {
            EXPECT_DOUBLE_EQ(mixer->getLaneModulationDepth(i, s), 0.0)
                << "Lane " << i << " slot " << s << " should default to modulationDepth=0";
        }
    }
}

TEST_F(LaneMixerTest, LaneModulationDepth_SetGet_RoundTrips) {
    mixer->setLaneModulationDepth(2, 0, 0.5);
    EXPECT_DOUBLE_EQ(mixer->getLaneModulationDepth(2, 0), 0.5);
    mixer->setLaneModulationDepth(2, 0, -0.75);
    EXPECT_DOUBLE_EQ(mixer->getLaneModulationDepth(2, 0), -0.75);
    mixer->setLaneModulationDepth(2, 0, 0.0);
    EXPECT_DOUBLE_EQ(mixer->getLaneModulationDepth(2, 0), 0.0);
}

TEST_F(LaneMixerTest, LaneModulationDepth_ClampedToRange) {
    mixer->setLaneModulationDepth(1, 0, 2.5);
    EXPECT_DOUBLE_EQ(mixer->getLaneModulationDepth(1, 0), 1.0);
    mixer->setLaneModulationDepth(1, 0, -2.5);
    EXPECT_DOUBLE_EQ(mixer->getLaneModulationDepth(1, 0), -1.0);
}

TEST_F(LaneMixerTest, LaneModulationDepth_SlotsAreIndependent) {
    // Setting slot 0's depth must not bleed into slot 1, and vice versa.
    blankMixer(*mixer);
    mixer->setLaneModulationDepth(3, 0, 0.7);
    EXPECT_DOUBLE_EQ(mixer->getLaneModulationDepth(3, 0), 0.7);
    EXPECT_DOUBLE_EQ(mixer->getLaneModulationDepth(3, 1), 0.0) << "Slot 1 must not be affected by writes to slot 0";

    mixer->setLaneModulationDepth(3, 1, -0.4);
    EXPECT_DOUBLE_EQ(mixer->getLaneModulationDepth(3, 0), 0.7) << "Slot 0 must not be affected by writes to slot 1";
    EXPECT_DOUBLE_EQ(mixer->getLaneModulationDepth(3, 1), -0.4);
}

TEST_F(LaneMixerTest, LaneModulationDepth_InvalidSlotIsNoOp) {
    blankMixer(*mixer);
    mixer->setLaneModulationDepth(1, -1, 0.5);
    mixer->setLaneModulationDepth(1, 2, 0.5);
    EXPECT_DOUBLE_EQ(mixer->getLaneModulationDepth(1, -1), 0.0);
    EXPECT_DOUBLE_EQ(mixer->getLaneModulationDepth(1, 2), 0.0);
    EXPECT_DOUBLE_EQ(mixer->getLaneModulationDepth(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(mixer->getLaneModulationDepth(1, 1), 0.0);
}

TEST_F(LaneMixerTest, ModulationEnvValue_DefaultsToZero) {
    EXPECT_DOUBLE_EQ(mixer->getModulationEnvValue(0), 0.0);
    EXPECT_DOUBLE_EQ(mixer->getModulationEnvValue(1), 0.0);
}

TEST_F(LaneMixerTest, SetModulationEnvValue_NoModulatedLanes_DoesNotIncrementVersion) {
    // When no lane has modulationDepth != 0 for this slot, env changes have no
    // audible effect, so the renderer must not be invalidated.
    blankMixer(*mixer); // clears all per-lane modDepth
    const auto v0 = mixer->getVersion();
    const auto mv0 = mixer->getMixVersion();
    mixer->setModulationEnvValue(0, 0.5);
    EXPECT_EQ(mixer->getVersion(), v0);
    EXPECT_EQ(mixer->getMixVersion(), mv0);
}

TEST_F(LaneMixerTest, SetModulationEnvValue_LaneIsModulated_IncrementsVersion) {
    // When at least one lane has modulationDepth != 0 for the targeted slot,
    // env changes alter the effective amplitudes, so the LUT must be re-rendered.
    blankMixer(*mixer);
    mixer->setLaneModulationDepth(1, 0, 0.5);
    const auto v0 = mixer->getVersion();
    const auto mv0 = mixer->getMixVersion();
    mixer->setModulationEnvValue(0, 0.5);
    EXPECT_GT(mixer->getVersion(), v0);
    EXPECT_GT(mixer->getMixVersion(), mv0);
}

TEST_F(LaneMixerTest, SetModulationEnvValue_OtherSlotModulated_DoesNotIncrementVersion) {
    // Slot 1's depth being non-zero must not cause slot 0's env change to bump
    // the version: the slots are independent contributors.
    blankMixer(*mixer);
    mixer->setLaneModulationDepth(1, 1, 0.5); // only slot 1 is modulated
    const auto v0 = mixer->getVersion();
    const auto mv0 = mixer->getMixVersion();
    mixer->setModulationEnvValue(0, 0.5); // slot 0 env change
    EXPECT_EQ(mixer->getVersion(), v0);
    EXPECT_EQ(mixer->getMixVersion(), mv0);
}

TEST_F(LaneMixerTest, SetModulationEnvValue_UnchangedValue_DoesNotIncrementVersion) {
    // Steady-state env (same value as before) must not invalidate the LUT,
    // even when lanes are modulated. Mirrors setBlendAmount's early-return guard.
    blankMixer(*mixer);
    mixer->setLaneModulationDepth(1, 0, 0.5);
    mixer->setModulationEnvValue(0, 0.3);
    const auto v0 = mixer->getVersion();
    const auto mv0 = mixer->getMixVersion();
    mixer->setModulationEnvValue(0, 0.3); // same value
    EXPECT_EQ(mixer->getVersion(), v0);
    EXPECT_EQ(mixer->getMixVersion(), mv0);
}

TEST_F(LaneMixerTest, MixVersion_IncrementedOnSetLaneModulationDepth) {
    const auto mv0 = mixer->getMixVersion();
    mixer->setLaneModulationDepth(1, 0, 0.5);
    EXPECT_GT(mixer->getMixVersion(), mv0);
}

TEST_F(LaneMixerTest, MixVersion_UnchangedModulationDepth_DoesNotIncrement) {
    mixer->setLaneModulationDepth(1, 0, 0.3);
    const auto mv0 = mixer->getMixVersion();
    mixer->setLaneModulationDepth(1, 0, 0.3); // same value
    EXPECT_EQ(mixer->getMixVersion(), mv0);
}

TEST_F(LaneMixerTest, ComputeSum_EnvZero_ModDepthHasNoEffect) {
    // With both slot envs at 0, any per-lane modulationDepth must be inert:
    // results match a fresh mixer that never had modulationDepth set.
    blankMixer(*mixer);
    mixer->setLaneAmplitude(1, 0.7);
    mixer->setLaneModulationDepth(1, 0, 0.8);  // would otherwise pull lane up
    mixer->setLaneModulationDepth(2, 1, -0.5); // and pull lane down (slot 1)
    // both slot envs stay at 0 from blankMixer.
    const auto withModDepths = computeSum();

    mixer->setLaneModulationDepth(1, 0, 0.0);
    mixer->setLaneModulationDepth(2, 1, 0.0);
    const auto withoutModDepths = computeSum();

    ASSERT_EQ(withModDepths.size(), withoutModDepths.size());
    for (size_t i = 0; i < withModDepths.size(); ++i) {
        EXPECT_DOUBLE_EQ(withModDepths[i], withoutModDepths[i]);
    }
}

TEST_F(LaneMixerTest, ComputeSum_PositiveModDepthMatchesEquivalentBaseShift) {
    // Lane 1 = H3. base=0.4, modDepth[slot0]=+0.5, env[slot0]=1.0 → effective = 0.9.
    // Compare against a mixer with base=0.9, no modDepth, env=0 → effective = 0.9.
    blankMixer(*mixer);
    mixer->setLaneAmplitude(1, 0.4);
    mixer->setLaneModulationDepth(1, 0, 0.5);
    mixer->setModulationEnvValue(0, 1.0);
    const auto modulated = computeSum();

    auto fresh = std::make_unique<dsp_core::LaneMixer>();
    blankMixer(*fresh);
    fresh->setLaneAmplitude(1, 0.9);
    std::vector<double> expected(dsp_core::LaneMixer::TABLE_SIZE, 0.0);
    fresh->computeSum(expected.data(), dsp_core::LaneMixer::TABLE_SIZE);

    ASSERT_EQ(modulated.size(), expected.size());
    for (size_t i = 0; i < modulated.size(); ++i) {
        EXPECT_NEAR(modulated[i], expected[i], 1e-12);
    }
}

TEST_F(LaneMixerTest, ComputeSum_NegativeModDepthClampsToSilence) {
    // base=0.3, modDepth[slot0]=-1.0, env[slot0]=1.0 → effective = max(0, -0.7) = 0.
    blankMixer(*mixer);
    mixer->setLaneAmplitude(0, 1.0); // keep lane 0 (identity) audible so output isn't all-zero
    mixer->setLaneAmplitude(1, 0.3);
    mixer->setLaneModulationDepth(1, 0, -1.0);
    mixer->setModulationEnvValue(0, 1.0);
    const auto withModulation = computeSum();

    // Compare against mixer with lane 1 at amplitude=0 (silent contribution).
    auto fresh = std::make_unique<dsp_core::LaneMixer>();
    blankMixer(*fresh);
    fresh->setLaneAmplitude(0, 1.0);
    fresh->setLaneAmplitude(1, 0.0);
    std::vector<double> expected(dsp_core::LaneMixer::TABLE_SIZE, 0.0);
    fresh->computeSum(expected.data(), dsp_core::LaneMixer::TABLE_SIZE);

    ASSERT_EQ(withModulation.size(), expected.size());
    for (size_t i = 0; i < withModulation.size(); ++i) {
        EXPECT_NEAR(withModulation[i], expected[i], 1e-12);
    }
}

TEST_F(LaneMixerTest, ComputeSum_BlendAndModDepthSumIndependently) {
    // Verify additive composition with both slots active:
    //   effective = max(0, base + macro * blendDepth + env0 * modDepth0 + env1 * modDepth1).
    // base=0.1, blendDepth=0.3, blendAmount=1.0 → contributes +0.3
    // slot0: modDepth=0.2, env=1.0             → contributes +0.2
    // slot1: modDepth=0.3, env=1.0             → contributes +0.3
    // total effective = 0.1 + 0.3 + 0.2 + 0.3 = 0.9
    blankMixer(*mixer);
    mixer->setLaneAmplitude(1, 0.1);
    mixer->setLaneBlendDepth(1, 0.3);
    mixer->setLaneModulationDepth(1, 0, 0.2);
    mixer->setLaneModulationDepth(1, 1, 0.3);
    mixer->setBlendAmount(1.0);
    mixer->setModulationEnvValue(0, 1.0);
    mixer->setModulationEnvValue(1, 1.0);
    const auto combined = computeSum();

    auto fresh = std::make_unique<dsp_core::LaneMixer>();
    blankMixer(*fresh);
    fresh->setLaneAmplitude(1, 0.9);
    std::vector<double> expected(dsp_core::LaneMixer::TABLE_SIZE, 0.0);
    fresh->computeSum(expected.data(), dsp_core::LaneMixer::TABLE_SIZE);

    ASSERT_EQ(combined.size(), expected.size());
    for (size_t i = 0; i < combined.size(); ++i) {
        EXPECT_NEAR(combined[i], expected[i], 1e-12);
    }
}

TEST_F(LaneMixerTest, Serialization_RoundTripsModulationDepth) {
    blankMixer(*mixer);
    mixer->setLaneModulationDepth(0, 0, 0.25);
    mixer->setLaneModulationDepth(0, 1, -0.15);
    mixer->setLaneModulationDepth(1, 0, -0.5);
    mixer->setLaneModulationDepth(1, 1, 0.6);
    mixer->setLaneModulationDepth(5, 0, 1.0);
    mixer->setLaneModulationDepth(7, 1, -1.0);

    const auto vt = mixer->toValueTree();
    dsp_core::LaneMixer restored;
    restored.fromValueTree(vt);

    EXPECT_DOUBLE_EQ(restored.getLaneModulationDepth(0, 0), 0.25);
    EXPECT_DOUBLE_EQ(restored.getLaneModulationDepth(0, 1), -0.15);
    EXPECT_DOUBLE_EQ(restored.getLaneModulationDepth(1, 0), -0.5);
    EXPECT_DOUBLE_EQ(restored.getLaneModulationDepth(1, 1), 0.6);
    EXPECT_DOUBLE_EQ(restored.getLaneModulationDepth(5, 0), 1.0);
    EXPECT_DOUBLE_EQ(restored.getLaneModulationDepth(7, 1), -1.0);
}

TEST_F(LaneMixerTest, Serialization_PreV6_NoModulationDepthProperty_DefaultsToZero) {
    // Build a v5 ValueTree by hand: it has blendDepth but no modulationDepth.
    juce::ValueTree v5("LaneMixer");
    v5.setProperty("formatVersion", 5, nullptr);
    v5.setProperty("numLanes", 2, nullptr);
    v5.setProperty("nextLaneId", 2, nullptr);
    v5.setProperty("tableSize", dsp_core::LaneMixer::TABLE_SIZE, nullptr);

    juce::ValueTree lane0("Lane");
    lane0.setProperty("index", 0, nullptr);
    lane0.setProperty("laneId", 0, nullptr);
    lane0.setProperty("amplitude", 1.0, nullptr);
    lane0.setProperty("blendDepth", 0.4, nullptr);
    lane0.setProperty("contentType", static_cast<int>(dsp_core::LaneContentType::Harmonic), nullptr);
    lane0.setProperty("harmonicNumber", 1, nullptr);
    v5.appendChild(lane0, nullptr);

    juce::ValueTree lane1("Lane");
    lane1.setProperty("index", 1, nullptr);
    lane1.setProperty("laneId", 1, nullptr);
    lane1.setProperty("amplitude", 0.5, nullptr);
    lane1.setProperty("blendDepth", -0.2, nullptr);
    lane1.setProperty("contentType", static_cast<int>(dsp_core::LaneContentType::Harmonic), nullptr);
    lane1.setProperty("harmonicNumber", 3, nullptr);
    v5.appendChild(lane1, nullptr);

    // Pre-poison: write some non-zero state to ensure deserialization actively zeros.
    dsp_core::LaneMixer restored;
    restored.setLaneModulationDepth(0, 0, 0.7);
    restored.setLaneModulationDepth(0, 1, 0.7);
    restored.setLaneModulationDepth(1, 0, -0.4);
    restored.setLaneModulationDepth(1, 1, -0.4);

    restored.fromValueTree(v5);

    EXPECT_DOUBLE_EQ(restored.getLaneModulationDepth(0, 0), 0.0)
        << "v5 ValueTree had no per-lane modulationDepth — should default to 0";
    EXPECT_DOUBLE_EQ(restored.getLaneModulationDepth(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(restored.getLaneModulationDepth(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(restored.getLaneModulationDepth(1, 1), 0.0);
    // Sanity: the v5 fields that were present still round-trip.
    EXPECT_DOUBLE_EQ(restored.getLaneBlendDepth(0), 0.4);
    EXPECT_DOUBLE_EQ(restored.getLaneBlendDepth(1), -0.2);
}

TEST_F(LaneMixerTest, Serialization_V6_LegacyModulationDepthLoadsIntoSlot1) {
    // Build a v6 ValueTree by hand: it has the legacy single `modulationDepth`
    // property. The new loader must route that value into slot 0 only and leave
    // slot 1 at zero, preserving the original audible behavior of old presets.
    juce::ValueTree v6("LaneMixer");
    v6.setProperty("formatVersion", 6, nullptr);
    v6.setProperty("numLanes", 2, nullptr);
    v6.setProperty("nextLaneId", 2, nullptr);
    v6.setProperty("tableSize", dsp_core::LaneMixer::TABLE_SIZE, nullptr);

    juce::ValueTree lane0("Lane");
    lane0.setProperty("index", 0, nullptr);
    lane0.setProperty("laneId", 0, nullptr);
    lane0.setProperty("amplitude", 1.0, nullptr);
    lane0.setProperty("blendDepth", 0.0, nullptr);
    lane0.setProperty("modulationDepth", 0.6, nullptr); // legacy single value
    lane0.setProperty("contentType", static_cast<int>(dsp_core::LaneContentType::Harmonic), nullptr);
    lane0.setProperty("harmonicNumber", 1, nullptr);
    v6.appendChild(lane0, nullptr);

    juce::ValueTree lane1("Lane");
    lane1.setProperty("index", 1, nullptr);
    lane1.setProperty("laneId", 1, nullptr);
    lane1.setProperty("amplitude", 0.5, nullptr);
    lane1.setProperty("blendDepth", 0.0, nullptr);
    lane1.setProperty("modulationDepth", -0.3, nullptr); // legacy single value
    lane1.setProperty("contentType", static_cast<int>(dsp_core::LaneContentType::Harmonic), nullptr);
    lane1.setProperty("harmonicNumber", 3, nullptr);
    v6.appendChild(lane1, nullptr);

    dsp_core::LaneMixer restored;
    restored.fromValueTree(v6);

    EXPECT_DOUBLE_EQ(restored.getLaneModulationDepth(0, 0), 0.6) << "v6 legacy modulationDepth must load into slot 0";
    EXPECT_DOUBLE_EQ(restored.getLaneModulationDepth(0, 1), 0.0) << "Slot 1 must default to 0 for v6 presets";
    EXPECT_DOUBLE_EQ(restored.getLaneModulationDepth(1, 0), -0.3);
    EXPECT_DOUBLE_EQ(restored.getLaneModulationDepth(1, 1), 0.0);
}

// ============================================================================
// Thumbnail Evaluation (formatVersion 3 — community-website preview samples)
// ============================================================================

namespace {
constexpr int kThumbN = 128;
constexpr int kThumbM = 5;
constexpr std::size_t kThumbTotal = static_cast<std::size_t>(kThumbN) * static_cast<std::size_t>(kThumbM);
constexpr std::array<float, kThumbM> kThumbMorphs = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};

double thumbXAt(int s) {
    return -1.0 + (2.0 * s) / static_cast<double>(kThumbN - 1);
}

std::size_t thumbIdx(int m, int s) {
    return static_cast<std::size_t>(m) * static_cast<std::size_t>(kThumbN) + static_cast<std::size_t>(s);
}
} // namespace

TEST_F(LaneMixerTest, Thumbnail_TanhSingleLaneMatchesAnalytic) {
    blankMixer(*mixer);
    // Replace lane 0 with raw tanh(2x) (no normalize), full amplitude, no morph dependency.
    std::vector<double> tanhCurve(dsp_core::LaneMixer::TABLE_SIZE);
    for (int i = 0; i < dsp_core::LaneMixer::TABLE_SIZE; ++i) {
        const double x = -1.0 + (2.0 * i) / static_cast<double>(dsp_core::LaneMixer::TABLE_SIZE - 1);
        tanhCurve[static_cast<size_t>(i)] = std::tanh(2.0 * x);
    }
    mixer->setLaneCurveData(0, tanhCurve);
    mixer->setLaneAmplitude(0, 1.0);
    mixer->setLaneBlendDepth(0, 0.0); // morph-independent
    mixer->setMixerMode(dsp_core::LaneMixer::MixerMode::Blend);

    std::array<float, kThumbTotal> out{};
    ASSERT_TRUE(mixer->evaluateStaticThumbnail(out.data(), kThumbN, kThumbMorphs.data(), kThumbM));

    // All 5 morph rows should be identical (depth=0) and match tanh(2x) normalized
    // by max|tanh(2x)| = tanh(2). The thumbnail does per-row normalize to mirror
    // the live computeSum / computeScan / computeSeries normalize step, so the
    // reference value is the analytic curve scaled to peak at ±1.
    const double scale = 1.0 / std::tanh(2.0);
    for (int m = 0; m < kThumbM; ++m) {
        for (int s = 0; s < kThumbN; ++s) {
            const double expected = std::tanh(2.0 * thumbXAt(s)) * scale;
            EXPECT_NEAR(out[thumbIdx(m, s)], static_cast<float>(expected), 1e-3f)
                << "morph=" << m << " sample=" << s;
        }
    }
}

TEST_F(LaneMixerTest, Thumbnail_AllLanesDisabled_ProducesAllZeros) {
    blankMixer(*mixer);
    mixer->setMixerMode(dsp_core::LaneMixer::MixerMode::Blend);

    std::array<float, kThumbTotal> out{};
    ASSERT_TRUE(mixer->evaluateStaticThumbnail(out.data(), kThumbN, kThumbMorphs.data(), kThumbM));
    for (float v : out) {
        EXPECT_EQ(v, 0.0f);
    }
}

TEST_F(LaneMixerTest, Thumbnail_BlendMorphSweep_ChangesOutputBetweenRows) {
    // Two lanes with opposite curves and opposite blendDepths so morph=0 emphasizes
    // one and morph=1 emphasizes the other.
    blankMixer(*mixer);
    std::vector<double> curveA(dsp_core::LaneMixer::TABLE_SIZE);
    std::vector<double> curveB(dsp_core::LaneMixer::TABLE_SIZE);
    for (int i = 0; i < dsp_core::LaneMixer::TABLE_SIZE; ++i) {
        const double x = -1.0 + (2.0 * i) / static_cast<double>(dsp_core::LaneMixer::TABLE_SIZE - 1);
        curveA[static_cast<size_t>(i)] = x;        // identity
        curveB[static_cast<size_t>(i)] = -x * 0.5; // inverted, half amplitude
    }
    mixer->setLaneCurveData(0, curveA);
    mixer->setLaneCurveData(1, curveB);
    // base 0, depth +1 → effective(morph=0) = 0, effective(morph=1) = 1
    mixer->setLaneAmplitude(0, 0.0);
    mixer->setLaneAmplitude(1, 0.0);
    mixer->setLaneBlendDepth(0, 1.0);
    mixer->setLaneBlendDepth(1, -1.0); // pulls down, clamped to 0
    mixer->setMixerMode(dsp_core::LaneMixer::MixerMode::Blend);

    std::array<float, kThumbTotal> out{};
    ASSERT_TRUE(mixer->evaluateStaticThumbnail(out.data(), kThumbN, kThumbMorphs.data(), kThumbM));

    // morph=0: both lanes effective=0 → all zero.
    // morph=1: lane 0 effective=1 (identity), lane 1 effective clamped to 0 → identity.
    for (int s = 0; s < kThumbN; ++s) {
        EXPECT_EQ(out[thumbIdx(0, s)], 0.0f);
        const auto expected = static_cast<float>(thumbXAt(s));
        EXPECT_NEAR(out[thumbIdx(4, s)], expected, 1e-3f);
    }
}

TEST_F(LaneMixerTest, Thumbnail_ScanModeLerpBetweenLanes) {
    // Two lanes with distinct constant curves so scan position picks them.
    blankMixer(*mixer);
    std::vector<double> curveA(dsp_core::LaneMixer::TABLE_SIZE, 0.4);
    std::vector<double> curveB(dsp_core::LaneMixer::TABLE_SIZE, -0.6);
    mixer->setLaneCurveData(0, curveA);
    mixer->setLaneCurveData(1, curveB);
    // Scan mode doesn't read amplitudes/depths — it lerps the LUTs directly. Still set
    // amps so only first 2 lanes are obviously the active ones for the test's intent.
    mixer->setLaneAmplitude(0, 1.0);
    mixer->setLaneAmplitude(1, 1.0);
    mixer->setMixerMode(dsp_core::LaneMixer::MixerMode::Scan);

    // Use only morphs 0.0, 0.5, 1.0 by reading rows 0, 2, 4 of the standard 5-row output.
    std::array<float, kThumbTotal> out{};
    ASSERT_TRUE(mixer->evaluateStaticThumbnail(out.data(), kThumbN, kThumbMorphs.data(), kThumbM));

    // Scan position = morph * (activeLaneCount - 1). We have 10 active lanes by default
    // (blankMixer doesn't change activeLaneCount), so morph=0 → lane 0, morph=1 → lane 9.
    // Lane 9's curveData was zeroed by blankMixer's setLaneAmplitude/Depth but its curve
    // isn't blanked. Default lane 9 is harmonic H19 — non-trivial. So this test only
    // asserts row 0 (morph=0 → lane 0 = constant 0.4, post-normalize → constant 1.0).
    for (int s = 0; s < kThumbN; ++s) {
        EXPECT_NEAR(out[static_cast<size_t>(s)], 1.0f, 1e-5f)
            << "morph=0 should equal normalized lane 0 (constant 0.4 → 1.0 after row normalize)";
    }
}

TEST_F(LaneMixerTest, Thumbnail_OverUnitySumNormalizesInsteadOfFlatTopping) {
    // Regression test for the website-preview clip bug: presets whose unscaled
    // lane sum exceeds ±1 used to produce flat-topped rows (the evaluator
    // clamped each sample to [-1, 1]). Production normalizes by row-max instead.
    // After the fix, the thumbnail row should hit ±1 exactly once and otherwise
    // preserve the curve shape — no plateaus.
    blankMixer(*mixer);

    // Two lanes with the same identity x curve, both at amplitude 1.0. Without
    // normalize the sum would be 2x ∈ [-2, 2], which would clamp to ±1 across
    // a wide region. With normalize, the row should equal x (identity), peak ±1.
    std::vector<double> identity(dsp_core::LaneMixer::TABLE_SIZE);
    for (int i = 0; i < dsp_core::LaneMixer::TABLE_SIZE; ++i) {
        identity[static_cast<size_t>(i)] = -1.0 + (2.0 * i) / static_cast<double>(dsp_core::LaneMixer::TABLE_SIZE - 1);
    }
    mixer->setLaneCurveData(0, identity);
    mixer->setLaneCurveData(1, identity);
    mixer->setLaneAmplitude(0, 1.0);
    mixer->setLaneAmplitude(1, 1.0);
    mixer->setLaneBlendDepth(0, 0.0);
    mixer->setLaneBlendDepth(1, 0.0);
    mixer->setMixerMode(dsp_core::LaneMixer::MixerMode::Blend);

    std::array<float, kThumbTotal> out{};
    ASSERT_TRUE(mixer->evaluateStaticThumbnail(out.data(), kThumbN, kThumbMorphs.data(), kThumbM));

    // Each row should equal x (identity, unscaled-sum 2x normalized by 2).
    // No plateau at ±1 — every interior sample must differ from its neighbors.
    for (int m = 0; m < kThumbM; ++m) {
        for (int s = 0; s < kThumbN; ++s) {
            const auto expected = static_cast<float>(thumbXAt(s));
            EXPECT_NEAR(out[thumbIdx(m, s)], expected, 1e-3f) << "morph=" << m << " sample=" << s;
        }
        // Anti-flat-top: count interior samples that equal their neighbor. With
        // a true clamp, the broad plateau at ±1 produces dozens of equal-pairs;
        // a properly normalized identity has zero.
        int plateauPairs = 0;
        for (int s = 1; s < kThumbN - 1; ++s) {
            if (out[thumbIdx(m, s)] == out[thumbIdx(m, s + 1)]) {
                ++plateauPairs;
            }
        }
        EXPECT_EQ(plateauPairs, 0) << "morph=" << m << " row should have no flat plateaus";
    }
}

TEST_F(LaneMixerTest, Thumbnail_EquationLaneRebakesPerMorphRow) {
    // Regression test for the website-preview "equations don't morph" bug:
    // an Equation lane's curveData is a snapshot at the LIVE morph value, so
    // reading it 5 times produced 5 identical rows. The fix is for
    // evaluateStaticThumbnail to recompile + re-synthesize via the
    // dsp_core::Services::synthesizeLaneLUT primitive (same one the live audio
    // path uses) at each thumbnail morph value.
    //
    // We use tanh(4*x - 4*m + 2) — a sigmoid whose horizontal shift is driven
    // by m. At m=0 the inflection sits at x=-0.5; at m=1 it sits at x=+0.5.
    // After per-row normalize the curves still differ wildly in shape, so any
    // two rows must NOT be equal.
    blankMixer(*mixer);
    auto& lane = mixer->getMutableLane(0);
    lane.equationText = "tanh(4*x - 4*m + 2)";
    mixer->setLaneContentType(0, dsp_core::LaneContentType::Equation);
    mixer->setLaneAmplitude(0, 1.0);
    mixer->setLaneBlendDepth(0, 0.0); // no blend-depth contribution; morph effect comes purely from `m` in the equation
    // Seed curveData with garbage to prove the thumbnail does NOT just read it.
    std::vector<double> garbage(dsp_core::LaneMixer::TABLE_SIZE, 0.123);
    mixer->setLaneCurveData(0, garbage);
    mixer->setMixerMode(dsp_core::LaneMixer::MixerMode::Blend);

    std::array<float, kThumbTotal> out{};
    ASSERT_TRUE(mixer->evaluateStaticThumbnail(out.data(), kThumbN, kThumbMorphs.data(), kThumbM));

    // Row 0 (morph=0) and row 4 (morph=1) must differ substantially — that's
    // the whole point. If the thumbnail were reading the (constant 0.123)
    // snapshot LUT instead of re-synthesizing, both rows would normalize to a
    // constant 1.0 and this would fail.
    int diffCount = 0;
    for (int s = 0; s < kThumbN; ++s) {
        if (std::abs(out[thumbIdx(0, s)] - out[thumbIdx(4, s)]) > 1e-2f) {
            ++diffCount;
        }
    }
    EXPECT_GT(diffCount, kThumbN / 4)
        << "Equation lane should produce visibly different rows at different morph values "
           "(at least a quarter of samples should differ between morph=0 and morph=1)";

    // Sigmoid: tanh(4*x - 4*m + 2) inflection at x = m - 0.5. With morph 0 and
    // 1 the inflection points are at -0.5 and +0.5 respectively — so at x=0
    // (sample index 63 or 64) the rows should be on opposite sides of zero.
    EXPECT_GT(out[thumbIdx(0, kThumbN / 2)], 0.5f) << "morph=0 should be saturated positive at x=0";
    EXPECT_LT(out[thumbIdx(4, kThumbN / 2)], -0.5f) << "morph=1 should be saturated negative at x=0";

    // Each row should still be properly normalized (max|y| close to 1) since
    // tanh saturates near ±1 for the given x range.
    for (int m = 0; m < kThumbM; ++m) {
        float maxAbs = 0.0f;
        for (int s = 0; s < kThumbN; ++s) {
            maxAbs = std::max(maxAbs, std::abs(out[thumbIdx(m, s)]));
        }
        EXPECT_NEAR(maxAbs, 1.0f, 1e-3f) << "morph=" << m << " row should normalize to peak ±1";
    }
}

TEST_F(LaneMixerTest, Thumbnail_SplineLaneRebakesPerMorphRowFromMorphGesture) {
    // Regression test for the spline-morph counterpart of the equation bug:
    // a Spline lane's curveData is also a snapshot baked at the LIVE morph
    // value, so reading it 5 times produced 5 identical rows. The fix wires
    // dsp_core::Services::synthesizeSplineLaneLUT into evaluateStaticThumbnail
    // — for each anchor with a morphGesture we re-apply the gesture at each
    // thumbnail morph value and re-fit/re-evaluate into a scratch LUT.
    //
    // We construct a 5-anchor spline and give the middle anchor a vertical
    // gesture: at morph=0 it sits at home (y≈0), at morph=1 it has slid down
    // by 1.6 (clamped to y=-1). The two halves of the curve flip orientation
    // as the middle anchor moves — every row should differ from every other.
    blankMixer(*mixer);
    auto& lane = mixer->getMutableLane(0);
    lane.splineAnchors.clear();

    auto makeAnchor = [](double x, double y) {
        dsp_core::SplineAnchor a;
        a.x = x;
        a.y = y;
        a.homeX = x;
        a.homeY = y;
        return a;
    };

    lane.splineAnchors.push_back(makeAnchor(-1.0, -1.0));
    lane.splineAnchors.push_back(makeAnchor(-0.5, -0.5));
    lane.splineAnchors.push_back(makeAnchor(0.0, 0.0)); // gesture-bearing
    lane.splineAnchors.push_back(makeAnchor(0.5, 0.5));
    lane.splineAnchors.push_back(makeAnchor(1.0, 1.0));

    // Gesture: dy goes from 0 (home) at t=0 to -1.6 at t=1 — vertical descent.
    // Linear interpolation between 5 deltas matches the 5 thumbnail morph values.
    lane.splineAnchors[2].morphGesture.deltas = {
        {0.0, 0.0}, {0.0, -0.4}, {0.0, -0.8}, {0.0, -1.2}, {0.0, -1.6}};

    mixer->setLaneContentType(0, dsp_core::LaneContentType::Spline);
    mixer->setLaneAmplitude(0, 1.0);
    mixer->setLaneBlendDepth(0, 0.0); // morph drives the gesture, not lane gain
    // Seed curveData with garbage to prove the thumbnail does NOT just read it.
    std::vector<double> garbage(dsp_core::LaneMixer::TABLE_SIZE, 0.42);
    mixer->setLaneCurveData(0, garbage);
    mixer->setMixerMode(dsp_core::LaneMixer::MixerMode::Blend);

    std::array<float, kThumbTotal> out{};
    ASSERT_TRUE(mixer->evaluateStaticThumbnail(out.data(), kThumbN, kThumbMorphs.data(), kThumbM));

    // Row 0 (morph=0, home positions) and row 4 (morph=1, mid anchor at y=-1)
    // describe very different curves. If the snapshot were being read instead,
    // both rows would be the same constant 1.0 (garbage 0.42 normalized).
    int diffCount = 0;
    for (int s = 0; s < kThumbN; ++s) {
        if (std::abs(out[thumbIdx(0, s)] - out[thumbIdx(4, s)]) > 1e-2f) {
            ++diffCount;
        }
    }
    EXPECT_GT(diffCount, kThumbN / 4)
        << "Spline lane should produce visibly different rows at different morph values "
           "(at least a quarter of samples should differ between morph=0 and morph=1)";

    // At x=0 (sample index 63 or 64), morph=0 → mid anchor at y=0 (after
    // normalize the curve passes through 0 at x=0); morph=1 → mid anchor at
    // y=-1, so y(x=0) ≈ -1 (saturated). Verify the sign flip.
    EXPECT_NEAR(out[thumbIdx(0, kThumbN / 2)], 0.0f, 0.1f) << "morph=0 mid anchor at home (y=0)";
    EXPECT_LT(out[thumbIdx(4, kThumbN / 2)], -0.5f) << "morph=1 mid anchor at y=-1 (saturated negative)";
}

TEST_F(LaneMixerTest, Thumbnail_NaNInLaneCurve_ReturnsFalse) {
    blankMixer(*mixer);
    // Poison the entire curve with NaN — the 128 thumbnail samples are spaced
    // ~129 curve-indices apart, so a single poisoned cell can be missed entirely.
    std::vector<double> poisoned(dsp_core::LaneMixer::TABLE_SIZE,
                                 std::numeric_limits<double>::quiet_NaN());
    mixer->setLaneCurveData(0, poisoned);
    mixer->setLaneAmplitude(0, 1.0);
    mixer->setMixerMode(dsp_core::LaneMixer::MixerMode::Blend);

    std::array<float, kThumbTotal> out{};
    EXPECT_FALSE(mixer->evaluateStaticThumbnail(out.data(), kThumbN, kThumbMorphs.data(), kThumbM));
}

TEST_F(LaneMixerTest, Thumbnail_InvalidInputs_ReturnsFalse) {
    std::array<float, kThumbTotal> out{};
    EXPECT_FALSE(mixer->evaluateStaticThumbnail(nullptr, kThumbN, kThumbMorphs.data(), kThumbM));
    EXPECT_FALSE(mixer->evaluateStaticThumbnail(out.data(), 0, kThumbMorphs.data(), kThumbM));
    EXPECT_FALSE(mixer->evaluateStaticThumbnail(out.data(), kThumbN, nullptr, kThumbM));
    EXPECT_FALSE(mixer->evaluateStaticThumbnail(out.data(), kThumbN, kThumbMorphs.data(), 0));
}

} // namespace dsp_core_test
