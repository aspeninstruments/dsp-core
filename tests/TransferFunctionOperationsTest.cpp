#include <dsp_core/dsp_core.h>
#include <gtest/gtest.h>
#include <cmath>

using namespace dsp_core::Services;

// ============================================================================
// CurveData Operations Tests
// ============================================================================

class CurveDataOperationsTest : public ::testing::Test {
  protected:
    void SetUp() override {
        curve.resize(64);
        for (int i = 0; i < 64; ++i) {
            curve[static_cast<size_t>(i)] = -1.0 + (2.0 * i / 63.0);
        }
    }
    std::vector<double> curve;
};

TEST_F(CurveDataOperationsTest, Invert_FlipsAllValues) {
    std::vector<double> original = curve;
    TransferFunctionOperations::invert(curve);
    for (size_t i = 0; i < curve.size(); ++i) {
        EXPECT_NEAR(curve[i], -original[i], 1e-10);
    }
}

TEST_F(CurveDataOperationsTest, Invert_DoubleInvertRestores) {
    std::vector<double> original = curve;
    TransferFunctionOperations::invert(curve);
    TransferFunctionOperations::invert(curve);
    for (size_t i = 0; i < curve.size(); ++i) {
        EXPECT_NEAR(curve[i], original[i], 1e-10);
    }
}

TEST_F(CurveDataOperationsTest, InvertHorizontal_ReversesTable) {
    std::vector<double> original = curve;
    TransferFunctionOperations::invertHorizontal(curve);
    for (size_t i = 0; i < curve.size(); ++i) {
        EXPECT_DOUBLE_EQ(curve[i], original[curve.size() - 1 - i]);
    }
}

TEST_F(CurveDataOperationsTest, InvertHorizontal_DoubleInvertRestores) {
    std::vector<double> original = curve;
    TransferFunctionOperations::invertHorizontal(curve);
    TransferFunctionOperations::invertHorizontal(curve);
    for (size_t i = 0; i < curve.size(); ++i) {
        EXPECT_DOUBLE_EQ(curve[i], original[i]);
    }
}

TEST_F(CurveDataOperationsTest, RemoveDCInstantaneous_CentersAtMidpoint) {
    for (auto& v : curve) {
        v += 0.5;
    }
    TransferFunctionOperations::removeDCInstantaneous(curve);
    EXPECT_NEAR(curve[curve.size() / 2], 0.0, 1e-10);
}

TEST_F(CurveDataOperationsTest, RemoveDCSteadyState_ZerosAverage) {
    for (auto& v : curve) {
        v += 0.5;
    }
    TransferFunctionOperations::removeDCSteadyState(curve);
    double sum = 0.0;
    for (const auto v : curve) {
        sum += v;
    }
    EXPECT_NEAR(sum / static_cast<double>(curve.size()), 0.0, 1e-10);
}

TEST_F(CurveDataOperationsTest, Normalize_ScalesToUnit) {
    for (auto& v : curve) {
        v *= 0.25;
    }
    TransferFunctionOperations::normalize(curve);
    double maxAbs = 0.0;
    for (const auto v : curve) {
        maxAbs = std::max(maxAbs, std::abs(v));
    }
    EXPECT_NEAR(maxAbs, 1.0, 1e-10);
}

TEST_F(CurveDataOperationsTest, Normalize_ZeroCurve_NoOp) {
    std::vector<double> zeros(64, 0.0);
    TransferFunctionOperations::normalize(zeros);
    for (const auto v : zeros) {
        EXPECT_DOUBLE_EQ(v, 0.0);
    }
}

TEST_F(CurveDataOperationsTest, Normalize_PreservesRelativeShape) {
    for (auto& v : curve) {
        v *= 0.3;
    }
    std::vector<double> original = curve;
    TransferFunctionOperations::normalize(curve);
    const double scale = 1.0 / 0.3;
    for (size_t i = 0; i < curve.size(); ++i) {
        EXPECT_NEAR(curve[i], original[i] * scale, 1e-10);
    }
}

// ============================================================================
// Smooth Tests
// ============================================================================

TEST_F(CurveDataOperationsTest, Smooth_ReducesHighFrequencyNoise) {
    // Alternating +1/-1 is the highest frequency signal
    std::vector<double> noisy(1024);
    for (size_t i = 0; i < noisy.size(); ++i) {
        noisy[i] = (i % 2 == 0) ? 1.0 : -1.0;
    }
    TransferFunctionOperations::smooth(noisy);
    // Interior values should be heavily attenuated by the 201-wide box filter
    double maxAbsInterior = 0.0;
    for (size_t i = 200; i < 800; ++i) {
        maxAbsInterior = std::max(maxAbsInterior, std::abs(noisy[i]));
    }
    EXPECT_LT(maxAbsInterior, 0.01);
}

TEST_F(CurveDataOperationsTest, Smooth_PreservesConstantCurve) {
    std::vector<double> flat(256, 0.5);
    TransferFunctionOperations::smooth(flat);
    for (const auto v : flat) {
        EXPECT_NEAR(v, 0.5, 1e-10);
    }
}

TEST_F(CurveDataOperationsTest, Smooth_PreservesLinearRamp) {
    std::vector<double> ramp(1024);
    for (size_t i = 0; i < ramp.size(); ++i) {
        ramp[i] = -1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(ramp.size() - 1);
    }
    std::vector<double> original = ramp;
    TransferFunctionOperations::smooth(ramp);
    // Interior should be nearly unchanged (box filter preserves linear functions)
    for (size_t i = 150; i < 874; ++i) {
        EXPECT_NEAR(ramp[i], original[i], 0.01) << "Deviation at index " << i;
    }
}

TEST_F(CurveDataOperationsTest, Smooth_EmptyCurve_NoOp) {
    std::vector<double> empty;
    TransferFunctionOperations::smooth(empty);
    EXPECT_TRUE(empty.empty());
}

TEST_F(CurveDataOperationsTest, Smooth_SingleElement_NoOp) {
    std::vector<double> single = {0.7};
    TransferFunctionOperations::smooth(single);
    EXPECT_DOUBLE_EQ(single[0], 0.7);
}

TEST_F(CurveDataOperationsTest, Smooth_SmallCurve_Works) {
    std::vector<double> small = {-1.0, 0.0, 1.0, 0.5, -0.5};
    TransferFunctionOperations::smooth(small);
    EXPECT_EQ(small.size(), 5u);
    for (const auto v : small) {
        EXPECT_GE(v, -1.0);
        EXPECT_LE(v, 1.0);
    }
}

TEST_F(CurveDataOperationsTest, Smooth_SoftensStepFunction) {
    std::vector<double> step(512);
    for (size_t i = 0; i < step.size(); ++i) {
        step[i] = (i < step.size() / 2) ? -1.0 : 1.0;
    }
    TransferFunctionOperations::smooth(step);
    // Transition region should have intermediate values
    const size_t mid = step.size() / 2;
    EXPECT_GT(step[mid], -0.9); // No longer a hard -1→1 jump
    EXPECT_LT(step[mid - 1], 0.9);
}

// ============================================================================
// Smoother Tests
// ============================================================================

TEST_F(CurveDataOperationsTest, Smoother_StrongerThanSmooth) {
    // Create identical step functions
    std::vector<double> stepSmooth(1024);
    std::vector<double> stepSmoother(1024);
    for (size_t i = 0; i < 1024; ++i) {
        double val = (i < 512) ? -1.0 : 1.0;
        stepSmooth[i] = val;
        stepSmoother[i] = val;
    }
    TransferFunctionOperations::smooth(stepSmooth);
    TransferFunctionOperations::smoother(stepSmoother);
    // Smoother should produce a wider, more gradual transition
    // Check that smoother deviates more from the original step at a point
    // well within the smooth radius but outside the smoother radius
    double smoothDeviation = std::abs(stepSmooth[350] - (-1.0));
    double smootherDeviation = std::abs(stepSmoother[350] - (-1.0));
    EXPECT_GT(smootherDeviation, smoothDeviation);
}

TEST_F(CurveDataOperationsTest, Smoother_PreservesConstantCurve) {
    std::vector<double> flat(256, 0.5);
    TransferFunctionOperations::smoother(flat);
    for (const auto v : flat) {
        EXPECT_NEAR(v, 0.5, 1e-10);
    }
}

// ============================================================================
// Horizontal Shift Tests
// ============================================================================

namespace {
// Build an identity curve y=x on [-1, 1] with `n` samples.
std::vector<double> makeIdentity(size_t n) {
    std::vector<double> c(n);
    for (size_t i = 0; i < n; ++i) {
        c[i] = -1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(n - 1);
    }
    return c;
}

// Sample a sampled curve at arbitrary x using linear interpolation.
double sampleAt(const std::vector<double>& c, double x) {
    const auto n = static_cast<int>(c.size());
    const double fIdx = (x + 1.0) * 0.5 * (n - 1);
    const int i0 = std::max(0, std::min(static_cast<int>(fIdx), n - 1));
    const int i1 = std::min(i0 + 1, n - 1);
    const double f = fIdx - static_cast<double>(i0);
    return c[static_cast<size_t>(i0)] * (1.0 - f) + c[static_cast<size_t>(i1)] * f;
}
} // namespace

TEST(HorizontalShiftTest, ShiftHorizontal_ZeroDelta_NoOp) {
    auto c = makeIdentity(1024);
    auto original = c;
    TransferFunctionOperations::shiftHorizontal(c, 0.0);
    for (size_t i = 0; i < c.size(); ++i) {
        EXPECT_DOUBLE_EQ(c[i], original[i]);
    }
}

TEST(HorizontalShiftTest, ShiftHorizontal_PreservesEndpoints) {
    auto c = makeIdentity(2048);
    const double firstOriginal = c.front();
    const double lastOriginal = c.back();
    TransferFunctionOperations::shiftHorizontal(c, 0.05);
    EXPECT_NEAR(c.front(), firstOriginal, 1e-10);
    EXPECT_NEAR(c.back(), lastOriginal, 1e-10);
}

TEST(HorizontalShiftTest, ShiftRight_PivotAtDelta) {
    // A non-trivial curve so that f(0) is a distinctive value.
    std::vector<double> c(4096);
    for (size_t i = 0; i < c.size(); ++i) {
        const double x = -1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(c.size() - 1);
        c[i] = std::sin(x * 3.0);
    }
    const double fOrig0 = sampleAt(c, 0.0);
    TransferFunctionOperations::shiftHorizontalRight(c);
    // Value at x = +0.05 in the shifted curve should equal f(0) from the original.
    EXPECT_NEAR(sampleAt(c, 0.05), fOrig0, 1e-3);
}

TEST(HorizontalShiftTest, ShiftLeftThenRight_ApproximatelyRoundTrips) {
    std::vector<double> c(4096);
    for (size_t i = 0; i < c.size(); ++i) {
        const double x = -1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(c.size() - 1);
        c[i] = std::tanh(x * 2.0);
    }
    auto original = c;
    TransferFunctionOperations::shiftHorizontalRight(c);
    TransferFunctionOperations::shiftHorizontalLeft(c);
    // Interior should approximately round-trip (boundaries accumulate most error).
    const size_t lo = c.size() / 4;
    const size_t hi = 3 * c.size() / 4;
    for (size_t i = lo; i < hi; ++i) {
        EXPECT_NEAR(c[i], original[i], 0.02) << "at i=" << i;
    }
}

TEST(HorizontalShiftTest, ShiftToZeroCrossing_CrossingEndsAtZero) {
    // Curve with a zero crossing at x = 0.2, by construction.
    std::vector<double> c(4096);
    constexpr double crossingX = 0.2;
    for (size_t i = 0; i < c.size(); ++i) {
        const double x = -1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(c.size() - 1);
        c[i] = x - crossingX; // zero at x=0.2, monotone
    }
    TransferFunctionOperations::shiftToZeroCrossing(c);
    EXPECT_NEAR(sampleAt(c, 0.0), 0.0, 1e-3);
}

TEST(HorizontalShiftTest, ShiftToZeroCrossing_NoCrossing_NoOp) {
    std::vector<double> c(1024, 0.5);
    auto original = c;
    TransferFunctionOperations::shiftToZeroCrossing(c);
    for (size_t i = 0; i < c.size(); ++i) {
        EXPECT_DOUBLE_EQ(c[i], original[i]);
    }
}

TEST(HorizontalShiftTest, ShiftToZeroCrossing_PicksNearestToZero) {
    // Curve with crossings at x ~ -0.3 and x ~ +0.1. Nearest to 0 is +0.1.
    std::vector<double> c(4096);
    for (size_t i = 0; i < c.size(); ++i) {
        const double x = -1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(c.size() - 1);
        c[i] = (x + 0.3) * (x - 0.1);
    }
    TransferFunctionOperations::shiftToZeroCrossing(c);
    // After shift, x=0 in new domain == x=+0.1 in original, which was a crossing.
    EXPECT_NEAR(sampleAt(c, 0.0), 0.0, 1e-3);
}
