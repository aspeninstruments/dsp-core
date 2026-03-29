#include <dsp_core/dsp_core.h>
#include <gtest/gtest.h>
#include <cmath>

using namespace dsp_core;
using namespace dsp_core::Services;

// ============================================================================
// Test Fixture
// ============================================================================

class TransferFunctionOperationsTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Create a small LTF for testing (64 points for simplicity)
        ltf = std::make_unique<LayeredTransferFunction>(64, -1.0, 1.0);
    }

    std::unique_ptr<LayeredTransferFunction> ltf;
};

// ============================================================================
// Invert Tests
// ============================================================================

TEST_F(TransferFunctionOperationsTest, Invert_FlipsAllValues) {
    // Set up a simple linear ramp
    const int tableSize = ltf->getTableSize();
    for (int i = 0; i < tableSize; ++i) {
        double const x = -1.0 + (2.0 * i / (tableSize - 1));
        ltf->setBaseLayerValue(i, x); // Linear ramp from -1 to 1
    }

    // Invert
    TransferFunctionOperations::invert(*ltf);

    // Verify all values are negated
    for (int i = 0; i < tableSize; ++i) {
        double const expected = -(-1.0 + (2.0 * i / (tableSize - 1)));
        EXPECT_NEAR(ltf->getBaseLayerValue(i), expected, 1e-10) << "Value at index " << i << " not correctly inverted";
    }
}

TEST_F(TransferFunctionOperationsTest, Invert_DoubleInvertRestoresOriginal) {
    // Set up arbitrary values
    const int tableSize = ltf->getTableSize();
    std::vector<double> original(tableSize);
    for (int i = 0; i < tableSize; ++i) {
        double const val = std::sin(2.0 * M_PI * i / tableSize);
        ltf->setBaseLayerValue(i, val);
        original[i] = val;
    }

    // Double invert
    TransferFunctionOperations::invert(*ltf);
    TransferFunctionOperations::invert(*ltf);

    // Should restore original
    for (int i = 0; i < tableSize; ++i) {
        EXPECT_NEAR(ltf->getBaseLayerValue(i), original[i], 1e-10);
    }
}

TEST_F(TransferFunctionOperationsTest, Invert_ZeroLayerUnchanged) {
    // Set base layer to zero explicitly
    const int tableSize = ltf->getTableSize();
    ltf->clearBaseLayer();

    TransferFunctionOperations::invert(*ltf);

    // All values should remain zero
    for (int i = 0; i < tableSize; ++i) {
        EXPECT_DOUBLE_EQ(ltf->getBaseLayerValue(i), 0.0);
    }
}

// ============================================================================
// RemoveDCInstantaneous Tests
// ============================================================================

TEST_F(TransferFunctionOperationsTest, RemoveDCInstantaneous_CentersAtOrigin) {
    // Set up a function with DC offset at origin
    const int tableSize = ltf->getTableSize();
    const double dcOffset = 0.5;
    for (int i = 0; i < tableSize; ++i) {
        double const x = -1.0 + (2.0 * i / (tableSize - 1));
        ltf->setBaseLayerValue(i, x + dcOffset); // Ramp with DC offset
    }

    // Remove DC
    TransferFunctionOperations::removeDCInstantaneous(*ltf);

    // Value at center (x=0) should now be zero
    const int midIndex = tableSize / 2;
    EXPECT_NEAR(ltf->getBaseLayerValue(midIndex), 0.0, 1e-10);
}

TEST_F(TransferFunctionOperationsTest, RemoveDCInstantaneous_ShiftsAllValuesByOffset) {
    // Set constant function with offset
    const int tableSize = ltf->getTableSize();
    const double offset = 0.3;
    for (int i = 0; i < tableSize; ++i) {
        ltf->setBaseLayerValue(i, offset);
    }

    TransferFunctionOperations::removeDCInstantaneous(*ltf);

    // All values should now be zero
    for (int i = 0; i < tableSize; ++i) {
        EXPECT_NEAR(ltf->getBaseLayerValue(i), 0.0, 1e-10);
    }
}

// ============================================================================
// RemoveDCSteadyState Tests
// ============================================================================

TEST_F(TransferFunctionOperationsTest, RemoveDCSteadyState_RemovesAverageOffset) {
    // Set up asymmetric function with non-zero average
    const int tableSize = ltf->getTableSize();
    for (int i = 0; i < tableSize; ++i) {
        // Positive-biased sine: average is approximately 0.5
        double const val = 0.5 + 0.5 * std::sin(2.0 * M_PI * i / tableSize);
        ltf->setBaseLayerValue(i, val);
    }

    TransferFunctionOperations::removeDCSteadyState(*ltf);

    // Verify average is now zero
    double sum = 0.0;
    for (int i = 0; i < tableSize; ++i) {
        sum += ltf->getBaseLayerValue(i);
    }
    double const average = sum / tableSize;
    EXPECT_NEAR(average, 0.0, 1e-10);
}

TEST_F(TransferFunctionOperationsTest, RemoveDCSteadyState_PreservesShape) {
    // Set up a simple function
    const int tableSize = ltf->getTableSize();
    std::vector<double> original(tableSize);
    for (int i = 0; i < tableSize; ++i) {
        double const val = std::sin(2.0 * M_PI * i / tableSize);
        ltf->setBaseLayerValue(i, val);
        original[i] = val;
    }

    // Calculate original average
    double origSum = 0.0;
    for (double const v : original) {
        origSum += v;
}
    double const origAvg = origSum / tableSize;

    TransferFunctionOperations::removeDCSteadyState(*ltf);

    // Shape should be preserved (just shifted)
    for (int i = 0; i < tableSize; ++i) {
        EXPECT_NEAR(ltf->getBaseLayerValue(i), original[i] - origAvg, 1e-10);
    }
}

// ============================================================================
// Normalize Tests
// ============================================================================

TEST_F(TransferFunctionOperationsTest, Normalize_ScalesToUnitRange) {
    // Set up a function with max value of 0.5
    const int tableSize = ltf->getTableSize();
    for (int i = 0; i < tableSize; ++i) {
        double const val = 0.5 * std::sin(2.0 * M_PI * i / tableSize);
        ltf->setBaseLayerValue(i, val);
    }

    TransferFunctionOperations::normalize(*ltf);

    // Find max absolute value
    double maxAbs = 0.0;
    for (int i = 0; i < tableSize; ++i) {
        maxAbs = std::max(maxAbs, std::abs(ltf->getBaseLayerValue(i)));
    }
    EXPECT_NEAR(maxAbs, 1.0, 1e-10);
}

TEST_F(TransferFunctionOperationsTest, Normalize_PreservesRelativeShape) {
    // Set up a function
    const int tableSize = ltf->getTableSize();
    std::vector<double> original(tableSize);
    for (int i = 0; i < tableSize; ++i) {
        double const val = 0.25 * std::sin(2.0 * M_PI * i / tableSize);
        ltf->setBaseLayerValue(i, val);
        original[i] = val;
    }

    // Find original max for expected scale factor
    double origMax = 0.0;
    for (double const v : original) {
        origMax = std::max(origMax, std::abs(v));
}

    TransferFunctionOperations::normalize(*ltf);

    // Verify relative proportions are maintained
    double const scaleFactor = 1.0 / origMax;
    for (int i = 0; i < tableSize; ++i) {
        EXPECT_NEAR(ltf->getBaseLayerValue(i), original[i] * scaleFactor, 1e-10);
    }
}

TEST_F(TransferFunctionOperationsTest, Normalize_AlreadyNormalized_NoChange) {
    // Set up a function already at full scale
    const int tableSize = ltf->getTableSize();
    for (int i = 0; i < tableSize; ++i) {
        double const val = std::sin(2.0 * M_PI * i / tableSize); // Already peaks at ±1
        ltf->setBaseLayerValue(i, val);
    }

    // Store original
    std::vector<double> original(tableSize);
    for (int i = 0; i < tableSize; ++i) {
        original[i] = ltf->getBaseLayerValue(i);
    }

    TransferFunctionOperations::normalize(*ltf);

    // Should be essentially unchanged
    for (int i = 0; i < tableSize; ++i) {
        EXPECT_NEAR(ltf->getBaseLayerValue(i), original[i], 1e-10);
    }
}

TEST_F(TransferFunctionOperationsTest, Normalize_ZeroLayer_NoOp) {
    // Set base layer to zero explicitly
    const int tableSize = ltf->getTableSize();
    ltf->clearBaseLayer();

    TransferFunctionOperations::normalize(*ltf);

    // All values should remain zero (no division by zero)
    for (int i = 0; i < tableSize; ++i) {
        EXPECT_DOUBLE_EQ(ltf->getBaseLayerValue(i), 0.0);
    }
}

TEST_F(TransferFunctionOperationsTest, Normalize_NegativeOnlyValues) {
    // Set up all negative values
    const int tableSize = ltf->getTableSize();
    for (int i = 0; i < tableSize; ++i) {
        ltf->setBaseLayerValue(i, -0.5);
    }

    TransferFunctionOperations::normalize(*ltf);

    // Max abs should now be 1.0
    double maxAbs = 0.0;
    for (int i = 0; i < tableSize; ++i) {
        maxAbs = std::max(maxAbs, std::abs(ltf->getBaseLayerValue(i)));
    }
    EXPECT_NEAR(maxAbs, 1.0, 1e-10);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(TransferFunctionOperationsTest, ChainedOperations_InvertThenNormalize) {
    // Set up a function
    const int tableSize = ltf->getTableSize();
    for (int i = 0; i < tableSize; ++i) {
        double const val = 0.5 * std::sin(2.0 * M_PI * i / tableSize);
        ltf->setBaseLayerValue(i, val);
    }

    // Chain operations
    TransferFunctionOperations::invert(*ltf);
    TransferFunctionOperations::normalize(*ltf);

    // Result should be inverted and normalized
    double maxAbs = 0.0;
    for (int i = 0; i < tableSize; ++i) {
        maxAbs = std::max(maxAbs, std::abs(ltf->getBaseLayerValue(i)));
    }
    EXPECT_NEAR(maxAbs, 1.0, 1e-10);
}

// ============================================================================
// CurveData Overload Tests
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
