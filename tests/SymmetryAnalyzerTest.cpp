#include <dsp_core/dsp_core.h>
#include <gtest/gtest.h>
#include <cmath>

using namespace dsp_core;
using namespace dsp_core::Services;

//==============================================================================
// Test Fixture
//==============================================================================

class SymmetryAnalyzerTest : public ::testing::Test {
  protected:
    std::unique_ptr<LayeredTransferFunction> ltf;
    const int tableSize = 16384; // Power-of-2 for FFT/SIMD

    void SetUp() override {
        ltf = std::make_unique<LayeredTransferFunction>(tableSize, -1.0, 1.0);
        // Reset coefficients to traditional test defaults (WT=1.0, all harmonics=0.0)
        ltf->setCoefficient(0, 1.0);
        for (int i = 1; i < ltf->getNumCoefficients(); ++i) {
            ltf->setCoefficient(i, 0.0);
        }
    }

    std::vector<double> extractCurveData() const {
        std::vector<double> data(ltf->getTableSize());
        for (int i = 0; i < ltf->getTableSize(); ++i) {
            data[i] = ltf->getBaseLayerValue(i);
        }
        return data;
    }

    // Helper: Set curve to polynomial: y = x^n
    void setPolynomial(int power) {
        for (int i = 0; i < ltf->getTableSize(); ++i) {
            double const x = ltf->normalizeIndex(i);
            ltf->setBaseLayerValue(i, std::pow(x, power));
        }
    }

    // Helper: Set curve to tanh
    void setTanh(double steepness = 5.0) {
        for (int i = 0; i < ltf->getTableSize(); ++i) {
            double const x = ltf->normalizeIndex(i);
            ltf->setBaseLayerValue(i, std::tanh(steepness * x));
        }
    }

    // Helper: Set curve to harmonic (Chebyshev polynomial)
    void setHarmonic(int n) {
        for (int i = 0; i < ltf->getTableSize(); ++i) {
            double const x = ltf->normalizeIndex(i);
            double y = 0.0;

            // Chebyshev polynomial evaluation
            if (n == 1) {
                y = x;
            } else if (n == 2) {
                y = 2.0 * x * x - 1.0;
            } else if (n == 3) {
                y = 4.0 * x * x * x - 3.0 * x;
            } else if (n == 5) {
                y = 16.0 * std::pow(x, 5) - 20.0 * std::pow(x, 3) + 5.0 * x;
            }

            ltf->setBaseLayerValue(i, y);
        }
    }
};

//==============================================================================
// Test 1: Identity Function - Linear Symmetry
//==============================================================================

TEST_F(SymmetryAnalyzerTest, Symmetry_IdentityFunction_PerfectSymmetry) {
    // Setup: y = x (odd function: f(-x) = -f(x))
    setPolynomial(1);

    // Execute
    auto curveData = extractCurveData();
    auto result = SymmetryAnalyzer::analyzeOddSymmetry(curveData);

    // Verify
    EXPECT_GE(result.score, 0.99) << "Identity function should have near-perfect symmetry";
    EXPECT_EQ(result.classification, SymmetryAnalyzer::Result::Classification::Perfect);
    EXPECT_TRUE(result.shouldUsePairedAnchors());
}

//==============================================================================
// Test 2: Cubic Polynomial - Perfect Odd Symmetry
//==============================================================================

TEST_F(SymmetryAnalyzerTest, Symmetry_CubicPolynomial_PerfectSymmetry) {
    // Setup: y = x³ (classic odd function)
    setPolynomial(3);

    // Execute
    auto curveData = extractCurveData();
    auto result = SymmetryAnalyzer::analyzeOddSymmetry(curveData);

    // Verify
    EXPECT_GE(result.score, 0.99) << "Cubic polynomial should have perfect symmetry";
    EXPECT_EQ(result.classification, SymmetryAnalyzer::Result::Classification::Perfect);
    EXPECT_TRUE(result.shouldUsePairedAnchors());
    EXPECT_DOUBLE_EQ(result.centerX, 0.0) << "Center should be at origin";
}

//==============================================================================
// Test 3: Tanh Curve - Perfect Odd Symmetry
//==============================================================================

TEST_F(SymmetryAnalyzerTest, Symmetry_TanhCurve_PerfectSymmetry) {
    // Setup: y = tanh(5x) (smooth odd function)
    setTanh(5.0);

    // Execute
    auto curveData = extractCurveData();
    auto result = SymmetryAnalyzer::analyzeOddSymmetry(curveData);

    // Verify
    EXPECT_GE(result.score, 0.99) << "Tanh should have perfect symmetry";
    EXPECT_EQ(result.classification, SymmetryAnalyzer::Result::Classification::Perfect);
    EXPECT_TRUE(result.shouldUsePairedAnchors());
}

//==============================================================================
// Test 4: Tanh with Asymmetric Bump - Approximate Symmetry
//==============================================================================

TEST_F(SymmetryAnalyzerTest, Symmetry_TanhWithBump_ApproximateSymmetry) {
    // Setup: tanh + small asymmetric bump
    setTanh(5.0);

    // Add asymmetric bump to right side only (make it larger to ensure score < 0.99)
    for (int i = ltf->getTableSize() / 2; i < ltf->getTableSize(); ++i) {
        double const x = ltf->normalizeIndex(i);
        if (x > 0.4 && x < 0.6) {
            double const currentY = ltf->getBaseLayerValue(i);
            ltf->setBaseLayerValue(i, currentY + 0.15); // Larger bump to break perfect symmetry
        }
    }

    // Execute
    auto curveData = extractCurveData();
    auto result = SymmetryAnalyzer::analyzeOddSymmetry(curveData);

    // Verify
    EXPECT_GE(result.score, 0.85) << "Should still detect approximate symmetry";
    EXPECT_LT(result.score, 0.99) << "But not perfect due to asymmetric bump";
    // Classification could be Approximate or Asymmetric depending on bump size
    EXPECT_TRUE(result.score >= 0.90 || result.classification == SymmetryAnalyzer::Result::Classification::Asymmetric);
}

//==============================================================================
// Test 5: Asymmetric Curve (Even Function) - Not Odd Symmetric
//==============================================================================

TEST_F(SymmetryAnalyzerTest, Symmetry_AsymmetricCurve_NotSymmetric) {
    // Setup: y = x² (even function: f(-x) = f(x), NOT odd symmetric!)
    setPolynomial(2);

    // Execute
    auto curveData = extractCurveData();
    auto result = SymmetryAnalyzer::analyzeOddSymmetry(curveData);

    // Verify
    EXPECT_LT(result.score, 0.90) << "Even function should NOT be odd-symmetric";
    EXPECT_EQ(result.classification, SymmetryAnalyzer::Result::Classification::Asymmetric);
    EXPECT_FALSE(result.shouldUsePairedAnchors());
}

//==============================================================================
// Test 6: Harmonic 3 (Odd) - Perfect Symmetry
//==============================================================================

TEST_F(SymmetryAnalyzerTest, Symmetry_HarmonicOdd_PerfectSymmetry) {
    // Setup: Harmonic 3 (Chebyshev T₃, odd function)
    setHarmonic(3);

    // Execute
    auto curveData = extractCurveData();
    auto result = SymmetryAnalyzer::analyzeOddSymmetry(curveData);

    // Verify
    EXPECT_GE(result.score, 0.99) << "Odd harmonic should have perfect symmetry";
    EXPECT_EQ(result.classification, SymmetryAnalyzer::Result::Classification::Perfect);
    EXPECT_TRUE(result.shouldUsePairedAnchors());
}

//==============================================================================
// Test 7: Harmonic 2 (Even) - Asymmetric
//==============================================================================

TEST_F(SymmetryAnalyzerTest, Symmetry_HarmonicEven_Asymmetric) {
    // Setup: Harmonic 2 (Chebyshev T₂, even function)
    setHarmonic(2);

    // Execute
    auto curveData = extractCurveData();
    auto result = SymmetryAnalyzer::analyzeOddSymmetry(curveData);

    // Verify
    EXPECT_LT(result.score, 0.90) << "Even harmonic should NOT be odd-symmetric";
    EXPECT_EQ(result.classification, SymmetryAnalyzer::Result::Classification::Asymmetric);
    EXPECT_FALSE(result.shouldUsePairedAnchors());
}

//==============================================================================
// Test 8: Config Thresholds - Affect Classification
//==============================================================================

TEST_F(SymmetryAnalyzerTest, Symmetry_ConfigThresholds_AffectClassification) {
    // Setup: Create curve with score around 0.95
    // Use tanh with small perturbation to get specific score
    setTanh(5.0);

    // Add very small asymmetry (score should be ~0.95-0.98)
    for (int i = ltf->getTableSize() / 2; i < ltf->getTableSize(); ++i) {
        double const x = ltf->normalizeIndex(i);
        if (x > 0.7) {
            double const currentY = ltf->getBaseLayerValue(i);
            ltf->setBaseLayerValue(i, currentY + 0.01); // Tiny bump
        }
    }

    // Execute with Config 1: perfectThreshold = 0.99 (default)
    SymmetryAnalyzer::Config config1;
    config1.perfectThreshold = 0.99;
    config1.approximateThreshold = 0.90;
    auto curveData = extractCurveData();
    auto result1 = SymmetryAnalyzer::analyzeOddSymmetry(curveData, config1);

    // Execute with Config 2: perfectThreshold = 0.90
    SymmetryAnalyzer::Config config2;
    config2.perfectThreshold = 0.90;
    config2.approximateThreshold = 0.80;
    auto result2 = SymmetryAnalyzer::analyzeOddSymmetry(curveData, config2);

    // Verify scores are identical (thresholds don't affect computation)
    EXPECT_DOUBLE_EQ(result1.score, result2.score);

    // Verify classifications differ based on thresholds
    if (result1.score >= 0.90 && result1.score < 0.99) {
        EXPECT_EQ(result1.classification, SymmetryAnalyzer::Result::Classification::Approximate)
            << "Config 1: score in [0.90, 0.99) should be Approximate";
        EXPECT_EQ(result2.classification, SymmetryAnalyzer::Result::Classification::Perfect)
            << "Config 2: same score but lower threshold should be Perfect";
    }
}

//==============================================================================
// CurveData Overload Tests
//==============================================================================

TEST_F(SymmetryAnalyzerTest, CurveData_TanhVector_PerfectSymmetry) {
    // Build a tanh curve as raw vector
    std::vector<double> curveData(tableSize);
    for (int i = 0; i < tableSize; ++i) {
        const double x = -1.0 + 2.0 * static_cast<double>(i) / (tableSize - 1);
        curveData[static_cast<size_t>(i)] = std::tanh(5.0 * x);
    }

    auto result = SymmetryAnalyzer::analyzeOddSymmetry(curveData);

    EXPECT_GE(result.score, 0.99);
    EXPECT_EQ(result.classification, SymmetryAnalyzer::Result::Classification::Perfect);
}

TEST_F(SymmetryAnalyzerTest, CurveData_EvenFunction_Asymmetric) {
    // x^2 is even, not odd-symmetric
    std::vector<double> curveData(tableSize);
    for (int i = 0; i < tableSize; ++i) {
        const double x = -1.0 + 2.0 * static_cast<double>(i) / (tableSize - 1);
        curveData[static_cast<size_t>(i)] = x * x;
    }

    auto result = SymmetryAnalyzer::analyzeOddSymmetry(curveData);

    EXPECT_LT(result.score, 0.90);
    EXPECT_EQ(result.classification, SymmetryAnalyzer::Result::Classification::Asymmetric);
}

TEST_F(SymmetryAnalyzerTest, CurveData_FlatZero_PerfectSymmetry) {
    std::vector<double> curveData(tableSize, 0.0);

    auto result = SymmetryAnalyzer::analyzeOddSymmetry(curveData);

    EXPECT_GE(result.score, 0.99);
}

TEST_F(SymmetryAnalyzerTest, CurveData_TooSmall_Asymmetric) {
    std::vector<double> curveData(2, 0.0);

    auto result = SymmetryAnalyzer::analyzeOddSymmetry(curveData);

    EXPECT_EQ(result.classification, SymmetryAnalyzer::Result::Classification::Asymmetric);
}

TEST_F(SymmetryAnalyzerTest, CurveData_MatchesExtractedData) {
    // Verify extractCurveData helper produces consistent results
    setTanh(5.0);

    // Extract via helper
    auto helperData = extractCurveData();

    // Extract manually
    std::vector<double> manualData(tableSize);
    for (int i = 0; i < tableSize; ++i) {
        manualData[static_cast<size_t>(i)] = ltf->getBaseLayerValue(i);
    }

    auto helperResult = SymmetryAnalyzer::analyzeOddSymmetry(helperData);
    auto manualResult = SymmetryAnalyzer::analyzeOddSymmetry(manualData);

    EXPECT_NEAR(helperResult.score, manualResult.score, 0.02);
    EXPECT_EQ(helperResult.classification, manualResult.classification);
}
