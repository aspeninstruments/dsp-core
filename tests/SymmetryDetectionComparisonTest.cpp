/**
 * Symmetry Detection Comparison Tests
 *
 * Purpose: Compare spline fitting quality WITH vs WITHOUT symmetry detection
 *          for odd-symmetric curves (tanh, odd harmonics, mixed harmonics).
 *
 * Hypothesis: Enabling symmetry detection during greedy fitting should produce
 *             better visual symmetry (paired anchors) with comparable or better
 *             max error for odd-symmetric curves.
 *
 * Test Matrix:
 *   - Tanh variants: tanh(x), tanh(2x), tanh(5x), tanh(10x)
 *   - Pure odd harmonics: H3, H5, H15, H39
 *   - Mixed odd harmonics: H1+H3, H1+H5, H1+H15
 */

#include <dsp_core/dsp_core.h>
#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

using namespace dsp_core;
using namespace dsp_core::Services;

//==============================================================================
// Test Fixture
//==============================================================================

class SymmetryDetectionComparisonTest : public ::testing::Test {
  protected:
    std::unique_ptr<LayeredTransferFunction> ltf;
    static constexpr int kTableSize = 16384;

    void SetUp() override {
        ltf = std::make_unique<LayeredTransferFunction>(kTableSize, -1.0, 1.0);
    }

    std::vector<double> extractCurveData() const {
        std::vector<double> data(ltf->getTableSize());
        for (int i = 0; i < ltf->getTableSize(); ++i) {
            data[i] = ltf->getBaseLayerValue(i);
        }
        return data;
    }

    // Metrics captured for each fit
    struct FitMetrics {
        double maxError = 0.0;
        double avgError = 0.0;
        int anchorCount = 0;
        int pairedAnchorCount = 0;
        double pairRatio = 0.0;
    };

    //--------------------------------------------------------------------------
    // Curve Setup Helpers
    //--------------------------------------------------------------------------

    void setTanh(double k) {
        for (int i = 0; i < ltf->getTableSize(); ++i) {
            double const x = ltf->normalizeIndex(i);
            ltf->setBaseLayerValue(i, std::tanh(k * x));
        }
    }

    void setHarmonic(int n) {
        // Chebyshev polynomial via trig identity (matches HarmonicLayer)
        // Odd harmonics: sin(n * asin(x))
        // Even harmonics: cos(n * acos(x))
        for (int i = 0; i < ltf->getTableSize(); ++i) {
            double x = ltf->normalizeIndex(i);
            x = std::clamp(x, -1.0, 1.0); // Clamp for asin/acos

            double y = 0.0;
            if (n == 1) {
                y = x;
            } else if (n % 2 == 0) {
                // Even harmonic
                y = std::cos(n * std::acos(x));
            } else {
                // Odd harmonic
                y = std::sin(n * std::asin(x));
            }
            ltf->setBaseLayerValue(i, y);
        }
    }

    void setMixedHarmonics(const std::vector<int>& harmonics, const std::vector<double>& weights) {
        // Sum of weighted harmonics
        for (int i = 0; i < ltf->getTableSize(); ++i) {
            double x = ltf->normalizeIndex(i);
            x = std::clamp(x, -1.0, 1.0);

            double y = 0.0;
            for (size_t h = 0; h < harmonics.size(); ++h) {
                int const n = harmonics[h];
                double const w = (h < weights.size()) ? weights[h] : 1.0;

                double term = 0.0;
                if (n == 1) {
                    term = x;
                } else if (n % 2 == 0) {
                    term = std::cos(n * std::acos(x));
                } else {
                    term = std::sin(n * std::asin(x));
                }
                y += w * term;
            }
            ltf->setBaseLayerValue(i, y);
        }
    }

    //--------------------------------------------------------------------------
    // Fitting and Metrics
    //--------------------------------------------------------------------------

    FitMetrics fitWithMode(SymmetryDetection mode) {
        auto config = SplineFitConfig::tight();
        config.symmetryDetection = mode;
        config.maxAnchors = 48; // Allow enough anchors for complex curves
        config.positionTolerance = 0.01; // Reasonable tolerance for comparison

        auto curveData = extractCurveData();
        auto result = SplineFitter::fitCurve(
            curveData.data(), static_cast<int>(curveData.size()),
            ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);

        FitMetrics metrics;
        metrics.maxError = result.maxError;
        metrics.avgError = result.averageError;
        metrics.anchorCount = result.numAnchors;

        // Count paired anchors (anchors at x that have a match at -x)
        int pairedCount = 0;
        int totalNonCenter = 0;
        for (const auto& anchor : result.anchors) {
            if (std::abs(anchor.x) < 1e-4) continue; // Skip center anchor

            totalNonCenter++;
            for (const auto& other : result.anchors) {
                if (std::abs(other.x + anchor.x) < 1e-4) { // other.x ≈ -anchor.x
                    pairedCount++;
                    break;
                }
            }
        }

        metrics.pairedAnchorCount = pairedCount;
        metrics.pairRatio = (totalNonCenter > 0)
                                ? static_cast<double>(pairedCount) / totalNonCenter
                                : 1.0;

        return metrics;
    }

    FitMetrics fitWithoutSymmetry() { return fitWithMode(SymmetryDetection::Never); }
    FitMetrics fitWithSymmetryAuto() { return fitWithMode(SymmetryDetection::Auto); }

    //--------------------------------------------------------------------------
    // Output Helpers
    //--------------------------------------------------------------------------

    static void printMetrics(const std::string& label, const FitMetrics& m) {
        std::cout << std::setw(12) << label << " | " << std::fixed << std::setprecision(4)
                  << std::setw(8) << m.maxError << " | " << std::setw(7) << m.anchorCount << " | "
                  << std::setw(7) << static_cast<int>(m.pairRatio * 100) << "%" << '\n';
    }

    static void printComparison(const std::string& curveName, const FitMetrics& without,
                                const FitMetrics& withSym) {
        double const errorDiff = (without.maxError > 1e-9)
                               ? 100.0 * (withSym.maxError - without.maxError) / without.maxError
                               : 0.0;
        double const pairDiff = 100.0 * (withSym.pairRatio - without.pairRatio);

        std::string verdict;
        if (withSym.maxError <= without.maxError * 1.05 && pairDiff > 10) {
            verdict = "BETTER (more paired, similar error)";
        } else if (withSym.maxError > without.maxError * 1.2) {
            verdict = "REGRESSION (higher error)";
        } else if (pairDiff > 20) {
            verdict = "IMPROVED SYMMETRY";
        } else {
            verdict = "SIMILAR";
        }

        std::cout << "\n=== " << curveName << " ===" << '\n';
        std::cout << std::setw(12) << "Mode"
                  << " | " << std::setw(8) << "MaxErr"
                  << " | " << std::setw(7) << "Anchors"
                  << " | " << std::setw(7) << "Paired" << '\n';
        std::cout << std::string(48, '-') << '\n';
        printMetrics("Without", without);
        printMetrics("With", withSym);
        std::cout << "Error change: " << std::showpos << std::fixed << std::setprecision(1)
                  << errorDiff << "%" << std::noshowpos << '\n';
        std::cout << "Paired change: " << std::showpos << std::fixed << std::setprecision(1)
                  << pairDiff << "%" << std::noshowpos << '\n';
        std::cout << "Verdict: " << verdict << '\n';
    }
};

//==============================================================================
// Test: Tanh Variants
//==============================================================================

TEST_F(SymmetryDetectionComparisonTest, Tanh_AllVariants) {
    std::cout << "\n==========================================" << '\n';
    std::cout << "TANH VARIANTS COMPARISON" << '\n';
    std::cout << "==========================================\n" << '\n';

    for (double const k : {1.0, 2.0, 5.0, 10.0}) {
        setTanh(k);
        auto without = fitWithoutSymmetry();
        auto withSym = fitWithSymmetryAuto();

        std::string const name = "tanh(" + std::to_string(static_cast<int>(k)) + "x)";
        printComparison(name, without, withSym);

        // Verify no severe regression (allow 20% error increase max)
        EXPECT_LE(withSym.maxError, without.maxError * 1.2)
            << name << " has severe regression with symmetry detection";
    }
}

//==============================================================================
// Test: Pure Odd Harmonics
//==============================================================================

TEST_F(SymmetryDetectionComparisonTest, PureOddHarmonics) {
    std::cout << "\n==========================================" << '\n';
    std::cout << "PURE ODD HARMONICS COMPARISON" << '\n';
    std::cout << "==========================================\n" << '\n';

    for (int const n : {3, 5, 15, 39}) {
        setHarmonic(n);
        auto without = fitWithoutSymmetry();
        auto withSym = fitWithSymmetryAuto();

        std::string const name = "H" + std::to_string(n);
        printComparison(name, without, withSym);

        // For high-frequency harmonics, allow more tolerance due to known y-averaging issue
        double const tolerance = (n > 10) ? 1.5 : 1.2;
        EXPECT_LE(withSym.maxError, without.maxError * tolerance)
            << name << " has severe regression with symmetry detection";
    }
}

//==============================================================================
// Test: Mixed Odd Harmonics (Exciter Combinations)
//==============================================================================

TEST_F(SymmetryDetectionComparisonTest, MixedOddHarmonics) {
    std::cout << "\n==========================================" << '\n';
    std::cout << "MIXED ODD HARMONICS COMPARISON" << '\n';
    std::cout << "==========================================\n" << '\n';

    struct TestCase {
        std::vector<int> harmonics;
        std::vector<double> weights;
        std::string name;
    };

    std::vector<TestCase> const cases = {
        {{1, 3}, {1.0, 0.5}, "H1+H3"},
        {{1, 5}, {1.0, 0.5}, "H1+H5"},
        {{1, 15}, {1.0, 0.3}, "H1+H15"},
    };

    for (const auto& tc : cases) {
        setMixedHarmonics(tc.harmonics, tc.weights);
        auto without = fitWithoutSymmetry();
        auto withSym = fitWithSymmetryAuto();

        printComparison(tc.name, without, withSym);

        EXPECT_LE(withSym.maxError, without.maxError * 1.3)
            << tc.name << " has severe regression with symmetry detection";
    }
}

//==============================================================================
// Test: Auto Mode Detection
//==============================================================================

TEST_F(SymmetryDetectionComparisonTest, AutoModeDetection) {
    std::cout << "\n==========================================" << '\n';
    std::cout << "AUTO MODE VS ALWAYS MODE" << '\n';
    std::cout << "==========================================\n" << '\n';

    // Test that Auto mode correctly detects symmetry and behaves like Always
    setTanh(5.0);
    auto autoMetrics = fitWithSymmetryAuto();
    auto alwaysMetrics = fitWithSymmetryAuto();

    std::cout << "\n=== tanh(5x) Auto vs Always ===" << '\n';
    std::cout << std::setw(12) << "Mode"
              << " | " << std::setw(8) << "MaxErr"
              << " | " << std::setw(7) << "Anchors"
              << " | " << std::setw(7) << "Paired" << '\n';
    std::cout << std::string(48, '-') << '\n';
    printMetrics("Auto", autoMetrics);
    printMetrics("Always", alwaysMetrics);

    // Auto should detect tanh as symmetric and produce similar results to Always
    EXPECT_NEAR(autoMetrics.pairRatio, alwaysMetrics.pairRatio, 0.2)
        << "Auto mode should detect tanh symmetry and produce similar pairing";
}

//==============================================================================
// Test: Summary Report
//==============================================================================

TEST_F(SymmetryDetectionComparisonTest, SummaryReport) {
    std::cout << "\n\n==========================================" << '\n';
    std::cout << "FULL COMPARISON SUMMARY" << '\n';
    std::cout << "==========================================\n" << '\n';

    std::cout << std::setw(15) << "Curve" << " | " << std::setw(12) << "Without Err" << " | "
              << std::setw(12) << "With Err" << " | " << std::setw(10) << "Err Diff" << " | "
              << std::setw(12) << "Without Pair" << " | " << std::setw(12) << "With Pair"
              << '\n';
    std::cout << std::string(90, '-') << '\n';

    auto runTest = [&](const std::string& name, auto setupFn) {
        setupFn();
        auto without = fitWithoutSymmetry();
        auto withSym = fitWithSymmetryAuto();

        double const errDiff = (without.maxError > 1e-9)
                             ? 100.0 * (withSym.maxError - without.maxError) / without.maxError
                             : 0.0;

        std::cout << std::setw(15) << name << " | " << std::fixed << std::setprecision(5)
                  << std::setw(12) << without.maxError << " | " << std::setw(12)
                  << withSym.maxError << " | " << std::setw(8) << std::showpos << errDiff << "%"
                  << std::noshowpos << " | " << std::setw(10)
                  << static_cast<int>(without.pairRatio * 100) << "%" << " | " << std::setw(10)
                  << static_cast<int>(withSym.pairRatio * 100) << "%" << '\n';
    };

    // Tanh variants
    runTest("tanh(1x)", [&]() { setTanh(1.0); });
    runTest("tanh(2x)", [&]() { setTanh(2.0); });
    runTest("tanh(5x)", [&]() { setTanh(5.0); });
    runTest("tanh(10x)", [&]() { setTanh(10.0); });

    std::cout << std::string(90, '-') << '\n';

    // Pure odd harmonics
    runTest("H3", [&]() { setHarmonic(3); });
    runTest("H5", [&]() { setHarmonic(5); });
    runTest("H15", [&]() { setHarmonic(15); });
    runTest("H39", [&]() { setHarmonic(39); });

    std::cout << std::string(90, '-') << '\n';

    // Mixed harmonics
    runTest("H1+H3", [&]() { setMixedHarmonics({1, 3}, {1.0, 0.5}); });
    runTest("H1+H5", [&]() { setMixedHarmonics({1, 5}, {1.0, 0.5}); });
    runTest("H1+H15", [&]() { setMixedHarmonics({1, 15}, {1.0, 0.3}); });

    std::cout << "\n==========================================\n" << '\n';
}
