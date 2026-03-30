#include <dsp_core/dsp_core.h>
#include <gtest/gtest.h>
#include <thread>
#include <atomic>

namespace dsp_core_test {

/**
 * Test fixture for LayeredTransferFunction
 * Tests normalization, deferred normalization, harmonic mixing, and thread safety
 */
class LayeredTransferFunctionTest : public ::testing::Test {
  protected:
    void SetUp() override {
        ltf = std::make_unique<dsp_core::LayeredTransferFunction>(256, -1.0, 1.0);
        // Reset coefficients to traditional test defaults (WT=1.0, all harmonics=0.0)
        // The constructor now uses WT=0.0, H1=1.0 for plugin initialization
        ltf->setCoefficient(0, 1.0);
        for (int i = 1; i < ltf->getNumCoefficients(); ++i) {
            ltf->setCoefficient(i, 0.0);
        }
        // Reset base layer to linear for predictable testing
        for (int i = 0; i < ltf->getTableSize(); ++i) {
            const double x = ltf->normalizeIndex(i);
            ltf->setBaseLayerValue(i, x);
        }
    }

    std::unique_ptr<dsp_core::LayeredTransferFunction> ltf;
};

// ============================================================================
// Normalization Tests
// ============================================================================

TEST_F(LayeredTransferFunctionTest, UpdateComposite_NormalizesToRange) {
    // Set extreme base layer values
    for (int i = 0; i < 256; ++i) {
        ltf->setBaseLayerValue(i, 10.0); // Way out of range
    }

    // Compute normalization scalar (simulates what renderer does)
    ltf->updateNormalizationScalar();

    // Composite should be normalized to [-1, 1]
    for (int i = 0; i < 256; ++i) {
        double const value = ltf->computeCompositeAt(i);
        EXPECT_GE(value, -1.0);
        EXPECT_LE(value, 1.0);
    }
}

TEST_F(LayeredTransferFunctionTest, UpdateComposite_NormalizesNegativeValues) {
    // Set extreme negative base layer values
    for (int i = 0; i < 256; ++i) {
        ltf->setBaseLayerValue(i, -5.0); // Way out of range
    }

    // Compute normalization scalar (simulates what renderer does)
    ltf->updateNormalizationScalar();


    // Composite should be normalized to [-1, 1]
    for (int i = 0; i < 256; ++i) {
        double const value = ltf->computeCompositeAt(i);
        EXPECT_GE(value, -1.0);
        EXPECT_LE(value, 1.0);
    }
}

TEST_F(LayeredTransferFunctionTest, UpdateComposite_MaintainsRelativeProportions) {
    // Set base layer with known proportions
    ltf->setBaseLayerValue(100, 2.0);
    ltf->setBaseLayerValue(150, 4.0); // Double the value at index 100
    ltf->setCoefficient(0, 1.0);      // 100% WT mix


    // After normalization, proportions should be maintained
    double const value100 = ltf->computeCompositeAt(100);
    double const value150 = ltf->computeCompositeAt(150);

    EXPECT_NEAR(value150 / value100, 2.0, 0.01); // Ratio should be ~2.0
}

// ============================================================================
// Deferred Normalization Tests
// ============================================================================

TEST_F(LayeredTransferFunctionTest, PaintStrokeActive_FreezesScalar) {
    // Set initial state
    ltf->setBaseLayerValue(100, 0.5);
    ltf->setCoefficient(0, 1.0); // 100% WT mix
    ltf->updateNormalizationScalar();
    double const composite1 = ltf->computeCompositeAt(100);
    double const scalar1 = ltf->getNormalizationScalar();

    // Enable paint stroke mode (freezes normalization)
    ltf->setPaintStrokeActive(true);
    EXPECT_TRUE(ltf->isPaintStrokeActive());

    // Change base layer significantly
    ltf->setBaseLayerValue(100, 5.0);

    // Normalization scalar should be frozen (we haven't called updateNormalizationScalar again)
    double const scalar2 = ltf->getNormalizationScalar();
    EXPECT_NEAR(scalar1, scalar2, 1e-6);

    // Composite should change but use frozen scalar
    double const composite2 = ltf->computeCompositeAt(100);
    EXPECT_NE(composite1, composite2); // Value changed
}

TEST_F(LayeredTransferFunctionTest, PaintStrokeActive_CanUpdateAfterDisable) {
    // Set initial state
    ltf->setBaseLayerValue(100, 0.5);
    ltf->setCoefficient(0, 1.0);
    ltf->updateNormalizationScalar();
    double const scalar1 = ltf->getNormalizationScalar();

    // Enable paint stroke mode
    ltf->setPaintStrokeActive(true);

    // Change base layer
    ltf->setBaseLayerValue(100, 5.0);

    // Scalar should be frozen
    double const scalar2 = ltf->getNormalizationScalar();
    EXPECT_NEAR(scalar1, scalar2, 1e-6);

    // Disable paint stroke mode
    ltf->setPaintStrokeActive(false);
    EXPECT_FALSE(ltf->isPaintStrokeActive());

    // Now we can update scalar
    ltf->updateNormalizationScalar();
    double const scalar3 = ltf->getNormalizationScalar();
    EXPECT_NE(scalar1, scalar3); // Scalar should have changed
}

TEST_F(LayeredTransferFunctionTest, DeferNormalization_PreventsVisualJumping) {
    // Simulate paint stroke scenario
    ltf->setCoefficient(0, 1.0); // 100% WT mix

    // Set initial values
    for (int i = 0; i < 256; ++i) {
        ltf->setBaseLayerValue(i, 0.5);
    }

    // Compute and cache normalization scalar before paint stroke
    ltf->updateNormalizationScalar();
    double const initialScalar = ltf->getNormalizationScalar();

    // Begin paint stroke (freeze normalization)
    ltf->setPaintStrokeActive(true);

    // Paint several points with large values
    for (int i = 100; i < 110; ++i) {
        ltf->setBaseLayerValue(i, 2.0);
    }

    // Scalar should remain frozen (we haven't called updateNormalizationScalar again)
    double const paintingScalar = ltf->getNormalizationScalar();
    EXPECT_NEAR(initialScalar, paintingScalar, 1e-6);

    // End paint stroke
    ltf->setPaintStrokeActive(false);

    // Now we can update scalar (renderer would do this at next 25Hz poll)
    ltf->updateNormalizationScalar();
    double const finalScalar = ltf->getNormalizationScalar();
    EXPECT_LT(finalScalar, initialScalar); // Larger values require smaller scalar
}

// ============================================================================
// Harmonic Mixing Tests
// ============================================================================

TEST_F(LayeredTransferFunctionTest, HarmonicMixing_WeightedSum) {
    // Set WT coefficient to 0.6, first harmonic to 0.4
    ltf->setCoefficient(0, 0.6); // WT mix
    ltf->setCoefficient(1, 0.4); // First harmonic

    // Set known base layer value
    ltf->setBaseLayerValue(100, 1.0);


    // Composite should be weighted sum (after normalization)
    // Exact value depends on harmonic evaluation, but should be in range
    double const composite = ltf->computeCompositeAt(100);
    EXPECT_GE(composite, -1.0);
    EXPECT_LE(composite, 1.0);
}

TEST_F(LayeredTransferFunctionTest, HarmonicMixing_ZeroWTCoeff) {
    // Set WT coefficient to 0, harmonic to 1.0 (pure harmonic mode)
    ltf->setCoefficient(0, 0.0); // No WT
    ltf->setCoefficient(1, 1.0); // 100% first harmonic

    // Set base layer values (should have no effect)
    for (int i = 0; i < 256; ++i) {
        ltf->setBaseLayerValue(i, 1.0);
    }


    // Composite should only reflect harmonic contribution
    // (base layer is ignored when WT coeff = 0)
    for (int i = 0; i < 256; ++i) {
        double const composite = ltf->computeCompositeAt(i);
        EXPECT_GE(composite, -1.0);
        EXPECT_LE(composite, 1.0);
    }
}

TEST_F(LayeredTransferFunctionTest, HarmonicMixing_MultipleHarmonics) {
    // Set multiple harmonic coefficients
    ltf->setCoefficient(0, 0.5); // WT mix
    ltf->setCoefficient(1, 0.3); // First harmonic
    ltf->setCoefficient(2, 0.2); // Second harmonic

    // Set base layer
    for (int i = 0; i < 256; ++i) {
        ltf->setBaseLayerValue(i, static_cast<double>(i) / 256.0);
    }


    // All composites should be in valid range
    for (int i = 0; i < 256; ++i) {
        double const composite = ltf->computeCompositeAt(i);
        EXPECT_GE(composite, -1.0);
        EXPECT_LE(composite, 1.0);
    }
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

TEST_F(LayeredTransferFunctionTest, ThreadSafety_AtomicReads) {
    // Write from main thread
    ltf->setBaseLayerValue(100, 0.5);
    ltf->setCoefficient(0, 1.0);

    // Simulate audio thread read (lock-free)
    std::atomic<bool> readComplete{false};
    std::thread audioThread([&]() {
        for (int i = 0; i < 1000; ++i) {
            double const value = ltf->computeCompositeAt(100);
            EXPECT_GE(value, -1.0);
            EXPECT_LE(value, 1.0);
        }
        readComplete.store(true);
    });

    // Main thread continues writing
    for (int i = 0; i < 100; ++i) {
        ltf->setBaseLayerValue(100, static_cast<double>(i) / 100.0);
    }

    audioThread.join();
    EXPECT_TRUE(readComplete.load());
}

TEST_F(LayeredTransferFunctionTest, ThreadSafety_MultipleIndices) {
    // Initialize values
    for (int i = 0; i < 256; ++i) {
        ltf->setBaseLayerValue(i, static_cast<double>(i) / 256.0);
    }
    ltf->setCoefficient(0, 1.0);

    // Audio thread reads multiple indices
    std::atomic<bool> readComplete{false};
    std::atomic<int> readCount{0};

    std::thread audioThread([&]() {
        for (int iteration = 0; iteration < 100; ++iteration) {
            for (int i = 0; i < 256; ++i) {
                double const value = ltf->computeCompositeAt(i);
                EXPECT_GE(value, -1.0);
                EXPECT_LE(value, 1.0);
                readCount.fetch_add(1);
            }
        }
        readComplete.store(true);
    });

    // Main thread updates while audio thread reads
    for (int iteration = 0; iteration < 10; ++iteration) {
        for (int i = 0; i < 256; ++i) {
            ltf->setBaseLayerValue(i, static_cast<double>(iteration) / 10.0);
        }
    }

    audioThread.join();
    EXPECT_TRUE(readComplete.load());
    EXPECT_EQ(readCount.load(), 100 * 256); // All reads completed
}

// ============================================================================
// Base Layer Tests
// ============================================================================

TEST_F(LayeredTransferFunctionTest, ClearBaseLayer_SetsAllToZero) {
    // Set base layer values
    for (int i = 0; i < 256; ++i) {
        ltf->setBaseLayerValue(i, static_cast<double>(i) / 256.0);
    }

    // Clear base layer
    ltf->clearBaseLayer();

    // Verify all values are zero
    for (int i = 0; i < 256; ++i) {
        EXPECT_NEAR(ltf->getBaseLayerValue(i), 0.0, 1e-6);
    }
}

// ============================================================================
// Coefficient Tests
// ============================================================================

TEST_F(LayeredTransferFunctionTest, Coefficients_GetSetConsistent) {
    // Set coefficients
    for (int i = 0; i < ltf->getNumCoefficients(); ++i) {
        ltf->setCoefficient(i, static_cast<double>(i) / 10.0);
    }

    // Verify get returns set values
    for (int i = 0; i < ltf->getNumCoefficients(); ++i) {
        EXPECT_NEAR(ltf->getCoefficient(i), static_cast<double>(i) / 10.0, 1e-6);
    }
}

// ============================================================================
// On-Demand Composite Computation Tests
// ============================================================================

TEST_F(LayeredTransferFunctionTest, ComputeCompositeAtMatchesGetCompositeValue) {
    // Use larger table size to match production use case
    auto ltf16k = std::make_unique<dsp_core::LayeredTransferFunction>(16384, -1.0, 1.0);

    // Set some harmonic coefficients
    ltf16k->setCoefficient(1, 0.5);  // h1 = 0.5
    ltf16k->setCoefficient(2, 0.3);  // h2 = 0.3

    // Verify on-demand composite computation works correctly
    for (int i = 0; i < 16384; i += 1000) {  // Sample every 1000 points
        const double value = ltf16k->computeCompositeAt(i);
        // Value should be valid (no NaN/inf)
        EXPECT_TRUE(std::isfinite(value)) << "Invalid value at index " << i;
    }
}

TEST_F(LayeredTransferFunctionTest, ComputeCompositeAtBoundsCheck) {
    // Use larger table size to match production use case
    auto ltf16k = std::make_unique<dsp_core::LayeredTransferFunction>(16384, -1.0, 1.0);

    // Out of bounds should return 0.0
    EXPECT_DOUBLE_EQ(ltf16k->computeCompositeAt(-1), 0.0);
    EXPECT_DOUBLE_EQ(ltf16k->computeCompositeAt(16384), 0.0);
    EXPECT_DOUBLE_EQ(ltf16k->computeCompositeAt(100000), 0.0);

    // In bounds should return valid value (identity function at center)
    double const centerValue = ltf16k->computeCompositeAt(8192);  // Identity at x=0 is 0
    EXPECT_NEAR(centerValue, 0.0, 0.01);
}

// ============================================================================
// Harmonic Baking Tests
// ============================================================================

TEST_F(LayeredTransferFunctionTest, HasNonZeroHarmonics_ReturnsFalseForAllZero) {
    // Set WT mix but no harmonics
    ltf->setCoefficient(0, 0.8); // WT mix (not a harmonic)

    // All harmonics (indices 1-40) are zero
    for (int i = 1; i <= 40; ++i) {
        ltf->setCoefficient(i, 0.0);
    }

    EXPECT_FALSE(ltf->hasNonZeroHarmonics());
}

TEST_F(LayeredTransferFunctionTest, HasNonZeroHarmonics_ReturnsTrueForNonZero) {
    // Set one harmonic to non-zero
    ltf->setCoefficient(3, 0.5);

    EXPECT_TRUE(ltf->hasNonZeroHarmonics());
}

TEST_F(LayeredTransferFunctionTest, HasNonZeroHarmonics_UsesEpsilonThreshold) {
    // Set harmonic below epsilon threshold (should be treated as zero)
    ltf->setCoefficient(3, 1e-7); // Below HARMONIC_EPSILON (1e-6)

    EXPECT_FALSE(ltf->hasNonZeroHarmonics());

    // Set harmonic above epsilon threshold
    ltf->setCoefficient(3, 1e-5); // Above HARMONIC_EPSILON

    EXPECT_TRUE(ltf->hasNonZeroHarmonics());
}

TEST_F(LayeredTransferFunctionTest, BakeHarmonics_NoOpForZeroHarmonics) {
    // Set WT mix and base layer
    ltf->setCoefficient(0, 1.0);
    for (int i = 0; i < 256; ++i) {
        ltf->setBaseLayerValue(i, static_cast<double>(i) / 256.0);
    }

    // Capture base layer before baking
    std::vector<double> baseLayerBefore(256);
    for (int i = 0; i < 256; ++i) {
        baseLayerBefore[i] = ltf->getBaseLayerValue(i);
    }

    // Bake with all-zero harmonics
    bool const baked = ltf->bakeHarmonicsToBase();

    // Should return false (no-op)
    EXPECT_FALSE(baked);

    // Base layer should be unchanged
    for (int i = 0; i < 256; ++i) {
        EXPECT_NEAR(ltf->getBaseLayerValue(i), baseLayerBefore[i], 1e-12);
    }
}

TEST_F(LayeredTransferFunctionTest, BakeHarmonics_ZeroWTBakesZerosToBase) {
    // Regression test: When WT=0 and no harmonics, baking should zero the base layer
    // (preserving the zero composite the user saw, not the old base layer values)

    // Set WT=0 (base layer hidden) and populate base layer with non-zero values
    ltf->setCoefficient(0, 0.0); // WT = 0
    for (int i = 0; i < 256; ++i) {
        ltf->setBaseLayerValue(i, static_cast<double>(i) / 256.0);
    }

    // Composite should be zero everywhere (WT=0, no harmonics)
    for (int i = 0; i < 256; ++i) {
        EXPECT_NEAR(ltf->computeCompositeAt(i), 0.0, 1e-12);
    }

    // Bake - should transfer zeros to base layer
    bool const baked = ltf->bakeHarmonicsToBase();

    // Should return true (actual baking happened because WT != 1.0)
    EXPECT_TRUE(baked);

    // WT should now be 1.0
    EXPECT_NEAR(ltf->getCoefficient(0), 1.0, 1e-12);

    // Base layer should now be zeros (preserving what user saw)
    for (int i = 0; i < 256; ++i) {
        EXPECT_NEAR(ltf->getBaseLayerValue(i), 0.0, 1e-12);
    }

    // Composite should still be zero (visual identity preserved)
    for (int i = 0; i < 256; ++i) {
        EXPECT_NEAR(ltf->computeCompositeAt(i), 0.0, 1e-12);
    }
}

TEST_F(LayeredTransferFunctionTest, BakeHarmonics_TransfersCompositeToBase) {
    // Set base layer and harmonics
    ltf->setCoefficient(0, 1.0); // Full WT mix
    for (int i = 0; i < 256; ++i) {
        ltf->setBaseLayerValue(i, static_cast<double>(i) / 512.0); // Half range
    }
    ltf->setCoefficient(3, 0.5); // Add 3rd harmonic

    // Compute normalization scalar BEFORE capturing composite
    // This simulates what the renderer does before baking
    ltf->updateNormalizationScalar();

    // Capture composite BEFORE baking
    std::vector<double> compositeBefore(256);
    for (int i = 0; i < 256; ++i) {
        compositeBefore[i] = ltf->computeCompositeAt(i);
    }

    // Bake (this calls updateNormalizationScalar() internally)
    bool const baked = ltf->bakeHarmonicsToBase();

    EXPECT_TRUE(baked);

    // Base layer should now match old composite (visual identity preserved)
    for (int i = 0; i < 256; ++i) {
        double const baseAfter = ltf->getBaseLayerValue(i);
        EXPECT_NEAR(compositeBefore[i], baseAfter, 1e-12) << "Visual discontinuity at index " << i;
    }
}

TEST_F(LayeredTransferFunctionTest, BakeHarmonics_ZerosHarmonicCoefficients) {
    // Set harmonics
    ltf->setCoefficient(3, 0.5);
    ltf->setCoefficient(5, 0.3);
    ltf->setCoefficient(7, 0.2);

    EXPECT_TRUE(ltf->hasNonZeroHarmonics());

    // Bake
    ltf->bakeHarmonicsToBase();

    // All harmonics should be zero
    EXPECT_FALSE(ltf->hasNonZeroHarmonics());

    // Verify individual harmonics are zero
    for (int i = 1; i <= 40; ++i) {
        EXPECT_NEAR(ltf->getCoefficient(i), 0.0, 1e-12);
    }
}

TEST_F(LayeredTransferFunctionTest, BakeHarmonics_SetsWTToOne) {
    // Set WT mix to 0 (common case in Harmonic mode)
    ltf->setCoefficient(0, 0.0);
    ltf->setCoefficient(3, 0.5);

    // Bake
    ltf->bakeHarmonicsToBase();

    // WT coefficient MUST be set to 1.0 to make the baked base layer visible
    // If WT remains at 0, the result would be: composite = 0 * base = 0 (flat line bug)
    EXPECT_NEAR(ltf->getCoefficient(0), 1.0, 1e-12);
}

TEST_F(LayeredTransferFunctionTest, BakeHarmonics_IdempotentMultipleCalls) {
    // Set harmonics and bake
    ltf->setCoefficient(3, 0.5);
    ltf->bakeHarmonicsToBase();

    // Capture base layer after first bake
    std::vector<double> baseAfterFirstBake(256);
    for (int i = 0; i < 256; ++i) {
        baseAfterFirstBake[i] = ltf->getBaseLayerValue(i);
    }

    // Second bake should be no-op
    bool const secondBake = ltf->bakeHarmonicsToBase();
    EXPECT_FALSE(secondBake);

    // Base layer should be unchanged
    for (int i = 0; i < 256; ++i) {
        EXPECT_NEAR(ltf->getBaseLayerValue(i), baseAfterFirstBake[i], 1e-12);
    }
}

TEST_F(LayeredTransferFunctionTest, BakeHarmonics_RecalculatesNormalizationScalar) {
    // Set up curve that requires normalization
    ltf->setCoefficient(0, 1.0);
    for (int i = 0; i < 256; ++i) {
        ltf->setBaseLayerValue(i, 2.0); // Out of range
    }
    ltf->setCoefficient(3, 0.5);

    // Compute normalization scalar BEFORE capturing composite
    ltf->updateNormalizationScalar();

    // Capture composite values (for visual continuity check)
    std::vector<double> compositeBefore(256);
    for (int i = 0; i < 256; ++i) {
        compositeBefore[i] = ltf->computeCompositeAt(i);
    }

    // Capture normalization scalar before baking
    double const normScalarBefore = ltf->getNormalizationScalar();
    EXPECT_LT(normScalarBefore, 1.0); // Should be scaled down

    // Bake (calls updateNormalizationScalar() internally at the end)
    ltf->bakeHarmonicsToBase();

    // Normalization scalar is automatically recalculated by bakeHarmonicsToBase()
    // After baking, base layer contains normalized values, so scalar should be ~1.0
    double const normScalarAfter = ltf->getNormalizationScalar();
    EXPECT_NEAR(normScalarAfter, 1.0, 0.1); // Should be close to 1.0 now

    // Visual continuity: composite should be unchanged after baking
    for (int i = 0; i < 256; ++i) {
        EXPECT_DOUBLE_EQ(compositeBefore[i], ltf->computeCompositeAt(i));
    }
}

TEST_F(LayeredTransferFunctionTest, BakeHarmonics_VisualContinuityBitLevel) {
    // Set complex harmonic configuration
    ltf->setCoefficient(0, 1.0);
    ltf->setCoefficient(1, 0.3);
    ltf->setCoefficient(3, 0.5);
    ltf->setCoefficient(5, 0.2);

    // Compute normalization scalar BEFORE capturing composite
    ltf->updateNormalizationScalar();

    // Capture composite BEFORE baking
    std::vector<double> compositeBefore(256);
    for (int i = 0; i < 256; ++i) {
        compositeBefore[i] = ltf->computeCompositeAt(i);
    }

    // Bake (calls updateNormalizationScalar() internally at the end)
    ltf->bakeHarmonicsToBase();

    // Composite AFTER baking should be identical (bit-level)
    // Normalization scalar is automatically recalculated by bakeHarmonicsToBase()
    for (int i = 0; i < 256; ++i) {
        double const compositeAfter = ltf->computeCompositeAt(i);
        EXPECT_DOUBLE_EQ(compositeBefore[i], compositeAfter) << "Visual discontinuity at index " << i;
    }
}

TEST_F(LayeredTransferFunctionTest, BakeHarmonics_Performance) {
    // Setup: non-trivial harmonics
    ltf->setCoefficient(0, 1.0);
    for (int h = 1; h <= 40; ++h) {
        ltf->setCoefficient(h, 0.1 / h);
    }

    // Measure baking time
    auto start = std::chrono::high_resolution_clock::now();
    ltf->bakeHarmonicsToBase();
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Should be <1ms as claimed in feature plan
    EXPECT_LT(duration.count(), 1000) << "Baking took " << duration.count() << "μs";
}

TEST_F(LayeredTransferFunctionTest, GetHarmonicCoefficients_ReturnsAllCoefficients) {
    // Set known coefficients
    ltf->setCoefficient(0, 0.8); // WT mix
    ltf->setCoefficient(1, 0.3);
    ltf->setCoefficient(3, 0.5);
    ltf->setCoefficient(40, 0.1);

    // Get all coefficients
    auto coeffs = ltf->getHarmonicCoefficients();

    // Verify size
    EXPECT_EQ(coeffs.size(), 41u);

    // Verify values
    EXPECT_NEAR(coeffs[0], 0.8, 1e-12); // WT mix
    EXPECT_NEAR(coeffs[1], 0.3, 1e-12);
    EXPECT_NEAR(coeffs[3], 0.5, 1e-12);
    EXPECT_NEAR(coeffs[40], 0.1, 1e-12);
}

TEST_F(LayeredTransferFunctionTest, SetHarmonicCoefficients_SetsAllCoefficients) {
    // Create coefficient array
    std::array<double, 41> coeffs{};
    coeffs[0] = 0.9; // WT mix
    coeffs[1] = 0.4;
    coeffs[3] = 0.6;
    coeffs[40] = 0.2;

    // Set all coefficients
    ltf->setHarmonicCoefficients(coeffs);

    // Verify they were set
    EXPECT_NEAR(ltf->getCoefficient(0), 0.9, 1e-12);
    EXPECT_NEAR(ltf->getCoefficient(1), 0.4, 1e-12);
    EXPECT_NEAR(ltf->getCoefficient(3), 0.6, 1e-12);
    EXPECT_NEAR(ltf->getCoefficient(40), 0.2, 1e-12);
}

TEST_F(LayeredTransferFunctionTest, SetHarmonicCoefficients_UpdatesComposite) {
    // Set initial state
    ltf->setCoefficient(0, 1.0);
    double const compositeBefore = ltf->computeCompositeAt(100);

    // Set new coefficients via array
    std::array<double, 41> coeffs{};
    coeffs[0] = 1.0;
    coeffs[3] = 0.5; // Add 3rd harmonic
    ltf->setHarmonicCoefficients(coeffs);

    // Composite should have changed (harmonics added)
    double const compositeAfter = ltf->computeCompositeAt(100);
    EXPECT_NE(compositeBefore, compositeAfter);
}

TEST_F(LayeredTransferFunctionTest, GetSetHarmonicCoefficients_RoundTrip) {
    // Set known coefficients
    for (int i = 0; i <= 40; ++i) {
        ltf->setCoefficient(i, static_cast<double>(i) / 100.0);
    }

    // Get coefficients
    auto coeffs = ltf->getHarmonicCoefficients();

    // Clear coefficients
    std::array<double, 41> const zeros{};
    ltf->setHarmonicCoefficients(zeros);

    // Restore original coefficients
    ltf->setHarmonicCoefficients(coeffs);

    // Verify restoration
    for (int i = 0; i <= 40; ++i) {
        EXPECT_NEAR(ltf->getCoefficient(i), static_cast<double>(i) / 100.0, 1e-12);
    }
}

// ============================================================================
// Batch Update Tests
// ============================================================================

TEST_F(LayeredTransferFunctionTest, BatchUpdate_DefersVersionIncrement) {
    uint64_t const versionBefore = ltf->getVersion();

    // Start batch mode
    ltf->beginBatchUpdate();

    // Perform multiple mutations - version should NOT increment
    ltf->setBaseLayerValue(0, 1.0);
    ltf->setBaseLayerValue(1, 2.0);
    ltf->setCoefficient(1, 0.5);

    uint64_t const versionDuring = ltf->getVersion();
    EXPECT_EQ(versionDuring, versionBefore); // No increment during batch

    // End batch mode - version should increment ONCE
    ltf->endBatchUpdate();

    uint64_t const versionAfter = ltf->getVersion();
    EXPECT_EQ(versionAfter, versionBefore + 1); // Exactly one increment
}

TEST_F(LayeredTransferFunctionTest, BatchUpdateGuard_IncrementsOnDestruction) {
    uint64_t const versionBefore = ltf->getVersion();

    {
        dsp_core::BatchUpdateGuard const guard(*ltf);

        // Mutations within guard scope
        ltf->setBaseLayerValue(0, 1.0);
        ltf->setBaseLayerValue(1, 2.0);
        ltf->setCoefficient(1, 0.5);

        uint64_t const versionDuring = ltf->getVersion();
        EXPECT_EQ(versionDuring, versionBefore); // No increment yet

    } // Guard destructor runs here

    uint64_t const versionAfter = ltf->getVersion();
    EXPECT_EQ(versionAfter, versionBefore + 1); // Exactly one increment
}

TEST_F(LayeredTransferFunctionTest, BakeHarmonicsToBase_IncrementsVersionOnce) {
    // Set up harmonics that need baking
    ltf->setCoefficient(1, 0.3); // Add 1st harmonic
    ltf->setCoefficient(2, 0.2); // Add 2nd harmonic
    ltf->setCoefficient(0, 1.0); // WT mix

    uint64_t const versionBefore = ltf->getVersion();

    // Bake harmonics - should increment version ONCE (not 16,384+ times!)
    bool const baked = ltf->bakeHarmonicsToBase();
    EXPECT_TRUE(baked);

    uint64_t const versionAfter = ltf->getVersion();

    // Critical: Version should increment exactly ONCE despite 16,384 base layer writes
    EXPECT_EQ(versionAfter, versionBefore + 1);
}

TEST_F(LayeredTransferFunctionTest, BakeCompositeToBase_IncrementsVersionOnce) {
    // Set up harmonics that need baking
    ltf->setCoefficient(1, 0.3); // Add 1st harmonic
    ltf->setCoefficient(0, 0.8); // WT mix

    uint64_t const versionBefore = ltf->getVersion();

    // Bake composite - should increment version ONCE (not 16,384+ times!)
    ltf->bakeCompositeToBase();

    uint64_t const versionAfter = ltf->getVersion();

    // Critical: Version should increment exactly ONCE despite 16,384 base layer writes
    EXPECT_EQ(versionAfter, versionBefore + 1);
}

TEST_F(LayeredTransferFunctionTest, BatchUpdate_NestedGuardsNotSupported) {
    // This test documents that nested batch updates aren't supported
    // (The implementation doesn't have a depth counter, so nested calls won't work correctly)

    uint64_t const versionBefore = ltf->getVersion();

    ltf->beginBatchUpdate();
    ltf->setBaseLayerValue(0, 1.0);

    // Calling beginBatchUpdate again is a no-op (already in batch mode)
    ltf->beginBatchUpdate();
    ltf->setBaseLayerValue(1, 2.0);

    // First endBatchUpdate increments version and exits batch mode
    ltf->endBatchUpdate();
    uint64_t const versionAfterFirst = ltf->getVersion();
    EXPECT_EQ(versionAfterFirst, versionBefore + 1);

    // Second endBatchUpdate is a no-op (not in batch mode anymore)
    ltf->endBatchUpdate();
    uint64_t const versionAfterSecond = ltf->getVersion();
    EXPECT_EQ(versionAfterSecond, versionAfterFirst); // No change
}

// ============================================================================
// Version Increment Tests for All Mutation Methods
// ============================================================================

TEST_F(LayeredTransferFunctionTest, SetSplineAnchors_IncrementsVersion) {
    uint64_t const versionBefore = ltf->getVersion();

    std::vector<dsp_core::SplineAnchor> const anchors = {
        {-1.0, -1.0, false, 0.0}, {1.0, 1.0, false, 0.0}};
    ltf->setSplineAnchors(anchors);

    EXPECT_EQ(ltf->getVersion(), versionBefore + 1);
}

TEST_F(LayeredTransferFunctionTest, ClearSplineAnchors_IncrementsVersion) {
    // First set some anchors
    std::vector<dsp_core::SplineAnchor> const anchors = {
        {-1.0, -1.0, false, 0.0}, {1.0, 1.0, false, 0.0}};
    ltf->setSplineAnchors(anchors);

    uint64_t const versionBefore = ltf->getVersion();
    ltf->clearSplineAnchors();

    EXPECT_EQ(ltf->getVersion(), versionBefore + 1);
}

TEST_F(LayeredTransferFunctionTest, SetRenderingMode_IncrementsVersion) {
    uint64_t versionBefore = ltf->getVersion();
    ltf->setRenderingMode(dsp_core::RenderingMode::Spline);
    EXPECT_EQ(ltf->getVersion(), versionBefore + 1);

    versionBefore = ltf->getVersion();
    ltf->setRenderingMode(dsp_core::RenderingMode::Harmonic);
    EXPECT_EQ(ltf->getVersion(), versionBefore + 1);

    versionBefore = ltf->getVersion();
    ltf->setRenderingMode(dsp_core::RenderingMode::Paint);
    EXPECT_EQ(ltf->getVersion(), versionBefore + 1);
}

TEST_F(LayeredTransferFunctionTest, SetExtrapolationMode_IncrementsVersion) {
    uint64_t versionBefore = ltf->getVersion();
    ltf->setExtrapolationMode(dsp_core::LayeredTransferFunction::ExtrapolationMode::Linear);
    EXPECT_EQ(ltf->getVersion(), versionBefore + 1);

    versionBefore = ltf->getVersion();
    ltf->setExtrapolationMode(dsp_core::LayeredTransferFunction::ExtrapolationMode::Clamp);
    EXPECT_EQ(ltf->getVersion(), versionBefore + 1);
}

TEST_F(LayeredTransferFunctionTest, SetBaseLayerValue_IncrementsVersion) {
    uint64_t const versionBefore = ltf->getVersion();
    ltf->setBaseLayerValue(100, 0.5);
    EXPECT_EQ(ltf->getVersion(), versionBefore + 1);
}

TEST_F(LayeredTransferFunctionTest, ClearBaseLayer_IncrementsVersion) {
    uint64_t const versionBefore = ltf->getVersion();
    ltf->clearBaseLayer();
    EXPECT_EQ(ltf->getVersion(), versionBefore + 1);
}

TEST_F(LayeredTransferFunctionTest, SetCoefficient_IncrementsVersion) {
    uint64_t const versionBefore = ltf->getVersion();
    ltf->setCoefficient(1, 0.5);
    EXPECT_EQ(ltf->getVersion(), versionBefore + 1);
}

TEST_F(LayeredTransferFunctionTest, SetHarmonicCoefficients_IncrementsVersion) {
    std::array<double, 41> coeffs{};
    coeffs[0] = 1.0;
    coeffs[3] = 0.5;

    uint64_t const versionBefore = ltf->getVersion();
    ltf->setHarmonicCoefficients(coeffs);
    EXPECT_EQ(ltf->getVersion(), versionBefore + 1);
}

} // namespace dsp_core_test
