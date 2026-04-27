#include <dsp_core/dsp_core.h>
#include <gtest/gtest.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <map>

namespace dsp_core_test {

/**
 * Anchor Clustering Analysis Test
 *
 * Purpose: Validate hypothesis about why anchors cluster near boundaries for symmetric functions.
 *
 * Hypothesis: For symmetric function H3, if adding an anchor at x=-0.970 results in max error
 * staying on the left side (x=-0.961) instead of switching to the symmetric right side (x=+0.970),
 * this indicates:
 * 1. Poor "radius of influence" - anchors only help tiny regions
 * 2. Tangent computation creates artifacts near boundaries
 * 3. Cubic Hermite interpolation quality issues near boundaries
 */
class AnchorClusteringAnalysisTest : public ::testing::Test {
  protected:
    // Helper: Sample struct for internal use
    struct Sample {
        double x, y;
    };

    void SetUp() override {
        ltf = std::make_unique<dsp_core::LayeredTransferFunction>(16384, -1.0, 1.0);
    }

    void setH3Curve() {
        for (int i = 0; i < ltf->getTableSize(); ++i) {
            double x = ltf->normalizeIndex(i);
            x = std::clamp(x, -1.0, 1.0);
            double const y = std::sin(3.0 * std::asin(x)); // H3
            ltf->setBaseLayerValue(i, y);
        }
    }

    void setH5Curve() {
        for (int i = 0; i < ltf->getTableSize(); ++i) {
            double x = ltf->normalizeIndex(i);
            x = std::clamp(x, -1.0, 1.0);
            double const y = std::sin(5.0 * std::asin(x)); // H5
            ltf->setBaseLayerValue(i, y);
        }
    }

    // Structure to record each anchor addition
    struct AnchorAdditionRecord {
        int iteration = 0;
        dsp_core::SplineAnchor addedAnchor;
        double maxErrorBefore = 0.0;
        double maxErrorAfter = 0.0;
        double maxErrorLocationBefore = 0.0; // x coordinate
        double maxErrorLocationAfter = 0.0;
        double errorReductionPercent = 0.0;
        bool errorSwitchedSides = false;               // For symmetric functions
        std::map<std::string, double> regionMaxErrors; // By region after addition
    };

    // Instrumented greedy fitting - simplified version that logs each step
    std::vector<AnchorAdditionRecord> performInstrumentedGreedyFit(const dsp_core::SplineFitConfig& config) {
        std::vector<AnchorAdditionRecord> records;

        // Sample the curve (same as SplineFitter does)
        const int tableSize = ltf->getTableSize();
        std::vector<Sample> samples;
        samples.reserve(static_cast<size_t>(tableSize) * 2);

        // Sample entire curve
        for (int i = 0; i < tableSize; ++i) {
            double const x = ltf->normalizeIndex(i);
            double const y = ltf->evaluateBaseAndHarmonics(x);
            samples.push_back({x, y});
        }

        // Densify with midpoints
        std::vector<Sample> densified;
        densified.reserve(samples.size() * 2);
        for (size_t i = 0; i < samples.size(); ++i) {
            densified.push_back(samples[i]);
            if (i < samples.size() - 1) {
                double const midX = (samples[i].x + samples[i + 1].x) / 2.0;
                double const midY = ltf->evaluateBaseAndHarmonics(midX);
                densified.push_back({midX, midY});
            }
        }
        samples = std::move(densified);

        // Initialize with endpoints
        std::vector<dsp_core::SplineAnchor> anchors = {{samples.front().x, samples.front().y, false, 0.0},
                                                       {samples.back().x, samples.back().y, false, 0.0}};

        // Greedy iterations
        for (int iter = 0; iter < config.maxAnchors - 2; ++iter) {
            // Compute tangents
            dsp_core::Services::SplineFitter::computeTangents(anchors, config);

            // Find worst error BEFORE adding anchor
            size_t worstIdx = 0;
            double maxErrorBefore = 0.0;
            for (size_t i = 0; i < samples.size(); ++i) {
                double const splineY = dsp_core::Services::SplineEvaluator::evaluate(anchors, samples[i].x);
                double const error = std::abs(samples[i].y - splineY);
                if (error > maxErrorBefore) {
                    maxErrorBefore = error;
                    worstIdx = i;
                }
            }

            double const worstX = samples[worstIdx].x;

            // Check if converged (simplified - no adaptive tolerance for analysis)
            if (maxErrorBefore < config.positionTolerance) {
                break;
            }

            // Add anchor at worst location
            auto insertPos = std::lower_bound(anchors.begin(), anchors.end(), worstX,
                                              [](const dsp_core::SplineAnchor& a, double x) { return a.x < x; });

            dsp_core::SplineAnchor const newAnchor = {samples[worstIdx].x, samples[worstIdx].y, false, 0.0};

            anchors.insert(insertPos, newAnchor);

            // Recompute tangents with new anchor
            dsp_core::Services::SplineFitter::computeTangents(anchors, config);

            // Find worst error AFTER adding anchor
            size_t worstIdxAfter = 0;
            double maxErrorAfter = 0.0;
            for (size_t i = 0; i < samples.size(); ++i) {
                double const splineY = dsp_core::Services::SplineEvaluator::evaluate(anchors, samples[i].x);
                double const error = std::abs(samples[i].y - splineY);
                if (error > maxErrorAfter) {
                    maxErrorAfter = error;
                    worstIdxAfter = i;
                }
            }

            double const worstXAfter = samples[worstIdxAfter].x;

            // Record this addition
            AnchorAdditionRecord record;
            record.iteration = iter;
            record.addedAnchor = newAnchor;
            record.maxErrorBefore = maxErrorBefore;
            record.maxErrorAfter = maxErrorAfter;
            record.maxErrorLocationBefore = worstX;
            record.maxErrorLocationAfter = worstXAfter;
            record.errorReductionPercent = 100.0 * (maxErrorBefore - maxErrorAfter) / maxErrorBefore;

            // Check if error switched sides (for symmetric functions)
            bool const wasBeforeOnLeft = (worstX < 0);
            bool const isAfterOnRight = (worstXAfter > 0);
            record.errorSwitchedSides = (wasBeforeOnLeft && isAfterOnRight);

            // Compute regional errors
            record.regionMaxErrors = computeRegionalErrors(samples, anchors);

            records.push_back(record);
        }

        return records;
    }

    static std::map<std::string, double> computeRegionalErrors(const std::vector<Sample>& samples,
                                                               const std::vector<dsp_core::SplineAnchor>& anchors) {
        std::map<std::string, double> regionErrors;

        std::map<std::string, std::pair<double, double>> const regions = {{"x<-0.9", {-1.0, -0.9}},
                                                                          {"[-0.9,-0.3]", {-0.9, -0.3}},
                                                                          {"[-0.3,0.3]", {-0.3, 0.3}},
                                                                          {"[0.3,0.9]", {0.3, 0.9}},
                                                                          {"x>0.9", {0.9, 1.0}}};

        for (const auto& [name, range] : regions) {
            double maxErr = 0.0;
            for (const auto& s : samples) {
                if (s.x >= range.first && s.x <= range.second) {
                    double const splineY = dsp_core::Services::SplineEvaluator::evaluate(anchors, s.x);
                    double const err = std::abs(s.y - splineY);
                    maxErr = std::max(maxErr, err);
                }
            }
            regionErrors[name] = maxErr;
        }

        return regionErrors;
    }

    std::unique_ptr<dsp_core::LayeredTransferFunction> ltf;
};

/**
 * Test 1: Step-by-Step Greedy Fitting Analysis
 *
 * This test records every anchor addition and validates the user's hypothesis:
 * For symmetric H3, after adding anchor at x=-0.970, does max error switch to
 * the symmetric location x=+0.970, or does it stay on the left side?
 */
TEST_F(AnchorClusteringAnalysisTest, H3_GreedyFittingStepByStep) {
    setH3Curve();

    auto config = dsp_core::SplineFitConfig::tight();
    config.enableFeatureDetection = false; // Pure greedy for analysis

    std::cout << "\n========================================" << '\n';
    std::cout << "H3 GREEDY FITTING STEP-BY-STEP ANALYSIS" << '\n';
    std::cout << "========================================\n" << '\n';

    auto records = performInstrumentedGreedyFit(config);

    std::cout << "Total iterations: " << records.size() << "\n" << '\n';

    // Print first 10 iterations
    int const printCount = std::min(10, static_cast<int>(records.size()));

    for (int i = 0; i < printCount; ++i) {
        const auto& rec = records[i];

        std::cout << "--- Iteration " << rec.iteration << " ---" << '\n';
        std::cout << "  Max error BEFORE: " << std::fixed << std::setprecision(6) << rec.maxErrorBefore
                  << " at x=" << rec.maxErrorLocationBefore << '\n';
        std::cout << "  Added anchor: x=" << rec.addedAnchor.x << ", y=" << rec.addedAnchor.y << '\n';
        std::cout << "  Max error AFTER:  " << rec.maxErrorAfter << " at x=" << rec.maxErrorLocationAfter << '\n';
        std::cout << "  Error reduction: " << std::setprecision(1) << rec.errorReductionPercent << "%" << '\n';

        // CRITICAL CHECK: For symmetric function, did error switch sides?
        bool const anchorOnLeft = (rec.addedAnchor.x < 0);
        bool const errorStillOnLeft = (rec.maxErrorLocationAfter < 0);

        if (anchorOnLeft && errorStillOnLeft) {
            std::cout << "  ⚠️  SYMMETRY VIOLATION: Added anchor on LEFT, max error STILL on LEFT" << '\n';
            std::cout << "      Expected: Max error should switch to RIGHT side (x=" << -rec.addedAnchor.x << ")"
                      << '\n';
            std::cout << "      Actual: Max error at x=" << rec.maxErrorLocationAfter << '\n';
            std::cout << "      → Anchor has POOR radius of influence OR tangent artifact" << '\n';
        } else if (rec.errorSwitchedSides) {
            std::cout << "  ✓ SYMMETRY PRESERVED: Error switched from LEFT to RIGHT" << '\n';
        }

        // Print regional errors
        std::cout << "  Regional errors:" << '\n';
        for (const auto& [region, error] : rec.regionMaxErrors) {
            std::cout << "    " << std::setw(15) << std::left << region << ": " << std::fixed << std::setprecision(6)
                      << error << '\n';
        }
        std::cout << '\n';
    }

    // Summary analysis
    int symmetryViolations = 0;
    for (const auto& rec : records) {
        bool const anchorOnLeft = (rec.addedAnchor.x < -0.5);
        bool const errorStillOnLeft = (rec.maxErrorLocationAfter < -0.5);
        if (anchorOnLeft && errorStillOnLeft) {
            symmetryViolations++;
        }
    }

    std::cout << "\n=== SUMMARY ===" << '\n';
    std::cout << "Total anchor additions: " << records.size() << '\n';
    std::cout << "Symmetry violations: " << symmetryViolations << '\n';
    std::cout << "Percentage: "
              << (100.0 * static_cast<double>(symmetryViolations) / static_cast<double>(records.size())) << "%" << '\n';

    if (symmetryViolations > records.size() / 4) {
        std::cout << "\n⚠️  ROOT CAUSE IDENTIFIED: POOR ANCHOR INFLUENCE RADIUS" << '\n';
        std::cout << "Anchors near boundaries do NOT effectively reduce error" << '\n';
        std::cout << "in their local region, causing the greedy algorithm to add" << '\n';
        std::cout << "more anchors nearby instead of switching to the symmetric side." << '\n';
    }
}

} // namespace dsp_core_test
