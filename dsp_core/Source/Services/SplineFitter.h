#pragma once
#include "../SplineTypes.h"
#include <vector>

namespace dsp_core::Services {

/**
 * SplineFitter - Pure service for spline curve fitting
 *
 * Converts curve data (raw array) to control points + cubic Hermite spline
 * using feature-based initialization and greedy error-driven refinement.
 *
 * Algorithm (5 stages):
 *   1. Feature Detection: identify extrema and inflection points
 *   2. Sample & Sanitize: curve data → clean polyline
 *   3. Greedy Fitting: feature anchors + error-driven refinement
 *   4. Tangent Computation: compute Fritsch-Carlson monotone tangents
 *   5. Error Analysis: compute fit quality metrics
 *
 * Service Pattern (5/5 score):
 *   - Pure static methods (no state)
 *   - Unit testable in isolation
 *   - Reusable across modules
 */
class SplineFitter {
  public:
    /**
     * Fit a spline to raw curve data.
     *
     * @param curveData  Array of curve samples (e.g., Lane::curveData)
     * @param tableSize  Number of samples in the array
     * @param minValue   Minimum x value (typically -1.0)
     * @param maxValue   Maximum x value (typically 1.0)
     * @param config     Fitting configuration
     * @return Fit result with anchors and error metrics
     */
    static SplineFitResult fitCurve(const double* curveData, int tableSize, double minValue, double maxValue,
                                    const SplineFitConfig& config = SplineFitConfig::tight());

    // Tangent computation (exposed for manual anchor manipulation)
    // Recomputes tangents for anchors after position changes using configured algorithm
    static void computeTangents(std::vector<SplineAnchor>& anchors,
                                const SplineFitConfig& config = SplineFitConfig::tight());

  private:
    // Step 1: Sample & sanitize
    struct Sample {
        double x, y;
    };
    static std::vector<Sample> sampleAndSanitize(const double* curveData, int tableSize, double minValue,
                                                 double maxValue, const SplineFitConfig& config);

    // Sub-steps of sanitize
    static void sortByX(std::vector<Sample>& samples);
    static void deduplicateNearVerticals(std::vector<Sample>& samples);
    static void enforceMonotonicity(std::vector<Sample>& samples);
    static void clampToRange(std::vector<Sample>& samples);

    // Tangent computation algorithms
    static void computePCHIPTangentsImpl(std::vector<SplineAnchor>& anchors, const SplineFitConfig& config);

    static void computeFritschCarlsonTangents(std::vector<SplineAnchor>& anchors, const SplineFitConfig& config);

    static void computeAkimaTangents(std::vector<SplineAnchor>& anchors, const SplineFitConfig& config);

    static void computeFiniteDifferenceTangents(std::vector<SplineAnchor>& anchors, const SplineFitConfig& config);

    // Greedy spline fitting (replaces RDP + refinement)
    // Now uses feature-based anchor placement: starts with mandatory feature anchors
    static std::vector<SplineAnchor> greedySplineFit(const std::vector<Sample>& samples, const SplineFitConfig& config,
                                                     const double* curveData = nullptr, int tableSize = 0,
                                                     double minValue = -1.0, double maxValue = 1.0,
                                                     const std::vector<int>& mandatoryAnchorIndices = {});

    // Find sample with worst fit error
    struct WorstFitResult {
        size_t sampleIndex;
        double maxError;
    };

    static WorstFitResult findWorstFitSample(const std::vector<Sample>& samples,
                                             const std::vector<SplineAnchor>& anchors);

    // Helper methods for greedySplineFit to reduce cognitive complexity
    static std::vector<SplineAnchor> initializeAnchorsFromIndices(const std::vector<Sample>& samples, int tableSize,
                                                                  double minValue, double maxValue,
                                                                  const std::vector<int>& mandatoryIndices);

    // Insert anchor pair in symmetric mode. Returns number of anchors added (1 or 2)
    static int insertAnchorSymmetric(std::vector<SplineAnchor>& anchors, const std::vector<Sample>& samples,
                                     size_t worstSampleIndex, double adaptiveTolerance);

    // Insert single anchor in asymmetric mode. Returns true if anchor was inserted
    static bool insertAnchorAsymmetric(std::vector<SplineAnchor>& anchors, const Sample& worstSample);

    // Coordinate mapping helper
    static double normalizeIndex(int index, int tableSize, double minValue, double maxValue);

    // Catmull-Rom interpolation for reading curve data at arbitrary x positions
    static double interpolateCurve(const double* curveData, int tableSize, double minValue, double maxValue, double x);

    SplineFitter() = delete; // Pure static utility
};

} // namespace dsp_core::Services
