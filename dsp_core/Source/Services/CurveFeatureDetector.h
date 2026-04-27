#pragma once
#include <vector>

namespace dsp_core {

/**
 * Configuration for feature detection and significance filtering
 */
struct FeatureDetectionConfig {
    /**
     * Significance threshold as percentage of vertical range (0.0-1.0)
     * Extrema with LOCAL PROMINENCE less than this threshold are discarded
     * Prominence = how much an extremum stands out from nearby points (~0.6% domain window)
     * - Peak: prominence = y_peak - max(neighbors)
     * - Valley: prominence = min(neighbors) - y_valley
     * This correctly filters noise while preserving extrema at ANY y-value (e.g., peak at y=0)
     * Default: 0.001 (0.1% of vertical range)
     */
    double significanceThreshold;

    /**
     * Maximum number of features to detect (0 = unlimited)
     * If limited, keeps most significant features by amplitude/curvature
     */
    int maxFeatures;

    /**
     * First derivative noise floor - ignore extrema with |dy/dx| < threshold
     * Lower = more sensitive (detects subtle peaks)
     * Higher = more conservative (ignores noise)
     * Range: 1e-7 to 1e-5, Default: 1e-6
     */
    double derivativeThreshold;

    /**
     * Enable significance filtering based on local prominence
     * When enabled, filters extrema with prominence < significanceThreshold
     * When disabled, accepts all extrema with valid derivative sign changes
     * Default: false (disabled) - accepts all detected extrema
     *
     * Note: Significance filtering is experimental and may incorrectly filter
     * valid features. Disable if you observe missing extrema or peaks/valleys.
     */
    bool enableSignificanceFiltering;

    /**
     * Default constructor - uses sensible defaults
     * Note: Very low significance threshold (0.1%) to preserve all real features
     * Increase threshold (e.g., 2-5%) for noise filtering
     */
    FeatureDetectionConfig()
        : significanceThreshold(0.001), maxFeatures(100), derivativeThreshold(1e-06),
          enableSignificanceFiltering(false) {}

    bool operator==(const FeatureDetectionConfig&) const = default;
};

namespace Services {

/**
 * CurveFeatureDetector - Pure service for geometric feature detection in curves
 *
 * Detects critical geometric features (local extrema, inflection points) in
 * transfer function curves that require anchors for accurate spline fitting.
 *
 * Purpose:
 *   Eliminate ripple/oscillation artifacts by ensuring spline anchors are placed
 *   at all geometric features. A cubic segment can represent at most 1 inflection
 *   point and 2 local extrema. If data has more features between anchors, we're
 *   undersampling (like violating Nyquist frequency), which causes ripple.
 *
 * Algorithm:
 *   1. Detect local extrema (dy/dx sign changes) - peaks and valleys
 *   2. Detect inflection points (d²y/dx² sign changes) - curvature reversals
 *   3. Merge features into mandatory anchor list (always include endpoints)
 *
 * Service Pattern (5/5 score):
 *   - Pure static methods (no state)
 *   - Unit testable in isolation
 *   - Reusable across modules
 */
class CurveFeatureDetector {
  public:
    struct FeatureResult {
        std::vector<int> localExtrema;     // peaks and valleys (dy/dx sign changes)
        std::vector<int> inflectionPoints; // d²y/dx² sign changes
        std::vector<int> mandatoryAnchors; // merged + sorted (always include these)
    };

    /**
     * Detect all geometric features requiring spline anchors
     *
     * @param curveData Raw curve samples
     * @param tableSize Number of samples
     * @param minValue Minimum x value (typically -1.0)
     * @param maxValue Maximum x value (typically 1.0)
     * @param config Configuration for detection and filtering (optional)
     * @return Feature indices (table indices, not normalized coordinates)
     */
    static FeatureResult detectFeatures(const double* curveData, int tableSize, double minValue, double maxValue,
                                        const FeatureDetectionConfig& config = FeatureDetectionConfig{});

  private:
    CurveFeatureDetector() = delete; // Pure static service

    /**
     * Compute normalized x position from table index
     */
    static double normalizeIndex(int index, int tableSize, double minValue, double maxValue);

    /**
     * Estimate first derivative at index using central difference
     */
    static double estimateDerivative(const double* curveData, int tableSize, double minValue, double maxValue, int idx);

    // Internal feature representation for prioritization
    struct Feature {
        int index;
        double significance;
    };

    // Helper methods to reduce cognitive complexity
    static void detectLocalExtrema(const double* curveData, int tableSize, double minValue, double maxValue,
                                   const FeatureDetectionConfig& config, double amplitudeThreshold,
                                   double verticalCenter, FeatureResult& result, std::vector<Feature>& features);

    static void prioritizeFeatures(const FeatureDetectionConfig& config, int tableSize, FeatureResult& result,
                                   std::vector<Feature>& features);
};

} // namespace Services
} // namespace dsp_core
