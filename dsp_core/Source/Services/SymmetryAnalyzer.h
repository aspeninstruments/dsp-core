#pragma once
#include <vector>

namespace dsp_core {

// Forward declaration
class LayeredTransferFunction;

namespace Services {

/**
 * SymmetryAnalyzer - Detects symmetry in transfer function curves
 *
 * Purpose:
 *   Analyze curves for point symmetry about origin (odd functions)
 *   and mirror symmetry about y-axis (even functions).
 *
 *   For waveshaping, we primarily care about odd symmetry:
 *   f(-x) = -f(x) → Examples: tanh, x³, sin(x)
 *
 * Algorithm:
 *   1. Sample curve at complementary points (x, -x)
 *   2. Compute symmetry score: correlation(f(x), -f(-x))
 *   3. Classify: perfect (>0.99), approximate (>0.90), asymmetric
 *
 * Service Pattern: Pure static methods (no state)
 */
class SymmetryAnalyzer {
  public:
    /**
     * Configuration for symmetry analysis
     */
    struct Config {
        /**
         * Perfect symmetry threshold (0.99 = 99% correlation)
         * Above this: treat as perfectly symmetric
         */
        double perfectThreshold = 0.99;

        /**
         * Approximate symmetry threshold (0.90 = 90% correlation)
         * Above this: treat as "roughly symmetric"
         */
        double approximateThreshold = 0.90;

        /**
         * Sample count for correlation analysis (recommend: 64-256)
         * Higher = more accurate, slower
         */
        int sampleCount = 128;
    };

    /**
     * Result of symmetry analysis
     */
    struct Result {
        /**
         * Symmetry score (0.0-1.0)
         * 1.0 = perfectly symmetric
         * 0.0 = completely asymmetric
         * Negative = anti-symmetric (rare)
         */
        double score = 0.0;

        /**
         * Classification based on thresholds
         */
        enum class Classification {
            Perfect,     // score >= perfectThreshold
            Approximate, // score >= approximateThreshold
            Asymmetric   // score < approximateThreshold
        };
        Classification classification = Classification::Asymmetric;

        /**
         * Center of symmetry (x coordinate)
         * For odd functions: typically 0.0
         */
        double centerX = 0.0;

        /**
         * Is curve symmetric enough for paired anchor placement?
         */
        bool shouldUsePairedAnchors() const {
            return classification != Classification::Asymmetric;
        }
    };

    /**
     * Analyze curve symmetry
     *
     * @param ltf Transfer function to analyze
     * @param config Analysis configuration
     * @return Symmetry analysis result
     */
    static Result analyzeOddSymmetry(const LayeredTransferFunction& ltf, const Config& config);

    /**
     * Analyze curve symmetry with default configuration
     *
     * @param ltf Transfer function to analyze
     * @return Symmetry analysis result
     */
    static Result analyzeOddSymmetry(const LayeredTransferFunction& ltf);

    /**
     * Analyze odd symmetry directly from lane curve data
     *
     * @param curveData Raw curve samples (e.g., Lane::curveData)
     * @param config Analysis configuration
     * @return Symmetry analysis result
     */
    static Result analyzeOddSymmetry(const std::vector<double>& curveData, const Config& config);

    /**
     * Analyze odd symmetry directly from lane curve data with default configuration
     *
     * @param curveData Raw curve samples (e.g., Lane::curveData)
     * @return Symmetry analysis result
     */
    static Result analyzeOddSymmetry(const std::vector<double>& curveData);

  private:
    SymmetryAnalyzer() = delete; // Pure static service

    /**
     * Compute Pearson correlation between f(x) and -f(-x)
     */
    static double computeSymmetryScore(const std::vector<double>& fPositive, const std::vector<double>& fNegative);
};

} // namespace Services
} // namespace dsp_core
