#pragma once
#include <vector>
#include <juce_core/juce_core.h>
#include "Services/CurveFeatureDetector.h"
#include "MorphGesture.h"

namespace dsp_core {

// Tangent computation algorithms for spline fitting
enum class TangentAlgorithm {
    PCHIP,           // Piecewise Cubic Hermite Interpolating Polynomial (default)
    FritschCarlson,  // Monotone-preserving variant
    Akima,           // Local weighted average
    FiniteDifference // Simple baseline (for comparison)
};

// Symmetry detection for anchor placement
enum class SymmetryDetection {
    Auto, // Detect symmetry, enable paired anchors if score >= threshold (default)
    Never // Disable symmetric fitting (original greedy algorithm)
};

// Control point with optional tangent override
struct SplineAnchor {
    double x = 0.0; // Currently displayed position in [-1, 1]
    double y = 0.0; // Currently displayed value in [-1, 1]
    bool hasCustomTangent = false;
    double tangent = 0.0; // m_i in PCHIP notation

    // User-recorded morph gesture (deltas from this anchor's home position).
    // Empty (size < 2) means "no gesture". Carried inline so the gesture
    // survives anchor reordering, snapshot/restore, and serialization without
    // a separate index-keyed map that could go stale.
    AnchorMorphGesture morphGesture;

    // Home position - the anchor's position when the morph knob is at sample(0).
    // Meaningful only when morphGesture is non-empty: x = clamp(homeX + gesture.sampleAt(t).dx, [-1,1]).
    // When morphGesture is empty, these fields are unused (and stay at their default).
    // Set when the user records a gesture and updated when the user drags an anchor that has one.
    double homeX = 0.0;
    double homeY = 0.0;

    bool operator==(const SplineAnchor&) const = default;
};

// Result of curve fitting operation
struct SplineFitResult {
    bool success = false;
    std::vector<SplineAnchor> anchors;
    double maxError = 0.0; // Peak absolute error |y - ŷ|
    int numAnchors = 0;
    juce::String message; // User-facing feedback

    // Statistics for user feedback
    double averageError = 0.0;
    double maxDerivativeError = 0.0;
};

/**
 * Configuration for spline curve fitting algorithm.
 *
 * The fitting algorithm uses adaptive tolerance to prevent "anchor creeping" - the problem
 * where refitting curves produces exponentially more anchors on each cycle.
 *
 * Key behavioral characteristics:
 * - Adaptive tolerance increases with anchor density to prevent over-fitting
 * - Backtranslation stable: refit curves converge to minimal anchor count
 * - Anchor counts scale with geometric complexity (matches Chebyshev extrema)
 * - Typical results: H1=2, H2=2, H3=3, H5=5, H10=10 anchors
 *
 * See docs/architecture/spline-curve-fitting.md for detailed algorithm documentation.
 */
struct SplineFitConfig {
    /**
     * Position tolerance - baseline error threshold for curve fitting.
     *
     * This serves as both:
     * 1. Minimum floor for adaptive tolerance (early iterations use this value)
     * 2. Basis for computing adaptive tolerance: baselineTolerance = positionTolerance / verticalRange
     *
     * The adaptive tolerance formula:
     *   tolerance = baselineTolerance × (1 + anchorRatio × 2.0)
     *   where anchorRatio = currentAnchors / maxAnchors
     *
     * Effect on fitting:
     * - Lower values (0.001-0.002): Tighter fit, more anchors, higher quality
     * - Higher values (0.01-0.02): Looser fit, fewer anchors, smoother curves
     *
     * Range: 0.001 (tight) to 0.02 (smooth)
     * Recommended: 0.01 for general use, 0.002 for high-precision work
     */
    double positionTolerance = 0.01;

    /**
     * Maximum anchor budget - hard limit on anchor count.
     *
     * Interacts with adaptive tolerance:
     * - As anchor count approaches maxAnchors, tolerance increases to prevent over-fitting
     * - Prevents pathological cases where algorithm adds unlimited anchors
     * - Should be set based on UI performance requirements (more anchors = more drag latency)
     *
     * Effect on fitting:
     * - Lower values (16-24): Faster UI, fewer control points, may under-fit complex curves
     * - Higher values (64-128): More precision, can capture very complex curves, slower UI
     *
     * Range: 16 (minimal) to 128 (high precision)
     * Recommended: 24 for interactive editing, 64 for offline processing
     */
    int maxAnchors = 128;

    /**
     * Enable greedy refinement phase.
     *
     * When true: Uses feature detection + error-driven refinement (default, recommended)
     * When false: Uses only feature detection (fast but lower quality)
     *
     * Should always be true unless benchmarking or debugging.
     */
    bool enableRefinement = true;

    /**
     * Enforce monotonicity - prevent y-backtracking.
     *
     * When true: Transfer function is guaranteed to be monotonically increasing
     * When false: Allows folding distortion effects (y can decrease as x increases)
     *
     * For audio waveshaping:
     * - True: Standard waveshaping (no phase inversion artifacts)
     * - False: Creative effects (folding distortion, wavefolding)
     *
     * Recommended: true for production, false for experimental effects
     */
    bool enforceMonotonicity = true;

    /**
     * Slope bounds for anti-aliasing.
     *
     * Clamps tangent values to prevent extreme slopes that could cause aliasing
     * in audio processing. Range of [-8, 8] allows steep curves while preventing
     * numerical instability.
     *
     * Formula: tangent = clamp(computedTangent, minSlope, maxSlope)
     *
     * Recommended: Leave at defaults unless experiencing aliasing artifacts
     */
    double minSlope = -8.0; // Minimum tangent slope
    double maxSlope = 8.0;  // Maximum tangent slope

    /**
     * Pin endpoints to (-1, -1) and (1, 1).
     *
     * When true: First and last anchors are always at domain boundaries
     * When false: Endpoints can be placed anywhere
     *
     * Recommended: true (ensures full domain coverage)
     */
    bool pinEndpoints = true;

    /**
     * Tangent computation algorithm.
     *
     * Available algorithms:
     * - FritschCarlson: Monotone-preserving with no-overshoot guarantee (default)
     * - PCHIP: Smoother but may overshoot (requires iterative correction)
     * - Akima: Local weighted average (good for noisy data)
     * - FiniteDifference: Simple baseline (for testing/comparison)
     *
     * Recommendation: FritschCarlson for production (predictable UI behavior)
     * See docs/architecture/spline-algorithm-decision.md for full analysis.
     */
    TangentAlgorithm tangentAlgorithm = TangentAlgorithm::FritschCarlson;

    /**
     * Symmetry detection for anchor placement.
     *
     * Auto: Analyzes curve symmetry, enables paired anchors if score >= threshold (default)
     * Never: Disables symmetric fitting (original greedy algorithm)
     *
     * Symmetric fitting adds anchors in complementary pairs (x, -x) to
     * maintain visual symmetry for symmetric curves (tanh, x³, odd harmonics).
     *
     * Performance tested with 7-48% lower max error for tanh curves and
     * 13-41% lower max error for odd harmonics. No regressions found.
     *
     * Recommendation: Auto (default) for best quality on symmetric curves
     */
    SymmetryDetection symmetryDetection = SymmetryDetection::Auto;

    /**
     * Symmetry detection threshold (for Auto mode).
     *
     * If symmetry score >= threshold, enable paired anchor placement.
     *
     * Range: 0.0-1.0
     * Recommended: 0.90 (90% symmetric or better)
     */
    double symmetryThreshold = 0.90;

    /**
     * Enable feature-based anchor placement.
     *
     * ⚠️ EXPERIMENTAL FEATURE - Disabled by default due to backtranslation regressions
     *
     * When true: CurveFeatureDetector places mandatory anchors at geometric features
     *            (local extrema, inflection points) before greedy refinement
     * When false: Pure greedy algorithm without mandatory feature anchors (default)
     *
     * Known Issues:
     * - Breaks backtranslation stability (3 anchors → 7 anchors on simple parabola)
     * - Forces anchors at detected features even when not needed for error tolerance
     * - Can over-constrain the greedy algorithm
     *
     * Recommendation: false (default) - preserves backtranslation stability
     * Only enable for experimentation or if you need guaranteed feature preservation
     */
    bool enableFeatureDetection = true;

    /**
     * Feature detection configuration (only used if enableFeatureDetection = true)
     *
     * Controls how geometric features (extrema, inflection points) are detected
     * and prioritized for mandatory anchor placement.
     *
     * Key parameters:
     * - significanceThreshold: Noise filtering (0.001 = 0.1% of vertical range)
     * - maxFeatures: Hard limit on detected features (100 default)
     * - derivativeThreshold: First derivative noise floor (1e-6 default)
     * - secondDerivativeThreshold: Second derivative noise floor (1e-4 default)
     * - extremaInflectionRatio: Budget ratio for extrema vs inflections (0.8 default)
     *
     * See docs/feature-plans/feature-detection-parameter-tuning.md for tuning guidance.
     */
    FeatureDetectionConfig featureConfig;

    // Presets
    static SplineFitConfig tight() {
        SplineFitConfig cfg;
        cfg.positionTolerance = 0.005; // Relaxed from 0.002 for backtranslation stability
        cfg.maxAnchors = 128;          // Optimal for perfect stability (3→3, 4→4) + exceptional quality (0.01% error)
        cfg.tangentAlgorithm = TangentAlgorithm::FritschCarlson;
        return cfg;
    }
};

} // namespace dsp_core
