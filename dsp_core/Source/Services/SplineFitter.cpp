#include "SplineFitter.h"
#include "SplineEvaluator.h"
#include "CurveFeatureDetector.h"
#include "AdaptiveToleranceCalculator.h"
#include "SymmetryAnalyzer.h"
#include <algorithm>
#include <cmath>

namespace dsp_core::Services {

double SplineFitter::normalizeIndex(int index, int tableSize, double minValue, double maxValue) {
    if (tableSize <= 1) return minValue;
    return minValue + (maxValue - minValue) * static_cast<double>(index) / static_cast<double>(tableSize - 1);
}

double SplineFitter::interpolateCurve(const double* curveData, int tableSize, double minValue, double maxValue, double x) {
    // Catmull-Rom interpolation matching LTF's applyTransferFunction quality
    const double indexF = (x - minValue) / (maxValue - minValue) * static_cast<double>(tableSize - 1);
    const int lastIdx = tableSize - 1;
    const int i0 = std::clamp(static_cast<int>(std::floor(indexF)), 0, lastIdx);
    const double frac = indexF - i0;

    const int im1 = std::max(i0 - 1, 0);
    const int i1 = std::min(i0 + 1, lastIdx);
    const int i2 = std::min(i0 + 2, lastIdx);

    const double p0 = curveData[im1];
    const double p1 = curveData[i0];
    const double p2 = curveData[i1];
    const double p3 = curveData[i2];

    const double t = frac;
    const double t2 = t * t;
    const double t3 = t2 * t;
    return 0.5 * ((2.0 * p1) +
                   (-p0 + p2) * t +
                   (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
                   (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3);
}

SplineFitResult SplineFitter::fitCurve(const double* curveData, int tableSize,
                                        double minValue, double maxValue,
                                        const SplineFitConfig& config) {

    SplineFitResult result;

    // Step 0: Feature-based anchor placement (optional)
    // Detect and anchor at geometric features (structural correctness)
    // Limit mandatory features to 70% of maxAnchors, reserving 30% for error-driven refinement
    std::vector<int> mandatoryIndices;
    if (config.enableFeatureDetection) {
        // Use feature detection config from SplineFitConfig, but cap maxFeatures
        // to 70% of anchor budget to reserve 30% for error-driven refinement
        auto featureConfig = config.featureConfig;
        int maxFeaturesFromBudget = static_cast<int>(config.maxAnchors * 0.7);
        if (featureConfig.maxFeatures == 0 || featureConfig.maxFeatures > maxFeaturesFromBudget) {
            featureConfig.maxFeatures = maxFeaturesFromBudget;
        }
        auto features = CurveFeatureDetector::detectFeatures(curveData, tableSize, minValue, maxValue, featureConfig);
        mandatoryIndices = features.mandatoryAnchors;
    }

    // Step 1: Sample & sanitize
    auto samples = sampleAndSanitize(curveData, tableSize, minValue, maxValue, config);

    if (samples.empty()) {
        result.success = false;
        result.message = "No samples to fit";
        return result;
    }

    if (samples.size() < 2) {
        result.success = false;
        result.message = "Need at least 2 samples";
        return result;
    }

    // Step 2: Greedy spline fitting with feature-based initialization
    // Start with mandatory feature anchors, then iteratively refine (error-driven)
    auto anchors = greedySplineFit(samples, config, curveData, tableSize, minValue, maxValue, mandatoryIndices);

    if (anchors.empty()) {
        result.success = false;
        result.message = "Greedy fit failed to produce anchors";
        return result;
    }

    // Note: No need for endpoint truncation fix - greedy algorithm
    // naturally respects maxAnchors limit during iteration

    // Step 3: Error analysis
    // Compute error statistics by comparing original samples to fitted spline
    double maxErr = 0.0;
    double sumErr = 0.0;
    for (const auto& s : samples) {
        double fitted = SplineEvaluator::evaluate(anchors, s.x);
        double err = std::abs(s.y - fitted);
        maxErr = std::max(maxErr, err);
        sumErr += err;
    }

    result.success = true;
    result.anchors = std::move(anchors);
    result.numAnchors = static_cast<int>(result.anchors.size());
    result.maxError = maxErr;
    result.averageError = sumErr / static_cast<double>(samples.size());
    result.message = "Fitted " + juce::String(result.numAnchors) + " anchors, max error: " + juce::String(maxErr, 4);

    return result;
}

std::vector<SplineFitter::Sample> SplineFitter::sampleAndSanitize(const double* curveData, int tableSize,
                                                                  double minValue, double maxValue,
                                                                  const SplineFitConfig& config) {

    std::vector<Sample> samples;
    samples.reserve(tableSize * 2); // Reserve extra space for densification

    // Raster-to-polyline: sample entire curve data
    for (int i = 0; i < tableSize; ++i) {
        double x = normalizeIndex(i, tableSize, minValue, maxValue);
        double y = curveData[i];
        samples.push_back({x, y});
    }

    // Add midpoint samples for better coverage
    // Must sample actual curve, NOT linear interpolation!
    // Linear interpolation creates false errors for high-frequency curves (e.g., Harmonic 15)
    std::vector<Sample> densified;
    densified.reserve(samples.size() * 2);

    for (size_t i = 0; i < samples.size(); ++i) {
        densified.push_back(samples[i]);

        // Add midpoint sample between this and next point
        if (i < samples.size() - 1) {
            double midX = (samples[i].x + samples[i + 1].x) / 2.0;

            // Sample using Catmull-Rom interpolation for sub-sample accuracy
            double midY = interpolateCurve(curveData, tableSize, minValue, maxValue, midX);

            densified.push_back({midX, midY});
        }
    }

    samples = std::move(densified);

    // Sort by x (should already be sorted, but ensure)
    sortByX(samples);

    // Deduplicate near-verticals
    deduplicateNearVerticals(samples);

    // Enforce strict monotonicity (if enabled)
    if (config.enforceMonotonicity) {
        enforceMonotonicity(samples);
    }

    // Clamp to [-1, 1] range
    clampToRange(samples);

    return samples;
}

void SplineFitter::sortByX(std::vector<Sample>& samples) {
    std::sort(samples.begin(), samples.end(), [](const Sample& a, const Sample& b) { return a.x < b.x; });
}

void SplineFitter::deduplicateNearVerticals(std::vector<Sample>& samples) {
    // Average y values for samples with nearly identical x
    constexpr double kXEpsilon = 1e-6;

    std::vector<Sample> deduped;
    deduped.reserve(samples.size());

    for (size_t i = 0; i < samples.size();) {
        double xSum = samples[i].x;
        double ySum = samples[i].y;
        int count = 1;

        // Group samples with similar x
        while (i + count < samples.size() && std::abs(samples[i + count].x - samples[i].x) < kXEpsilon) {
            xSum += samples[i + count].x;
            ySum += samples[i + count].y;
            ++count;
        }

        // Average the group
        deduped.push_back({xSum / count, ySum / count});
        i += count;
    }

    samples = std::move(deduped);
}

void SplineFitter::enforceMonotonicity(std::vector<Sample>& samples) {
    // Light isotonic regression: ensure y is strictly increasing with x
    // Use Pool Adjacent Violators Algorithm (PAVA) with minimal deviation

    if (samples.size() < 2)
        return;

    // Forward pass: ensure y[i+1] >= y[i]
    for (size_t i = 1; i < samples.size(); ++i) {
        if (samples[i].y < samples[i - 1].y) {
            // Average violating pairs (minimal deviation)
            double avgY = (samples[i].y + samples[i - 1].y) / 2.0;
            samples[i].y = avgY;
            samples[i - 1].y = avgY;
        }
    }

    // TODO: More sophisticated isotonic regression if needed
    // For now, simple pairwise averaging is sufficient
}

void SplineFitter::clampToRange(std::vector<Sample>& samples) {
    for (auto& s : samples) {
        s.x = juce::jlimit(-1.0, 1.0, s.x);
        s.y = juce::jlimit(-1.0, 1.0, s.y);
    }
}

// Tangent Computation (Dispatcher)

void SplineFitter::computeTangents(std::vector<SplineAnchor>& anchors, const SplineFitConfig& config) {

    switch (config.tangentAlgorithm) {
    case TangentAlgorithm::PCHIP:
        computePCHIPTangentsImpl(anchors, config);
        break;
    case TangentAlgorithm::FritschCarlson:
        computeFritschCarlsonTangents(anchors, config);
        break;
    case TangentAlgorithm::Akima:
        computeAkimaTangents(anchors, config);
        break;
    case TangentAlgorithm::FiniteDifference:
        computeFiniteDifferenceTangents(anchors, config);
        break;
    default:
        jassertfalse;
        computePCHIPTangentsImpl(anchors, config);
        break;
    }
}

// PCHIP Tangent Computation (Implementation)

void SplineFitter::computePCHIPTangentsImpl(std::vector<SplineAnchor>& anchors, const SplineFitConfig& config) {

    const int n = static_cast<int>(anchors.size());
    if (n < 2)
        return;

    // Compute secant slopes d_i = (y_{i+1} - y_i) / (x_{i+1} - x_i)
    std::vector<double> secants(n - 1);
    for (int i = 0; i < n - 1; ++i) {
        double dx = anchors[i + 1].x - anchors[i].x;
        if (std::abs(dx) < 1e-12) {
            secants[i] = 0.0; // Degenerate segment
        } else {
            secants[i] = (anchors[i + 1].y - anchors[i].y) / dx;
        }
    }

    // Compute tangents m_i using Fritsch-Carlson rules
    for (int i = 0; i < n; ++i) {
        if (i == 0) {
            // Endpoint: use one-sided derivative
            anchors[i].tangent = secants[0];
        } else if (i == n - 1) {
            // Endpoint: use one-sided derivative
            anchors[i].tangent = secants[n - 2];
        } else {
            // Interior point: apply Fritsch-Carlson formula
            double d_prev = secants[i - 1];
            double d_next = secants[i];

            // If secants have opposite signs, set tangent to zero (local extremum)
            if (d_prev * d_next <= 0.0) {
                anchors[i].tangent = 0.0;
            } else {
                // Weighted harmonic mean
                double dx_prev = anchors[i].x - anchors[i - 1].x;
                double dx_next = anchors[i + 1].x - anchors[i].x;

                double w1 = 2.0 * dx_next + dx_prev;
                double w2 = dx_next + 2.0 * dx_prev;

                anchors[i].tangent = harmonicMean(d_prev, d_next, w1, w2);
            }
        }

        // Enforce slope caps (for anti-aliasing)
        anchors[i].tangent = juce::jlimit(config.minSlope, config.maxSlope, anchors[i].tangent);
    }

    // Overshoot detection and correction
    // Check if PCHIP cubic overshoots the monotonic range between anchors
    // This prevents visual "bulges" and sonic artifacts in smooth regions
    const int maxOvershootIterations = 3; // Iterative refinement
    for (int iter = 0; iter < maxOvershootIterations; ++iter) {
        bool hadOvershoot = false;

        for (int i = 0; i < n - 1; ++i) {
            // Sample cubic at 5 interior points between anchors
            const double yMin = std::min(anchors[i].y, anchors[i + 1].y);
            const double yMax = std::max(anchors[i].y, anchors[i + 1].y);
            const double overshootTolerance = 0.001; // Allow tiny numerical error

            for (int j = 1; j < 5; ++j) {
                const double t = j / 5.0;
                const double x = anchors[i].x + t * (anchors[i + 1].x - anchors[i].x);
                const double y = SplineEvaluator::evaluateSegment(anchors[i], anchors[i + 1], x);

                // If cubic overshoots endpoint range, scale tangents down
                if (y < yMin - overshootTolerance || y > yMax + overshootTolerance) {
                    const double dampingFactor = 0.7; // Reduce tangents by 30%
                    anchors[i].tangent *= dampingFactor;
                    anchors[i + 1].tangent *= dampingFactor;
                    hadOvershoot = true;
                    break; // Move to next segment
                }
            }
        }

        // If no overshoots detected, we're done
        if (!hadOvershoot) {
            break;
        }
    }

    // Length-based tangent scaling
    // Very long segments (sparse anchor distribution) need gentler tangents to avoid oscillation
    // Only apply scaling to segments longer than threshold (0.3 = 15% of full range)
    const double longSegmentThreshold = 0.3;
    for (int i = 0; i < n - 1; ++i) {
        const double segmentLength = anchors[i + 1].x - anchors[i].x;

        if (segmentLength > longSegmentThreshold) {
            // Scale factor for long segments: gradually reduce tangents
            // Formula: min(1.0, threshold / segmentLength)
            // Example: length=0.4 → factor=0.75, length=0.6 → factor=0.5
            const double lengthFactor = std::min(1.0, longSegmentThreshold / segmentLength);

            anchors[i].tangent *= lengthFactor;

            // Also scale the next anchor's tangent if it's the last one
            if (i == n - 2) {
                anchors[i + 1].tangent *= lengthFactor;
            }
        }
    }
}

double SplineFitter::harmonicMean(double a, double b, double wa, double wb) {
    // Formula: m = (w1 + w2) / (w1/d_prev + w2/d_next)
    if (std::abs(a) < 1e-12 || std::abs(b) < 1e-12) {
        return 0.0;
    }
    return (wa + wb) / (wa / a + wb / b);
}

// Fritsch-Carlson Tangent Computation (Monotone-Preserving)

void SplineFitter::computeFritschCarlsonTangents(std::vector<SplineAnchor>& anchors, const SplineFitConfig& config) {

    const int n = static_cast<int>(anchors.size());
    if (n < 2)
        return;

    std::vector<double> tangents(n, 0.0);
    std::vector<double> deltas(n - 1);

    // 1. Compute segment slopes
    for (int i = 0; i < n - 1; ++i) {
        double dx = anchors[i + 1].x - anchors[i].x;
        if (std::abs(dx) < 1e-12) {
            deltas[i] = 0.0;
        } else {
            deltas[i] = (anchors[i + 1].y - anchors[i].y) / dx;
        }
    }

    // 2. Initial tangent estimates
    for (int i = 1; i < n - 1; ++i) {
        if (deltas[i - 1] * deltas[i] <= 0.0) {
            tangents[i] = 0.0; // Local extremum - force horizontal tangent
        } else {
            // Weighted average (harmonic mean variant)
            double w1 = 2.0 * (anchors[i + 1].x - anchors[i].x) + (anchors[i].x - anchors[i - 1].x);
            double w2 = (anchors[i + 1].x - anchors[i].x) + 2.0 * (anchors[i].x - anchors[i - 1].x);
            tangents[i] = (w1 * deltas[i - 1] + w2 * deltas[i]) / (w1 + w2);
        }
    }

    // 3. Enforce Fritsch-Carlson monotonicity constraints
    for (int i = 0; i < n - 1; ++i) {
        if (std::abs(deltas[i]) < 1e-9)
            continue; // Skip flat segments

        double alpha = tangents[i] / deltas[i];
        double beta = tangents[i + 1] / deltas[i];

        // Constraint: α² + β² ≤ 9 (ensures no overshoot)
        if (alpha * alpha + beta * beta > 9.0) {
            double tau = 3.0 / std::sqrt(alpha * alpha + beta * beta);
            tangents[i] = tau * alpha * deltas[i];
            tangents[i + 1] = tau * beta * deltas[i];
        }
    }

    // 4. Boundary tangents (use one-sided slopes)
    tangents[0] = deltas[0];
    tangents[n - 1] = deltas[n - 2];

    // 5. Apply tangents to anchors and enforce slope bounds
    for (int i = 0; i < n; ++i) {
        anchors[i].tangent = juce::jlimit(config.minSlope, config.maxSlope, tangents[i]);
    }
}

// Akima Tangent Computation (Local Weighted Average)

void SplineFitter::computeAkimaTangents(std::vector<SplineAnchor>& anchors, const SplineFitConfig& config) {

    const int n = static_cast<int>(anchors.size());
    if (n < 2)
        return;

    std::vector<double> tangents(n);
    std::vector<double> slopes(n + 3); // Extended for boundary handling

    // 1. Compute slopes with extrapolation for boundaries
    for (int i = 2; i < static_cast<int>(slopes.size()) - 2; ++i) {
        int anchorIdx = i - 2;
        double dx = anchors[anchorIdx + 1].x - anchors[anchorIdx].x;
        if (std::abs(dx) < 1e-12) {
            slopes[i] = 0.0;
        } else {
            slopes[i] = (anchors[anchorIdx + 1].y - anchors[anchorIdx].y) / dx;
        }
    }

    // Extrapolate boundary slopes
    slopes[0] = 2.0 * slopes[2] - slopes[3];
    slopes[1] = 2.0 * slopes[2] - slopes[3];
    slopes[slopes.size() - 1] = 2.0 * slopes[slopes.size() - 3] - slopes[slopes.size() - 4];
    slopes[slopes.size() - 2] = 2.0 * slopes[slopes.size() - 3] - slopes[slopes.size() - 4];

    // 2. Akima weighted formula
    for (int i = 0; i < n; ++i) {
        double m1 = slopes[i];
        double m2 = slopes[i + 1];
        double m3 = slopes[i + 2];
        double m4 = slopes[i + 3];

        double w1 = std::abs(m4 - m3);
        double w2 = std::abs(m2 - m1);

        if (w1 + w2 < 1e-9) {
            tangents[i] = (m2 + m3) * 0.5; // Average if weights are zero
        } else {
            tangents[i] = (w1 * m2 + w2 * m3) / (w1 + w2);
        }
    }

    // 3. Apply tangents to anchors and enforce slope bounds
    for (int i = 0; i < n; ++i) {
        anchors[i].tangent = juce::jlimit(config.minSlope, config.maxSlope, tangents[i]);
    }
}

// Finite Difference Tangent Computation (Simple Baseline)

void SplineFitter::computeFiniteDifferenceTangents(std::vector<SplineAnchor>& anchors, const SplineFitConfig& config) {

    const int n = static_cast<int>(anchors.size());
    if (n < 2)
        return;

    for (int i = 0; i < n; ++i) {
        double tangent = 0.0;

        if (i == 0) {
            // Forward difference for first point
            double dx = anchors[1].x - anchors[0].x;
            if (std::abs(dx) >= 1e-12) {
                tangent = (anchors[1].y - anchors[0].y) / dx;
            }
        } else if (i == n - 1) {
            // Backward difference for last point
            double dx = anchors[i].x - anchors[i - 1].x;
            if (std::abs(dx) >= 1e-12) {
                tangent = (anchors[i].y - anchors[i - 1].y) / dx;
            }
        } else {
            // Central difference for interior points
            double dx = anchors[i + 1].x - anchors[i - 1].x;
            if (std::abs(dx) >= 1e-12) {
                tangent = (anchors[i + 1].y - anchors[i - 1].y) / dx;
            }
        }

        anchors[i].tangent = juce::jlimit(config.minSlope, config.maxSlope, tangent);
    }
}

// Greedy Spline Fitting (Replaces RDP)

SplineFitter::WorstFitResult SplineFitter::findWorstFitSample(const std::vector<Sample>& samples,
                                                              const std::vector<SplineAnchor>& anchors) {

    WorstFitResult result{0, 0.0};

    if (anchors.size() < 2) {
        return result;
    }

    // Scan all samples, find worst error
    for (size_t i = 0; i < samples.size(); ++i) {
        const auto& sample = samples[i];

        // Skip samples that already have anchors at their x position
        bool hasAnchor = std::any_of(anchors.begin(), anchors.end(),
                                     [&sample](const SplineAnchor& a) { return std::abs(a.x - sample.x) < 1e-9; });

        if (hasAnchor) {
            continue; // Skip this sample
        }

        // Evaluate PCHIP spline at this x position
        double splineY = SplineEvaluator::evaluate(anchors, sample.x);

        // Compute absolute error
        double error = std::abs(sample.y - splineY);

        if (error > result.maxError) {
            result.maxError = error;
            result.sampleIndex = i;
        }
    }

    return result;
}

std::vector<SplineAnchor> SplineFitter::greedySplineFit(const std::vector<Sample>& samples,
                                                        const SplineFitConfig& config,
                                                        const double* curveData, int tableSize,
                                                        double minValue, double maxValue,
                                                        const std::vector<int>& mandatoryAnchorIndices) {

    if (samples.size() < 2)
        return {};

    // Initialize anchors from mandatory features or endpoints
    std::vector<SplineAnchor> anchors;
    if (mandatoryAnchorIndices.empty() || curveData == nullptr) {
        anchors.push_back({samples.front().x, samples.front().y, false, 0.0});
        anchors.push_back({samples.back().x, samples.back().y, false, 0.0});
    } else {
        anchors = initializeAnchorsFromIndices(samples, tableSize, minValue, maxValue, mandatoryAnchorIndices);
    }

    // Calculate how many additional anchors we can add
    // CRITICAL: Clamp to 0 to prevent negative iteration count if mandatory anchors exceed maxAnchors
    int remainingAnchors = std::max(0, config.maxAnchors - static_cast<int>(anchors.size()));

    // Compute vertical range for adaptive tolerance calculation
    double minY = samples.front().y;
    double maxY = samples.front().y;
    for (const auto& sample : samples) {
        minY = std::min(minY, sample.y);
        maxY = std::max(maxY, sample.y);
    }
    double verticalRange = maxY - minY;

    // Configure adaptive tolerance calculator
    // Use a fixed relative error target (1% of vertical range) for good balance between
    // quality and anchor count. This is decoupled from positionTolerance to allow
    // adaptive tolerance to increase aggressively for backtranslation stability.
    AdaptiveToleranceCalculator::Config adaptiveConfig;
    adaptiveConfig.relativeErrorTarget = 0.01; // 1% of vertical range (balanced default)
    // anchorDensityMultiplier = 10.0 (default from AdaptiveToleranceCalculator::Config)

    // Analyze symmetry if needed (controlled by config.symmetryDetection)
    bool useSymmetricMode = false;
    if (config.symmetryDetection == SymmetryDetection::Auto && curveData != nullptr && tableSize > 0) {
        std::vector<double> curveVec(curveData, curveData + tableSize);
        auto symmetryResult = SymmetryAnalyzer::analyzeOddSymmetry(curveVec);
        useSymmetricMode = (symmetryResult.score >= config.symmetryThreshold);
    }
    // If Never: useSymmetricMode remains false

    // Iteratively refine with error-driven placement (quality)
    for (int iteration = 0; iteration < remainingAnchors; ++iteration) {
        computeTangents(anchors, config);

        const auto worst = findWorstFitSample(samples, anchors);

        const double adaptiveTolerance = AdaptiveToleranceCalculator::computeTolerance(
            verticalRange, static_cast<int>(anchors.size()), config.maxAnchors, adaptiveConfig);

        if (worst.maxError <= adaptiveTolerance)
            break;

        // Insert anchor(s) at worst-fit location
        if (useSymmetricMode) {
            const int anchorsAdded = insertAnchorSymmetric(anchors, samples, worst.sampleIndex, adaptiveTolerance);
            if (anchorsAdded == 0)
                break; // No progress possible
            if (anchorsAdded == 2)
                ++iteration; // Consume extra iteration for pair
        } else {
            if (!insertAnchorAsymmetric(anchors, samples[worst.sampleIndex]))
                break; // No progress possible
        }
    }

    // Final tangent computation using configured algorithm
    computeTangents(anchors, config);

    return anchors;
}

// Helper Methods for Greedy Spline Fitting

std::vector<SplineAnchor> SplineFitter::initializeAnchorsFromIndices(const std::vector<Sample>& samples,
                                                                      int tableSize, double minValue, double maxValue,
                                                                      const std::vector<int>& mandatoryIndices) {
    std::vector<SplineAnchor> anchors;

    for (const int tableIdx : mandatoryIndices) {
        const double targetX = normalizeIndex(tableIdx, tableSize, minValue, maxValue);

        // Find closest sample to this x coordinate
        size_t closestIdx = 0;
        double minDist = std::abs(samples[0].x - targetX);

        for (size_t i = 1; i < samples.size(); ++i) {
            const double dist = std::abs(samples[i].x - targetX);
            if (dist < minDist) {
                minDist = dist;
                closestIdx = i;
            }
        }

        anchors.push_back({samples[closestIdx].x, samples[closestIdx].y, false, 0.0});
    }

    // Sort anchors by x
    std::sort(anchors.begin(), anchors.end(),
              [](const SplineAnchor& a, const SplineAnchor& b) { return a.x < b.x; });

    // Remove duplicates
    anchors.erase(std::unique(anchors.begin(), anchors.end(),
                              [](const SplineAnchor& a, const SplineAnchor& b) { return std::abs(a.x - b.x) < 1e-9; }),
                  anchors.end());

    return anchors;
}

int SplineFitter::insertAnchorSymmetric(std::vector<SplineAnchor>& anchors, const std::vector<Sample>& samples,
                                        size_t worstSampleIndex, double adaptiveTolerance) {
    const auto& worstSample = samples[worstSampleIndex];

    // Find complementary x position and closest sample
    const double complementaryX = -worstSample.x;
    size_t complementaryIdx = 0;
    double minDist = std::abs(samples[0].x - complementaryX);
    for (size_t i = 1; i < samples.size(); ++i) {
        const double dist = std::abs(samples[i].x - complementaryX);
        if (dist < minDist) {
            minDist = dist;
            complementaryIdx = i;
        }
    }
    const auto& complementarySample = samples[complementaryIdx];

    // Check for existing anchors
    const bool hasOriginalAnchor = std::any_of(anchors.begin(), anchors.end(), [&worstSample](const SplineAnchor& a) {
        return std::abs(a.x - worstSample.x) < 1e-9;
    });

    const bool hasComplementaryAnchor =
        std::any_of(anchors.begin(), anchors.end(),
                    [&complementarySample](const SplineAnchor& a) { return std::abs(a.x - complementarySample.x) < 1e-9; });

    // Check complementary error
    double complementaryError = 0.0;
    if (!hasComplementaryAnchor) {
        const double splineY = SplineEvaluator::evaluate(anchors, complementarySample.x);
        complementaryError = std::abs(complementarySample.y - splineY);
    }

    // Determine if we can add a symmetric pair
    const bool canAddPair = !hasOriginalAnchor && !hasComplementaryAnchor && complementaryError > adaptiveTolerance * 0.5;

    if (canAddPair) {
        // Add both anchors with symmetric y values (odd symmetry)
        const double ySymmetric = (worstSample.y - complementarySample.y) / 2.0;

        auto insertPos1 = std::lower_bound(anchors.begin(), anchors.end(), worstSample.x,
                                           [](const SplineAnchor& a, double x) { return a.x < x; });
        anchors.insert(insertPos1, {worstSample.x, ySymmetric, false, 0.0});

        auto insertPos2 = std::lower_bound(anchors.begin(), anchors.end(), complementarySample.x,
                                           [](const SplineAnchor& a, double x) { return a.x < x; });
        anchors.insert(insertPos2, {complementarySample.x, -ySymmetric, false, 0.0});

        return 2;
    }

    // Fallback: add single anchor
    if (hasOriginalAnchor) {
        return 0; // No progress possible
    }

    auto insertPos = std::lower_bound(anchors.begin(), anchors.end(), worstSample.x,
                                      [](const SplineAnchor& a, double x) { return a.x < x; });
    anchors.insert(insertPos, {worstSample.x, worstSample.y, false, 0.0});
    return 1;
}

bool SplineFitter::insertAnchorAsymmetric(std::vector<SplineAnchor>& anchors, const Sample& worstSample) {
    // Check for duplicate
    const bool isDuplicate = std::any_of(anchors.begin(), anchors.end(), [&worstSample](const SplineAnchor& a) {
        return std::abs(a.x - worstSample.x) < 1e-9;
    });

    if (isDuplicate) {
        return false;
    }

    auto insertPos = std::lower_bound(anchors.begin(), anchors.end(), worstSample.x,
                                      [](const SplineAnchor& a, double x) { return a.x < x; });
    anchors.insert(insertPos, {worstSample.x, worstSample.y, false, 0.0});
    return true;
}

} // namespace dsp_core::Services
