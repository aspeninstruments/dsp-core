#include "LayeredTransferFunction.h"
#include "Services/SplineEvaluator.h"
#include <algorithm>
#include <cmath>
#include <juce_core/juce_core.h>
#include <juce_data_structures/juce_data_structures.h>

// Branch prediction hints (improve CPU branch predictor performance)
// GCC/Clang support __builtin_expect for optimization
#if defined(__GNUC__) || defined(__clang__)
#define juce_likely(x) __builtin_expect(!!(x), 1)
#define juce_unlikely(x) __builtin_expect(!!(x), 0)
#else
#define juce_likely(x) (x)
#define juce_unlikely(x) (x)
#endif

namespace dsp_core {

// Static instance counter for debugging
std::atomic<int> LayeredTransferFunction::instanceCounter{0};

namespace {
    // Number of harmonics for harmonic layer synthesis
    constexpr int kNumHarmonics = 40;
    // Total coefficients: [0] = wavetable mix, [1..40] = harmonic amplitudes
    constexpr int kTotalCoefficients = kNumHarmonics + 1;
    // Epsilon for zero comparison in normalization
    constexpr double kNormalizationEpsilon = 1e-12;
} // namespace

LayeredTransferFunction::LayeredTransferFunction(int tableSize, double minVal, double maxVal)
    : instanceId(instanceCounter.fetch_add(1, std::memory_order_relaxed)),
      tableSize(tableSize), minValue(minVal), maxValue(maxVal), harmonicLayer(std::make_unique<HarmonicLayer>(kNumHarmonics)),
      splineLayer(std::make_unique<SplineLayer>()), // NEW: Initialize spline layer
      coefficients(kTotalCoefficients, 0.0),        // kTotalCoefficients: [0] = WT, [1..40] = harmonics
      baseTable(tableSize) {

    // Initialize coefficients: WT=0.0, H1=1.0, others=0.0
    // This results in y=x (linear) at startup, but tanh(2x) is one slider move away
    coefficients[0] = 0.0; // WT mix = 0.0 (base layer not mixed in yet)
    coefficients[1] = 1.0; // H1 = 1.0 (linear harmonic)
    // coefficients[2..40] already initialized to 0.0

    // Initialize base layer to tanh(2x) - classic distortion ready to use
    for (int i = 0; i < tableSize; ++i) {
        const double x = normalizeIndex(i);
        baseTable[i].store(std::tanh(2.0 * x), std::memory_order_release);
    }

    // Precompute harmonic basis functions
    harmonicLayer->precomputeBasisFunctions(tableSize, minValue, maxValue);

    // No need to compute composite here - renderer will generate it at 25Hz
}

double LayeredTransferFunction::getBaseLayerValue(int index) const {
    if (index >= 0 && index < tableSize) {
        const double value = baseTable[index].load(std::memory_order_acquire);
        return value;
    }
    return 0.0;
}

void LayeredTransferFunction::setBaseLayerValue(int index, double value) {
    if (index >= 0 && index < tableSize) {
        baseTable[index].store(value, std::memory_order_release);

        incrementVersionIfNotBatching();
    }
}

void LayeredTransferFunction::clearBaseLayer() {
    for (int i = 0; i < tableSize; ++i) {
        baseTable[i].store(0.0, std::memory_order_release);
    }
    incrementVersionIfNotBatching();
}

const HarmonicLayer& LayeredTransferFunction::getHarmonicLayer() const {
    return *harmonicLayer;
}

const SplineLayer& LayeredTransferFunction::getSplineLayer() const {
    return *splineLayer;
}

void LayeredTransferFunction::beginBatchUpdate() {
    batchUpdateActive = true;
}

void LayeredTransferFunction::endBatchUpdate() {
    if (batchUpdateActive) {
        batchUpdateActive = false;
        versionCounter.fetch_add(1, std::memory_order_release);
    }
}

void LayeredTransferFunction::setSplineAnchors(const std::vector<SplineAnchor>& anchors) {
    splineLayer->setAnchors(anchors);
    incrementVersionIfNotBatching();
}

void LayeredTransferFunction::clearSplineAnchors() {
    splineLayer->setAnchors({});
    incrementVersionIfNotBatching();
}

void LayeredTransferFunction::setCoefficient(int index, double value) {
    if (index >= 0 && index < static_cast<int>(coefficients.size())) {
        coefficients[index] = value;

        incrementVersionIfNotBatching();
    }
}

double LayeredTransferFunction::getCoefficient(int index) const {
    if (index >= 0 && index < static_cast<int>(coefficients.size())) {
        return coefficients[index];
    }
    return 0.0;
}

double LayeredTransferFunction::computeCompositeAt(int index) const {
    // Bounds check
    if (index < 0 || index >= tableSize) {
        return 0.0;
    }

    // Normalize index to x-coordinate
    const double x = normalizeIndex(index);

    // Read base layer value
    const double baseValue = baseTable[index].load(std::memory_order_acquire);

    // Evaluate harmonics at x
    const double harmonicValue = harmonicLayer->evaluate(x, coefficients, tableSize);

    // Compute unnormalized composite: wtCoeff * base + harmonics
    const double wtCoeff = coefficients[0];
    const double unnormalized = wtCoeff * baseValue + harmonicValue;

    // Apply cached normalization scalar
    const double normScalar = normalizationScalar.load(std::memory_order_acquire);
    return normScalar * unnormalized;
}

void LayeredTransferFunction::updateNormalizationScalar() {
    // Compute max absolute value across entire composite
    double maxAbsValue = 0.0;
    const double wtCoeff = coefficients[0];

    for (int i = 0; i < tableSize; ++i) {
        const double x = normalizeIndex(i);
        const double baseValue = baseTable[i].load(std::memory_order_acquire);
        const double harmonicValue = harmonicLayer->evaluate(x, coefficients, tableSize);
        const double unnormalized = wtCoeff * baseValue + harmonicValue;
        maxAbsValue = std::max(maxAbsValue, std::abs(unnormalized));
    }

    // Update cached normalization scalar
    constexpr double kNormalizationEpsilon = 1e-12;
    const double normScalar = (maxAbsValue > kNormalizationEpsilon) ? (1.0 / maxAbsValue) : 1.0;
    normalizationScalar.store(normScalar, std::memory_order_release);
}

void LayeredTransferFunction::setPaintStrokeActive(bool active) {
    paintStrokeActive = active;
}

bool LayeredTransferFunction::isPaintStrokeActive() const {
    return paintStrokeActive;
}

bool LayeredTransferFunction::hasNonZeroHarmonics() const {
    // Check harmonics only (coefficients[1..40]), not WT mix at [0]
    for (int i = 1; i < static_cast<int>(coefficients.size()); ++i) {
        if (std::abs(coefficients[i]) > HARMONIC_EPSILON) {
            return true;
        }
    }
    return false;
}

bool LayeredTransferFunction::bakeHarmonicsToBase() {
    // Early exit only if composite equals base layer (no-op optimization)
    // This requires: WT=1.0 AND no harmonics (then composite = 1.0 * base + 0 = base)
    // If WT != 1.0, we must bake even with no harmonics because composite = WT * base
    const bool wtIsOne = std::abs(coefficients[0] - 1.0) <= HARMONIC_EPSILON;
    if (wtIsOne && !hasNonZeroHarmonics()) {
        return false;  // Composite already equals base layer, nothing to bake
    }

    // Batch update guard: Defer version increment until end of function
    // Without this, setBaseLayerValue() would increment 16,384 times!
    // NOLINTNEXTLINE(misc-const-correctness) - RAII guard, destructor modifies state
    BatchUpdateGuard guard(*this);

    // Step 1: Compute and cache normalization scalar BEFORE baking
    // This ensures baked values match what the user sees on screen
    updateNormalizationScalar();

    // Step 2: Bake composite curve (base + harmonics with normalization applied)
    // computeCompositeAt() will use the cached scalar we just computed
    for (int i = 0; i < tableSize; ++i) {
        const double compositeValue = computeCompositeAt(i);
        setBaseLayerValue(i, compositeValue);  // No version increment (batched)
    }

    // Step 3: Set WT coefficient to 1.0 (enables the baked base layer)
    // If WT was 0 during harmonic editing, the baked base layer would be invisible
    coefficients[0] = 1.0;

    // Step 4: Zero out all harmonic coefficients (h1..h40)
    for (int i = 1; i < static_cast<int>(coefficients.size()); ++i) {
        coefficients[i] = 0.0;
    }

    // Step 5: Recalculate normalization scalar for the new state
    // After baking, base layer contains normalized values, so scalar should be ~1.0
    updateNormalizationScalar();

    // Step 6: Version increment happens automatically when guard destructor runs
    // (16,384 base layer writes + coefficient changes → 1 version increment)

    return true;
}

void LayeredTransferFunction::bakeCompositeToBase() {
    // Batch update guard: Defer version increment until end of function
    // Without this, setBaseLayerValue() would increment 16,384 times!
    // NOLINTNEXTLINE(misc-const-correctness) - RAII guard, destructor modifies state
    BatchUpdateGuard guard(*this);

    // Step 1: Compute and cache normalization scalar BEFORE baking
    // This ensures baked values match what the user sees on screen
    updateNormalizationScalar();

    // Step 2: Bake composite curve (base + harmonics with normalization applied)
    // computeCompositeAt() will use the cached scalar we just computed
    for (int i = 0; i < tableSize; ++i) {
        const double compositeValue = computeCompositeAt(i);
        setBaseLayerValue(i, compositeValue);  // No version increment (batched)
    }

    // Step 3: Set WT coefficient to 1.0 (enable base layer)
    coefficients[0] = 1.0;

    // Step 4: Zero out all harmonic coefficients (h1..h40)
    for (int i = 1; i < static_cast<int>(coefficients.size()); ++i) {
        coefficients[i] = 0.0;
    }

    // Step 5: Recalculate normalization scalar for the new state
    // After baking, base layer contains normalized values, so scalar should be ~1.0
    updateNormalizationScalar();

    // Step 6: Version increment happens automatically when guard destructor runs
    // (16,384 base layer writes + coefficient changes → 1 version increment)
}

void LayeredTransferFunction::bakeSplineToBase() {
    // Batch update guard: Defer version increment until end of function
    // NOLINTNEXTLINE(misc-const-correctness) - RAII guard, destructor modifies state
    BatchUpdateGuard guard(*this);

    // Step 1: Get current spline anchors
    const auto& anchors = splineLayer->getAnchors();
    if (anchors.empty()) {
        return;  // No spline to bake
    }

    // Step 2: Batch evaluate spline at all table indices
    std::vector<double> xValues(static_cast<size_t>(tableSize));
    std::vector<double> yValues(static_cast<size_t>(tableSize));

    for (int i = 0; i < tableSize; ++i) {
        xValues[static_cast<size_t>(i)] = normalizeIndex(i);
    }

    Services::SplineEvaluator::evaluateBatch(anchors, xValues.data(), yValues.data(), tableSize);

    // Step 3: Write evaluated spline values to base layer
    for (int i = 0; i < tableSize; ++i) {
        setBaseLayerValue(i, yValues[static_cast<size_t>(i)]);
    }

    // Step 4: Set WT coefficient to 1.0 (enable base layer)
    coefficients[0] = 1.0;

    // Step 5: Zero out all harmonic coefficients (h1..h40)
    for (int i = 1; i < static_cast<int>(coefficients.size()); ++i) {
        coefficients[i] = 0.0;
    }

    // Step 6: Recalculate normalization scalar for the new state
    updateNormalizationScalar();

    // Step 7: Version increment happens automatically when guard destructor runs
}

std::array<double, LayeredTransferFunction::NUM_HARMONIC_COEFFICIENTS>
LayeredTransferFunction::getHarmonicCoefficients() const {
    std::array<double, NUM_HARMONIC_COEFFICIENTS> coeffs{};

    // Copy all coefficients: [0] = WT mix, [1..40] = harmonics
    for (int i = 0; i < NUM_HARMONIC_COEFFICIENTS && i < static_cast<int>(coefficients.size()); ++i) {
        coeffs[i] = coefficients[i];
    }

    return coeffs;
}

void LayeredTransferFunction::setHarmonicCoefficients(const std::array<double, NUM_HARMONIC_COEFFICIENTS>& coeffs) {
    // Set all coefficients: [0] = WT mix, [1..40] = harmonics
    for (int i = 0; i < NUM_HARMONIC_COEFFICIENTS && i < static_cast<int>(coefficients.size()); ++i) {
        coefficients[i] = coeffs[i];
    }

    // Increment version to trigger renderer update
    incrementVersionIfNotBatching();
}

void LayeredTransferFunction::setRenderingMode(RenderingMode mode) {
    renderingMode.store(static_cast<int>(mode), std::memory_order_release);
    incrementVersionIfNotBatching();
}

RenderingMode LayeredTransferFunction::getRenderingMode() const {
    return static_cast<RenderingMode>(renderingMode.load(std::memory_order_acquire));
}

double LayeredTransferFunction::normalizeIndex(int index) const {
    if (index < 0 || index >= tableSize) {
        return 0.0;
    }
    return juce::jmap(static_cast<double>(index), 0.0, static_cast<double>(tableSize - 1), minValue, maxValue);
}

double LayeredTransferFunction::applyTransferFunction(double x) const {
    return interpolateCatmullRom(x);
}

void LayeredTransferFunction::processBlock(double* samples, int numSamples) const {
    for (int i = 0; i < numSamples; ++i) {
        samples[i] = applyTransferFunction(samples[i]);
    }
}

double LayeredTransferFunction::interpolateCatmullRom(double x) const {
    // Map x from signal range to table index
    const double x_proj = (x - minValue) / (maxValue - minValue) * (tableSize - 1);
    const int index = static_cast<int>(x_proj);
    const double t = x_proj - index;

    // PERFORMANCE: Fast path for Clamp mode (most common, default case ~95%)
    // Computes on-demand from base layer + harmonics (no cached composite table)
    if (juce_likely(extrapMode == ExtrapolationMode::Clamp)) {
        const int idx0 = juce::jlimit(0, tableSize - 1, index - 1);
        const int idx1 = juce::jlimit(0, tableSize - 1, index);
        const int idx2 = juce::jlimit(0, tableSize - 1, index + 1);
        const int idx3 = juce::jlimit(0, tableSize - 1, index + 2);

        const double y0 = computeCompositeAt(idx0);
        const double y1 = computeCompositeAt(idx1);
        const double y2 = computeCompositeAt(idx2);
        const double y3 = computeCompositeAt(idx3);

        // NOLINTBEGIN(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
        // Standard Catmull-Rom interpolation formula
        return 0.5 * ((2.0 * y1) + (-y0 + y2) * t + (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) * t * t +
                      (-y0 + 3.0 * y1 - 3.0 * y2 + y3) * t * t * t);
        // NOLINTEND(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
    }

    if (extrapMode == ExtrapolationMode::Mirror) {
        auto mirrorIdx = [this](int i) -> int {
            const int period = 2 * (tableSize - 1);
            int wrapped = ((i % period) + period) % period;
            return wrapped <= tableSize - 1 ? wrapped : period - wrapped;
        };

        const double y0 = computeCompositeAt(mirrorIdx(index - 1));
        const double y1 = computeCompositeAt(mirrorIdx(index));
        const double y2 = computeCompositeAt(mirrorIdx(index + 1));
        const double y3 = computeCompositeAt(mirrorIdx(index + 2));

        // NOLINTBEGIN(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
        return 0.5 * ((2.0 * y1) + (-y0 + y2) * t + (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) * t * t +
                      (-y0 + 3.0 * y1 - 3.0 * y2 + y3) * t * t * t);
        // NOLINTEND(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
    }

    // Linear extrapolation path (helper lambda for readability)
    auto getSample = [this](int i) -> double {
        if (i < 0) {
            const double slope = computeCompositeAt(1) - computeCompositeAt(0);
            return computeCompositeAt(0) + slope * i;
        }
        if (i >= tableSize) {
            const double slope = computeCompositeAt(tableSize - 1) - computeCompositeAt(tableSize - 2);
            return computeCompositeAt(tableSize - 1) + slope * (i - tableSize + 1);
        }
        return computeCompositeAt(i);
    };

    const double y0 = getSample(index - 1);
    const double y1 = getSample(index);
    const double y2 = getSample(index + 1);
    const double y3 = getSample(index + 2);

    // NOLINTBEGIN(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
    // Standard Catmull-Rom interpolation formula
    return 0.5 * ((2.0 * y1) + (-y0 + y2) * t + (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) * t * t +
                  (-y0 + 3.0 * y1 - 3.0 * y2 + y3) * t * t * t);
    // NOLINTEND(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
}

double LayeredTransferFunction::getBaseValueAt(double x) const {
    // Map x to table index
    const double tableIndex = ((x - minValue) / (maxValue - minValue)) * (tableSize - 1);

    // Handle out-of-bounds based on extrapolation mode
    if (extrapMode == ExtrapolationMode::Clamp) {
        if (tableIndex <= 0.0) {
            return baseTable[0].load(std::memory_order_relaxed);
        }
        if (tableIndex >= tableSize - 1) {
            return baseTable[tableSize - 1].load(std::memory_order_relaxed);
        }
    }

    if (extrapMode == ExtrapolationMode::Mirror) {
        // Mirror-wrap the continuous index into [0, tableSize-1]
        const int period = 2 * (tableSize - 1);
        auto mirrorWrap = [period, this](double ti) -> double {
            // Fold into [0, period) then reflect upper half
            double wrapped = std::fmod(std::fmod(ti, period) + period, static_cast<double>(period));
            if (wrapped > tableSize - 1) {
                wrapped = period - wrapped;
            }
            // Linear interpolation at the wrapped position
            int i = static_cast<int>(wrapped);
            i = juce::jlimit(0, tableSize - 2, i);
            double f = wrapped - i;
            const double a = baseTable[i].load(std::memory_order_relaxed);
            const double b = baseTable[i + 1].load(std::memory_order_relaxed);
            return a + f * (b - a);
        };
        return mirrorWrap(tableIndex);
    }

    // Get integer and fractional parts
    const int idx = static_cast<int>(tableIndex);
    const double frac = tableIndex - idx;

    // Bounds check
    if (idx < 0 || idx >= tableSize - 1) {
        // Linear extrapolation (only reached if extrapMode == Linear)
        if (idx < 0) {
            const double y0 = baseTable[0].load(std::memory_order_relaxed);
            const double y1 = baseTable[1].load(std::memory_order_relaxed);
            const double slope = y1 - y0;
            return y0 + slope * tableIndex;
        } else {
            const double y0 = baseTable[tableSize - 2].load(std::memory_order_relaxed);
            const double y1 = baseTable[tableSize - 1].load(std::memory_order_relaxed);
            const double slope = y1 - y0;
            return y0 + slope * (tableIndex - (tableSize - 2));
        }
    }

    // Linear interpolation (sufficient for base layer reading)
    const double y0 = baseTable[idx].load(std::memory_order_relaxed);
    const double y1 = baseTable[idx + 1].load(std::memory_order_relaxed);
    return y0 + frac * (y1 - y0);
}

double LayeredTransferFunction::evaluateBaseAndHarmonics(double x) const {
    // Always evaluate base + harmonics, ignoring spline layer state
    // Used by SplineFitter to read the normalized composite when entering spline mode
    //
    // normalizationScalar is preserved when entering spline mode (not reset to 1.0),
    // so this returns the properly normalized values that SplineFitter needs to fit

    // Interpolate base layer (linear interpolation)
    const double x_proj = (x - minValue) / (maxValue - minValue) * (tableSize - 1);
    const int index = static_cast<int>(x_proj);
    const double t = x_proj - index;

    const int idx0 = juce::jlimit(0, tableSize - 1, index);
    const int idx1 = juce::jlimit(0, tableSize - 1, index + 1);

    const double y0 = baseTable[idx0].load(std::memory_order_relaxed);
    const double y1 = baseTable[idx1].load(std::memory_order_relaxed);

    const double baseValue = y0 + t * (y1 - y0);

    // Add harmonics
    const double harmonicValue = harmonicLayer->evaluate(x, coefficients, tableSize);
    const double wtCoeff = coefficients[0];
    const double result = wtCoeff * baseValue + harmonicValue;

    // Apply normalization scalar to match what user sees on screen
    const double normScalar = normalizationScalar.load(std::memory_order_acquire);
    return normScalar * result;
}

/**
 * Evaluate transfer function for LUT rendering based on current mode
 *
 * RENDERING MODE EVALUATION PATHS
 *
 * This method implements three distinct evaluation strategies optimized for
 * different editing workflows. Mode selection happens at runtime via atomic
 * renderingMode flag, allowing seamless mode switching without LUT rebuild.
 *
 * ────────────────────────────────────────────────────────────────────────────
 * PAINT MODE: wtCoeff × base (no normalization, ~10-15% faster)
 * ────────────────────────────────────────────────────────────────────────────
 *
 * Formula: output = base[x]
 *
 * Invariant: Harmonics already baked to base layer
 *   - wtCoeff = 1.0 (enables base layer)
 *   - harmonicCoeffs[1..40] = 0.0 (all harmonics zeroed)
 *   - Base layer contains final composite (visually identical to pre-bake)
 *
 * Why no normalization:
 *   - Paint strokes constrain output to [-1, 1] via UI
 *   - Base layer directly editable (user sees what they get)
 *   - Skipping normalization saves ~10-15% per sample
 *
 * Performance: ~5-10 CPU cycles per sample
 *   - Single interpolated read from baseTable
 *   - No harmonic basis function evaluations
 *   - No normalization multiply
 *
 * Output range: [-1, 1] (clamped during painting)
 *
 * Use cases:
 *   ✓ Direct curve drawing
 *   ✓ Algorithmic transforms (Magic mode)
 *   ✓ Post-harmonic baking
 *
 * ────────────────────────────────────────────────────────────────────────────
 * HARMONIC MODE: normScalar × (wtCoeff × base + Σ harmonics)
 * ────────────────────────────────────────────────────────────────────────────
 *
 * Formula: output = normScalar × (wtCoeff × base[x] + Σ(coeff[n] × basis_n(x)))
 *
 * Why normalization required:
 *   - Harmonics can amplify signal (max value unpredictable)
 *   - User adjusts coefficients freely (no UI clamping)
 *   - Normalization ensures output ∈ [-1, 1]
 *
 * Normalization computation (done by renderer once per LUT):
 *   1. Scan all 16,384 points: maxAbs = max(|unnormalized[i]|)
 *   2. Compute scalar: normScalar = 1.0 / maxAbs
 *   3. Pass scalar to evaluateForRendering() for each sample
 *
 * Performance: ~60-80 CPU cycles per sample
 *   - 1 interpolated base layer read
 *   - 40 harmonic basis function evaluations (sin/cos)
 *   - 41 coefficient loads + multiply-accumulate
 *   - 1 normalization multiply
 *
 * Optimization: Early-exit if hasNonZeroHarmonics() == false
 *   - Saves ~50 cycles when harmonics zeroed
 *   - Falls back to base-only path: normScalar × wtCoeff × base[x]
 *
 * Output range: [-1, 1] (normalized)
 *
 * Use cases:
 *   ✓ Harmonic synthesis (slider adjustments)
 *   ✓ Real-time harmonic preview
 *   ✓ Before baking (visualize composite)
 *
 * ────────────────────────────────────────────────────────────────────────────
 * SPLINE MODE: splineLayer.evaluate(x) (direct Catmull-Rom)
 * ────────────────────────────────────────────────────────────────────────────
 *
 * Formula: output = catmullRom(anchors, x)
 *
 * Why bypass base + harmonics:
 *   - Spline defines curve independently (not additive)
 *   - Anchors positioned directly in output space
 *   - No mixing needed (spline IS the transfer function)
 *
 * Anchor management:
 *   - Fitted from base+harmonics on mode entry (~5-20 anchors)
 *   - User drags anchors directly (UI-clamped to [-1, 1])
 *   - Catmull-Rom interpolation ensures smooth C1 continuity
 *
 * Performance: ~20-30 CPU cycles per sample
 *   - Binary search to find anchor interval
 *   - Catmull-Rom evaluation (4-point interpolation)
 *   - No harmonic evaluation, no normalization
 *
 * Output range: [-1, 1] (anchors UI-clamped during placement)
 *
 * Use cases:
 *   ✓ Precise control point editing
 *   ✓ Curve refinement after harmonic synthesis
 *   ✓ Manual tweaking of algorithmic results
 *
 * PERFORMANCE COMPARISON (per-sample cost on M1 Max)
 *
 * Paint Mode:    ~5-10 cycles  (baseline)
 * Spline Mode:   ~20-30 cycles (3× Paint, but fewer anchors than harmonics)
 * Harmonic Mode: ~60-80 cycles (8× Paint, 40 sin/cos evaluations)
 *
 * LUT render times (16,384 samples):
 *   Paint:    ~80 μs
 *   Spline:   ~400 μs
 *   Harmonic: ~1.2 ms (with all harmonics active)
 *
 * Worker budget at 25Hz: 40ms (ample headroom for all modes)
 */
double LayeredTransferFunction::evaluateForRendering(double x, double normScalar) const {
    const RenderingMode mode = getRenderingMode();

    switch (mode) {
        case RenderingMode::Spline:
            // Direct spline evaluation (bypasses base + harmonics)
            return splineLayer->evaluate(x);

        case RenderingMode::Paint:
        {
            // Paint mode: Direct base layer output (no normalization, no harmonics)
            // Invariant: Harmonics should be baked into base layer (wtCoeff = 1.0, harmonics = 0)
            // We skip wtCoeff multiplication for performance (always 1.0 in Paint mode)
            return getBaseValueAt(x);  // Direct base read, NO NORMALIZATION
        }

        case RenderingMode::Harmonic:
        default:
        {
            // Harmonic mode: Base + harmonics with normalization
            const double wtCoeff = coefficients[0];
            const double baseValue = getBaseValueAt(x);

            // OPTIMIZATION: Early-exit if all harmonics are zero
            // Saves 40 coefficient loads + harmonic evaluation (sin/cos computations)
            if (!hasNonZeroHarmonics()) {
                const double unnormalized = wtCoeff * baseValue;
                return normScalar * unnormalized;  // Base only, WITH NORMALIZATION
            }

            const double harmonicValue = harmonicLayer->evaluate(x, coefficients, tableSize);
            const double unnormalized = wtCoeff * baseValue + harmonicValue;
            return normScalar * unnormalized;  // WITH NORMALIZATION
        }
    }
}

juce::ValueTree LayeredTransferFunction::toValueTree() const {
    juce::ValueTree vt("LayeredTransferFunction");

    // Serialize coefficients
    juce::Array<juce::var> coeffArray;
    for (const double c : coefficients) {
        coeffArray.add(c);
    }
    vt.setProperty("coefficients", coeffArray, nullptr);

    // Serialize base layer
    if (tableSize > 0) {
        juce::ValueTree baseVT("BaseLayer");
        juce::MemoryBlock baseBlob;
        for (int i = 0; i < tableSize; ++i) {
            const double value = baseTable[i].load(std::memory_order_acquire);
            baseBlob.append(&value, sizeof(double));
        }
        baseVT.setProperty("tableData", baseBlob, nullptr);
        vt.addChild(baseVT, -1, nullptr);
    }

    // Serialize harmonic layer (algorithm settings only, no coefficients)
    if (harmonicLayer) {
        vt.addChild(harmonicLayer->toValueTree(), -1, nullptr);
    }

    // NEW: Serialize spline layer
    if (splineLayer) {
        vt.addChild(splineLayer->toValueTree(), -1, nullptr);
    }

    // Serialize normalization scalar
    vt.setProperty("normalizationScalar", normalizationScalar.load(std::memory_order_acquire), nullptr);

    // Serialize settings
    vt.setProperty("extrapolationMode", static_cast<int>(extrapMode), nullptr);

    return vt;
}

void LayeredTransferFunction::fromValueTree(const juce::ValueTree& vt) {
    if (!vt.isValid() || vt.getType().toString() != "LayeredTransferFunction") {
        return;
    }

    // Load coefficients
    // Always maintain exactly NUM_HARMONIC_COEFFICIENTS (41) coefficients
    // Old presets may have fewer coefficients - pad with zeros if needed
    coefficients.resize(NUM_HARMONIC_COEFFICIENTS, 0.0); // Reset to 41 zeros
    if (vt.hasProperty("coefficients")) {
        const juce::Array<juce::var>* coeffArray = vt.getProperty("coefficients").getArray();
        if (coeffArray != nullptr) {
            const int numToLoad = std::min(coeffArray->size(), NUM_HARMONIC_COEFFICIENTS);
            for (int i = 0; i < numToLoad; ++i) {
                coefficients[i] = static_cast<double>((*coeffArray)[i]);
            }
        }
    }

    // Load base layer
    const auto baseVT = vt.getChildWithName("BaseLayer");
    if (baseVT.isValid() && baseVT.hasProperty("tableData")) {
        const juce::MemoryBlock baseBlob = *baseVT.getProperty("tableData").getBinaryData();
        const double* data = static_cast<const double*>(baseBlob.getData());
        const int numValues = static_cast<int>(baseBlob.getSize() / sizeof(double));

        for (int i = 0; i < std::min(numValues, tableSize); ++i) {
            baseTable[i].store(data[i], std::memory_order_release);
        }
    }

    // Load harmonic layer (algorithm settings only)
    const auto harmonicVT = vt.getChildWithName("HarmonicLayer");
    if (harmonicVT.isValid()) {
        harmonicLayer->fromValueTree(harmonicVT);
        harmonicLayer->precomputeBasisFunctions(tableSize, minValue, maxValue);
    }

    // NEW: Load spline layer
    const auto splineVT = vt.getChildWithName("SplineLayer");
    if (splineVT.isValid()) {
        splineLayer->fromValueTree(splineVT);
    }

    // Load normalization scalar (optional - will be recomputed anyway)
    if (vt.hasProperty("normalizationScalar")) {
        normalizationScalar.store(static_cast<double>(vt.getProperty("normalizationScalar")),
                                  std::memory_order_release);
    }

    // Load settings
    // Note: interpolationMode is ignored for backward compatibility (Catmull-Rom is now hardcoded)
    if (vt.hasProperty("extrapolationMode")) {
        // Legacy presets may have persisted ExtrapolationMode::Surge (raw int = 3).
        // That enum value is gone — fall back to Linear (no crash, sensible default).
        // The user is responsible for re-saving presets that relied on Surge mode.
        const int raw = static_cast<int>(vt.getProperty("extrapolationMode"));
        extrapMode = (raw >= 0 && raw <= 2)
                         ? static_cast<ExtrapolationMode>(raw)
                         : ExtrapolationMode::Linear;
    }

    // Increment version to trigger LUT render on preset load
    incrementVersionIfNotBatching();
}

} // namespace dsp_core
