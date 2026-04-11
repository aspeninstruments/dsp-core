#pragma once
#include "HarmonicLayer.h"
#include "SplineLayer.h"
#include <juce_core/juce_core.h>
#include <vector>
#include <atomic>
#include <memory>

namespace dsp_core {

/**
 * RenderingMode - Determines which layer evaluation path to use
 *
 * Paint    → Direct base layer (no normalization, harmonics baked)
 * Harmonic → Base + harmonics with normalization (harmonics non-zero)
 * Spline   → Direct spline evaluation (bypasses base+harmonics)
 */
enum class RenderingMode {
    Paint,      // Direct base layer output (no normalization)
    Harmonic,   // Base + harmonics with normalization
    Spline      // Direct spline evaluation
};

/**
 * LayeredTransferFunction - Multi-layer waveshaping state container
 *
 * Stores UI state for transfer function editing: base layer (wavetable), harmonic
 * coefficients, and spline anchors. The renderer (SeamlessTransferFunction) uses
 * this state to generate production-ready LUTs at 25Hz for audio processing.
 *
 * Architecture:
 *   UI State → Renderer (25Hz) → Production LUT → Audio Thread
 *
 * Mixing formula (computed on-demand by renderer):
 *   UnNormMix[i] = wavetableCoeff × Base[i] + Σ(harmonicCoeff[n] × Harmonic_n[i])
 *   where wavetableCoeff = coefficients[0]
 *
 * Normalization (applied by renderer, not stored here):
 *   maxAbsValue = max(abs(UnNormMix[i]))
 *   normalizationScalar = 1.0 / maxAbsValue
 *   Output[i] = normalizationScalar × UnNormMix[i]
 *
 * Critical: Layers (baseTable, harmonic evaluations) are NEVER modified by normalization.
 * This allows seamless coefficient mixing in CurveEditorController.
 *
 * THREADING CONTRACT
 * ==================
 * - Message Thread: All mutations (setters, baking, mode changes)
 * - Worker Thread:  Read-only via const methods, getVersion() polling
 * - Audio Thread:   Read-only via evaluateForRendering(), applyTransferFunction()
 *
 * VERSION TRACKING INVARIANT
 * ==========================
 * Every mutation MUST increment the version counter via incrementVersionIfNotBatching().
 * Use public setters (setBaseLayerValue, setCoefficient, setSplineAnchors, etc.) or
 * BatchUpdateGuard for multiple mutations. Direct layer mutation is PROHIBITED
 * (const references only via getHarmonicLayer()/getSplineLayer()).
 *
 * MEMORY ORDERING
 * ===============
 * - Reads:  memory_order_acquire (visibility of prior writes)
 * - Writes: memory_order_release (visibility to readers)
 */
class LayeredTransferFunction {
  public:
    LayeredTransferFunction(int tableSize, double minVal, double maxVal);

    // Layer Access

    // Base layer (user-drawn wavetable)
    double getBaseLayerValue(int index) const;
    void setBaseLayerValue(int index, double value);
    void clearBaseLayer(); // Set all base values to 0.0

    // Harmonic layer (read-only access - use setCoefficient/setHarmonicCoefficients to mutate)
    const HarmonicLayer& getHarmonicLayer() const;

    // Spline layer (read-only access - use setSplineAnchors/clearSplineAnchors to mutate)
    const SplineLayer& getSplineLayer() const;

    // Coefficient access (includes WT mix at index 0 + harmonics at indices 1..N)
    void setCoefficient(int index, double value);
    double getCoefficient(int index) const;
    int getNumCoefficients() const {
        return static_cast<int>(coefficients.size());
    }

    /**
     * Compute composite value on-demand at specific index
     *
     * Computes: normScalar * (wtCoeff * baseValue + harmonicValue)
     *
     * This method computes the composite on-demand from UI state (base layer + harmonics).
     * Used for baking operations and legacy interpolation methods.
     *
     * Thread-safe: Can be called from any thread (reads atomics with acquire ordering)
     *
     * @param index Table index [0, tableSize)
     * @return Composite value with normalization applied, or 0.0 if index out of bounds
     */
    double computeCompositeAt(int index) const;

    // Normalization scalar (read-only access)
    double getNormalizationScalar() const {
        return normalizationScalar.load(std::memory_order_acquire);
    }

    // Normalization Control
    //
    // NORMALIZATION PATTERNS
    //
    // Normalization ensures transfer function output stays within [-1, 1] range
    // by scaling the composite curve (base + harmonics) by a computed scalar.
    //
    // ────────────────────────────────────────────────────────────────────────
    // WHEN NORMALIZATION HAPPENS
    // ────────────────────────────────────────────────────────────────────────
    //
    // 1. Harmonic mode (every worker render):
    //    - Worker scans 16,384 points: maxAbs = max(|wtCoeff*base + harmonics|)
    //    - Computes: normScalar = 1.0 / maxAbs
    //    - Renders LUT: output[i] = normScalar * (wtCoeff*base[i] + harmonics[i])
    //    - Cost: ~1-2ms for full scan + render
    //
    // 2. Paint mode (after paint stroke ends):
    //    - Controller calls updateNormalizationScalar()
    //    - Scans updated base layer: maxAbs = max(|base[i]|)
    //    - Caches scalar for next render
    //    - Cost: ~0.5ms (scan only, no render)
    //
    // 3. Baking operations (before capturing composite):
    //    - bakeHarmonicsToBase() calls updateNormalizationScalar()
    //    - Ensures baked values match what user sees on screen
    //    - computeCompositeAt() uses cached scalar
    //    - Cost: ~0.5ms (included in 5ms baking time)
    //
    // ────────────────────────────────────────────────────────────────────────
    // FREEZING DURING PAINT STROKES
    // ────────────────────────────────────────────────────────────────────────
    //
    // PROBLEM: Without freezing, curve "shrinks" as you paint
    //   User paints large peak → max increases → normScalar decreases → entire
    //   curve scales down → visually confusing (curve shifts while painting)
    //
    // SOLUTION: Freeze normalization during active stroke
    //   1. BEFORE stroke: updateNormalizationScalar() (cache current max)
    //   2. START stroke: setPaintStrokeActive(true) (freeze scalar)
    //   3. DURING stroke: Add points (max grows, but scalar stays frozen)
    //   4. END stroke: setPaintStrokeActive(false) (worker re-scans on next render)
    //
    // RESULT: Curve grows naturally without shrinking
    //   - New points appear at painted positions
    //   - Existing points stay fixed
    //   - After stroke ends, worker renormalizes once (smooth transition)
    //
    // ────────────────────────────────────────────────────────────────────────
    // CACHING STRATEGY
    // ────────────────────────────────────────────────────────────────────────
    //
    // normalizationScalar (atomic<double>):
    //   - Stores result of last scan (1.0 / maxAbsValue)
    //   - Read by: computeCompositeAt(), evaluateBaseAndHarmonics()
    //   - Written by: updateNormalizationScalar(), worker thread
    //   - Memory ordering: acquire/release (lock-free, thread-safe)
    //
    // Invalidation triggers:
    //   - Paint stroke end: Worker re-scans on next 25Hz poll
    //   - Harmonic slider drag: Worker re-scans every render
    //   - Mode switch: Worker re-scans in new mode
    //
    // Cache lifetime:
    //   - Harmonic mode: ~40ms (25Hz worker updates)
    //   - Paint mode: Until next stroke (could be minutes)
    //   - Frozen: Indefinite (until setPaintStrokeActive(false))
    //
    // ────────────────────────────────────────────────────────────────────────
    // API USAGE PATTERNS
    // ────────────────────────────────────────────────────────────────────────
    //
    // Pattern 1: Paint stroke (message thread)
    //   updateNormalizationScalar();        // Cache before stroke
    //   setPaintStrokeActive(true);          // Freeze
    //   for (each mouse point)
    //     setBaseLayerValue(i, value);       // Add points (max grows)
    //   setPaintStrokeActive(false);         // Un-freeze (worker re-scans)
    //
    // Pattern 2: Baking (message thread)
    //   updateNormalizationScalar();         // Ensure correct scalar
    //   BatchUpdateGuard guard(ltf);
    //   for (i = 0; i < 16384; ++i)
    //     setBaseLayerValue(i, computeCompositeAt(i));  // Uses cached scalar
    //
    // Pattern 3: Worker rendering (worker thread)
    //   if (paintStrokeActive)
    //     normScalar = getFrozenScalar();    // Use cached value
    //   else
    //     normScalar = computeFreshScalar(); // Re-scan
    //   renderLUT(normScalar);
    //
    /**
     * Compute and cache normalization scalar explicitly
     *
     * Scans the entire composite (base + harmonics) to find max absolute value,
     * then caches the normalization scalar (1.0 / max).
     *
     * Use this BEFORE baking operations to ensure baked values are properly normalized.
     * The cached scalar will be used by computeCompositeAt() until the next call.
     *
     * Thread-safe: Can be called from message thread (before baking) or worker thread (during rendering).
     */
    void updateNormalizationScalar();

    /**
     * Set paint stroke active state (message thread only)
     *
     * When true, renderer will use the frozen normalization scalar instead of
     * recomputing it. This prevents visual shifting during paint strokes.
     *
     * Call updateNormalizationScalar() BEFORE setting this to true to cache the
     * correct scalar for the paint stroke.
     *
     * @param active If true, freeze normalization during rendering; if false, compute fresh
     */
    void setPaintStrokeActive(bool active);

    /**
     * Check if paint stroke is currently active (message thread only)
     *
     * Used by renderer to determine whether to use frozen scalar or compute fresh.
     *
     * @return true if paint stroke active (use frozen scalar), false otherwise
     */
    bool isPaintStrokeActive() const;

    /**
     * Set rendering mode (determines evaluation path for LUT rendering)
     *
     * Paint    → Direct base layer (no normalization)
     * Harmonic → Base + harmonics with normalization
     * Spline   → Direct spline evaluation
     *
     * Thread-safe: Can be called from any thread
     *
     * @param mode The rendering mode to use
     */
    void setRenderingMode(RenderingMode mode);

    /**
     * Get current rendering mode
     *
     * Thread-safe: Can be called from any thread
     *
     * @return Current rendering mode
     */
    RenderingMode getRenderingMode() const;

    // Harmonic Layer Baking

    // Constants
    static constexpr int NUM_HARMONICS = 40;
    static constexpr int NUM_HARMONIC_COEFFICIENTS = NUM_HARMONICS + 1; // wtMix + h1..h40
    static constexpr double HARMONIC_EPSILON = 1e-6;                    // ~-120dB threshold for "effectively zero"

    /**
     * Check if any harmonic coefficients are non-zero
     *
     * Used for no-op optimization before baking.
     * Checks coefficients[1..40] (harmonics only, not WT mix at [0]).
     *
     * @return true if any harmonic amplitude exceeds HARMONIC_EPSILON
     */
    bool hasNonZeroHarmonics() const;

    /**
     * Bake harmonic layer into base layer and reset harmonics to zero
     *
     * This captures the current composite curve (base + harmonics) and writes it to base layer.
     * After baking:
     *   - Base layer contains the composite values (visually identical curve)
     *   - All harmonic coefficients are set to zero
     *   - WT mix coefficient remains at its current value
     *   - Normalization scalar is preserved (not reset)
     *
     * @return true if baking occurred (harmonics were non-zero), false if no-op
     *
     * THREAD SAFETY: Call from message thread only. The baking writes to base layer
     * and increments the version counter, triggering the renderer to generate a new
     * production LUT at the next 25Hz poll.
     */
    bool bakeHarmonicsToBase();

    /**
     * Bake composite layer (base + harmonics) into base layer and reset harmonics
     *
     * This captures the current composite curve (base + harmonics with normalization)
     * and writes it to base layer. After baking:
     *   - Base layer contains the composite values (visually identical curve)
     *   - All harmonic coefficients are set to zero
     *   - WT mix coefficient is set to 1.0 (enables base layer)
     *   - Normalization scalar will be recalculated by renderer
     *
     * THREAD SAFETY: Call from message thread only. Increments version counter
     * to trigger renderer update at next 25Hz poll.
     */
    void bakeCompositeToBase();

    /**
     * Bake spline layer into base layer and reset harmonics
     *
     * This evaluates the current spline curve at all table indices and writes
     * the values to the base layer. After baking:
     *   - Base layer contains the spline values (visually identical curve)
     *   - All harmonic coefficients are set to zero
     *   - WT mix coefficient is set to 1.0 (enables base layer)
     *   - Normalization scalar will be recalculated by renderer
     *
     * Use this when exiting Spline mode to preserve the edited spline shape.
     *
     * THREAD SAFETY: Call from message thread only. Increments version counter
     * to trigger renderer update at next 25Hz poll.
     */
    void bakeSplineToBase();

    /**
     * Get current harmonic coefficients for undo/redo
     *
     * @return array: [wtMix, h1, h2, ..., h40] (41 values)
     */
    std::array<double, NUM_HARMONIC_COEFFICIENTS> getHarmonicCoefficients() const;

    /**
     * Set all harmonic coefficients at once (for undo)
     *
     * @param coeffs Input array: [wtMix, h1, h2, ..., h40] (41 values)
     */
    void setHarmonicCoefficients(const std::array<double, NUM_HARMONIC_COEFFICIENTS>& coeffs);

    // Utilities (same API as TransferFunction for compatibility)
    int getTableSize() const {
        return tableSize;
    }
    double getMinSignalValue() const {
        return minValue;
    }
    double getMaxSignalValue() const {
        return maxValue;
    }

    /**
     * Map table index to normalized position x ∈ [minValue, maxValue]
     */
    double normalizeIndex(int index) const;

    // Processing (reads composite - thread-safe)

    /**
     * Apply transfer function to input sample
     * Uses composite table with interpolation
     */
    double applyTransferFunction(double x) const;

    /**
     * Evaluate base layer + harmonics explicitly (ignores spline layer)
     *
     * Used by SplineFitter to read the normalized composite when entering spline mode.
     * This ensures we fit the correct normalized curve, not the stale spline layer.
     *
     * CRITICAL: normalizationScalar is NOT reset when entering spline mode, so this
     * returns the properly normalized values that the user sees on screen.
     *
     * @param x Input value in [minValue, maxValue]
     * @return base + harmonics (with current normalization applied)
     */
    double evaluateBaseAndHarmonics(double x) const;

    /**
     * Evaluate transfer function for rendering based on current mode
     *
     * This is the primary evaluation method used by LUT renderer. It routes to the
     * appropriate evaluation path based on current rendering mode:
     *
     * Paint Mode:
     *   - Returns: wtCoeff * baseValue
     *   - No normalization applied
     *   - Harmonics assumed baked into base layer
     *
     * Harmonic Mode:
     *   - Returns: normScalar * (wtCoeff * baseValue + harmonicValue)
     *   - Normalization applied to composite
     *   - Evaluates both base and harmonic layers
     *
     * Spline Mode:
     *   - Returns: splineLayer.evaluate(x)
     *   - Direct spline evaluation
     *   - Bypasses base + harmonics entirely
     *
     * Thread-safe: Can be called from any thread (reads atomics with acquire ordering)
     *
     * @param x Normalized input in [minValue, maxValue]
     * @param normScalar Normalization scalar computed by renderer
     * @return Evaluated output value
     */
    double evaluateForRendering(double x, double normScalar) const;

    /**
     * Process block of samples in-place
     */
    void processBlock(double* samples, int numSamples) const;

    // Extrapolation Mode

    enum class ExtrapolationMode { Clamp, Linear, Mirror };

    void setExtrapolationMode(ExtrapolationMode mode) {
        extrapMode = mode;
        // Increment version to trigger LUT re-render with new extrapolation mode
        incrementVersionIfNotBatching();
    }

    ExtrapolationMode getExtrapolationMode() const {
        return extrapMode;
    }

    // Version Tracking (for seamless LUT updates)

    /**
     * Get current version of transfer function
     *
     * The version counter increments on every mutation to enable dirty detection.
     * Used by SeamlessTransferFunction to trigger asynchronous LUT rendering.
     *
     * NOTE: Version is NOT serialized - it's a runtime-only dirty tracking mechanism.
     *
     * @return Current version number
     */
    uint64_t getVersion() const {
        return versionCounter.load(std::memory_order_acquire);
    }

    /**
     * Begin batch update (defer version counter increment until endBatchUpdate)
     *
     * Use this to group multiple mutations into a single version increment.
     * Common use cases:
     *   - Baking operations (16,384 base layer writes → 1 version increment)
     *   - Preset loading (multiple state changes → 1 version increment)
     *
     * CRITICAL: Always use BatchUpdateGuard RAII wrapper instead of calling this directly.
     * Manual calls risk forgetting to call endBatchUpdate(), which breaks dirty tracking.
     *
     * Thread safety: Message thread only (not thread-safe)
     */
    void beginBatchUpdate();

    /**
     * End batch update (increment version counter once for all batched mutations)
     *
     * CRITICAL: Always use BatchUpdateGuard RAII wrapper instead of calling this directly.
     * Manual calls risk exception-safety issues and inconsistent state.
     *
     * Thread safety: Message thread only (not thread-safe)
     */
    void endBatchUpdate();

    /**
     * Set spline anchors (wrapper that increments version counter)
     *
     * Use this instead of getSplineLayer().setAnchors() to ensure version tracking works.
     */
    void setSplineAnchors(const std::vector<SplineAnchor>& anchors);

    /**
     * Clear all spline anchors (wrapper that increments version counter)
     */
    void clearSplineAnchors();

    /**
     * Set normalization scalar directly (for worker thread to restore frozen state)
     *
     * CRITICAL: Only used by LUT renderer to restore frozen normalization state.
     *
     * @param scalar The normalization scalar to set
     */
    void setNormalizationScalar(double scalar) {
        normalizationScalar.store(scalar, std::memory_order_release);
    }

    // Debugging

    /**
     * Get instance ID for debugging (identifies which LayeredTransferFunction instance this is)
     */
    int getInstanceId() const {
        return instanceId;
    }

    // Serialization

    juce::ValueTree toValueTree() const;
    void fromValueTree(const juce::ValueTree& vt);

  private:
    // Instance tracking for debugging
    static std::atomic<int> instanceCounter;
    int instanceId;

    int tableSize;
    double minValue, maxValue;

    // Layers (declare before tables since constructor initializes them first)
    std::unique_ptr<HarmonicLayer> harmonicLayer; // Harmonic basis function evaluator (no data ownership)
    std::unique_ptr<SplineLayer> splineLayer;     // NEW: Spline evaluator for spline mode

    // Coefficient storage (owned by LayeredTransferFunction)
    std::vector<double> coefficients; // [0] = WT mix, [1..N] = harmonics

    // Layers (NEVER normalized directly - preserved as-is)
    std::vector<std::atomic<double>> baseTable; // User-drawn wavetable

    // Normalization scalar (applied to mix, not to layers)
    // Cached value computed by updateNormalizationScalar() or set by renderer
    mutable std::atomic<double> normalizationScalar{1.0};

    // Paint stroke active flag (prevents normalization recomputation during strokes)
    // Message thread only - not atomic (single-threaded access)
    bool paintStrokeActive = false;

    // Rendering mode (determines evaluation path for LUT rendering)
    // This is the single source of truth for which evaluation path to use
    // Default to Harmonic mode to enable H1 linear blending with base layer
    std::atomic<int> renderingMode{static_cast<int>(RenderingMode::Harmonic)};

    /**
     * Version counter for dirty detection (used by SeamlessTransferFunction)
     *
     * INCREMENTS:
     *   - Single mutations: Version increments IMMEDIATELY after each mutation
     *     Example: ltf.setBaseLayerValue(0, 1.0); // version++
     *
     *   - Batched mutations: Version increments ONCE when batch ends
     *     Example:
     *       {
     *           BatchUpdateGuard guard(ltf);
     *           ltf.setBaseLayerValue(0, 1.0);  // No increment
     *           ltf.setBaseLayerValue(1, 2.0);  // No increment
     *           // ... 16,382 more writes ...
     *       } // version++ (ONCE for all 16,384 writes)
     *
     * MEMORY ORDERING:
     *   - Reads:  memory_order_acquire (ensures visibility of all prior writes)
     *   - Writes: memory_order_release (ensures all writes visible to readers)
     *   - Atomic operations guarantee lock-free, thread-safe dirty detection
     *
     * NOT SERIALIZED:
     *   - Version counter is runtime-only (never saved to presets/ValueTree)
     *   - Resets to 0 on plugin load, increments from there
     *   - Renderer only cares about "changed since last render", not absolute value
     *
     * USAGE PATTERNS:
     *   - Interactive edits (paint strokes, slider drags): Single mutations OK
     *     Version increments per-sample or per-drag are negligible overhead (~1ns)
     *
     *   - Expensive operations (baking, preset loading): Use BatchUpdateGuard
     *     Prevents 16,384+ version increments, improves clarity (not performance)
     *
     * CRITICAL INVARIANT:
     *   - All mutation methods MUST call incrementVersionIfNotBatching()
     *   - Never call versionCounter.fetch_add() directly (breaks batching)
     *   - See incrementVersionIfNotBatching() implementation below
     */
    std::atomic<uint64_t> versionCounter{0};

    // Batch update control (message thread only, not atomic)
    bool batchUpdateActive{false};

    /**
     * Increment version counter if not in batch mode
     *
     * Helper to consolidate version increment logic. All mutation methods should
     * call this instead of directly incrementing versionCounter.
     *
     * When batchUpdateActive = true:  No-op (defer increment until endBatchUpdate)
     * When batchUpdateActive = false: Increment immediately (default behavior)
     */
    void incrementVersionIfNotBatching() {
        if (!batchUpdateActive) {
            versionCounter.fetch_add(1, std::memory_order_release);
        }
    }

    ExtrapolationMode extrapMode = ExtrapolationMode::Clamp;

    // Interpolation helper (Catmull-Rom) with dual-path optimization
    // Two code paths selected by a single branch on extrapMode:
    //   - Fast path (Clamp): Direct clamped loads, relaxed memory order (default)
    //   - Slow path (Linear): Boundary checks + slope calculations for extrapolation
    double interpolateCatmullRom(double x) const;

    /**
     * Get base layer value at normalized position x using current interpolation mode
     *
     * This is a helper for evaluateForRendering() to read base layer values.
     *
     * @param x Normalized input in [minValue, maxValue]
     * @return Interpolated base layer value
     */
    double getBaseValueAt(double x) const;
};

/**
 * BatchUpdateGuard - RAII wrapper for batch updates
 *
 * Automatically calls beginBatchUpdate() on construction and endBatchUpdate() on destruction.
 * This ensures exception-safe cleanup and prevents forgetting to end the batch update.
 *
 * Usage:
 *   {
 *       BatchUpdateGuard guard(ltf);
 *       ltf.setBaseLayerValue(0, 1.0);  // No version increment
 *       ltf.setBaseLayerValue(1, 2.0);  // No version increment
 *       // ... thousands of mutations ...
 *   } // guard destructor increments version ONCE
 *
 * CRITICAL: Always use this instead of calling beginBatchUpdate/endBatchUpdate directly.
 */
class BatchUpdateGuard {
  public:
    explicit BatchUpdateGuard(LayeredTransferFunction& ltf) : ltf(ltf) {
        ltf.beginBatchUpdate();
    }

    ~BatchUpdateGuard() {
        ltf.endBatchUpdate();
    }

    // Non-copyable, non-movable (RAII guard pattern)
    BatchUpdateGuard(const BatchUpdateGuard&) = delete;
    BatchUpdateGuard& operator=(const BatchUpdateGuard&) = delete;
    BatchUpdateGuard(BatchUpdateGuard&&) = delete;
    BatchUpdateGuard& operator=(BatchUpdateGuard&&) = delete;

  private:
    LayeredTransferFunction& ltf;
};

} // namespace dsp_core
