#pragma once

#include "Lane.h"
#include "HarmonicLayer.h"
#include <juce_core/juce_core.h>
#include <juce_data_structures/juce_data_structures.h>
#include <array>
#include <atomic>
#include <memory>

namespace dsp_core {

/**
 * LaneMixer - 41-lane mixer for arbitrary waveshapes.
 *
 * Replaces the single-layer model (LayeredTransferFunction) with a multi-lane
 * architecture where each lane holds an independent curve that can be any
 * waveshape (harmonic, painted, spline, equation, preset capture).
 *
 * Mixing formula:
 *   sum[i] = Σ(lane[n].amplitude × lane[n].curveData[i])  for n = 0..NUM_LANES-1
 *   output[i] = normalizationEnabled ? sum[i] / max(|sum|) : sum[i]
 *
 * Default initialization:
 *   Lane 0:  tanh(2x), amplitude=0.0  ("WT" base layer)
 *   Lane 1:  T₁(x)=x, amplitude=1.0  (H1 — linear passthrough)
 *   Lane 2-40: T_n(x),  amplitude=0.0  (H2-H40 — Chebyshev harmonics)
 *
 * This default produces y=x (clean passthrough), identical to current behavior.
 *
 * MEMORY: 41 × 16384 × 8 bytes = 5.2 MB (message thread only)
 * Audio thread only sees the 128KB rendered sum LUT via triple buffering.
 *
 * THREADING CONTRACT
 * ==================
 * - Message Thread: All mutations (setLaneAmplitude, setLaneCurveData, etc.)
 * - Worker Thread:  Read-only via computeSum(), getVersion() polling
 * - Audio Thread:   Never accesses LaneMixer directly (reads rendered LUT)
 *
 * VERSION TRACKING
 * ================
 * Every mutation increments the version counter (or defers via BatchUpdateGuard).
 * SeamlessTransferFunction polls this at 20Hz to detect changes and render new LUTs.
 */
class LaneMixer {
  public:
    static constexpr int NUM_LANES = 41;
    static constexpr int TABLE_SIZE = 16384;
    static constexpr double MIN_VALUE = -1.0;
    static constexpr double MAX_VALUE = 1.0;

    LaneMixer();

    // ========================================================================
    // Lane Access
    // ========================================================================

    const Lane& getLane(int index) const;
    Lane& getMutableLane(int index);
    int getNumLanes() const { return NUM_LANES; }

    // ========================================================================
    // Lane Mutations (message thread only)
    // ========================================================================

    void setLaneAmplitude(int index, double amplitude);
    double getLaneAmplitude(int index) const;

    void setLaneContentType(int index, LaneContentType type);
    LaneContentType getLaneContentType(int index) const;

    void setLaneHarmonicNumber(int index, int harmonicNumber);
    int getLaneHarmonicNumber(int index) const;

    /**
     * Set the curve data for a lane.
     * Data must be TABLE_SIZE doubles. Copies the data.
     */
    void setLaneCurveData(int index, const std::vector<double>& data);
    void setLaneCurveData(int index, const double* data, int size);

    /**
     * Set a single sample in a lane's curve data.
     * Used by paint mode for per-sample writes during brush strokes.
     * Does NOT increment version — caller should use beginBatchUpdate()/endBatchUpdate()
     * or call incrementVersion() after a series of writes.
     */
    void setLaneCurveValue(int laneIndex, int sampleIndex, double value);

    /**
     * Explicitly increment the version counter.
     * Used after a batch of setLaneCurveValue() calls outside of BatchUpdateGuard.
     */
    void incrementVersion();

    /**
     * Fill a lane's curve with the Chebyshev polynomial T_n(x).
     * Uses precomputed basis functions from HarmonicLayer.
     */
    void fillLaneWithHarmonic(int index, int harmonicNumber);

    /**
     * Fill lane 0 with tanh(2x) — the default "WT" base layer curve.
     */
    void fillLaneWithTanh2x(int index);

    // ========================================================================
    // Mixing
    // ========================================================================

    /**
     * Compute the weighted sum of all lanes into an output buffer.
     *
     * sum[i] = Σ(lane[n].amplitude × lane[n].curveData[i])
     *
     * If normalization is enabled, the output is scaled so max(|output|) = 1.0.
     *
     * @param outputBuffer Pre-allocated buffer of at least TABLE_SIZE doubles
     * @param size Number of output samples (must be <= TABLE_SIZE)
     */
    void computeSum(double* outputBuffer, int size) const;

    /**
     * Evaluate the mixed sum at a normalized position x ∈ [-1, 1].
     * Uses Catmull-Rom interpolation on the internally maintained sum cache.
     *
     * Note: Call computeSum() first to update the internal cache, or use
     * this only after the sum has been computed at least once.
     */
    double evaluateSumAt(double x) const;

    // ========================================================================
    // Normalization
    // ========================================================================

    void setNormalizationEnabled(bool enabled);
    bool isNormalizationEnabled() const { return normalizationEnabled_; }

    // ========================================================================
    // Version Tracking
    // ========================================================================

    uint64_t getVersion() const {
        return versionCounter_.load(std::memory_order_acquire);
    }

    void beginBatchUpdate();
    void endBatchUpdate();

    // ========================================================================
    // Initialization
    // ========================================================================

    /**
     * Reset all lanes to default state (harmonics with default amplitudes).
     */
    void resetToDefaults();

    // ========================================================================
    // Serialization
    // ========================================================================

    juce::ValueTree toValueTree() const;
    void fromValueTree(const juce::ValueTree& vt);

    /**
     * Migrate from legacy LayeredTransferFunction ValueTree format.
     *
     * Maps: coefficients[0..40] → lane amplitudes, BaseLayer → lane 0 curve data,
     * lanes 1-40 filled with Chebyshev harmonics, normalization flag preserved.
     *
     * Does NOT handle editingMode/presetPath/equationText (plugin-level metadata).
     */
    void fromLegacyLTFValueTree(const juce::ValueTree& ltfVT);

    // ========================================================================
    // Utilities
    // ========================================================================

    int getTableSize() const { return TABLE_SIZE; }
    double normalizeIndex(int index) const;

  private:
    std::array<Lane, NUM_LANES> lanes_;
    bool normalizationEnabled_ = true;

    // Precomputed harmonic basis functions (shared across all harmonic lanes)
    std::unique_ptr<HarmonicLayer> harmonicLayer_;

    // Version tracking (same pattern as LayeredTransferFunction)
    std::atomic<uint64_t> versionCounter_{0};
    bool batchUpdateActive_ = false;

    void incrementVersionIfNotBatching() {
        if (!batchUpdateActive_) {
            versionCounter_.fetch_add(1, std::memory_order_release);
        }
    }

    void initializeDefaults();
    bool isValidIndex(int index) const { return index >= 0 && index < NUM_LANES; }
};

/**
 * LaneMixerBatchUpdateGuard - RAII wrapper for batch updates.
 *
 * Groups multiple mutations into a single version increment.
 */
class LaneMixerBatchUpdateGuard {
  public:
    explicit LaneMixerBatchUpdateGuard(LaneMixer& mixer) : mixer_(mixer) {
        mixer_.beginBatchUpdate();
    }

    ~LaneMixerBatchUpdateGuard() {
        mixer_.endBatchUpdate();
    }

    LaneMixerBatchUpdateGuard(const LaneMixerBatchUpdateGuard&) = delete;
    LaneMixerBatchUpdateGuard& operator=(const LaneMixerBatchUpdateGuard&) = delete;

  private:
    LaneMixer& mixer_;
};

} // namespace dsp_core
