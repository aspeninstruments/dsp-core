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
 * LaneMixer - Dynamic-count mixer for arbitrary waveshapes (up to MAX_LANES).
 *
 * Replaces the single-layer model (LayeredTransferFunction) with a multi-lane
 * architecture where each lane holds an independent curve that can be any
 * waveshape (harmonic, painted, spline, equation, preset capture).
 *
 * Mixing formula:
 *   sum[i] = Sigma(lane[n].amplitude * lane[n].curveData[i])  for n = 0..activeLaneCount_-1
 *   output[i] = normalizationEnabled ? sum[i] / max(|sum|) : sum[i]
 *
 * Default initialization (11 lanes):
 *   Lane 0:    tanh(x),  amplitude=0.0  ("WT" base layer, equation mode)
 *   Lane 1:    T_1(x)=x, amplitude=1.0  (H1 -- linear passthrough)
 *   Lane 2-10: T_n(x),   amplitude=0.0  (H3,H5,...,H19 -- odd Chebyshev harmonics)
 *   All lanes: oddSymmetryEnabled=true
 *
 * This default produces y=x (clean passthrough).
 *
 * MEMORY: Only active lanes have populated curveData (128 KB each).
 * Default 13 lanes = 1.6 MB. MAX_LANES fully active = 12.8 MB.
 * Audio thread only sees the 128KB rendered sum LUT via triple buffering.
 *
 * THREADING CONTRACT
 * ==================
 * - Message Thread: All mutations (setLaneAmplitude, setLaneCurveData, addLane, removeLane, etc.)
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
    static constexpr int MAX_LANES = 100;
    static constexpr int TABLE_SIZE = 16384;
    static constexpr double MIN_VALUE = -1.0;
    static constexpr double MAX_VALUE = 1.0;
    static constexpr int MAX_HARMONIC_NUMBER = 200;

    // Extrapolation mode -- controls LUT boundary behavior in AudioEngine.
    // Defined here (not on LTF) so LaneMixer is the sole source of truth.
    enum class ExtrapolationMode { Clamp, Linear };

    LaneMixer();

    // ========================================================================
    // Lane Access
    // ========================================================================

    const Lane& getLane(int index) const;
    Lane& getMutableLane(int index);
    int getNumLanes() const { return activeLaneCount_; }
    int getActiveLaneCount() const { return activeLaneCount_; }

    // ========================================================================
    // Dynamic Lane Management (message thread only)
    // ========================================================================

    /**
     * Insert a new lane after insertAfterIndex (-1 = append at end).
     * Returns the new lane's index, or -1 if at MAX_LANES.
     */
    int addLane(int insertAfterIndex = -1);

    /**
     * Remove lane at index. Returns false if invalid or activeLaneCount_ <= 1.
     */
    bool removeLane(int index);

    // ========================================================================
    // Lane Identity
    // ========================================================================

    uint32_t getLaneId(int index) const;
    int findLaneById(uint32_t laneId) const;

    // ========================================================================
    // Version Counter Access (for Phase 9 audio-thread-safe writes)
    // ========================================================================

    std::atomic<uint64_t>& getVersionCounter() { return versionCounter_; }

    // ========================================================================
    // Lane Mutations (message thread only)
    // ========================================================================

    void setLaneAmplitude(int index, double amplitude);
    double getLaneAmplitude(int index) const;

    void setLaneContentType(int index, LaneContentType type);
    LaneContentType getLaneContentType(int index) const;

    void setLaneHarmonicNumber(int index, int harmonicNumber);
    int getLaneHarmonicNumber(int index) const;

    void setLaneOddSymmetryEnabled(int index, bool enabled);
    bool isLaneOddSymmetryEnabled(int index) const;

    /**
     * Set the curve data for a lane.
     * Data must be TABLE_SIZE doubles. Copies the data.
     */
    void setLaneCurveData(int index, const std::vector<double>& data);
    void setLaneCurveData(int index, const double* data, int size);

    /**
     * Set a single sample in a lane's curve data.
     * Used by paint mode for per-sample writes during brush strokes.
     * Does NOT increment version -- caller should use beginBatchUpdate()/endBatchUpdate()
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
     * Fill a lane with tanh(2x). Used by legacy migration and harmonic-0 paths.
     */
    void fillLaneWithTanh2x(int index);

    // ========================================================================
    // Mixing
    // ========================================================================

    /**
     * Compute the weighted sum of all lanes into an output buffer.
     *
     * sum[i] = Sigma(lane[n].amplitude * lane[n].curveData[i])
     *
     * If normalization is enabled, the output is scaled so max(|output|) = 1.0.
     *
     * @param outputBuffer Pre-allocated buffer of at least TABLE_SIZE doubles
     * @param size Number of output samples (must be <= TABLE_SIZE)
     */
    void computeSum(double* outputBuffer, int size) const;

    /**
     * Evaluate the mixed sum at a normalized position x in [-1, 1].
     * Uses linear interpolation on the internally maintained sum cache.
     *
     * Note: Call computeSum() first to update the internal cache, or use
     * this only after the sum has been computed at least once.
     */
    double evaluateSumAt(double x) const;

    // ========================================================================
    // Extrapolation Mode
    // ========================================================================

    void setExtrapolationMode(ExtrapolationMode mode);
    ExtrapolationMode getExtrapolationMode() const { return extrapolationMode_; }

    // ========================================================================
    // Convenience Accessors
    // ========================================================================

    /**
     * Returns all active lane amplitudes as a vector.
     * Useful for snapshot capture and bulk reads.
     */
    std::vector<double> getAmplitudes() const;

    /**
     * Zero out a lane's curve data and increment version.
     */
    void clearLaneCurveData(int index);

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
     * Maps: coefficients[0..40] -> lane amplitudes, BaseLayer -> lane 0 curve data,
     * lanes 1-40 filled with Chebyshev harmonics, normalization flag preserved.
     *
     * Does NOT handle editingMode/presetPath/equationText (plugin-level metadata).
     */
    void fromLegacyLTFValueTree(const juce::ValueTree& ltfVT);

    /**
     * Migrate from legacy LTF format with editing mode context.
     *
     * @param wasHarmonicMode  Whether the saved editing mode was Harmonic
     * @param oddSymmetryWasEnabled  Whether odd symmetry was enabled system-wide
     *
     * Case B (harmonic + odd symmetry): Lane 0 = base, odd harmonics only (21 lanes)
     * Case B (harmonic, no symmetry): Lane 0 + H1-H39 (41 lanes)
     * Case C (non-harmonic): Lane 0 = saved state, Lanes 1-12 = default H1-H12 (13 lanes)
     */
    void fromLegacyLTFValueTree(const juce::ValueTree& ltfVT,
                                 bool wasHarmonicMode,
                                 bool oddSymmetryWasEnabled);

    // ========================================================================
    // Utilities
    // ========================================================================

    int getTableSize() const { return TABLE_SIZE; }
    double normalizeIndex(int index) const;

  private:
    std::array<Lane, MAX_LANES> lanes_;
    int activeLaneCount_ = 0;
    uint32_t nextLaneId_ = 0;

    ExtrapolationMode extrapolationMode_ = ExtrapolationMode::Clamp;

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
    bool isValidIndex(int index) const { return index >= 0 && index < activeLaneCount_; }

    int findNextUnusedHarmonicNumber() const;
    void shiftLanesRight(int fromIndex);
    void shiftLanesLeft(int fromIndex);
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
