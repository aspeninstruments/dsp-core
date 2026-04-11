#pragma once

#include "Lane.h"
#include "HarmonicLayer.h"
#include <juce_core/juce_core.h>
#include <juce_data_structures/juce_data_structures.h>
#include <array>
#include <atomic>
#include <functional>
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
 *   Lane 0:    tanh(2x), amplitude=0.0  ("WT" base layer, equation mode)
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
    enum class ExtrapolationMode { Clamp, Linear, Mirror };

    // Mirror-wrap an index into [0, tableSize-1] by reflecting at boundaries.
    static inline int mirrorIndex(int i, int tableSize) {
        const int period = 2 * (tableSize - 1);
        int wrapped = ((i % period) + period) % period;
        return wrapped <= tableSize - 1 ? wrapped : period - wrapped;
    }

    // Mixer mode -- controls how lanes are combined into the output curve.
    //   Blend: Weighted sum of all lanes (existing behavior)
    //   Scan:  Wavetable-style sweep across lanes using scan position
    enum class MixerMode { Blend = 0, Scan = 1 };

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
     * Duplicate an existing lane and insert the copy immediately after it.
     * The new lane gets a fresh laneId and amplitude=0.0.
     * Adjusts scanPosition_ to maintain scan continuity.
     * Returns the new lane's index, or -1 if at MAX_LANES or invalid source.
     */
    int duplicateLane(int sourceLaneIndex);

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

    /**
     * Per-lane blend depth [-1, 1]. Modulated by the global blend amount in Blend mode:
     *     effectiveAmplitude = amplitude + blendAmount * blendDepth
     * Default 0 means the lane is inert with respect to the Morph knob in Blend mode.
     * Increments both mix and full version counters (amplitude-class change).
     * Out-of-range values are clamped to [-1, 1].
     */
    void setLaneBlendDepth(int index, double depth);
    double getLaneBlendDepth(int index) const;

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
     * Compute the weighted sum of all lanes into an output buffer (Blend mode).
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
     * Compute the scan output into an output buffer (Scan mode).
     *
     * Interpolates between adjacent lanes based on scan position.
     * Lane amplitudes are ignored — raw curve data is used directly.
     * Output is normalized to [-1, 1].
     *
     * @param outputBuffer Pre-allocated buffer of at least TABLE_SIZE doubles
     * @param size Number of output samples (must be <= TABLE_SIZE)
     */
    void computeScan(double* outputBuffer, int size) const;

    /**
     * Evaluate the mixed sum at a normalized position x in [-1, 1].
     * Uses linear interpolation on the internally maintained sum cache.
     *
     * Note: Call computeSum() first to update the internal cache, or use
     * this only after the sum has been computed at least once.
     */
    double evaluateSumAt(double x) const;

    // ========================================================================
    // Mixer Mode
    // ========================================================================

    void setMixerMode(MixerMode mode);
    MixerMode getMixerMode() const { return mixerMode_; }

    /**
     * Set the scan position for Scan mode.
     * Clamped to [0, 1]. 0 = first lane, 1 = last lane.
     * Increments version counter so the render pipeline picks up the change.
     */
    void setScanPosition(double position);
    double getScanPosition() const { return scanPosition_.load(std::memory_order_acquire); }

    /**
     * Global blend amount [0, 1] driving per-lane depth modulation in Blend mode.
     * Effective lane amplitude = amplitude + blendAmount * blendDepth (no per-lane clamp;
     * computeSum's post-sum normalize bounds the output). Increments both counters.
     * In Scan mode this should remain 0 — invariant: at most one of scanPosition/blendAmount
     * is non-zero at any time, enforced by PluginAudioProcessor::setMixerMode.
     */
    void setBlendAmount(double amount);
    double getBlendAmount() const { return blendAmount_.load(std::memory_order_acquire); }

    // ========================================================================
    // Extrapolation Mode
    // ========================================================================

    void setExtrapolationMode(ExtrapolationMode mode);
    ExtrapolationMode getExtrapolationMode() const { return extrapolationMode_; }

    // ========================================================================
    // Soft Clipping (unified with extrapolation — controls LUT input bounding)
    // ========================================================================

    void setSoftClipEnabled(bool enabled);
    bool getSoftClipEnabled() const { return softClipEnabled_; }

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

    /**
     * Get the mix-only version counter.
     * Incremented only for amplitude and scan position changes (cheap re-mix).
     * NOT incremented for curve data, structural, or content type changes.
     */
    uint64_t getMixVersion() const {
        return mixVersionCounter_.load(std::memory_order_acquire);
    }

    /**
     * Get a reference to the mix version counter atomic (for AutomationSlot).
     */
    std::atomic<uint64_t>& getMixVersionCounter() { return mixVersionCounter_; }

    /**
     * Set a callback invoked whenever the version counter increments.
     * Called from incrementVersionIfNotBatching() and endBatchUpdate().
     * Must be safe to call from the message thread (the only thread that mutates LaneMixer).
     */
    void setOnVersionChanged(std::function<void()> callback) {
        onVersionChanged_ = std::move(callback);
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

    MixerMode mixerMode_ = MixerMode::Blend;
    std::atomic<double> scanPosition_{0.0};
    std::atomic<double> blendAmount_{0.0};

    ExtrapolationMode extrapolationMode_ = ExtrapolationMode::Clamp;
    bool softClipEnabled_ = false;

    // Precomputed harmonic basis functions (shared across all harmonic lanes)
    std::unique_ptr<HarmonicLayer> harmonicLayer_;

    // Version tracking (same pattern as LayeredTransferFunction)
    std::atomic<uint64_t> versionCounter_{0};
    std::atomic<uint64_t> mixVersionCounter_{0};  // Incremented only for amplitude/scan changes
    bool batchUpdateActive_ = false;
    bool batchHasMixChange_ = false;  // Track if batch contains amplitude/scan changes
    std::function<void()> onVersionChanged_;  // Callback for event-driven rendering

    void incrementVersionIfNotBatching() {
        if (!batchUpdateActive_) {
            versionCounter_.fetch_add(1, std::memory_order_release);
            if (onVersionChanged_) onVersionChanged_();
        }
    }

    void incrementMixVersionIfNotBatching() {
        if (!batchUpdateActive_) {
            mixVersionCounter_.fetch_add(1, std::memory_order_release);
        } else {
            batchHasMixChange_ = true;
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
