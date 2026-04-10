#pragma once

#include "SplineTypes.h"
#include <juce_core/juce_core.h>
#include <atomic>
#include <vector>

namespace dsp_core {

/**
 * LaneContentType - What kind of data a mixer lane holds.
 *
 * All lanes are structurally identical regardless of content type.
 * The content type determines how the lane's curveData was populated
 * and which metadata fields are meaningful.
 *
 * IMPORTANT: Do not reorder values - they are serialized as integers.
 */
enum class LaneContentType {
    Harmonic = 0, // Curve = Chebyshev T_n(x) for a specific harmonic number
    Paint = 1,    // Curve = freehand drawn
    Spline = 2,   // Curve = evaluated from spline anchors
    Equation = 3, // Curve = evaluated from math expression
    Preset = 4    // Curve = loaded from .tfunc file
};

/**
 * Lane - A single mixer lane holding an arbitrary waveshape.
 *
 * Each lane contains:
 *   - curveData: The waveshape itself (TABLE_SIZE points)
 *   - amplitude: How much this lane contributes to the mix [-1, 1]
 *   - contentType: What kind of data this lane holds
 *   - Mode-specific metadata (harmonicNumber, splineAnchors, equationText, etc.)
 *
 * Lane 0 defaults to tanh(x) ("WT" base layer, equation mode).
 * Lanes 1+ default to Chebyshev harmonics.
 * All lanes are interchangeable and can hold any content type.
 *
 * THREADING: Message thread only. Audio thread reads the rendered sum LUT
 * via SeamlessTransferFunction's triple buffer, never individual lanes.
 */
struct Lane {
    std::vector<double> curveData; // TABLE_SIZE points — the lane's waveshape

    std::atomic<double> amplitude{0.0}; // Mix amount [-1, 1], negative = phase inversion

    // Per-lane modulation depth from the global Blend Amount macro [-1, 1].
    // Effective amplitude in Blend mode = amplitude + blendAmount * blendDepth.
    // Default 0 means the lane is inert with respect to the Morph knob in Blend mode.
    std::atomic<double> blendDepth{0.0};

    LaneContentType contentType = LaneContentType::Harmonic;
    int harmonicNumber = 0; // Which harmonic (0 = WT/base, 1+ = T_n). Only meaningful for Harmonic type.

    // Mode-specific state (only populated for the relevant content type)
    std::vector<SplineAnchor> splineAnchors;
    juce::String equationText;
    juce::String presetSourcePath;

    bool oddSymmetryEnabled = false; // Per-lane odd symmetry enforcement (f(-x) = -f(x))

    uint32_t laneId = 0; // Stable identity, never reused

    Lane() = default;
    ~Lane() = default;

    // Move constructor — std::atomic is not movable, so load/store explicitly.
    // memory_order_relaxed is correct: shifting is message-thread-only, and the
    // version counter increment after the shift provides the release fence.
    Lane(Lane&& other) noexcept
        : curveData(std::move(other.curveData)),
          amplitude(other.amplitude.load(std::memory_order_relaxed)),
          blendDepth(other.blendDepth.load(std::memory_order_relaxed)),
          contentType(other.contentType),
          harmonicNumber(other.harmonicNumber),
          splineAnchors(std::move(other.splineAnchors)),
          equationText(std::move(other.equationText)),
          presetSourcePath(std::move(other.presetSourcePath)),
          oddSymmetryEnabled(other.oddSymmetryEnabled),
          laneId(other.laneId) {}

    Lane& operator=(Lane&& other) noexcept {
        curveData = std::move(other.curveData);
        amplitude.store(other.amplitude.load(std::memory_order_relaxed),
                        std::memory_order_relaxed);
        blendDepth.store(other.blendDepth.load(std::memory_order_relaxed),
                         std::memory_order_relaxed);
        contentType = other.contentType;
        harmonicNumber = other.harmonicNumber;
        splineAnchors = std::move(other.splineAnchors);
        equationText = std::move(other.equationText);
        presetSourcePath = std::move(other.presetSourcePath);
        oddSymmetryEnabled = other.oddSymmetryEnabled;
        laneId = other.laneId;
        return *this;
    }

    // Copy constructor/assignment — needed for snapshot capture/restore
    Lane(const Lane& other)
        : curveData(other.curveData),
          amplitude(other.amplitude.load(std::memory_order_relaxed)),
          blendDepth(other.blendDepth.load(std::memory_order_relaxed)),
          contentType(other.contentType),
          harmonicNumber(other.harmonicNumber),
          splineAnchors(other.splineAnchors),
          equationText(other.equationText),
          presetSourcePath(other.presetSourcePath),
          oddSymmetryEnabled(other.oddSymmetryEnabled),
          laneId(other.laneId) {}

    Lane& operator=(const Lane& other) {
        if (this != &other) {
            curveData = other.curveData;
            amplitude.store(other.amplitude.load(std::memory_order_relaxed),
                            std::memory_order_relaxed);
            blendDepth.store(other.blendDepth.load(std::memory_order_relaxed),
                             std::memory_order_relaxed);
            contentType = other.contentType;
            harmonicNumber = other.harmonicNumber;
            splineAnchors = other.splineAnchors;
            equationText = other.equationText;
            presetSourcePath = other.presetSourcePath;
            oddSymmetryEnabled = other.oddSymmetryEnabled;
            laneId = other.laneId;
        }
        return *this;
    }

    /**
     * Initialize curveData to a given size with all zeros.
     */
    void initCurveData(int tableSize) {
        curveData.assign(tableSize, 0.0);
    }

    /**
     * Check if this lane contributes to the mix.
     * A lane contributes if its amplitude is non-zero and it has curve data.
     */
    bool isActive() const {
        return std::abs(amplitude.load(std::memory_order_acquire)) > 1e-12 && !curveData.empty();
    }
};

} // namespace dsp_core
