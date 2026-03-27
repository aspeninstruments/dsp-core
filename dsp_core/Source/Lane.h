#pragma once

#include "SplineTypes.h"
#include <juce_core/juce_core.h>
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
 * Lane 0 defaults to tanh(2x) ("WT" base layer).
 * Lanes 1-40 default to Chebyshev harmonics T_1(x) through T_40(x).
 * All lanes are interchangeable and can hold any content type.
 *
 * THREADING: Message thread only. Audio thread reads the rendered sum LUT
 * via SeamlessTransferFunction's triple buffer, never individual lanes.
 */
struct Lane {
    std::vector<double> curveData; // TABLE_SIZE points — the lane's waveshape

    double amplitude = 0.0; // Mix amount [-1, 1], negative = phase inversion

    LaneContentType contentType = LaneContentType::Harmonic;
    int harmonicNumber = 0; // Which harmonic (0 = WT/base, 1-40 = T_n). Only meaningful for Harmonic type.

    // Mode-specific state (only populated for the relevant content type)
    std::vector<SplineAnchor> splineAnchors;
    juce::String equationText;
    juce::String presetSourcePath;

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
        return std::abs(amplitude) > 1e-12 && !curveData.empty();
    }
};

} // namespace dsp_core
