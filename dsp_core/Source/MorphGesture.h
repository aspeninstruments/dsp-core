#pragma once
#include <juce_data_structures/juce_data_structures.h>
#include <vector>

namespace dsp_core {

struct GestureSample {
    double dx = 0.0;
    double dy = 0.0;

    bool operator==(const GestureSample&) const = default;
};

/**
 * AnchorMorphGesture - A user-recorded 2D motion path stored as deltas
 * relative to an anchor's home position.
 *
 * deltas[0] is always (0, 0); deltas.back() is the captured endpoint.
 * Empty deltas (or fewer than 2 samples) means "no gesture".
 *
 * sampleAt(t) linearly interpolates between adjacent stored samples.
 * The morph knob's 0..1 value is the t input.
 */
struct AnchorMorphGesture {
    std::vector<GestureSample> deltas;

    bool empty() const {
        return deltas.size() < 2;
    }

    GestureSample sampleAt(double t) const;

    // Returns a copy with every delta negated (for symmetry-paired anchor).
    AnchorMorphGesture mirrored() const;

    juce::ValueTree toValueTree() const;
    static AnchorMorphGesture fromValueTree(const juce::ValueTree& vt);

    bool operator==(const AnchorMorphGesture&) const = default;
};

} // namespace dsp_core
