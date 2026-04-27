#include "MorphGesture.h"
#include <algorithm>
#include <cmath>

namespace dsp_core {

GestureSample AnchorMorphGesture::sampleAt(double t) const {
    if (deltas.empty()) {
        return {0.0, 0.0};
    }
    if (deltas.size() == 1) {
        return deltas.front();
    }

    const double clamped = std::clamp(t, 0.0, 1.0);
    const double scaled = clamped * static_cast<double>(deltas.size() - 1);
    const auto i0 = static_cast<size_t>(std::floor(scaled));
    const auto i1 = std::min(i0 + 1, deltas.size() - 1);
    const double frac = scaled - static_cast<double>(i0);

    const auto& a = deltas[i0];
    const auto& b = deltas[i1];
    return {a.dx + (b.dx - a.dx) * frac, a.dy + (b.dy - a.dy) * frac};
}

AnchorMorphGesture AnchorMorphGesture::mirrored() const {
    AnchorMorphGesture out;
    out.deltas.reserve(deltas.size());
    for (const auto& s : deltas) {
        out.deltas.push_back({-s.dx, -s.dy});
    }
    return out;
}

juce::ValueTree AnchorMorphGesture::toValueTree() const {
    juce::ValueTree vt("Gesture");
    const juce::MemoryBlock raw(deltas.data(), deltas.size() * sizeof(GestureSample));
    vt.setProperty("samples", juce::var(raw), nullptr);
    vt.setProperty("count", static_cast<int>(deltas.size()), nullptr);
    return vt;
}

AnchorMorphGesture AnchorMorphGesture::fromValueTree(const juce::ValueTree& vt) {
    AnchorMorphGesture out;
    if (!vt.isValid() || vt.getType().toString() != "Gesture") {
        return out;
    }

    const auto* raw = vt.getProperty("samples").getBinaryData();
    if (raw == nullptr) {
        return out;
    }

    const auto count = static_cast<size_t>(static_cast<int>(vt.getProperty("count", 0)));
    const size_t expectedBytes = count * sizeof(GestureSample);
    if (raw->getSize() < expectedBytes || count == 0) {
        return out;
    }

    out.deltas.resize(count);
    std::memcpy(out.deltas.data(), raw->getData(), expectedBytes);
    return out;
}

} // namespace dsp_core
