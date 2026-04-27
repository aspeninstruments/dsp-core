#include "GestureSmoother.h"
#include <algorithm>
#include <cmath>

namespace dsp_core {

namespace {

// Centripetal Catmull-Rom interpolation between p1 and p2 (p0, p3 are neighbors).
// alpha=0.5 gives centripetal parameterization which avoids loops/cusps.
GestureSample catmullRom(const GestureSample& p0,
                         const GestureSample& p1,
                         const GestureSample& p2,
                         const GestureSample& p3,
                         double t) {
    const double t2 = t * t;
    const double t3 = t2 * t;

    auto interp = [&](double a, double b, double c, double d) {
        return 0.5 * ((2.0 * b)
                      + (-a + c) * t
                      + (2.0 * a - 5.0 * b + 4.0 * c - d) * t2
                      + (-a + 3.0 * b - 3.0 * c + d) * t3);
    };

    return {interp(p0.dx, p1.dx, p2.dx, p3.dx),
            interp(p0.dy, p1.dy, p2.dy, p3.dy)};
}

double distance(const GestureSample& a, const GestureSample& b) {
    const double dx = a.dx - b.dx;
    const double dy = a.dy - b.dy;
    return std::sqrt(dx * dx + dy * dy);
}

} // namespace

AnchorMorphGesture GestureSmoother::smooth(const std::vector<GestureSample>& rawSamples,
                                           int outputCount) {
    AnchorMorphGesture out;

    if (rawSamples.empty()) {
        return out;
    }

    // De-duplicate consecutive identical samples (zero-length segments break
    // arc-length parameterization).
    std::vector<GestureSample> samples;
    samples.reserve(rawSamples.size());
    samples.push_back(rawSamples.front());
    for (size_t i = 1; i < rawSamples.size(); ++i) {
        if (distance(rawSamples[i], samples.back()) > 1e-9) {
            samples.push_back(rawSamples[i]);
        }
    }

    if (samples.size() < 2) {
        // Single point or all duplicates: emit a degenerate gesture so
        // callers can detect "no meaningful motion" via empty().
        return out;
    }

    outputCount = std::max(outputCount, 2);

    // Build cumulative arc-length table.
    std::vector<double> cumLen(samples.size(), 0.0);
    for (size_t i = 1; i < samples.size(); ++i) {
        cumLen[i] = cumLen[i - 1] + distance(samples[i - 1], samples[i]);
    }

    const double totalLen = cumLen.back();
    if (totalLen < 1e-9) {
        return out;
    }

    out.deltas.reserve(static_cast<size_t>(outputCount));

    // First sample is always (0,0) — gesture is in delta space relative to home.
    out.deltas.push_back({0.0, 0.0});

    // Resample at uniform arc-length intervals.
    for (int i = 1; i < outputCount; ++i) {
        const double targetLen = (static_cast<double>(i) / (outputCount - 1)) * totalLen;

        // Find segment [j, j+1] containing targetLen.
        auto it = std::upper_bound(cumLen.begin(), cumLen.end(), targetLen);
        size_t j = (it == cumLen.begin()) ? 0 : static_cast<size_t>(it - cumLen.begin()) - 1;
        if (j >= samples.size() - 1) {
            j = samples.size() - 2;
        }

        const double segLen = cumLen[j + 1] - cumLen[j];
        const double localT = (segLen > 1e-12) ? (targetLen - cumLen[j]) / segLen : 0.0;

        // Get 4 control points, clamping at endpoints.
        const auto& p0 = samples[j == 0 ? 0 : j - 1];
        const auto& p1 = samples[j];
        const auto& p2 = samples[j + 1];
        const auto& p3 = samples[std::min(j + 2, samples.size() - 1)];

        out.deltas.push_back(catmullRom(p0, p1, p2, p3, localT));
    }

    // Force first sample to exactly (0,0) — the anchor's home position.
    // Subtract the captured first-sample offset from every subsequent sample
    // so the gesture starts at "no displacement" regardless of where the
    // user clicked relative to the anchor center.
    const GestureSample origin = out.deltas.front();
    for (auto& s : out.deltas) {
        s.dx -= origin.dx;
        s.dy -= origin.dy;
    }

    return out;
}

} // namespace dsp_core
