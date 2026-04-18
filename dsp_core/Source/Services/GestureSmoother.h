#pragma once
#include "../MorphGesture.h"
#include <vector>

namespace dsp_core {

/**
 * GestureSmoother - resample raw 2D drag samples into a fixed-length,
 * arc-length-uniform AnchorMorphGesture using Catmull-Rom interpolation.
 *
 * Input samples are deltas from the anchor's home position; the smoother
 * does not interpret coordinate space (any consistent unit works).
 *
 * PCHIP is rejected here because gesture paths can loop or double back
 * (non-monotone in x). Catmull-Rom over arc-length parameter is robust
 * to arbitrary 2D point clouds.
 */
class GestureSmoother {
  public:
    static constexpr int kDefaultOutputCount = 64;

    static AnchorMorphGesture smooth(const std::vector<GestureSample>& rawSamples,
                                     int outputCount = kDefaultOutputCount);

  private:
    GestureSmoother() = delete;
};

} // namespace dsp_core
