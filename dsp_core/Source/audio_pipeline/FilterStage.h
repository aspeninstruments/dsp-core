#pragma once

#include "Tanh2xLUT.h"

namespace dsp_core::audio_pipeline {

/**
 * Polymorphic base for a one-pole virtual-analog filter stage.
 *
 * Each subclass implements one nonlinear integrator (dy/dt) given the stage's
 * input, its current state, and the prewarped cutoff g. A VirtualAnalogFilter-
 * Stage wrapper owns N of these and cascades them with RK4 + feedback.
 *
 * This is the counterpart to the templated VirtualAnalogFilterImpl<StageOp>:
 * runtime polymorphism allows heterogeneous cascades (e.g., Ladder→OTA→Ladder
 * →OTA) but incurs one vcall per stage per RK4 step.
 */
class FilterStage {
  public:
    virtual ~FilterStage() = default;

    /**
     * @param in     signal entering this stage (after prior stage's stepped output)
     * @param state  this stage's current integrator state y_i
     * @param gVal   2*pi*cutoff at this RK4 sub-step
     * @return       dy_i/dt (the wrapper multiplies by T = 1/fs)
     */
    virtual double computeDelta(double in, double state, double gVal) const noexcept = 0;
};

/** Moog-style: nonlinearity wraps (input - state). */
class LadderFilterStage final : public FilterStage {
  public:
    double computeDelta(double in, double state, double gVal) const noexcept override {
        // g_tanh2xLUT.lookup(z) = tanh(2*z), so this is gVal * tanh(2*(in - state)).
        return gVal * g_tanh2xLUT.lookup(in - state);
    }
};

/** OTA-style: each side saturated independently, then differenced. */
class OTAFilterStage final : public FilterStage {
  public:
    double computeDelta(double in, double state, double gVal) const noexcept override {
        return gVal * (g_tanh2xLUT.lookup(in) - g_tanh2xLUT.lookup(state));
    }
};

} // namespace dsp_core::audio_pipeline
