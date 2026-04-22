#pragma once

#include "AudioProcessingStage.h"
#include <juce_audio_basics/juce_audio_basics.h>
#include <atomic>
#include <cmath>

namespace dsp_core::audio_pipeline {

/**
 * Unity-gain soft clipper with extended transition region.
 *
 * Shape: Linear from -a to a (slope = 1), quarter-sine transition to ±1.
 *
 * Mathematical Definition:
 * Let u = |x|, b = a + (1 - a) * (π/2)
 *
 * For u < a:  y = u                                    (linear, unity gain)
 * For a ≤ u < b: y = a + (1 - a) * sin(π/2 * (u - a) / (b - a))  (sine transition)
 * For u >= b: y = 1                                    (flat saturation)
 * Return: sign(x) * y
 *
 * Properties:
 * - C1 continuous (smooth first derivative at |x| = a and |x| = b)
 * - Unity gain below knee: completely transparent for |x| < a
 * - Derivative at |x| = b is zero (sine peak)
 * - Odd symmetry: f(-x) = -f(x)
 * - f(±b) = ±1 (saturation point is beyond 0 dBFS)
 *
 * Example with a = 0.95 (-0.45 dBFS knee):
 *   b = 0.95 + 0.05 * 1.571 = 1.029 (+0.25 dBFS)
 *   Signal must reach +0.25 dBFS input for output to hit 0 dBFS
 *
 * All constants are precomputed at construction for minimal audio thread overhead.
 */
class SoftClippingSolver {
  public:
    /**
     * Construct solver with transition point parameter.
     * @param a Knee threshold in (0, 1). Default 0.95 (~-0.45 dBFS).
     *          Smaller values create earlier, more aggressive limiting.
     */
    explicit SoftClippingSolver(double a = 0.95) {
        setTransitionPoint(a);
    }

    /**
     * Set the transition point (knee) parameter.
     * @param a Knee threshold in (0, 1). Clamped to valid range.
     */
    void setTransitionPoint(double a) {
        // Clamp to valid range (avoid division by zero at a=1)
        a_ = juce::jlimit(0.001, 0.999, a);

        // Calculate saturation point b for C1 continuity with slope 1 at knee
        // dy/dx at knee = (1 - a) * (π/2) / (b - a) = 1
        // Therefore: b - a = (1 - a) * (π/2)
        constexpr double kPiOver2 = juce::MathConstants<double>::pi / 2.0;
        b_ = a_ + (1.0 - a_) * kPiOver2;

        // Precompute for efficiency
        outputRange_ = 1.0 - a_;       // Output goes from a to 1
        inputRange_ = b_ - a_;         // Input goes from a to b
        invInputRange_ = 1.0 / inputRange_;
    }

    /**
     * Get the current transition point (knee).
     */
    double getTransitionPoint() const {
        return a_;
    }

    /**
     * Get the saturation point (where output reaches ±1).
     */
    [[nodiscard]] double getSaturationPoint() const {
        return b_;
    }

    /**
     * Process a single sample through the soft clipper.
     * Transparent below knee, smooth limiting above.
     */
    double process(double x) const {
        const double sign = (x >= 0.0) ? 1.0 : -1.0;
        double u = std::abs(x);

        if (u <= a_) {
            // Linear region with unity gain - completely transparent
            return x;
        }

        if (u >= b_) {
            // Hard saturation at ±1
            return sign;
        }

        // Quarter sine transition from (a, a) to (b, 1)
        double t = (u - a_) * invInputRange_;  // t ∈ [0, 1]
        constexpr double kPiOver2 = juce::MathConstants<double>::pi / 2.0;
        double y = a_ + outputRange_ * std::sin(kPiOver2 * t);

        return sign * y;
    }

    /**
     * Analytical derivative dy/dx of process() at x.
     *
     * Piecewise:
     *   |x| ≤ a: 1 (linear unity gain)
     *   |x| ≥ b: 0 (flat saturation)
     *   otherwise: cos(π/2 · (|x|-a)/(b-a)) — the (1-a)(π/2)/(b-a) prefactor
     *              is identically 1 by C1 construction. Since y = sign(x)·f(|x|),
     *              sign² cancels and dy/dx = f'(|x|).
     */
    [[nodiscard]] double derivative(double x) const {
        const double u = std::abs(x);
        if (u <= a_) return 1.0;
        if (u >= b_) return 0.0;
        constexpr double kPiOver2 = juce::MathConstants<double>::pi / 2.0;
        const double t = (u - a_) * invInputRange_;
        return std::cos(kPiOver2 * t);
    }

  private:
    double a_ = 0.95;           // Knee threshold (input level where limiting starts)
    double b_ = 1.0285;         // Saturation threshold (input level where output = 1)
    double outputRange_ = 0.05; // 1 - a
    double inputRange_ = 0.0785; // b - a
    double invInputRange_ = 12.74; // 1 / (b - a)
};

/**
 * Audio processing stage applying soft clipping.
 *
 * Applies a smooth soft clipper that transitions from linear response
 * to sine-shaped limiting at the transition point. This provides gentle,
 * musical saturation for signals approaching ±1.
 *
 * Pipeline Position: After input gain, before waveshaper (within dry/wet mix).
 *
 * Thread Safety:
 *   - enabled_: atomic (UI writes, audio reads)
 *   - Solver: immutable after construction (safe for audio reads)
 *
 * CPU Cost: Low (~1-2% per channel, mostly transcendental function for clipped samples)
 */
class SoftClippingStage : public AudioProcessingStage {
  public:
    /**
     * Construct with default transition point (0.95).
     */
    SoftClippingStage() = default;

    /**
     * Construct with custom transition point.
     * @param transitionPoint Transition from linear to sine (0.001 to 0.999)
     */
    explicit SoftClippingStage(double transitionPoint) : solver_(transitionPoint) {
    }

    // AudioProcessingStage interface
    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void process(juce::AudioBuffer<double>& buffer) override;
    void reset() override;
    juce::String getName() const override {
        return "SoftClipping";
    }

    // Control interface (thread-safe)
    void setEnabled(bool shouldBeEnabled) {
        enabled_.store(shouldBeEnabled, std::memory_order_release);
    }

    bool isEnabled() const {
        return enabled_.load(std::memory_order_acquire);
    }

    /**
     * Set transition point parameter.
     * @param transitionPoint Value in (0, 1). Default 0.95.
     *
     * Thread Safety: UI thread only
     */
    void setTransitionPoint(double transitionPoint) {
        solver_.setTransitionPoint(transitionPoint);
    }

    double getTransitionPoint() const {
        return solver_.getTransitionPoint();
    }

  private:
    std::atomic<bool> enabled_{true}; // Default: ON
    SoftClippingSolver solver_;
};

} // namespace dsp_core::audio_pipeline
