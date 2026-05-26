#include "FirstOrderShelfStage.h"

#include <cmath>

namespace dsp_core::audio_pipeline {

namespace {
// File-scoped names to avoid collisions under the dsp-core unity build (every
// audio_pipeline/*.cpp is concatenated through dsp_core.cpp).
constexpr double kFirstOrderShelfMinCutoffHz = 1.0;
constexpr double kFirstOrderShelfNyquistMarginFactor = 0.49;
constexpr double kFirstOrderShelfMaxGainDbAbs = 36.0; // hard cap for sanity
} // namespace

void FirstOrderShelfStage::prepareToPlay(double sampleRate, int /*samplesPerBlock*/, int numChannels) {
    sampleRate_ = sampleRate;
    const int ch = std::max(1, numChannels);
    x1_.assign(static_cast<std::size_t>(ch), 0.0);
    y1_.assign(static_cast<std::size_t>(ch), 0.0);
    recomputeCoefficients();
}

void FirstOrderShelfStage::process(juce::AudioBuffer<double>& buffer) {
    const juce::ScopedNoDenormals noDenormals;

    if (!enabled_.load(std::memory_order_acquire)) {
        return;
    }

    const double b0 = b0_.load(std::memory_order_acquire);
    const double b1 = b1_.load(std::memory_order_acquire);
    const double a1 = a1_.load(std::memory_order_acquire);

    const int numChannels = std::min(buffer.getNumChannels(), static_cast<int>(x1_.size()));
    const int numSamples = buffer.getNumSamples();

    for (int ch = 0; ch < numChannels; ++ch) {
        auto* data = buffer.getWritePointer(ch);
        double x1 = x1_[static_cast<std::size_t>(ch)];
        double y1 = y1_[static_cast<std::size_t>(ch)];
        for (int i = 0; i < numSamples; ++i) {
            const double x = data[i];
            const double y = b0 * x + b1 * x1 - a1 * y1;
            data[i] = y;
            x1 = x;
            y1 = y;
        }
        x1_[static_cast<std::size_t>(ch)] = x1;
        y1_[static_cast<std::size_t>(ch)] = y1;
    }
}

void FirstOrderShelfStage::reset() {
    std::fill(x1_.begin(), x1_.end(), 0.0);
    std::fill(y1_.begin(), y1_.end(), 0.0);
}

void FirstOrderShelfStage::setCutoffFrequency(double frequencyHz) {
    const double maxHz = std::max(kFirstOrderShelfMinCutoffHz, sampleRate_ * kFirstOrderShelfNyquistMarginFactor);
    const double clamped = juce::jlimit(kFirstOrderShelfMinCutoffHz, maxHz, frequencyHz);
    cutoffHz_.store(clamped, std::memory_order_release);
    recomputeCoefficients();
}

void FirstOrderShelfStage::setGainDb(double gainDb) {
    const double clamped = juce::jlimit(-kFirstOrderShelfMaxGainDbAbs, kFirstOrderShelfMaxGainDbAbs, gainDb);
    gainDb_.store(clamped, std::memory_order_release);
    recomputeCoefficients();
}

void FirstOrderShelfStage::recomputeCoefficients() {
    // Bilinear-transform of the analog 1st-order shelf prototype, with
    // prewarping at the corner. Derivation:
    //
    //   HighShelf:  H(s) = A · (s + ω0/α) / (s + ω0·α),   α = sqrt(A)
    //   LowShelf:   H(s) =       (s + ω0·α) / (s + ω0/α)
    //
    // where A = 10^(gainDb/20) is linear gain. Both forms reduce to identity
    // at 0 dB and give half the gain (in dB) at the corner — standard 1st-order
    // shelf behavior. Substituting s = K·(z-1)/(z+1) with K=2/T and ω0 prewarped
    // to ωd = K·tan(ω0·T/2) yields the closed-form coefficients below.

    const double sr = sampleRate_;
    const double cutoffHz = cutoffHz_.load(std::memory_order_acquire);
    const double gainDb = gainDb_.load(std::memory_order_acquire);

    const double A = std::pow(10.0, gainDb / 20.0);
    const double alpha = std::sqrt(A);
    const double K = 2.0 * sr;
    const double omega0 = 2.0 * juce::MathConstants<double>::pi * cutoffHz;
    const double wd = K * std::tan(omega0 / (2.0 * sr));

    double b0Raw = 0.0;
    double b1Raw = 0.0;
    double a0Raw = 0.0;
    double a1Raw = 0.0;

    if (mode_ == Mode::HighShelf) {
        // Numerator A·(K + ωd/α), A·(ωd/α − K)
        // Denominator     (K + ωd·α),    (ωd·α − K)
        // A/α = α, so A·ωd/α = α·ωd, A·K stays as-is.
        b0Raw = A * K + alpha * wd;
        b1Raw = alpha * wd - A * K;
        a0Raw = K + wd * alpha;
        a1Raw = wd * alpha - K;
    } else {
        // LowShelf: numerator (K + ωd·α), (ωd·α − K)
        //          denominator (K + ωd/α), (ωd/α − K)
        b0Raw = K + wd * alpha;
        b1Raw = wd * alpha - K;
        a0Raw = K + wd / alpha;
        a1Raw = wd / alpha - K;
    }

    const double inv = 1.0 / a0Raw;
    b0_.store(b0Raw * inv, std::memory_order_release);
    b1_.store(b1Raw * inv, std::memory_order_release);
    a1_.store(a1Raw * inv, std::memory_order_release);
}

} // namespace dsp_core::audio_pipeline
