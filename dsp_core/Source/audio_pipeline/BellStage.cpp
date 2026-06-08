#include "BellStage.h"
#include <cmath>

namespace dsp_core::audio_pipeline {

namespace {
// File-scoped names to avoid collisions with sibling stages under the
// dsp-core unity build (every audio_pipeline/*.cpp is concatenated into one
// translation unit via dsp_core.cpp, so anonymous-namespace constants share
// scope across files).
constexpr double kBellMinCentreHz = 20.0;
constexpr double kBellMaxCentreHz = 20000.0;
constexpr double kBellMinGainDb = -24.0;
constexpr double kBellMaxGainDb = 24.0;
constexpr double kBellMinQ = 0.1;
constexpr double kBellMaxQ = 10.0;
} // namespace

void BellStage::prepareToPlay(double sampleRate, int /*samplesPerBlock*/, int /*numChannels*/) {
    sampleRate_ = sampleRate;

    // Resize filters for stereo (will be resized in process() if needed)
    const int numChannels = 2;
    filters_.resize(numChannels);

    // Size the coefficient arrays to a 2nd-order biquad (allocates — prepare only).
    updateFilterCoefficients();

    smoothCutoff_.reset(sampleRate, kSmoothingTimeSec);
    smoothGainDb_.reset(sampleRate, kSmoothingTimeSec);
    smoothQ_.reset(sampleRate, kSmoothingTimeSec);

    // Snap smoothers to the current targets and clear filter state.
    reset();
}

void BellStage::process(juce::AudioBuffer<double>& buffer) {
    const int numChannels = buffer.getNumChannels();
    const int numSamples = buffer.getNumSamples();

    // Resize filters if channel count changed. Rare (layout change) — the
    // allocation here is acceptable outside the steady-state path.
    if (filters_.size() != static_cast<size_t>(numChannels)) {
        filters_.resize(numChannels);
        updateFilterCoefficients();
        for (auto& filter : filters_) {
            filter.reset();
        }
    }

    // Early return if disabled
    if (!enabled_.load(std::memory_order_acquire)) {
        return;
    }

    // Chase the atomic targets. setTargetValue only on an actual change so an
    // unchanging parameter doesn't restart its ramp every block.
    const double cutTarget = centreHz_.load(std::memory_order_acquire);
    const double gainTarget = gainDb_.load(std::memory_order_acquire);
    const double qTarget = q_.load(std::memory_order_acquire);
    if (cutTarget != lastCutoffTarget_) {
        smoothCutoff_.setTargetValue(cutTarget);
        lastCutoffTarget_ = cutTarget;
    }
    if (gainTarget != lastGainTarget_) {
        smoothGainDb_.setTargetValue(gainTarget);
        lastGainTarget_ = gainTarget;
    }
    if (qTarget != lastQTarget_) {
        smoothQ_.setTargetValue(qTarget);
        lastQTarget_ = qTarget;
    }

    // Steady state: one coefficient set for the whole block (cheap path).
    if (!smoothCutoff_.isSmoothing() && !smoothGainDb_.isSmoothing() && !smoothQ_.isSmoothing()) {
        writeCoefficients(smoothCutoff_.getCurrentValue(), smoothGainDb_.getCurrentValue(),
                          smoothQ_.getCurrentValue());
        for (int ch = 0; ch < numChannels; ++ch) {
            auto* data = buffer.getWritePointer(ch);
            for (int i = 0; i < numSamples; ++i) {
                data[i] = filters_[ch].processSample(data[i]);
            }
        }
        return;
    }

    // Smoothing: advance the shared smoothers once per sample (hoisted above the
    // channel loop so both channels see identical coefficients and the ramp
    // isn't advanced twice — mirrors LadderTPTStage).
    for (int i = 0; i < numSamples; ++i) {
        const double cut = smoothCutoff_.getNextValue();
        const double gain = smoothGainDb_.getNextValue();
        const double q = smoothQ_.getNextValue();
        writeCoefficients(cut, gain, q);
        for (int ch = 0; ch < numChannels; ++ch) {
            auto* data = buffer.getWritePointer(ch);
            data[i] = filters_[ch].processSample(data[i]);
        }
    }
}

void BellStage::reset() {
    // Snap the smoothers to the current targets so a freshly prepared / reset
    // stage starts converged (no ramp on the first block). Mirrors the
    // VirtualAnalog/Ladder stages' prepareToPlay convention.
    const double cut = centreHz_.load(std::memory_order_acquire);
    const double gain = gainDb_.load(std::memory_order_acquire);
    const double q = q_.load(std::memory_order_acquire);
    smoothCutoff_.setCurrentAndTargetValue(cut);
    smoothGainDb_.setCurrentAndTargetValue(gain);
    smoothQ_.setCurrentAndTargetValue(q);
    lastCutoffTarget_ = cut;
    lastGainTarget_ = gain;
    lastQTarget_ = q;

    for (auto& filter : filters_) {
        filter.reset();
    }
}

void BellStage::setCutoffFrequency(double frequencyHz) {
    frequencyHz = juce::jlimit(kBellMinCentreHz, kBellMaxCentreHz, frequencyHz);
    centreHz_.store(frequencyHz, std::memory_order_release);
}

void BellStage::setGainDb(double gainDb) {
    gainDb = juce::jlimit(kBellMinGainDb, kBellMaxGainDb, gainDb);
    gainDb_.store(gainDb, std::memory_order_release);
}

void BellStage::setQ(double q) {
    q = juce::jlimit(kBellMinQ, kBellMaxQ, q);
    q_.store(q, std::memory_order_release);
}

void BellStage::updateFilterCoefficients() {
    const double centreHz = centreHz_.load(std::memory_order_acquire);
    const double gainDb = gainDb_.load(std::memory_order_acquire);
    const double q = q_.load(std::memory_order_acquire);
    const double gainLinear = juce::Decibels::decibelsToGain(gainDb);

    // RBJ parametric peaking biquad. Allocates a fresh coefficient object — only
    // called from prepareToPlay / channel-count changes to size the arrays;
    // writeCoefficients() updates them in place on the audio thread thereafter.
    auto coefficients = juce::dsp::IIR::Coefficients<double>::makePeakFilter(
        sampleRate_, centreHz, q, gainLinear);

    for (auto& filter : filters_) {
        *filter.coefficients = *coefficients;
    }
}

void BellStage::writeCoefficients(double centreHz, double gainDb, double q) {
    // RBJ peaking biquad computed in place to match
    // juce::dsp::IIR::Coefficients<double>::makePeakFilter exactly. (JUCE floors
    // the gain at -300 dB before sqrt; that's a no-op for the ±24 dB control
    // range, so plain sqrt(gainLinear) is bit-faithful here.)
    const double gainLinear = juce::Decibels::decibelsToGain(gainDb);
    const double A = std::sqrt(gainLinear);
    const double omega =
        (2.0 * juce::MathConstants<double>::pi * juce::jmax(centreHz, 2.0)) / sampleRate_;
    const double alpha = std::sin(omega) / (q * 2.0);
    const double c2 = -2.0 * std::cos(omega);
    const double alphaTimesA = alpha * A;
    const double alphaOverA = alpha / A;

    const double b0 = 1.0 + alphaTimesA;
    const double b1 = c2;
    const double b2 = 1.0 - alphaTimesA;
    const double a0 = 1.0 + alphaOverA;
    const double a1 = c2;
    const double a2 = 1.0 - alphaOverA;

    // JUCE stores the normalised array {b0/a0, b1/a0, b2/a0, a1/a0, a2/a0}.
    const double a0Inv = 1.0 / a0;
    for (auto& filter : filters_) {
        auto* c = filter.coefficients->getRawCoefficients();
        c[0] = b0 * a0Inv;
        c[1] = b1 * a0Inv;
        c[2] = b2 * a0Inv;
        c[3] = a1 * a0Inv;
        c[4] = a2 * a0Inv;
    }
}

} // namespace dsp_core::audio_pipeline
