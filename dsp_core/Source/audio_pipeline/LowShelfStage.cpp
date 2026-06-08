#include "LowShelfStage.h"
#include <cmath>

namespace dsp_core::audio_pipeline {

namespace {
// File-scoped names to avoid collisions with sibling stages under the
// dsp-core unity build (every audio_pipeline/*.cpp is concatenated into one
// translation unit via dsp_core.cpp, so anonymous-namespace constants share
// scope across files).
constexpr double kLowShelfMinCutoffHz = 20.0;
constexpr double kLowShelfMaxCutoffHz = 20000.0;
constexpr double kLowShelfMinGainDb = -24.0;
constexpr double kLowShelfMaxGainDb = 24.0;
constexpr double kLowShelfButterworthQ = 0.7071067811865476; // 1 / sqrt(2)
} // namespace

void LowShelfStage::prepareToPlay(double sampleRate, int /*samplesPerBlock*/, int /*numChannels*/) {
    sampleRate_ = sampleRate;

    // Resize filters for stereo (will be resized in process() if needed)
    const int numChannels = 2;
    filters_.resize(numChannels);

    // Size the coefficient arrays to a 2nd-order biquad (allocates — prepare only).
    updateFilterCoefficients();

    smoothCutoff_.reset(sampleRate, kSmoothingTimeSec);
    smoothGainDb_.reset(sampleRate, kSmoothingTimeSec);

    // Snap smoothers to the current targets and clear filter state.
    reset();
}

void LowShelfStage::process(juce::AudioBuffer<double>& buffer) {
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
    const double cutTarget = cutoffHz_.load(std::memory_order_acquire);
    const double gainTarget = gainDb_.load(std::memory_order_acquire);
    if (cutTarget != lastCutoffTarget_) {
        smoothCutoff_.setTargetValue(cutTarget);
        lastCutoffTarget_ = cutTarget;
    }
    if (gainTarget != lastGainTarget_) {
        smoothGainDb_.setTargetValue(gainTarget);
        lastGainTarget_ = gainTarget;
    }

    // Steady state: one coefficient set for the whole block (cheap path).
    if (!smoothCutoff_.isSmoothing() && !smoothGainDb_.isSmoothing()) {
        writeCoefficients(smoothCutoff_.getCurrentValue(), smoothGainDb_.getCurrentValue());
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
        writeCoefficients(cut, gain);
        for (int ch = 0; ch < numChannels; ++ch) {
            auto* data = buffer.getWritePointer(ch);
            data[i] = filters_[ch].processSample(data[i]);
        }
    }
}

void LowShelfStage::reset() {
    // Snap the smoothers to the current targets so a freshly prepared / reset
    // stage starts converged (no ramp on the first block). Mirrors the
    // VirtualAnalog/Ladder stages' prepareToPlay convention.
    const double cut = cutoffHz_.load(std::memory_order_acquire);
    const double gain = gainDb_.load(std::memory_order_acquire);
    smoothCutoff_.setCurrentAndTargetValue(cut);
    smoothGainDb_.setCurrentAndTargetValue(gain);
    lastCutoffTarget_ = cut;
    lastGainTarget_ = gain;

    for (auto& filter : filters_) {
        filter.reset();
    }
}

void LowShelfStage::setCutoffFrequency(double frequencyHz) {
    frequencyHz = juce::jlimit(kLowShelfMinCutoffHz, kLowShelfMaxCutoffHz, frequencyHz);
    cutoffHz_.store(frequencyHz, std::memory_order_release);
}

void LowShelfStage::setGainDb(double gainDb) {
    gainDb = juce::jlimit(kLowShelfMinGainDb, kLowShelfMaxGainDb, gainDb);
    gainDb_.store(gainDb, std::memory_order_release);
}

void LowShelfStage::updateFilterCoefficients() {
    const double cutoffHz = cutoffHz_.load(std::memory_order_acquire);
    const double gainDb = gainDb_.load(std::memory_order_acquire);
    const double gainLinear = juce::Decibels::decibelsToGain(gainDb);

    // RBJ low-shelf biquad. Q fixed at Butterworth (1/sqrt(2)) — no user control.
    // Allocates a fresh coefficient object — only called from prepareToPlay /
    // channel-count changes to size the arrays; writeCoefficients() updates them
    // in place on the audio thread thereafter.
    auto coefficients = juce::dsp::IIR::Coefficients<double>::makeLowShelf(
        sampleRate_, cutoffHz, kLowShelfButterworthQ, gainLinear);

    for (auto& filter : filters_) {
        *filter.coefficients = *coefficients;
    }
}

void LowShelfStage::writeCoefficients(double cutoffHz, double gainDb) {
    // RBJ low-shelf biquad computed in place to match
    // juce::dsp::IIR::Coefficients<double>::makeLowShelf exactly (Butterworth Q).
    const double gainLinear = juce::Decibels::decibelsToGain(gainDb);
    const double A = std::sqrt(gainLinear);
    const double aminus1 = A - 1.0;
    const double aplus1 = A + 1.0;
    const double omega =
        (2.0 * juce::MathConstants<double>::pi * juce::jmax(cutoffHz, 2.0)) / sampleRate_;
    const double coso = std::cos(omega);
    const double beta = std::sin(omega) * std::sqrt(A) / kLowShelfButterworthQ;
    const double aminus1TimesCoso = aminus1 * coso;

    const double b0 = A * (aplus1 - aminus1TimesCoso + beta);
    const double b1 = A * 2.0 * (aminus1 - aplus1 * coso);
    const double b2 = A * (aplus1 - aminus1TimesCoso - beta);
    const double a0 = aplus1 + aminus1TimesCoso + beta;
    const double a1 = -2.0 * (aminus1 + aplus1 * coso);
    const double a2 = aplus1 + aminus1TimesCoso - beta;

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
