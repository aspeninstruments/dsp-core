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

    // Configure all filter coefficients
    updateFilterCoefficients();
}

void LowShelfStage::process(juce::AudioBuffer<double>& buffer) {
    const int numChannels = buffer.getNumChannels();
    const int numSamples = buffer.getNumSamples();

    // Resize filters if channel count changed
    if (filters_.size() != static_cast<size_t>(numChannels)) {
        filters_.resize(numChannels);
        updateFilterCoefficients();
    }

    // Early return if disabled
    if (!enabled_.load(std::memory_order_acquire)) {
        return;
    }

    // Process each channel: apply shelf
    for (int ch = 0; ch < numChannels; ++ch) {
        auto* data = buffer.getWritePointer(ch);

        for (int i = 0; i < numSamples; ++i) {
            data[i] = filters_[ch].processSample(data[i]);
        }
    }
}

void LowShelfStage::reset() {
    for (auto& filter : filters_) {
        filter.reset();
    }
}

void LowShelfStage::setCutoffFrequency(double frequencyHz) {
    frequencyHz = juce::jlimit(kLowShelfMinCutoffHz, kLowShelfMaxCutoffHz, frequencyHz);
    cutoffHz_.store(frequencyHz, std::memory_order_release);
    updateFilterCoefficients();
}

void LowShelfStage::setGainDb(double gainDb) {
    gainDb = juce::jlimit(kLowShelfMinGainDb, kLowShelfMaxGainDb, gainDb);
    gainDb_.store(gainDb, std::memory_order_release);
    updateFilterCoefficients();
}

void LowShelfStage::updateFilterCoefficients() {
    const double cutoffHz = cutoffHz_.load(std::memory_order_acquire);
    const double gainDb = gainDb_.load(std::memory_order_acquire);
    const double gainLinear = juce::Decibels::decibelsToGain(gainDb);

    // RBJ low-shelf biquad. Q fixed at Butterworth (1/sqrt(2)) — no user control.
    auto coefficients = juce::dsp::IIR::Coefficients<double>::makeLowShelf(
        sampleRate_, cutoffHz, kLowShelfButterworthQ, gainLinear);

    for (auto& filter : filters_) {
        *filter.coefficients = *coefficients;
    }
}

} // namespace dsp_core::audio_pipeline
