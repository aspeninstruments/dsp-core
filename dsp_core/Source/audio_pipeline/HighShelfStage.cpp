#include "HighShelfStage.h"
#include <cmath>

namespace dsp_core::audio_pipeline {

namespace {
// File-scoped names to avoid collisions with sibling stages under the
// dsp-core unity build (every audio_pipeline/*.cpp is concatenated into one
// translation unit via dsp_core.cpp, so anonymous-namespace constants share
// scope across files).
constexpr double kHighShelfMinCutoffHz = 20.0;
constexpr double kHighShelfMaxCutoffHz = 20000.0;
constexpr double kHighShelfMinGainDb = -24.0;
constexpr double kHighShelfMaxGainDb = 24.0;
constexpr double kHighShelfButterworthQ = 0.7071067811865476; // 1 / sqrt(2)
} // namespace

void HighShelfStage::prepareToPlay(double sampleRate, int /*samplesPerBlock*/, int /*numChannels*/) {
    sampleRate_ = sampleRate;

    // Resize filters for stereo (will be resized in process() if needed)
    const int numChannels = 2;
    filters_.resize(numChannels);

    // Configure all filter coefficients
    updateFilterCoefficients();
}

void HighShelfStage::process(juce::AudioBuffer<double>& buffer) {
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

void HighShelfStage::reset() {
    for (auto& filter : filters_) {
        filter.reset();
    }
}

void HighShelfStage::setCutoffFrequency(double frequencyHz) {
    frequencyHz = juce::jlimit(kHighShelfMinCutoffHz, kHighShelfMaxCutoffHz, frequencyHz);
    cutoffHz_.store(frequencyHz, std::memory_order_release);
    updateFilterCoefficients();
}

void HighShelfStage::setGainDb(double gainDb) {
    gainDb = juce::jlimit(kHighShelfMinGainDb, kHighShelfMaxGainDb, gainDb);
    gainDb_.store(gainDb, std::memory_order_release);
    updateFilterCoefficients();
}

void HighShelfStage::updateFilterCoefficients() {
    const double cutoffHz = cutoffHz_.load(std::memory_order_acquire);
    const double gainDb = gainDb_.load(std::memory_order_acquire);
    const double gainLinear = juce::Decibels::decibelsToGain(gainDb);

    // RBJ high-shelf biquad. Q fixed at Butterworth (1/sqrt(2)) — no user control.
    auto coefficients = juce::dsp::IIR::Coefficients<double>::makeHighShelf(
        sampleRate_, cutoffHz, kHighShelfButterworthQ, gainLinear);

    for (auto& filter : filters_) {
        *filter.coefficients = *coefficients;
    }
}

} // namespace dsp_core::audio_pipeline
