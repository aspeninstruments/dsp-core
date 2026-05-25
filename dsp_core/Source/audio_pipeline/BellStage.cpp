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

    // Configure all filter coefficients
    updateFilterCoefficients();
}

void BellStage::process(juce::AudioBuffer<double>& buffer) {
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

    // Process each channel: apply bell
    for (int ch = 0; ch < numChannels; ++ch) {
        auto* data = buffer.getWritePointer(ch);

        for (int i = 0; i < numSamples; ++i) {
            data[i] = filters_[ch].processSample(data[i]);
        }
    }
}

void BellStage::reset() {
    for (auto& filter : filters_) {
        filter.reset();
    }
}

void BellStage::setCutoffFrequency(double frequencyHz) {
    frequencyHz = juce::jlimit(kBellMinCentreHz, kBellMaxCentreHz, frequencyHz);
    centreHz_.store(frequencyHz, std::memory_order_release);
    updateFilterCoefficients();
}

void BellStage::setGainDb(double gainDb) {
    gainDb = juce::jlimit(kBellMinGainDb, kBellMaxGainDb, gainDb);
    gainDb_.store(gainDb, std::memory_order_release);
    updateFilterCoefficients();
}

void BellStage::setQ(double q) {
    q = juce::jlimit(kBellMinQ, kBellMaxQ, q);
    q_.store(q, std::memory_order_release);
    updateFilterCoefficients();
}

void BellStage::updateFilterCoefficients() {
    const double centreHz = centreHz_.load(std::memory_order_acquire);
    const double gainDb = gainDb_.load(std::memory_order_acquire);
    const double q = q_.load(std::memory_order_acquire);
    const double gainLinear = juce::Decibels::decibelsToGain(gainDb);

    // RBJ parametric peaking biquad.
    auto coefficients = juce::dsp::IIR::Coefficients<double>::makePeakFilter(
        sampleRate_, centreHz, q, gainLinear);

    for (auto& filter : filters_) {
        *filter.coefficients = *coefficients;
    }
}

} // namespace dsp_core::audio_pipeline
