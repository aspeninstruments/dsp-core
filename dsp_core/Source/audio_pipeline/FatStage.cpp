#include "FatStage.h"

namespace dsp_core::audio_pipeline {

namespace {
// File-scoped names to avoid collisions with sibling stages under the
// dsp-core unity build (every audio_pipeline/*.cpp is concatenated into one
// translation unit via dsp_core.cpp).
constexpr double kFatMinPercent = 0.0;
constexpr double kFatMaxPercent = 100.0;
constexpr double kFatMaxGainDb = 6.0;

// Hardcoded shelf parameters — not user-facing. 200 Hz sits in the bass-fundamental
// band that ladder filters thin out as resonance rises; Butterworth Q (set by
// LowShelfStage itself) keeps the corner smooth and click-free during sweeps.
constexpr double kFatShelfCutoffHz = 200.0;
} // namespace

void FatStage::prepareToPlay(double sampleRate, int samplesPerBlock, int numChannels) {
    lowShelf_.prepareToPlay(sampleRate, samplesPerBlock, numChannels);
    lowShelf_.setCutoffFrequency(kFatShelfCutoffHz);
    const double percent = fat_.load(std::memory_order_acquire);
    lowShelf_.setGainDb((percent / kFatMaxPercent) * kFatMaxGainDb);
}

void FatStage::process(juce::AudioBuffer<double>& buffer) {
    const juce::ScopedNoDenormals noDenormals;

    if (!enabled_.load(std::memory_order_acquire)) {
        return;
    }

    lowShelf_.process(buffer);
}

void FatStage::reset() {
    lowShelf_.reset();
}

void FatStage::setFat(double percent) {
    percent = juce::jlimit(kFatMinPercent, kFatMaxPercent, percent);
    fat_.store(percent, std::memory_order_release);
    lowShelf_.setGainDb((percent / kFatMaxPercent) * kFatMaxGainDb);
}

} // namespace dsp_core::audio_pipeline
