#include "LowpassStage.h"
#include <algorithm>

namespace dsp_core::audio_pipeline {

namespace {
constexpr double kMinCutoffHz = 20.0;
constexpr double kNyquistMargin = 0.45; // cutoff <= 0.45 * fs (TPT prewarp safety)
constexpr double kMinResonance = 0.5;
constexpr double kMaxResonance = 10.0;

double maxCutoffForSampleRate(double sampleRate) {
    return std::max(kMinCutoffHz, sampleRate * kNyquistMargin);
}
} // namespace

void LowpassStage::prepareToPlay(double sampleRate, int samplesPerBlock, int numChannels) {
    sampleRate_ = sampleRate;

    juce::dsp::ProcessSpec spec{};
    spec.sampleRate = sampleRate;
    spec.maximumBlockSize = static_cast<juce::uint32>(std::max(1, samplesPerBlock));
    spec.numChannels = static_cast<juce::uint32>(std::max(1, numChannels));

    filter_.prepare(spec);
    filter_.setType(juce::dsp::StateVariableTPTFilter<double>::Type::lowpass);

    // Re-clamp stored cutoff against the new Nyquist ceiling and write back so
    // the UI reads what's actually in use.
    const double maxCutoff = maxCutoffForSampleRate(sampleRate_);
    const double clampedCutoff =
        juce::jlimit(kMinCutoffHz, maxCutoff, cutoffHz_.load(std::memory_order_acquire));
    cutoffHz_.store(clampedCutoff, std::memory_order_release);

    filter_.setCutoffFrequency(clampedCutoff);
    filter_.setResonance(resonance_.load(std::memory_order_acquire));
}

void LowpassStage::process(juce::AudioBuffer<double>& buffer) {
    if (!enabled_.load(std::memory_order_acquire)) {
        return;
    }

    // Pull atomic params once per block. TPT coefficient update is cheap, but
    // not free (a tan() prewarp); per-sample updates are deferred until
    // modulation routing lands.
    filter_.setCutoffFrequency(cutoffHz_.load(std::memory_order_acquire));
    filter_.setResonance(resonance_.load(std::memory_order_acquire));

    const int numChannels = buffer.getNumChannels();
    const int numSamples = buffer.getNumSamples();

    for (int ch = 0; ch < numChannels; ++ch) {
        auto* data = buffer.getWritePointer(ch);
        for (int i = 0; i < numSamples; ++i) {
            data[i] = filter_.processSample(ch, data[i]);
        }
    }
}

void LowpassStage::reset() {
    filter_.reset();
}

void LowpassStage::setCutoffFrequency(double frequencyHz) {
    const double maxCutoff = maxCutoffForSampleRate(sampleRate_);
    frequencyHz = juce::jlimit(kMinCutoffHz, maxCutoff, frequencyHz);
    cutoffHz_.store(frequencyHz, std::memory_order_release);
    // filter_ is audio-thread-owned; the next process() block picks up the new value.
}

void LowpassStage::setResonance(double q) {
    q = juce::jlimit(kMinResonance, kMaxResonance, q);
    resonance_.store(q, std::memory_order_release);
}

} // namespace dsp_core::audio_pipeline
