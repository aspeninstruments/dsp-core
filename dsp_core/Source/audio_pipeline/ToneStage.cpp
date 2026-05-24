#include "ToneStage.h"

namespace dsp_core::audio_pipeline {

namespace {
// File-scoped names to avoid collisions with sibling stages under the
// dsp-core unity build (every audio_pipeline/*.cpp is concatenated into one
// translation unit via dsp_core.cpp, so anonymous-namespace constants share
// scope across files). See LowShelfStage.cpp for the canonical pattern.
constexpr double kToneMaxResonance = 3.5; // Safely below LadderTPT self-osc onset (~3.95).
} // namespace

void ToneStage::prepareToPlay(double sampleRate, int samplesPerBlock, int numChannels) {
    // Prepare both inner filters; only one runs per block but both must
    // have valid coefficients/state in case Type is switched mid-playback.
    lp12_.prepareToPlay(sampleRate, samplesPerBlock, numChannels);
    lp24_.prepareToPlay(sampleRate, samplesPerBlock, numChannels);
    lastType_ = type_.load(std::memory_order_acquire);
}

void ToneStage::process(juce::AudioBuffer<double>& buffer) {
    if (!enabled_.load(std::memory_order_acquire)) {
        return;
    }

    const Type t = type_.load(std::memory_order_acquire);
    if (t == Type::Off) {
        return;
    }

    if (t != lastType_) {
        if (t == Type::Lowpass12dB) {
            lp12_.reset();
        } else {
            lp24_.reset();
        }
        lastType_ = t;
    }

    if (t == Type::Lowpass12dB) {
        lp12_.process(buffer);
    } else {
        lp24_.process(buffer);
    }
}

void ToneStage::reset() {
    lp12_.reset();
    lp24_.reset();
}

void ToneStage::setCutoffFrequency(double frequencyHz) {
    lp12_.setCutoffFrequency(frequencyHz);
    lp24_.setCutoffFrequency(frequencyHz);
}

void ToneStage::setResonance(double zeroToOne) {
    const double clamped = juce::jlimit(0.0, 1.0, zeroToOne);
    const double r = clamped * kToneMaxResonance;
    lp12_.setResonance(r);
    lp24_.setResonance(r);
}

} // namespace dsp_core::audio_pipeline
