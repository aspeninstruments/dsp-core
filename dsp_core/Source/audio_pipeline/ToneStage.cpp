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
    // Prepare every strategy — only one runs per block, but all must have
    // valid coefficients/state in case Type is switched mid-playback.
    lp12_.prepareToPlay(sampleRate, samplesPerBlock, numChannels);
    lp24_.prepareToPlay(sampleRate, samplesPerBlock, numChannels);
    lowShelf_.prepareToPlay(sampleRate, samplesPerBlock, numChannels);
    lastType_ = type_.load(std::memory_order_acquire);
}

ToneFilterStrategy* ToneStage::strategyFor(Type t) {
    switch (t) {
        case Type::Lowpass12dB:
            return &lp12_;
        case Type::Lowpass24dB:
            return &lp24_;
        case Type::LowShelf:
            return &lowShelf_;
        case Type::Off:
        default:
            return nullptr;
    }
}

void ToneStage::process(juce::AudioBuffer<double>& buffer) {
    if (!enabled_.load(std::memory_order_acquire)) {
        return;
    }

    const Type t = type_.load(std::memory_order_acquire);
    if (t == Type::Off) {
        // Fat / shelf state is intentionally inert here: the user-facing
        // contract binds tone processing to a non-Off type. Skipping process()
        // means we never advance per-strategy bypass caches while Off — so the
        // first non-Off block enters cleanly.
        return;
    }

    if (t != lastType_) {
        if (auto* s = strategyFor(t)) {
            // Reset the newly-active strategy on the audio thread to clear
            // any stale internal state before it re-enters the signal path.
            // (LP12 <-> LP24 transitions worked under the pre-refactor
            // shared-Fat code because the per-block forwarding kept cutoff/
            // resonance synchronised; that still holds. LP <-> LS topology
            // switches accept a brief RBJ settle.)
            s->reset();
        }
        lastType_ = t;
    }

    if (auto* s = strategyFor(t)) {
        s->process(buffer);
    }
}

void ToneStage::reset() {
    lp12_.reset();
    lp24_.reset();
    lowShelf_.reset();
}

void ToneStage::setCutoffFrequency(double frequencyHz) {
    lp12_.setFrequency(frequencyHz);
    lp24_.setFrequency(frequencyHz);
    lowShelf_.setFrequency(frequencyHz);
}

void ToneStage::setResonance(double zeroToOne) {
    const double clamped = juce::jlimit(0.0, 1.0, zeroToOne);
    const double r = clamped * kToneMaxResonance;
    lp12_.setResonance(r);
    lp24_.setResonance(r);
    lowShelf_.setResonance(r); // no-op
}

void ToneStage::setShelfGainDb(double gainDb) {
    lp12_.setShelfGainDb(gainDb);    // no-op
    lp24_.setShelfGainDb(gainDb);    // no-op
    lowShelf_.setShelfGainDb(gainDb);
}

void ToneStage::setFat(double percent) {
    lp12_.setFat(percent);
    lp24_.setFat(percent);
    lowShelf_.setFat(percent); // no-op
}

double ToneStage::getFat() const {
    // Both lowpass strategies are kept primed with the same Fat value via
    // the universal setter, so either is a valid source.
    return lp12_.getFatPercent();
}

} // namespace dsp_core::audio_pipeline
