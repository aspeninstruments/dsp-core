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
    highShelf_.prepareToPlay(sampleRate, samplesPerBlock, numChannels);
    smile_.prepareToPlay(sampleRate, samplesPerBlock, numChannels);
    bell_.prepareToPlay(sampleRate, samplesPerBlock, numChannels);
    hysteresis_.prepareToPlay(sampleRate, samplesPerBlock, numChannels);
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
        case Type::HighShelf:
            return &highShelf_;
        case Type::Smile:
            return &smile_;
        case Type::Bell:
            return &bell_;
        case Type::Hysteresis:
            return &hysteresis_;
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
    highShelf_.reset();
    smile_.reset();
    bell_.reset();
    hysteresis_.reset();
}

void ToneStage::setCutoffFrequency(double frequencyHz) {
    lp12_.setFrequency(frequencyHz);
    lp24_.setFrequency(frequencyHz);
    lowShelf_.setFrequency(frequencyHz);
    highShelf_.setFrequency(frequencyHz);
    smile_.setFrequency(frequencyHz); // drives Smile's HS corner; LS recomputes from ratio
    bell_.setFrequency(frequencyHz);  // bell centre frequency
    hysteresis_.setFrequency(frequencyHz); // no-op (marker strategy)
}

void ToneStage::setResonance(double zeroToOne) {
    const double clamped = juce::jlimit(0.0, 1.0, zeroToOne);
    const double r = clamped * kToneMaxResonance;
    lp12_.setResonance(r);
    lp24_.setResonance(r);
    lowShelf_.setResonance(r);  // no-op
    highShelf_.setResonance(r); // no-op
    smile_.setResonance(r);     // no-op
    bell_.setResonance(r);      // no-op
    hysteresis_.setResonance(r); // no-op (marker strategy)
}

void ToneStage::setShelfGainDb(double gainDb) {
    lp12_.setShelfGainDb(gainDb); // no-op
    lp24_.setShelfGainDb(gainDb); // no-op
    lowShelf_.setShelfGainDb(gainDb);
    highShelf_.setShelfGainDb(gainDb);
    smile_.setShelfGainDb(gainDb); // linked: drives both Smile shelves
    bell_.setShelfGainDb(gainDb);  // bell peak gain (reuses Tone_Gain)
    hysteresis_.setShelfGainDb(gainDb); // no-op (marker strategy)
}

void ToneStage::setFat(double percent) {
    lp12_.setFat(percent);
    lp24_.setFat(percent);
    lowShelf_.setFat(percent);  // no-op
    highShelf_.setFat(percent); // no-op
    smile_.setFat(percent);     // no-op
    bell_.setFat(percent);      // no-op
    hysteresis_.setFat(percent); // no-op (marker strategy)
}

double ToneStage::getFat() const {
    // Both lowpass strategies are kept primed with the same Fat value via
    // the universal setter, so either is a valid source.
    return lp12_.getFatPercent();
}

void ToneStage::setLowShelfRatio(double ratio) {
    lp12_.setLowShelfRatio(ratio);     // no-op
    lp24_.setLowShelfRatio(ratio);     // no-op
    lowShelf_.setLowShelfRatio(ratio); // no-op
    highShelf_.setLowShelfRatio(ratio); // no-op
    smile_.setLowShelfRatio(ratio);    // applied here (Smile-only)
    bell_.setLowShelfRatio(ratio);     // no-op
    hysteresis_.setLowShelfRatio(ratio); // no-op (marker strategy)
}

void ToneStage::setQ(double q) {
    lp12_.setQ(q);     // no-op
    lp24_.setQ(q);     // no-op
    lowShelf_.setQ(q); // no-op
    highShelf_.setQ(q); // no-op
    smile_.setQ(q);    // no-op
    bell_.setQ(q);     // applied here (Bell-only)
    hysteresis_.setQ(q); // no-op (marker strategy)
}

} // namespace dsp_core::audio_pipeline
