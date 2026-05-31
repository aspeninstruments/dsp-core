#include "ToneStage.h"
#include "ToneFrequencyResponse.h"

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
    hp12_.prepareToPlay(sampleRate, samplesPerBlock, numChannels);
    hp24_.prepareToPlay(sampleRate, samplesPerBlock, numChannels);
    ir_.prepareToPlay(sampleRate, samplesPerBlock, numChannels);
    lastType_ = type_.load(std::memory_order_acquire);

    cachedSampleRate_.store(sampleRate, std::memory_order_release);
    initFrequencyResponseFrequencies();
}

void ToneStage::initFrequencyResponseFrequencies() {
    // Log-spaced from kFrequencyResponseMinHz to kFrequencyResponseMaxHz.
    const double logMin = std::log10(kFrequencyResponseMinHz);
    const double logMax = std::log10(kFrequencyResponseMaxHz);
    const double denom = static_cast<double>(kFrequencyResponseSize - 1);
    for (int i = 0; i < kFrequencyResponseSize; ++i) {
        const double t = static_cast<double>(i) / denom;
        frequencyResponseHz_[static_cast<std::size_t>(i)] = std::pow(10.0, logMin + t * (logMax - logMin));
    }
}

void ToneStage::recomputeFrequencyResponse(const ToneFrequencyResponseParams& params) {
    // Write in place into the single LUT buffer and bump the version. Writer
    // (editor 60 Hz timer) and reader (visualizer VBlank paint) both run on
    // the message thread, so there's no concurrent access to guard against —
    // see the buffer comment in ToneStage.h for why the previous double-buffer
    // was removed.
    computeFrequencyResponseDb(params, frequencyResponseHz_.data(),
                               frequencyResponseBuffer_.data(), kFrequencyResponseSize);
    frequencyResponseVersion_.fetch_add(1, std::memory_order_release);
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
        case Type::Highpass12dB:
            return &hp12_;
        case Type::Highpass24dB:
            return &hp24_;
        case Type::ImpulseResponse:
            return &ir_;
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
    hp12_.reset();
    hp24_.reset();
    ir_.reset();
}

void ToneStage::setCutoffFrequency(double frequencyHz) {
    lp12_.setFrequency(frequencyHz);
    lp24_.setFrequency(frequencyHz);
    lowShelf_.setFrequency(frequencyHz);
    highShelf_.setFrequency(frequencyHz);
    smile_.setFrequency(frequencyHz); // drives Smile's HS corner; LS recomputes from ratio
    bell_.setFrequency(frequencyHz);  // bell centre frequency
    hysteresis_.setFrequency(frequencyHz); // no-op (marker strategy)
    hp12_.setFrequency(frequencyHz);
    hp24_.setFrequency(frequencyHz);
    ir_.setFrequency(frequencyHz); // no-op (IR is the cabinet character)
    cachedCutoffHz_.store(frequencyHz, std::memory_order_release);
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
    hp12_.setResonance(r);
    hp24_.setResonance(r);
    ir_.setResonance(r); // no-op
    cachedResonanceNorm_.store(clamped, std::memory_order_release);
}

void ToneStage::setShelfGainDb(double gainDb) {
    lp12_.setShelfGainDb(gainDb); // no-op
    lp24_.setShelfGainDb(gainDb); // no-op
    lowShelf_.setShelfGainDb(gainDb);
    highShelf_.setShelfGainDb(gainDb);
    smile_.setShelfGainDb(gainDb); // linked: drives both Smile shelves
    bell_.setShelfGainDb(gainDb);  // bell peak gain (reuses Tone_Gain)
    hysteresis_.setShelfGainDb(gainDb); // no-op (marker strategy)
    hp12_.setShelfGainDb(gainDb); // no-op
    hp24_.setShelfGainDb(gainDb); // no-op
    ir_.setShelfGainDb(gainDb); // no-op
    cachedShelfGainDb_.store(gainDb, std::memory_order_release);
}

void ToneStage::setFat(double percent) {
    lp12_.setFat(percent);
    lp24_.setFat(percent);
    lowShelf_.setFat(percent);  // no-op
    highShelf_.setFat(percent); // no-op
    smile_.setFat(percent);     // no-op
    bell_.setFat(percent);      // no-op
    hysteresis_.setFat(percent); // no-op (marker strategy)
    hp12_.setFat(percent); // no-op (HPF has no Fat sub-stage)
    hp24_.setFat(percent); // no-op
    ir_.setFat(percent); // no-op
    cachedFatPercent_.store(percent, std::memory_order_release);
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
    hp12_.setLowShelfRatio(ratio); // no-op
    hp24_.setLowShelfRatio(ratio); // no-op
    ir_.setLowShelfRatio(ratio); // no-op
    cachedLowShelfRatio_.store(ratio, std::memory_order_release);
}

void ToneStage::setQ(double q) {
    lp12_.setQ(q);     // no-op
    lp24_.setQ(q);     // no-op
    lowShelf_.setQ(q); // no-op
    highShelf_.setQ(q); // no-op
    smile_.setQ(q);    // no-op
    bell_.setQ(q);     // applied here (Bell-only)
    hysteresis_.setQ(q); // no-op (marker strategy)
    hp12_.setQ(q); // no-op
    hp24_.setQ(q); // no-op
    ir_.setQ(q); // no-op
    cachedBellQ_.store(q, std::memory_order_release);
}

} // namespace dsp_core::audio_pipeline
