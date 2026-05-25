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
    fat_.prepareToPlay(sampleRate, samplesPerBlock, numChannels);
    lastType_ = type_.load(std::memory_order_acquire);
    lastFatActive_ = (fat_.getFat() > 0.0);
}

void ToneStage::process(juce::AudioBuffer<double>& buffer) {
    if (!enabled_.load(std::memory_order_acquire)) {
        return;
    }

    const Type t = type_.load(std::memory_order_acquire);
    if (t == Type::Off) {
        // Fat is intentionally inert here: the user-facing contract binds it
        // to the lowpass branches only, and skipping process() also means we
        // never advance lastFatActive_ while Off — so the first lowpass
        // block sees the 0->nonzero transition correctly.
        return;
    }

    if (t != lastType_) {
        if (t == Type::Lowpass12dB) {
            lp12_.reset();
        } else {
            lp24_.reset();
        }
        // Realign Fat's shelf state with whichever LP runs next so an
        // in-flight topology switch doesn't carry stale shelf output into a
        // freshly-reset filter.
        fat_.reset();
        lastType_ = t;
    }

    // Strict 0% bypass: single atomic load, single branch — when Fat is at
    // zero we incur no biquad work and no FatStage::process call.
    const double fatPct = fat_.getFat();
    const bool fatActive = (fatPct > 0.0);
    if (fatActive) {
        if (!lastFatActive_) {
            // Just emerged from bypass — flush whatever state was frozen
            // when we last skipped processing so the shelf re-enters cleanly.
            fat_.reset();
        }
        fat_.process(buffer);
    }
    lastFatActive_ = fatActive;

    if (t == Type::Lowpass12dB) {
        lp12_.process(buffer);
    } else {
        lp24_.process(buffer);
    }
}

void ToneStage::reset() {
    lp12_.reset();
    lp24_.reset();
    fat_.reset();
    lastFatActive_ = false;
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

void ToneStage::setFat(double percent) {
    fat_.setFat(percent);
}

double ToneStage::getFat() const {
    return fat_.getFat();
}

} // namespace dsp_core::audio_pipeline
