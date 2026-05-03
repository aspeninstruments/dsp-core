#include "LfoStage.h"
#include "../Services/PerlinNoiseService.h"
#include <cmath>

namespace dsp_core::audio_pipeline {

namespace {
constexpr double kTwoPi = 6.283185307179586476925286766559;
constexpr double kPerlinCellsPerCycle = 4.0;
} // namespace

LfoStage::LfoStage(std::atomic<double>& lfoStorage) : lfoStorage_(lfoStorage) {}

void LfoStage::prepareToPlay(double sampleRate, int /*samplesPerBlock*/, int /*numChannels*/) {
    sampleRate_ = sampleRate > 0.0 ? sampleRate : 48000.0;
}

void LfoStage::reset() {
    phase_ = 0.0;
    lfoStorage_.store(0.0, std::memory_order_release);
}

double LfoStage::periodInBeats(Division d, Flavor f) {
    double base = 1.0;
    switch (d) {
    case Division::Sixteenth:
        base = 0.25;
        break;
    case Division::Eighth:
        base = 0.5;
        break;
    case Division::Quarter:
        base = 1.0;
        break;
    case Division::Half:
        base = 2.0;
        break;
    case Division::Bar:
        base = 4.0;
        break;
    case Division::TwoBar:
        base = 8.0;
        break;
    case Division::FourBar:
        base = 16.0;
        break;
    }
    switch (f) {
    case Flavor::Straight:
        return base;
    case Flavor::Triplet:
        return base * (2.0 / 3.0);
    case Flavor::Dotted:
        return base * (3.0 / 2.0);
    }
    return base;
}

double LfoStage::evaluateShape(Shape s, double phase, unsigned int seed) {
    // phase is in [0, 1)
    switch (s) {
    case Shape::Off:
        return 0.0;
    case Shape::Sin: {
        return 0.5 + 0.5 * std::sin(kTwoPi * phase);
    }
    case Shape::Tri: {
        return 1.0 - std::fabs(2.0 * phase - 1.0);
    }
    case Shape::Saw: {
        return phase;
    }
    case Shape::Sq: {
        return phase < 0.5 ? 1.0 : 0.0;
    }
    case Shape::Random: {
        // perlinNoise1D returns ~[-1, 1]; map to [0, 1].
        const double n = Services::PerlinNoiseService::perlinNoise1D(phase * kPerlinCellsPerCycle, seed);
        return 0.5 + 0.5 * n;
    }
    }
    return 0.0;
}

void LfoStage::process(juce::AudioBuffer<double>& buffer) {
    const auto shape = static_cast<Shape>(shape_.load(std::memory_order_acquire));
    if (!enabled_.load(std::memory_order_acquire) || shape == Shape::Off) {
        lfoStorage_.store(0.0, std::memory_order_release);
        return;
    }

    const auto units = static_cast<Units>(units_.load(std::memory_order_acquire));
    const auto div = static_cast<Division>(division_.load(std::memory_order_acquire));
    const auto flv = static_cast<Flavor>(flavor_.load(std::memory_order_acquire));
    const double rateHz = rateHz_.load(std::memory_order_acquire);
    const double phaseOffset = phaseOffset_.load(std::memory_order_acquire);
    const unsigned int seed = seed_.load(std::memory_order_acquire);

    const int numSamples = buffer.getNumSamples();

    if (units == Units::BPM && hostIsPlaying_ && hostBpm_ > 0.0) {
        // Host-locked: derive phase directly from ppq each sample so we stay
        // tight to the beat regardless of DAW jitter.
        const double periodBeats = periodInBeats(div, flv);
        const double beatsPerSample = hostBpm_ / (60.0 * sampleRate_);
        for (int i = 0; i < numSamples; ++i) {
            const double ppq = hostPpq_ + beatsPerSample * static_cast<double>(i);
            double cyclePhase = std::fmod(ppq / periodBeats, 1.0);
            if (cyclePhase < 0.0)
                cyclePhase += 1.0;
            phase_ = cyclePhase;
        }
    } else if (units == Units::BPM && hostBpm_ > 0.0) {
        // Transport stopped: free-run at the BPM-derived rate so the LFO keeps
        // moving for monitoring/preview, but drop the host lock.
        const double periodBeats = periodInBeats(div, flv);
        const double cyclesPerSecond = (hostBpm_ / 60.0) / periodBeats;
        const double inc = cyclesPerSecond / sampleRate_;
        for (int i = 0; i < numSamples; ++i) {
            phase_ += inc;
            if (phase_ >= 1.0)
                phase_ -= std::floor(phase_);
        }
    } else {
        const double inc = (rateHz > 0.0 ? rateHz : 0.0) / sampleRate_;
        for (int i = 0; i < numSamples; ++i) {
            phase_ += inc;
            if (phase_ >= 1.0)
                phase_ -= std::floor(phase_);
        }
    }

    double effectivePhase = std::fmod(phase_ + phaseOffset, 1.0);
    if (effectivePhase < 0.0)
        effectivePhase += 1.0;
    const double value = evaluateShape(shape, effectivePhase, seed);
    lfoStorage_.store(value, std::memory_order_release);
}

} // namespace dsp_core::audio_pipeline
