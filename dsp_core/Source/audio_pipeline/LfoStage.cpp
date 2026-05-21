#include "LfoStage.h"
#include "../Services/PerlinNoiseService.h"
#include <cmath>

namespace dsp_core::audio_pipeline {

namespace {
constexpr double kTwoPi = 6.283185307179586476925286766559;
constexpr double kPerlinCellsPerCycle = 4.0;
// Keep the Random shape's noise coordinate bounded so the int cast inside
// perlinNoise1D() can never overflow. 2^24 cycles is months of continuous
// free-running playback before the noise window recurs — effectively never.
constexpr double kRandomWrapCycles = 16777216.0;
} // namespace

LfoStage::LfoStage(std::atomic<double>& lfoStorage) : lfoStorage_(lfoStorage) {}

void LfoStage::prepareToPlay(double sampleRate, int /*samplesPerBlock*/, int /*numChannels*/) {
    sampleRate_ = sampleRate > 0.0 ? sampleRate : 48000.0;
}

void LfoStage::reset() {
    phase_ = 0.0;
    cyclePosition_ = 0.0;
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

double LfoStage::evaluateShape(Shape s, double phase, double cyclePosition, unsigned int seed) {
    // phase is in [0, 1); cyclePosition is the unwrapped running cycle count.
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
        // Sample Perlin noise along the continuously advancing cycle position,
        // so every cycle traverses a fresh region of noise space and the shape
        // never repeats. Sampling phase * cells instead would retrace the same
        // [0, cells) window each cycle, turning Random into a fixed wavetable.
        // perlinNoise1D returns ~[-1, 1]; map to [0, 1].
        const double n =
            Services::PerlinNoiseService::perlinNoise1D(cyclePosition * kPerlinCellsPerCycle, seed);
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
        // tight to the beat regardless of DAW jitter. cyclePosition tracks the
        // unwrapped cycle count so Random follows the song timeline.
        const double periodBeats = periodInBeats(div, flv);
        const double beatsPerSample = hostBpm_ / (60.0 * sampleRate_);
        for (int i = 0; i < numSamples; ++i) {
            const double ppq = hostPpq_ + beatsPerSample * static_cast<double>(i);
            const double cyclesTotal = ppq / periodBeats;
            cyclePosition_ = cyclesTotal;
            double cyclePhase = std::fmod(cyclesTotal, 1.0);
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
            cyclePosition_ += inc;
            if (phase_ >= 1.0)
                phase_ -= std::floor(phase_);
        }
    } else {
        const double inc = (rateHz > 0.0 ? rateHz : 0.0) / sampleRate_;
        for (int i = 0; i < numSamples; ++i) {
            phase_ += inc;
            cyclePosition_ += inc;
            if (phase_ >= 1.0)
                phase_ -= std::floor(phase_);
        }
    }

    // Wrap the unbounded cycle counter into a safe range for the Random shape.
    // Wrapping by a whole number of cycles leaves phase alignment untouched.
    cyclePosition_ = std::fmod(cyclePosition_, kRandomWrapCycles);

    double effectivePhase = std::fmod(phase_ + phaseOffset, 1.0);
    if (effectivePhase < 0.0)
        effectivePhase += 1.0;
    const double value = evaluateShape(shape, effectivePhase, cyclePosition_ + phaseOffset, seed);
    lfoStorage_.store(value, std::memory_order_release);
}

} // namespace dsp_core::audio_pipeline
