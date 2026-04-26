#include "EnvelopeFollowerStage.h"
#include <algorithm>
#include <cmath>

namespace dsp_core::audio_pipeline {

namespace {

// RMS averaging time constant. Industry-standard value — short enough to
// track transients, long enough to smooth instantaneous samples into a
// perceptual loudness signal. Not exposed as a parameter; the user's
// Attack/Release knobs control the post-RMS smoother.
constexpr double kRmsTauSec = 0.030;

// Nonlinear release shape. rel_eff = rel * (1 + kAccelGain * dropDb / kAccelScaleDb),
// with dropDb capped at kAccelMaxDropDb so deep drops (e.g. peak-detector troughs
// on a sine, or silence after a transient) don't accelerate infinitely. The cap
// keeps the envelope able to ride the peaks of cyclic signals while still letting
// kicks/gates decay several × faster than the nominal release τ.
constexpr double kAccelGain = 3.0;
constexpr double kAccelScaleDb = 12.0;
constexpr double kAccelMaxDropDb = 24.0;

} // namespace

EnvelopeFollowerStage::EnvelopeFollowerStage(std::atomic<double>& envelopeStorage)
    : envelopeStorage_(envelopeStorage) {
    recomputeCoefficients();
}

void EnvelopeFollowerStage::prepareToPlay(double sampleRate, int /*samplesPerBlock*/) {
    sampleRate_ = sampleRate;
    recomputeCoefficients();
}

void EnvelopeFollowerStage::setAttackReleaseSec(double attackSec, double releaseSec) {
    attackSec_ = attackSec;
    releaseSec_ = releaseSec;
    recomputeCoefficients();
}

void EnvelopeFollowerStage::recomputeCoefficients() {
    if (sampleRate_ <= 0.0) return;
    attackCoef_ = (attackSec_ > 0.0)
                      ? 1.0 - std::exp(-1.0 / (sampleRate_ * attackSec_))
                      : 1.0;
    releaseCoef_ = (releaseSec_ > 0.0)
                       ? 1.0 - std::exp(-1.0 / (sampleRate_ * releaseSec_))
                       : 1.0;
    rmsCoef_ = 1.0 - std::exp(-1.0 / (sampleRate_ * kRmsTauSec));
}

void EnvelopeFollowerStage::process(juce::AudioBuffer<double>& buffer) {
    if (!enabled_.load(std::memory_order_acquire)) {
        return;
    }

    const int numChannels = buffer.getNumChannels();
    const int numSamples = buffer.getNumSamples();

    // Block-start cache. Loading atomics once amortises the acquire cost
    // and freezes the parameter set for the duration of the block.
    const double sens = sensitivityLinear_.load(std::memory_order_acquire);
    const bool rmsMode =
        detectionMode_.load(std::memory_order_acquire) == static_cast<int>(DetectionMode::Rms);
    const double atk = attackCoef_;
    const double rel = releaseCoef_;
    const double rmsCoef = rmsCoef_;

    double envDb = envDb_;
    double ms2 = ms2_;

    for (int i = 0; i < numSamples; ++i) {
        // 1. Per-channel detector input — read-only on the buffer.
        double detect;
        if (rmsMode) {
            double sumSq = 0.0;
            for (int ch = 0; ch < numChannels; ++ch) {
                const double x = buffer.getSample(ch, i);
                sumSq += x * x;
            }
            const double meanSq = sumSq / std::max(1, numChannels);
            ms2 += rmsCoef * (meanSq - ms2);
            detect = std::sqrt(std::max(ms2, 0.0));
        } else {
            double peak = 0.0;
            for (int ch = 0; ch < numChannels; ++ch) {
                peak = std::max(peak, std::abs(buffer.getSample(ch, i)));
            }
            detect = peak;
        }

        // 2. Apply pre-detection sensitivity, clamp to unity.
        const double targetLin = std::min(detect * sens, 1.0);
        const double targetDb =
            juce::Decibels::gainToDecibels(targetLin, kEnvDbFloor_);

        // 3. Choose smoother coefficient. Attack is straight; release
        //    accelerates with dB-drop magnitude (nonlinear release), capped
        //    so cyclic signals can still ride peaks without infinite-trough
        //    acceleration.
        double coef;
        if (targetDb > envDb) {
            coef = atk;
        } else {
            const double dropDb = std::min(envDb - targetDb, kAccelMaxDropDb);
            const double accel = 1.0 + kAccelGain * (dropDb / kAccelScaleDb);
            coef = std::min(rel * accel, 1.0);
        }

        // 4. dB-domain one-pole. Walks envDb toward targetDb at the chosen rate.
        envDb += coef * (targetDb - envDb);
    }

    envDb_ = envDb;
    ms2_ = ms2;

    // Publish linear envelope. decibelsToGain returns 0 when envDb hits the
    // floor, so silence reads as exactly 0 downstream.
    const double envLin = juce::Decibels::decibelsToGain(envDb, kEnvDbFloor_);
    envelopeStorage_.store(envLin, std::memory_order_release);
}

void EnvelopeFollowerStage::reset() {
    envDb_ = kEnvDbFloor_;
    ms2_ = 0.0;
    envelopeStorage_.store(0.0, std::memory_order_release);
}

} // namespace dsp_core::audio_pipeline
