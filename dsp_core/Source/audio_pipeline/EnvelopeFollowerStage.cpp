#include "EnvelopeFollowerStage.h"
#include <algorithm>
#include <cmath>

namespace dsp_core::audio_pipeline {

namespace {

// RMS averaging time constant. Short enough that the integrator's
// linear-amplitude tail doesn't dominate release feel (the leaky integrator
// decays exponentially in linear space, which stretches into a many-τ tail
// when viewed through the dB-domain smoother), still long enough to smooth
// per-sample x² into a meaningful loudness signal across most musical
// fundamentals. Not exposed as a parameter; the user's Attack/Release knobs
// control the post-RMS smoother on top of this.
constexpr double kRmsTauSec = 0.015;

// Nonlinear release shape. rel_eff = rel * (1 + kAccelGain * dropDb / kAccelScaleDb),
// with dropDb capped at kAccelMaxDropDb so deep drops (e.g. peak-detector troughs
// on a sine, or silence after a transient) don't accelerate infinitely. The cap
// keeps the envelope able to ride the peaks of cyclic signals while still letting
// kicks/gates decay several × faster than the nominal release τ.
constexpr double kAccelGain = 3.0;
constexpr double kAccelScaleDb = 12.0;
constexpr double kAccelMaxDropDb = 24.0;

// HPF cutoff safety range. 20 Hz is the lowest useful cutoff; 500 Hz is well
// above the kick/bass-fundamental band the HPF is meant to suppress.
constexpr double kMinHpfHz = 20.0;
constexpr double kMaxHpfHz = 500.0;

} // namespace

EnvelopeFollowerStage::EnvelopeFollowerStage(std::atomic<double>& envelopeStorage) : envelopeStorage_(envelopeStorage) {
    recomputeCoefficients();
}

void EnvelopeFollowerStage::prepareToPlay(double sampleRate, int /*samplesPerBlock*/) {
    sampleRate_ = sampleRate;
    recomputeCoefficients();

    // Allocate the shared coefficients object once here (non-audio thread);
    // subsequent updates mutate it in place.
    const double cutoffHz = hpfCutoffHz_.load(std::memory_order_acquire);
    hpfCoefficients_ = juce::dsp::IIR::Coefficients<double>::makeFirstOrderHighPass(sampleRate_, cutoffHz);
    resizeHpfFiltersForChannels(2);
}

void EnvelopeFollowerStage::setAttackReleaseSec(double attackSec, double releaseSec) {
    attackSec_ = attackSec;
    releaseSec_ = releaseSec;
    recomputeCoefficients();
}

void EnvelopeFollowerStage::setSidechainHpfCutoffHz(double cutoffHz) {
    cutoffHz = juce::jlimit(kMinHpfHz, kMaxHpfHz, cutoffHz);
    hpfCutoffHz_.store(cutoffHz, std::memory_order_release);
    updateHpfCoefficients();
}

void EnvelopeFollowerStage::updateHpfCoefficients() {
    if (sampleRate_ <= 0.0 || hpfCoefficients_ == nullptr)
        return;
    const double cutoffHz = hpfCutoffHz_.load(std::memory_order_acquire);

    // 1st-order Butterworth HPF coefficients computed in place — matches
    // juce::dsp::IIR::Coefficients<>::makeFirstOrderHighPass but without the
    // heap allocation, so this is safe to call from the audio thread.
    //
    // JUCE stores normalized 1st-order coefficients as 3 entries in the order
    // [b0/a0, b1/a0, a1/a0] (a0 is divided out — see assignImpl in JUCE source).
    // Source values for makeFirstOrderHighPass: {b0=1, b1=-1, a0=n+1, a1=n-1}.
    const double n = std::tan(juce::MathConstants<double>::pi * cutoffHz / sampleRate_);
    const double c = 1.0 / (1.0 + n);
    auto& arr = hpfCoefficients_->coefficients;
    arr.set(0, c);         // b0 / a0 =  1 / (n + 1)
    arr.set(1, -c);        // b1 / a0 = -1 / (n + 1)
    arr.set(2, n * c - c); // a1 / a0 = (n - 1) / (n + 1)
}

void EnvelopeFollowerStage::resizeHpfFiltersForChannels(int numChannels) {
    hpfFilters_.resize(static_cast<size_t>(std::max(1, numChannels)));
    for (auto& f : hpfFilters_) {
        f.coefficients = hpfCoefficients_;
        f.reset();
    }
}

void EnvelopeFollowerStage::recomputeCoefficients() {
    if (sampleRate_ <= 0.0)
        return;
    attackCoef_ = (attackSec_ > 0.0) ? 1.0 - std::exp(-1.0 / (sampleRate_ * attackSec_)) : 1.0;
    releaseCoef_ = (releaseSec_ > 0.0) ? 1.0 - std::exp(-1.0 / (sampleRate_ * releaseSec_)) : 1.0;
    rmsCoef_ = 1.0 - std::exp(-1.0 / (sampleRate_ * kRmsTauSec));
}

void EnvelopeFollowerStage::process(juce::AudioBuffer<double>& buffer) {
    if (!enabled_.load(std::memory_order_acquire)) {
        // Publish silence so downstream targets snap to 0 when the input is OFF;
        // otherwise the last computed value would persist in the shared atomic.
        envelopeStorage_.store(0.0, std::memory_order_release);
        return;
    }

    const auto& src = (externalInputBuffer_ != nullptr) ? *externalInputBuffer_ : buffer;
    const int numChannels = src.getNumChannels();
    const int numSamples = src.getNumSamples();
    if (numChannels <= 0 || numSamples <= 0) {
        return;
    }

    // Block-start cache. Loading atomics once amortises the acquire cost
    // and freezes the parameter set for the duration of the block.
    const double sens = sensitivityLinear_.load(std::memory_order_acquire);
    const bool hpfOn = hpfEnabled_.load(std::memory_order_acquire);
    const double atk = attackCoef_;
    const double rel = releaseCoef_;
    const double rmsCoef = rmsCoef_;

    // Resize HPF state to match the current input channel count if it changed.
    // Reuses the shared coefficients object, so no per-block allocation in
    // the steady state.
    if (hpfOn && hpfFilters_.size() != static_cast<size_t>(numChannels)) {
        resizeHpfFiltersForChannels(numChannels);
    }

    double envDb = envDb_;
    double ms2 = ms2_;

    for (int i = 0; i < numSamples; ++i) {
        // 1. Per-channel RMS — read-only on the buffer; HPF operates on a side
        //    copy via filter state, never writes back. Sum-of-squares linking
        //    keeps stereo loudness coherent.
        double sumSq = 0.0;
        for (int ch = 0; ch < numChannels; ++ch) {
            double x = src.getSample(ch, i);
            if (hpfOn)
                x = hpfFilters_[static_cast<size_t>(ch)].processSample(x);
            sumSq += x * x;
        }
        const double meanSq = sumSq / std::max(1, numChannels);
        ms2 += rmsCoef * (meanSq - ms2);
        const double detect = std::sqrt(std::max(ms2, 0.0));

        // 2. Apply pre-detection sensitivity, clamp to unity.
        const double targetLin = std::min(detect * sens, 1.0);
        const double targetDb = juce::Decibels::gainToDecibels(targetLin, kEnvDbFloor_);

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
    for (auto& f : hpfFilters_)
        f.reset();
    envelopeStorage_.store(0.0, std::memory_order_release);
}

} // namespace dsp_core::audio_pipeline
