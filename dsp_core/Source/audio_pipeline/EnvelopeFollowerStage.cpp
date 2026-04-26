#include "EnvelopeFollowerStage.h"
#include <algorithm>
#include <cmath>

namespace dsp_core::audio_pipeline {

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
}

void EnvelopeFollowerStage::process(juce::AudioBuffer<double>& buffer) {
    if (!enabled_.load(std::memory_order_acquire)) {
        return;
    }

    const double sens = sensitivityLinear_.load(std::memory_order_acquire);

    const int numChannels = buffer.getNumChannels();
    const int numSamples = buffer.getNumSamples();
    const double atk = attackCoef_;
    const double rel = releaseCoef_;
    double env = env_;

    for (int i = 0; i < numSamples; ++i) {
        double peak = 0.0;
        for (int ch = 0; ch < numChannels; ++ch) {
            peak = std::max(peak, std::abs(buffer.getSample(ch, i)));
        }
        const double target = std::min(peak * sens, 1.0);
        const double coef = (target > env) ? atk : rel;
        env += coef * (target - env);
    }

    env_ = env;
    envelopeStorage_.store(env, std::memory_order_release);
}

void EnvelopeFollowerStage::reset() {
    env_ = 0.0;
    envelopeStorage_.store(0.0, std::memory_order_release);
}

} // namespace dsp_core::audio_pipeline
