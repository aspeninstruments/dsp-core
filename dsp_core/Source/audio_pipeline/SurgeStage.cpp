#include "SurgeStage.h"

namespace dsp_core::audio_pipeline {

void SurgeStage::prepareToPlay(double sampleRate, int /*samplesPerBlock*/, int /*numChannels*/) {
    sampleRate_ = sampleRate;
    recomputeStep();
}

void SurgeStage::setSurgeDurationSec(double seconds) {
    surgeDurationSec_ = seconds;
    recomputeStep();
}

void SurgeStage::recomputeStep() {
    if (sampleRate_ > 0.0) {
        const double oneSample = 1.0 / sampleRate_;
        stepPerSample_ = (surgeDurationSec_ > oneSample) ? (1.0 / (sampleRate_ * surgeDurationSec_)) : 1.0;
    }
}

void SurgeStage::process(juce::AudioBuffer<double>& buffer) {
    if (!enabled_.load(std::memory_order_acquire)) {
        return;
    }

    const int numChannels = std::min(buffer.getNumChannels(), static_cast<int>(state_.size()));
    const int numSamples = buffer.getNumSamples();
    const double step = stepPerSample_;

    for (int ch = 0; ch < numChannels; ++ch) {
        auto& s = state_[ch];
        auto* data = buffer.getWritePointer(ch);

        for (int i = 0; i < numSamples; ++i) {
            const double x = data[i];

            if (x > 1.0) {
                s.wPos = std::min(1.0, s.wPos + step);
                s.wNeg = std::max(0.0, s.wNeg - step);
            } else if (x < -1.0) {
                s.wNeg = std::min(1.0, s.wNeg + step);
                s.wPos = std::max(0.0, s.wPos - step);
            } else {
                s.wPos = std::max(0.0, s.wPos - step);
                s.wNeg = std::max(0.0, s.wNeg - step);
            }

            const double w = shapedWeight((x >= 0.0) ? s.wPos : s.wNeg);
            const double clamped = std::clamp(x, -1.0, 1.0);
            data[i] = (1.0 - w) * x + w * clamped;
        }
    }
}

void SurgeStage::reset() {
    for (auto& s : state_) {
        s.wPos = 0.0;
        s.wNeg = 0.0;
    }
}

} // namespace dsp_core::audio_pipeline
