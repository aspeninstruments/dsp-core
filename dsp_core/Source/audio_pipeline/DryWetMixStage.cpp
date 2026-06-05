#include "DryWetMixStage.h"
#include <juce_core/juce_core.h>

namespace dsp_core::audio_pipeline {

namespace {
constexpr int kMaxChannels = 8;              // Support stereo, 5.1, 7.1
constexpr double kMixRampTimeSeconds = 0.01; // 10ms ramp, matches GainStage

// Power-of-two ceiling for the dry-path latency-compensation delay. Half-band
// IIR group delay at base rate is only a handful of samples even at 16x, so 1024
// is comfortably above any real value while staying cheap (1024 * 8ch * 8B = 64KB).
constexpr int kMaxDelaySamples = 1024;
constexpr int kDelayMask = kMaxDelaySamples - 1;
} // namespace

DryWetMixStage::DryWetMixStage(std::unique_ptr<AudioPipeline> effectsPipeline)
    : effectsPipeline_(std::move(effectsPipeline)) {
    jassert(effectsPipeline_ != nullptr);
}

void DryWetMixStage::prepareToPlay(double sampleRate, int samplesPerBlock, int numChannels) {
    dryBuffer_.setSize(kMaxChannels, samplesPerBlock, false, true, true);

    // Pre-allocate the dry-path delay line (per-channel ring) for latency
    // compensation. Cleared so the first blocks read silence — which is the
    // correct alignment while the wet path's latency is still "filling".
    delayCapacity_ = kMaxDelaySamples;
    delayLine_.setSize(kMaxChannels, delayCapacity_, false, true, true);
    delayLine_.clear();
    delayWritePos_ = 0;

    effectsPipeline_->prepareToPlay(sampleRate, samplesPerBlock, numChannels);

    // The effects pipeline now reports its latency (including any interior
    // oversampler) — delay the dry path to match.
    refreshLatencyCompensation();

    mixSmoothed_.reset(sampleRate, kMixRampTimeSeconds);
    mixSmoothed_.setCurrentAndTargetValue(1.0); // 100% wet by default
}

void DryWetMixStage::process(juce::AudioBuffer<double>& buffer) {
    captureDrySignal(buffer);
    effectsPipeline_->process(buffer);
    applyMix(buffer);
}

void DryWetMixStage::reset() {
    effectsPipeline_->reset();
    dryBuffer_.clear();
    delayLine_.clear();
    delayWritePos_ = 0;
}

void DryWetMixStage::refreshLatencyCompensation() {
    const int latency = effectsPipeline_->getLatencySamples();
    delaySamples_.store(juce::jlimit(0, delayCapacity_ > 0 ? delayCapacity_ - 1 : 0, latency),
                        std::memory_order_release);
}

juce::String DryWetMixStage::getName() const {
    return "DryWetMix(" + effectsPipeline_->getName() + ")";
}

int DryWetMixStage::getLatencySamples() const {
    return effectsPipeline_->getLatencySamples();
}

AudioPipeline* DryWetMixStage::getEffectsPipeline() {
    return effectsPipeline_.get();
}

void DryWetMixStage::setMixAmount(double mix) {
    mixSmoothed_.setTargetValue(juce::jlimit(0.0, 1.0, mix));
}

void DryWetMixStage::captureDrySignal(const juce::AudioBuffer<double>& buffer) {
    const int numChannels = buffer.getNumChannels();
    const int numSamples = buffer.getNumSamples();

    jassert(numChannels <= dryBuffer_.getNumChannels());
    jassert(numSamples <= dryBuffer_.getNumSamples());

    const int delay = delaySamples_.load(std::memory_order_acquire);

    if (delay <= 0) {
        // Zero-latency fast path: direct copy, bit-identical to a wet chain with
        // no latency (the common case when oversampling is off / linear).
        for (int ch = 0; ch < numChannels; ++ch) {
            dryBuffer_.copyFrom(ch, 0, buffer, ch, 0, numSamples);
        }
        return;
    }

    // Latency-compensated path: push the input through a per-channel ring and
    // read it back `delay` samples later so the dry blend aligns with the wet
    // output. writePos advances once per sample, shared across channels.
    int writePos = delayWritePos_;
    for (int i = 0; i < numSamples; ++i) {
        const int readPos = (writePos - delay + delayCapacity_) & kDelayMask;
        for (int ch = 0; ch < numChannels; ++ch) {
            double* ring = delayLine_.getWritePointer(ch);
            ring[writePos] = buffer.getSample(ch, i);
            dryBuffer_.setSample(ch, i, ring[readPos]);
        }
        writePos = (writePos + 1) & kDelayMask;
    }
    delayWritePos_ = writePos;
}

void DryWetMixStage::applyMix(juce::AudioBuffer<double>& wetBuffer) {
    const int numChannels = wetBuffer.getNumChannels();
    const int numSamples = wetBuffer.getNumSamples();

    for (int i = 0; i < numSamples; ++i) {
        const double wetGain = mixSmoothed_.getNextValue();
        const double dryGain = 1.0 - wetGain;

        for (int ch = 0; ch < numChannels; ++ch) {
            double* wetData = wetBuffer.getWritePointer(ch);
            const double* dryData = dryBuffer_.getReadPointer(ch);
            wetData[i] = wetGain * wetData[i] + dryGain * dryData[i];
        }
    }
}

} // namespace dsp_core::audio_pipeline
