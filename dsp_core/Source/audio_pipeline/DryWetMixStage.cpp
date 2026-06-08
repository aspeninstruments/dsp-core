#include "DryWetMixStage.h"
#include <juce_core/juce_core.h>

namespace dsp_core::audio_pipeline {

namespace {
constexpr int kMaxChannels = 8;              // Support stereo, 5.1, 7.1
constexpr double kMixRampTimeSeconds = 0.01; // 10ms ramp, matches GainStage

// Power-of-two ceiling for the residual dry-path delay. The shared up/down
// already aligns dry to wet; this only covers any extra wet-path latency, which
// is a handful of samples at most — 1024 is comfortably above it and cheap.
constexpr int kMaxDelaySamples = 1024;
constexpr int kDelayMask = kMaxDelaySamples - 1;

// No-op stage: lets the dry path reuse OversamplingWrapper purely for its
// half-band up/down (so dry picks up the same IIR phase as the wet core).
class IdentityStage : public AudioProcessingStage {
  public:
    void prepareToPlay(double /*sampleRate*/, int /*samplesPerBlock*/, int /*numChannels*/) override {}
    void process(juce::AudioBuffer<double>& /*buffer*/) override {}
    void reset() override {}
    juce::String getName() const override {
        return "Identity";
    }
};
} // namespace

DryWetMixStage::DryWetMixStage(std::unique_ptr<AudioPipeline> effectsPipeline)
    : effectsPipeline_(std::move(effectsPipeline)),
      dryOversampler_(std::make_unique<OversamplingWrapper>(std::make_unique<IdentityStage>(), 0)) {
    jassert(effectsPipeline_ != nullptr);
}

void DryWetMixStage::prepareToPlay(double sampleRate, int samplesPerBlock, int numChannels) {
    dryBuffer_.setSize(kMaxChannels, samplesPerBlock, false, true, true);

    // Pre-allocate the residual dry-path delay line (per-channel ring), cleared
    // so the first blocks read silence — correct alignment while latency fills.
    delayCapacity_ = kMaxDelaySamples;
    delayLine_.setSize(kMaxChannels, delayCapacity_, false, true, true);
    delayLine_.clear();
    delayWritePos_ = 0;

    dryOversampler_->prepareToPlay(sampleRate, samplesPerBlock, numChannels);
    effectsPipeline_->prepareToPlay(sampleRate, samplesPerBlock, numChannels);

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
    dryOversampler_->reset();
    dryBuffer_.clear();
    delayLine_.clear();
    delayWritePos_ = 0;
}

juce::String DryWetMixStage::getName() const {
    return "DryWetMix(" + effectsPipeline_->getName() + ")";
}

int DryWetMixStage::getLatencySamples() const {
    // Plugin-reported latency is the wet path's; the dry path is aligned to it.
    return effectsPipeline_->getLatencySamples();
}

AudioPipeline* DryWetMixStage::getEffectsPipeline() {
    return effectsPipeline_.get();
}

void DryWetMixStage::setMixAmount(double mix) {
    mixSmoothed_.setTargetValue(juce::jlimit(0.0, 1.0, mix));
}

void DryWetMixStage::setDryOversamplingOrder(int order) {
    dryOversampler_->setOversamplingOrder(order);
    refreshLatencyCompensation();
}

void DryWetMixStage::refreshLatencyCompensation() {
    // The dry oversampler already imparts the shared up/down latency; compensate
    // only any residual wet-path latency on top of it (≈0 when the wet path's
    // only latency is that same up/down).
    const int residual = effectsPipeline_->getLatencySamples() - dryOversampler_->getLatencySamples();
    const int maxDelay = delayCapacity_ > 0 ? delayCapacity_ - 1 : 0;
    delaySamples_.store(juce::jlimit(0, maxDelay, residual), std::memory_order_release);
}

void DryWetMixStage::captureDrySignal(const juce::AudioBuffer<double>& buffer) {
    const int numChannels = buffer.getNumChannels();
    const int numSamples = buffer.getNumSamples();

    jassert(numChannels <= dryBuffer_.getNumChannels());
    jassert(numSamples <= dryBuffer_.getNumSamples());

    for (int ch = 0; ch < numChannels; ++ch) {
        dryBuffer_.copyFrom(ch, 0, buffer, ch, 0, numSamples);
    }

    // Run the dry signal through the SAME half-band up/down as the wet core so
    // the blend stays phase-coherent (a sample delay can't reproduce the IIR's
    // per-frequency phase). Order 0 is a true passthrough — skip it so the
    // oversampling-off case is a bit-identical dry copy.
    if (dryOversampler_->getOversamplingOrder() > 0) {
        for (int ch = 0; ch < numChannels; ++ch) {
            dryChannelPtrs_[static_cast<size_t>(ch)] = dryBuffer_.getWritePointer(ch);
        }
        juce::AudioBuffer<double> dryView(dryChannelPtrs_.data(), numChannels, numSamples);
        dryOversampler_->process(dryView);
    }

    applyResidualDelay(numChannels, numSamples);
}

void DryWetMixStage::applyResidualDelay(int numChannels, int numSamples) {
    const int delay = delaySamples_.load(std::memory_order_acquire);
    if (delay <= 0) {
        return; // dry already aligned — the shared up/down handled the latency
    }

    int writePos = delayWritePos_;
    for (int i = 0; i < numSamples; ++i) {
        const int readPos = (writePos - delay + delayCapacity_) & kDelayMask;
        for (int ch = 0; ch < numChannels; ++ch) {
            double* ring = delayLine_.getWritePointer(ch);
            ring[writePos] = dryBuffer_.getSample(ch, i);
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
