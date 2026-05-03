#include "OversamplingWrapper.h"

namespace dsp_core::audio_pipeline {

namespace {
constexpr int kMaxOversamplingOrder = 4;
constexpr int kNumOversamplingModes = kMaxOversamplingOrder + 1; // Orders 0-4
} // namespace

OversamplingWrapper::OversamplingWrapper(std::unique_ptr<AudioProcessingStage> wrappedStage, int oversamplingOrder)
    : wrappedStage_(std::move(wrappedStage)), currentOrder_(oversamplingOrder) {
    jassert(wrappedStage_ != nullptr);
    jassert(oversamplingOrder >= 0 && oversamplingOrder <= kMaxOversamplingOrder);

    // Pre-create all oversamplers
    for (int i = 0; i < kNumOversamplingModes; ++i) {
        oversamplers_[i] = std::make_unique<juce::dsp::Oversampling<double>>(
            2, // 2 channels (stereo)
            i, // Oversampling order
            juce::dsp::Oversampling<double>::filterHalfBandPolyphaseIIR,
            true,   // isMaxQuality
            false); // useIntegerLatency — not needed, oversampling wraps both dry and wet paths
    }
}

void OversamplingWrapper::prepareToPlay(double sampleRate, int samplesPerBlock, int numChannels) {
    sampleRate_ = sampleRate;
    maxBlockSize_ = samplesPerBlock;

    // JUCE's Oversampling fixes its channel count at construction; processSamplesUp
    // asserts (and reads OOB in Release) when the input block's channel count differs.
    // Reconstruct all five if the host's channel count moved off what we previously built.
    if (numChannels != oversamplerChannels_) {
        for (int i = 0; i < kNumOversamplingModes; ++i) {
            oversamplers_[i] = std::make_unique<juce::dsp::Oversampling<double>>(
                numChannels,
                i,
                juce::dsp::Oversampling<double>::filterHalfBandPolyphaseIIR,
                true,
                false);
        }
        oversamplerChannels_ = numChannels;
    }

    for (auto& oversampler : oversamplers_) {
        oversampler->initProcessing(samplesPerBlock);
        oversampler->reset();
    }

    // Prepare wrapped stage at oversampled rate
    const int factor = 1 << currentOrder_; // 2^order
    const int oversampledBlockSize = samplesPerBlock * factor;
    wrappedStage_->prepareToPlay(sampleRate * factor, oversampledBlockSize, numChannels);
}

void OversamplingWrapper::process(juce::AudioBuffer<double>& buffer) {
    auto& oversampler = *oversamplers_[currentOrder_];

    // 1. Upsample
    juce::dsp::AudioBlock<double> block(buffer);
    auto oversampledBlock = oversampler.processSamplesUp(block);

    // 2. Create AudioBuffer view of oversampled data
    // PERFORMANCE FIX: Use pre-allocated array instead of std::vector to avoid allocation
    const size_t numChannels = oversampledBlock.getNumChannels();
    jassert(numChannels <= channelPointers_.size());

    for (size_t ch = 0; ch < numChannels; ++ch) {
        channelPointers_[ch] = oversampledBlock.getChannelPointer(ch);
    }

    juce::AudioBuffer<double> oversampledBuffer(channelPointers_.data(), static_cast<int>(numChannels),
                                                static_cast<int>(oversampledBlock.getNumSamples()));

    // 3. Process wrapped stage at high sample rate
    wrappedStage_->process(oversampledBuffer);

    // 4. Downsample
    oversampler.processSamplesDown(block);
}

void OversamplingWrapper::reset() {
    for (auto& oversampler : oversamplers_) {
        oversampler->reset();
    }
    wrappedStage_->reset();
}

juce::String OversamplingWrapper::getName() const {
    const int factor = 1 << currentOrder_;
    return juce::String(factor) + "x(" + wrappedStage_->getName() + ")";
}

int OversamplingWrapper::getLatencySamples() const {
    const auto& oversampler = *oversamplers_[currentOrder_];
    const int oversamplingLatency = static_cast<int>(oversampler.getLatencyInSamples());
    const int wrappedLatency = wrappedStage_->getLatencySamples();

    // Wrapped latency is at oversampled rate, convert to base rate
    const int factor = 1 << currentOrder_;
    return oversamplingLatency + (wrappedLatency / factor);
}

void OversamplingWrapper::setOversamplingOrder(int order) {
    jassert(order >= 0 && order <= kMaxOversamplingOrder);
    if (order == currentOrder_) {
        return;
    }

    currentOrder_ = order;

    // Re-prepare wrapped stage at new sample rate
    const int factor = 1 << currentOrder_;
    const int oversampledBlockSize = maxBlockSize_ * factor;
    wrappedStage_->prepareToPlay(sampleRate_ * factor, oversampledBlockSize, oversamplerChannels_);
}

} // namespace dsp_core::audio_pipeline
