#include "GainStage.h"

namespace dsp_core::audio_pipeline {

namespace {
constexpr double kGainRampTimeSeconds = 0.01; // 10ms ramp time
} // namespace

void GainStage::prepareToPlay(double sampleRate, int samplesPerBlock, int numChannels) {
    numChannels_ = numChannels;

    // CRITICAL: Prepare gainProcessor ONCE in prepareToPlay, NOT in process()
    // Calling prepare() in audio thread causes allocations!
    juce::dsp::ProcessSpec spec{};
    spec.sampleRate = sampleRate;
    spec.maximumBlockSize = static_cast<juce::uint32>(samplesPerBlock);
    spec.numChannels = static_cast<juce::uint32>(numChannels_);
    gainProcessor_.prepare(spec);

    // Configure per-sample smoothing to prevent clicks during gain changes
    gainProcessor_.setRampDurationSeconds(kGainRampTimeSeconds);

    isPrepared_ = true;
}

void GainStage::process(juce::AudioBuffer<double>& buffer) {
    if (!isPrepared_) {
        jassertfalse; // Must call prepareToPlay() first
        return;
    }

    // juce::dsp::Gain handles per-sample smoothing internally via setRampDurationSeconds()
    juce::dsp::AudioBlock<double> block(buffer);
    const juce::dsp::ProcessContextReplacing<double> context(block);
    gainProcessor_.process(context);
}

void GainStage::reset() {
    gainProcessor_.reset();
}

void GainStage::setGainDB(double gainDB) {
    const double linear = juce::Decibels::decibelsToGain(gainDB);
    gainProcessor_.setGainLinear(linear);
}

} // namespace dsp_core::audio_pipeline
