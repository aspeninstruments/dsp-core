#include <gtest/gtest.h>
#include "../dsp_core/Source/audio_pipeline/OversamplingWrapper.h"
#include <juce_audio_basics/juce_audio_basics.h>
#include <cmath>
#include <memory>

using namespace dsp_core::audio_pipeline;

namespace {

// Trivial passthrough stage — leaves the buffer untouched. Lets us isolate
// OversamplingWrapper's up/down sampling and channel-count handling from any
// downstream DSP behaviour.
class PassthroughStage : public AudioProcessingStage {
  public:
    void prepareToPlay(double /*sampleRate*/, int /*samplesPerBlock*/, int /*numChannels*/) override {}
    void process(juce::AudioBuffer<double>& /*buffer*/) override {}
    void reset() override {}
    juce::String getName() const override {
        return "Passthrough";
    }
};

void fillSine(juce::AudioBuffer<double>& buffer, double frequencyHz, double sampleRate, double amplitude = 0.5) {
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
        auto* data = buffer.getWritePointer(ch);
        for (int i = 0; i < buffer.getNumSamples(); ++i) {
            data[i] = amplitude * std::sin(2.0 * juce::MathConstants<double>::pi * frequencyHz * i / sampleRate);
        }
    }
}

double bufferPeak(const juce::AudioBuffer<double>& buffer) {
    double peak = 0.0;
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
        const auto* data = buffer.getReadPointer(ch);
        for (int i = 0; i < buffer.getNumSamples(); ++i) {
            peak = std::max(peak, std::abs(data[i]));
        }
    }
    return peak;
}

bool bufferIsFinite(const juce::AudioBuffer<double>& buffer) {
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
        const auto* data = buffer.getReadPointer(ch);
        for (int i = 0; i < buffer.getNumSamples(); ++i) {
            if (!std::isfinite(data[i])) {
                return false;
            }
        }
    }
    return true;
}

} // namespace

TEST(OversamplingWrapperTest, MonoPrepareThenProcess) {
    auto wrapper = std::make_unique<OversamplingWrapper>(std::make_unique<PassthroughStage>(), 3); // 8x

    constexpr double sampleRate = 48000.0;
    constexpr int blockSize = 512;
    wrapper->prepareToPlay(sampleRate, blockSize, 1);

    juce::AudioBuffer<double> buffer(1, blockSize);
    fillSine(buffer, 1000.0, sampleRate);
    const double inputPeak = bufferPeak(buffer);

    wrapper->process(buffer);

    EXPECT_TRUE(bufferIsFinite(buffer));
    EXPECT_GT(bufferPeak(buffer), 0.1) << "1 kHz sine should pass through 8x oversampling without being silenced";
    EXPECT_LT(bufferPeak(buffer), inputPeak * 1.5) << "passthrough should not amplify";
}

TEST(OversamplingWrapperTest, StereoPrepareThenProcess) {
    auto wrapper = std::make_unique<OversamplingWrapper>(std::make_unique<PassthroughStage>(), 3);

    constexpr double sampleRate = 48000.0;
    constexpr int blockSize = 512;
    wrapper->prepareToPlay(sampleRate, blockSize, 2);

    juce::AudioBuffer<double> buffer(2, blockSize);
    fillSine(buffer, 1000.0, sampleRate);

    wrapper->process(buffer);

    EXPECT_TRUE(bufferIsFinite(buffer));
    EXPECT_GT(bufferPeak(buffer), 0.1);
}

// Regression test for the bug this change fixes: OversamplingWrapper used to
// hardcode 2 channels at construction, so calling prepareToPlay then handing
// it a mismatched-channel-count buffer asserted in Debug and read OOB in
// Release. After the fix, prepareToPlay reconstructs the oversamplers when the
// channel count changes and subsequent process() calls must remain safe.
TEST(OversamplingWrapperTest, ChannelCountChangeRebuildsOversamplers) {
    auto wrapper = std::make_unique<OversamplingWrapper>(std::make_unique<PassthroughStage>(), 3);

    constexpr double sampleRate = 48000.0;
    constexpr int blockSize = 512;

    wrapper->prepareToPlay(sampleRate, blockSize, 1);
    {
        juce::AudioBuffer<double> mono(1, blockSize);
        fillSine(mono, 1000.0, sampleRate);
        wrapper->process(mono);
        EXPECT_TRUE(bufferIsFinite(mono));
    }

    wrapper->prepareToPlay(sampleRate, blockSize, 2);
    {
        juce::AudioBuffer<double> stereo(2, blockSize);
        fillSine(stereo, 1000.0, sampleRate);
        wrapper->process(stereo);
        EXPECT_TRUE(bufferIsFinite(stereo));
        EXPECT_GT(bufferPeak(stereo), 0.1);
    }
}
