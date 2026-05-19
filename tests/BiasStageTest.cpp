#include <gtest/gtest.h>
#include <dsp_core/dsp_core.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <memory>
#include <vector>

using namespace dsp_core::audio_pipeline;

class BiasStageTest : public ::testing::Test {
  protected:
    static constexpr double kSampleRate = 48000.0;

    void SetUp() override {
        stage_ = std::make_unique<BiasStage>();
        stage_->prepareToPlay(kSampleRate, 512, 2);
        stage_->setEnabled(true);
        stage_->setBias(0.0);
    }

    juce::AudioBuffer<double> makeConstantBuffer(double value, int numChannels = 2, int numSamples = 256) {
        juce::AudioBuffer<double> buf(numChannels, numSamples);
        for (int ch = 0; ch < numChannels; ++ch) {
            auto* d = buf.getWritePointer(ch);
            for (int i = 0; i < numSamples; ++i)
                d[i] = value;
        }
        return buf;
    }

    std::unique_ptr<BiasStage> stage_;
};

TEST_F(BiasStageTest, DisabledIsIdentity) {
    stage_->setEnabled(false);
    stage_->setBias(0.5);
    auto buf = makeConstantBuffer(0.25);
    stage_->process(buf);
    for (int ch = 0; ch < buf.getNumChannels(); ++ch) {
        for (int i = 0; i < buf.getNumSamples(); ++i) {
            EXPECT_EQ(buf.getSample(ch, i), 0.25);
        }
    }
}

TEST_F(BiasStageTest, EnabledWithZeroBiasIsIdentity) {
    stage_->setBias(0.0);
    auto buf = makeConstantBuffer(0.3);
    stage_->process(buf);
    for (int ch = 0; ch < buf.getNumChannels(); ++ch) {
        for (int i = 0; i < buf.getNumSamples(); ++i) {
            EXPECT_EQ(buf.getSample(ch, i), 0.3);
        }
    }
}

TEST_F(BiasStageTest, PositiveBiasAddsConstantOffset) {
    stage_->setBias(0.25);
    auto buf = makeConstantBuffer(0.0);
    stage_->process(buf);
    for (int ch = 0; ch < buf.getNumChannels(); ++ch) {
        for (int i = 0; i < buf.getNumSamples(); ++i) {
            EXPECT_DOUBLE_EQ(buf.getSample(ch, i), 0.25);
        }
    }
}

TEST_F(BiasStageTest, NegativeBiasShiftsDownward) {
    stage_->setBias(-0.5);
    auto buf = makeConstantBuffer(0.4);
    stage_->process(buf);
    for (int ch = 0; ch < buf.getNumChannels(); ++ch) {
        for (int i = 0; i < buf.getNumSamples(); ++i) {
            EXPECT_DOUBLE_EQ(buf.getSample(ch, i), -0.1);
        }
    }
}

TEST_F(BiasStageTest, BiasAppliedToVaryingSignal) {
    stage_->setBias(0.1);
    const std::vector<double> inputs = {-0.7, -0.3, 0.0, 0.2, 0.6};
    juce::AudioBuffer<double> buf(2, static_cast<int>(inputs.size()));
    for (int ch = 0; ch < 2; ++ch) {
        auto* d = buf.getWritePointer(ch);
        for (int i = 0; i < buf.getNumSamples(); ++i)
            d[i] = inputs[static_cast<size_t>(i)];
    }
    stage_->process(buf);
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < buf.getNumSamples(); ++i) {
            EXPECT_NEAR(buf.getSample(ch, i), inputs[static_cast<size_t>(i)] + 0.1, 1e-12);
        }
    }
}

TEST_F(BiasStageTest, BiasUpdateAppliesOnNextBlock) {
    stage_->setBias(0.2);
    auto buf1 = makeConstantBuffer(0.0);
    stage_->process(buf1);
    EXPECT_DOUBLE_EQ(buf1.getSample(0, 0), 0.2);

    stage_->setBias(-0.4);
    auto buf2 = makeConstantBuffer(0.0);
    stage_->process(buf2);
    EXPECT_DOUBLE_EQ(buf2.getSample(0, 0), -0.4);
}

TEST_F(BiasStageTest, ResetIsNoOp) {
    stage_->setBias(0.3);
    stage_->reset();
    auto buf = makeConstantBuffer(0.0);
    stage_->process(buf);
    EXPECT_DOUBLE_EQ(buf.getSample(0, 0), 0.3);
}

TEST_F(BiasStageTest, GetBiasReturnsLastSet) {
    stage_->setBias(0.42);
    EXPECT_DOUBLE_EQ(stage_->getBias(), 0.42);
    stage_->setBias(-0.99);
    EXPECT_DOUBLE_EQ(stage_->getBias(), -0.99);
}

TEST_F(BiasStageTest, IsEnabledTracksSetEnabled) {
    stage_->setEnabled(false);
    EXPECT_FALSE(stage_->isEnabled());
    stage_->setEnabled(true);
    EXPECT_TRUE(stage_->isEnabled());
}
