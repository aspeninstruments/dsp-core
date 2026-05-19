#include <gtest/gtest.h>
#include <dsp_core/dsp_core.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <memory>
#include <vector>

using namespace dsp_core::audio_pipeline;

class BiasStageTest : public ::testing::Test {
  protected:
    static constexpr double kSampleRate = 48000.0;
    // 10ms ramp at 48kHz = 480 samples. Use a comfortable margin so
    // steady-state tests can drive past the ramp without flakiness.
    static constexpr int kSettleSamples = 1024;

    void SetUp() override {
        stage_ = std::make_unique<BiasStage>();
        stage_->prepareToPlay(kSampleRate, 512, 2);
        stage_->setEnabled(true);
        stage_->setBias(0.0);
        stage_->reset(); // snap smoother to 0
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

    // Set bias and run silent samples until the smoother settles to the target,
    // so subsequent assertions can rely on the steady-state value.
    void setBiasAndSettle(double value) {
        stage_->setBias(value);
        juce::AudioBuffer<double> warmup(2, kSettleSamples);
        warmup.clear();
        stage_->process(warmup);
    }

    std::unique_ptr<BiasStage> stage_;
};

TEST_F(BiasStageTest, DisabledIsIdentity) {
    stage_->setEnabled(false);
    stage_->setBias(0.5);
    auto buf = makeConstantBuffer(0.25);
    stage_->process(buf);
    // Disabled drives the smoother toward 0; SetUp already snapped it to 0,
    // so output is exact identity.
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

TEST_F(BiasStageTest, PositiveBiasAddsConstantOffsetAtSteadyState) {
    setBiasAndSettle(0.25);
    auto buf = makeConstantBuffer(0.0);
    stage_->process(buf);
    for (int ch = 0; ch < buf.getNumChannels(); ++ch) {
        for (int i = 0; i < buf.getNumSamples(); ++i) {
            EXPECT_DOUBLE_EQ(buf.getSample(ch, i), 0.25);
        }
    }
}

TEST_F(BiasStageTest, NegativeBiasShiftsDownwardAtSteadyState) {
    setBiasAndSettle(-0.5);
    auto buf = makeConstantBuffer(0.4);
    stage_->process(buf);
    for (int ch = 0; ch < buf.getNumChannels(); ++ch) {
        for (int i = 0; i < buf.getNumSamples(); ++i) {
            EXPECT_DOUBLE_EQ(buf.getSample(ch, i), -0.1);
        }
    }
}

TEST_F(BiasStageTest, BiasAppliedToVaryingSignalAtSteadyState) {
    setBiasAndSettle(0.1);
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

TEST_F(BiasStageTest, RampsGraduallyOnBiasChange) {
    // Going from 0 → 0.5 over a 10ms ramp at 48kHz = 480 samples. The first
    // sample is one step into the ramp, not the full target. The last sample
    // of a 480-sample buffer should be at (or very near) the target.
    stage_->setBias(0.5);
    juce::AudioBuffer<double> buf(1, 480);
    buf.clear();
    stage_->process(buf);

    // First sample is partway, not 0.5
    EXPECT_GT(buf.getSample(0, 0), 0.0);
    EXPECT_LT(buf.getSample(0, 0), 0.5);

    // Monotonically increasing toward the target
    for (int i = 1; i < buf.getNumSamples(); ++i) {
        EXPECT_GE(buf.getSample(0, i), buf.getSample(0, i - 1));
    }

    // Final sample arrives at the target (allow tiny FP slack)
    EXPECT_NEAR(buf.getSample(0, buf.getNumSamples() - 1), 0.5, 1e-9);
}

TEST_F(BiasStageTest, DisablingRampsResidualOffsetToZero) {
    // With bias settled at +0.4, disabling the stage should ramp the offset
    // back to 0 instead of cutting it abruptly (which would click).
    setBiasAndSettle(0.4);
    stage_->setEnabled(false);

    juce::AudioBuffer<double> buf(1, 480);
    buf.clear();
    stage_->process(buf);

    // First sample is still close to +0.4 (one ramp step in); last sample is 0
    EXPECT_GT(buf.getSample(0, 0), 0.0);
    EXPECT_LT(buf.getSample(0, 0), 0.4);
    EXPECT_NEAR(buf.getSample(0, buf.getNumSamples() - 1), 0.0, 1e-9);
}

TEST_F(BiasStageTest, BiasUpdatePropagatesAcrossBlocks) {
    setBiasAndSettle(0.2);
    auto buf1 = makeConstantBuffer(0.0);
    stage_->process(buf1);
    EXPECT_DOUBLE_EQ(buf1.getSample(0, 0), 0.2);

    setBiasAndSettle(-0.4);
    auto buf2 = makeConstantBuffer(0.0);
    stage_->process(buf2);
    EXPECT_DOUBLE_EQ(buf2.getSample(0, 0), -0.4);
}

TEST_F(BiasStageTest, ResetSnapsSmootherToCurrentBias) {
    // After a stream restart (host stop/start) the smoother snaps to the
    // most recently set bias so the first audio block doesn't ramp from a
    // stale value.
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
