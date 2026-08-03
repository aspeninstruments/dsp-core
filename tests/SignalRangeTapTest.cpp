#include <gtest/gtest.h>
#include <dsp_core/dsp_core.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <cmath>
#include <memory>

using namespace dsp_core::audio_pipeline;

class SignalRangeTapTest : public ::testing::Test {
  protected:
    static constexpr double kSampleRate = 48000.0;
    static constexpr int kBlockSize = 512;
    // 0.1s hold at 48kHz = 4800 samples = 9.375 blocks; the window expires
    // during the 10th block after the one that opened it.
    static constexpr int kBlocksToOutlastHold = 10;

    void SetUp() override {
        tap_ = std::make_unique<SignalRangeTap>(state_);
        tap_->prepareToPlay(kSampleRate, kBlockSize, 2);
    }

    juce::AudioBuffer<double> makeConstantBuffer(double value, int numChannels = 2, int numSamples = kBlockSize) {
        juce::AudioBuffer<double> buf(numChannels, numSamples);
        for (int ch = 0; ch < numChannels; ++ch) {
            auto* d = buf.getWritePointer(ch);
            for (int i = 0; i < numSamples; ++i)
                d[i] = value;
        }
        return buf;
    }

    // Silent buffer except one negative and one positive extreme.
    juce::AudioBuffer<double> makeExtremesBuffer(double minValue, double maxValue) {
        auto buf = makeConstantBuffer(0.0);
        buf.setSample(0, 3, minValue);
        buf.setSample(0, 7, maxValue);
        return buf;
    }

    double publishedMin() const {
        return state_.minValue.load(std::memory_order_acquire);
    }
    double publishedMax() const {
        return state_.maxValue.load(std::memory_order_acquire);
    }

    SignalRangeState state_;
    std::unique_ptr<SignalRangeTap> tap_;
};

TEST_F(SignalRangeTapTest, PassthroughLeavesAudioUntouched) {
    juce::AudioBuffer<double> buf(2, kBlockSize);
    for (int ch = 0; ch < 2; ++ch) {
        auto* d = buf.getWritePointer(ch);
        for (int i = 0; i < kBlockSize; ++i)
            d[i] = std::sin(0.01 * i) * (ch == 0 ? 0.9 : -0.6);
    }
    juce::AudioBuffer<double> expected(2, kBlockSize);
    for (int ch = 0; ch < 2; ++ch)
        expected.copyFrom(ch, 0, buf, ch, 0, kBlockSize);

    tap_->process(buf);

    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < kBlockSize; ++i) {
            EXPECT_EQ(buf.getSample(ch, i), expected.getSample(ch, i));
        }
    }
}

TEST_F(SignalRangeTapTest, PublishesSignedMinMaxIndependently) {
    auto buf = makeExtremesBuffer(-0.2, 0.9);
    tap_->process(buf);
    EXPECT_DOUBLE_EQ(publishedMin(), -0.2);
    EXPECT_DOUBLE_EQ(publishedMax(), 0.9);
}

TEST_F(SignalRangeTapTest, MultiChannelMergesExtremes) {
    auto buf = makeConstantBuffer(0.0);
    buf.setSample(0, 5, -0.7); // min lives on ch0
    buf.setSample(1, 9, 0.4);  // max lives on ch1
    tap_->process(buf);
    EXPECT_DOUBLE_EQ(publishedMin(), -0.7);
    EXPECT_DOUBLE_EQ(publishedMax(), 0.4);
}

TEST_F(SignalRangeTapTest, HoldWindowRidesAcrossBlocks) {
    auto wide = makeExtremesBuffer(-0.8, 0.8);
    tap_->process(wide);

    // Quiet blocks inside the hold window keep the published extremes wide.
    for (int i = 0; i < kBlocksToOutlastHold - 1; ++i) {
        auto quiet = makeConstantBuffer(0.0);
        tap_->process(quiet);
        EXPECT_DOUBLE_EQ(publishedMin(), -0.8) << "collapsed early at quiet block " << i;
        EXPECT_DOUBLE_EQ(publishedMax(), 0.8) << "collapsed early at quiet block " << i;
    }

    // Once the window expires, the extremes restart from the current block.
    auto quiet = makeConstantBuffer(0.0);
    tap_->process(quiet);
    EXPECT_DOUBLE_EQ(publishedMin(), 0.0);
    EXPECT_DOUBLE_EQ(publishedMax(), 0.0);
}

TEST_F(SignalRangeTapTest, HoldWindowScalesWithPreparedRate) {
    // Inside an oversampled group prepareToPlay reports the oversampled rate;
    // the window must stretch with it. At 2x the 48k window (9600 samples),
    // the block count that collapsed the 48k test must still ride.
    tap_->prepareToPlay(kSampleRate * 2.0, kBlockSize, 2);
    auto wide = makeExtremesBuffer(-0.8, 0.8);
    tap_->process(wide);
    for (int i = 0; i < kBlocksToOutlastHold; ++i) {
        auto quiet = makeConstantBuffer(0.0);
        tap_->process(quiet);
    }
    EXPECT_DOUBLE_EQ(publishedMin(), -0.8);
    EXPECT_DOUBLE_EQ(publishedMax(), 0.8);
}

TEST_F(SignalRangeTapTest, DcInputCollapsesToPoint) {
    for (int i = 0; i <= kBlocksToOutlastHold; ++i) {
        auto dc = makeConstantBuffer(0.4);
        tap_->process(dc);
    }
    EXPECT_DOUBLE_EQ(publishedMin(), 0.4);
    EXPECT_DOUBLE_EQ(publishedMax(), 0.4);
}

TEST_F(SignalRangeTapTest, BlockCountHeartbeatIncrements) {
    const auto before = state_.blockCount.load(std::memory_order_acquire);
    auto buf = makeConstantBuffer(0.1);
    tap_->process(buf);
    tap_->process(buf);
    EXPECT_EQ(state_.blockCount.load(std::memory_order_acquire), before + 2);

    // Empty buffers are not audio activity — no heartbeat.
    juce::AudioBuffer<double> empty(2, 0);
    tap_->process(empty);
    EXPECT_EQ(state_.blockCount.load(std::memory_order_acquire), before + 2);
}

TEST_F(SignalRangeTapTest, ResetClearsPublishedRange) {
    auto buf = makeExtremesBuffer(-0.5, 0.6);
    tap_->process(buf);
    const auto heartbeat = state_.blockCount.load(std::memory_order_acquire);

    tap_->reset();
    EXPECT_DOUBLE_EQ(publishedMin(), 0.0);
    EXPECT_DOUBLE_EQ(publishedMax(), 0.0);
    // Heartbeat stays monotonic across reset.
    EXPECT_EQ(state_.blockCount.load(std::memory_order_acquire), heartbeat);

    // And the hold window restarts cleanly after reset.
    auto next = makeExtremesBuffer(-0.1, 0.2);
    tap_->process(next);
    EXPECT_DOUBLE_EQ(publishedMin(), -0.1);
    EXPECT_DOUBLE_EQ(publishedMax(), 0.2);
}
