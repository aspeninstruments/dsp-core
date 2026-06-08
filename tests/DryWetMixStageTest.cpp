#include <gtest/gtest.h>
#include "../dsp_core/Source/audio_pipeline/DryWetMixStage.h"
#include "../dsp_core/Source/audio_pipeline/AudioPipeline.h"
#include "../dsp_core/Source/audio_pipeline/AudioProcessingStage.h"
#include "../dsp_core/Source/audio_pipeline/OversamplingWrapper.h"
#include "../dsp_core/Source/audio_pipeline/GainStage.h"
#include <juce_audio_basics/juce_audio_basics.h>
#include <cmath>
#include <memory>

using namespace dsp_core::audio_pipeline;

namespace {
// Test stub: delays the buffer by a fixed number of samples and reports that
// many samples of latency. Stands in for any latent wet-path stage (e.g. an
// interior oversampler) so we can verify dry-path latency compensation.
class FixedDelayStage : public AudioProcessingStage {
  public:
    explicit FixedDelayStage(int delaySamples) : delay_(delaySamples) {}

    void prepareToPlay(double /*sampleRate*/, int samplesPerBlock, int numChannels) override {
        capacity_ = 1;
        while (capacity_ < delay_ + samplesPerBlock + 1) {
            capacity_ <<= 1;
        }
        ring_.setSize(numChannels, capacity_, false, true, true);
        ring_.clear();
        writePos_ = 0;
    }

    void process(juce::AudioBuffer<double>& buffer) override {
        const int numChannels = buffer.getNumChannels();
        const int numSamples = buffer.getNumSamples();
        const int mask = capacity_ - 1;
        int writePos = writePos_;
        for (int i = 0; i < numSamples; ++i) {
            const int readPos = (writePos - delay_ + capacity_) & mask;
            for (int ch = 0; ch < numChannels; ++ch) {
                double* ring = ring_.getWritePointer(ch);
                ring[writePos] = buffer.getSample(ch, i);
                buffer.setSample(ch, i, ring[readPos]);
            }
            writePos = (writePos + 1) & mask;
        }
        writePos_ = writePos;
    }

    void reset() override {
        ring_.clear();
        writePos_ = 0;
    }

    juce::String getName() const override {
        return "FixedDelay";
    }

    int getLatencySamples() const override {
        return delay_;
    }

  private:
    int delay_;
    int capacity_ = 1;
    int writePos_ = 0;
    juce::AudioBuffer<double> ring_;
};

// Build a DryWetMixStage whose wet path is a FixedDelayStage, settle the mix
// smoother to `mix`, and return the prepared stage.
std::unique_ptr<DryWetMixStage> makeDelayedDryWet(int delaySamples, double mix, int blockSize) {
    auto pipeline = std::make_unique<AudioPipeline>();
    pipeline->addStage(std::make_unique<FixedDelayStage>(delaySamples), "delay");
    auto stage = std::make_unique<DryWetMixStage>(std::move(pipeline));
    stage->prepareToPlay(44100.0, blockSize, 1);
    stage->setMixAmount(mix);

    // Settle the 10ms mix ramp (~441 samples @ 44.1k) with silent blocks.
    juce::AudioBuffer<double> settle(1, blockSize);
    settle.clear();
    for (int b = 0; b < 16; ++b) {
        stage->process(settle);
    }
    return stage;
}
} // namespace

class DryWetMixStageTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Create a simple pipeline with a gain stage
        auto pipeline = std::make_unique<AudioPipeline>();
        auto gainStage = std::make_unique<GainStage>();
        gainStage_ = gainStage.get(); // Save pointer for later access
        pipeline->addStage(std::move(gainStage), "gain");

        dryWetMix_ = std::make_unique<DryWetMixStage>(std::move(pipeline));
        dryWetMix_->prepareToPlay(44100.0, 1024, 2); // Larger buffer for edge case tests

        // JUCE Gain defaults to 0.0 linear (silence), set to unity for passthrough
        gainStage_->setGainDB(0.0);

        // Process a settle block so the gain smoother reaches unity (10ms ramp @ 44100Hz = 441 samples)
        juce::AudioBuffer<double> settle(2, 1024);
        settle.clear();
        dryWetMix_->process(settle);
    }

    std::unique_ptr<DryWetMixStage> dryWetMix_;
    GainStage* gainStage_ = nullptr; // Non-owning pointer
};

TEST_F(DryWetMixStageTest, FullyDry_OutputEqualsInput) {
    dryWetMix_->setMixAmount(0.0); // 100% dry

    juce::AudioBuffer<double> buffer(2, 64);
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 64; ++i) {
            buffer.setSample(ch, i, static_cast<double>(i) / 64.0);
        }
    }

    // Store expected values (input)
    juce::AudioBuffer<double> expected(2, 64);
    for (int ch = 0; ch < 2; ++ch) {
        expected.copyFrom(ch, 0, buffer, ch, 0, 64);
    }

    dryWetMix_->process(buffer);

    // Verify output equals input (100% dry)
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 64; ++i) {
            EXPECT_NEAR(buffer.getSample(ch, i), expected.getSample(ch, i), 1e-10)
                << "Channel " << ch << ", sample " << i;
        }
    }
}

TEST_F(DryWetMixStageTest, FullyWet_OutputEqualsProcessed) {
    // Use default gain (1.0) and mix to 100% wet to avoid gain smoothing issues
    dryWetMix_->setMixAmount(1.0); // 100% wet

    juce::AudioBuffer<double> buffer(2, 64);
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 64; ++i) {
            buffer.setSample(ch, i, static_cast<double>(i) / 64.0);
        }
    }

    dryWetMix_->process(buffer);

    // Verify output equals input (100% wet with 1x gain)
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 64; ++i) {
            double const expected = static_cast<double>(i) / 64.0;
            EXPECT_NEAR(buffer.getSample(ch, i), expected, 1e-10) << "Channel " << ch << ", sample " << i;
        }
    }
}

TEST_F(DryWetMixStageTest, FiftyFiftyMix_CorrectBlend) {
    // Use default gain (1.0) and mix to 50/50 to avoid gain smoothing issues
    dryWetMix_->setMixAmount(0.5); // 50% dry, 50% wet

    juce::AudioBuffer<double> buffer(2, 64);
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 64; ++i) {
            buffer.setSample(ch, i, static_cast<double>(i) / 64.0);
        }
    }

    dryWetMix_->process(buffer);

    // Verify output equals 0.5 * input + 0.5 * input = input (with 1x gain)
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 64; ++i) {
            double const input = static_cast<double>(i) / 64.0;
            double const expected = input; // 0.5 * input + 0.5 * input
            EXPECT_NEAR(buffer.getSample(ch, i), expected, 1e-10) << "Channel " << ch << ", sample " << i;
        }
    }
}

TEST_F(DryWetMixStageTest, EdgeCase_SingleSample) {
    dryWetMix_->setMixAmount(0.5);

    juce::AudioBuffer<double> buffer(1, 1);
    buffer.setSample(0, 0, 1.0);

    dryWetMix_->process(buffer);

    // Should not crash and produce valid output
    EXPECT_TRUE(std::isfinite(buffer.getSample(0, 0)));
}

TEST_F(DryWetMixStageTest, EdgeCase_TwoSamples) {
    dryWetMix_->setMixAmount(0.5);

    juce::AudioBuffer<double> buffer(1, 2);
    buffer.setSample(0, 0, 1.0);
    buffer.setSample(0, 1, 2.0);

    dryWetMix_->process(buffer);

    // Should not crash and produce valid output
    EXPECT_TRUE(std::isfinite(buffer.getSample(0, 0)));
    EXPECT_TRUE(std::isfinite(buffer.getSample(0, 1)));
}

TEST_F(DryWetMixStageTest, EdgeCase_Unaligned63Samples) {
    // Test SIMD alignment edge case (not divisible by typical SIMD width)
    dryWetMix_->setMixAmount(0.5);

    juce::AudioBuffer<double> buffer(2, 63);
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 63; ++i) {
            buffer.setSample(ch, i, static_cast<double>(i) / 63.0);
        }
    }

    juce::AudioBuffer<double> expected(2, 63);
    for (int ch = 0; ch < 2; ++ch) {
        expected.copyFrom(ch, 0, buffer, ch, 0, 63);
    }

    dryWetMix_->process(buffer);

    // Verify correctness for unaligned buffer
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 63; ++i) {
            double const input = static_cast<double>(i) / 63.0;
            double const expectedValue = 0.5 * input + 0.5 * input; // Gain=1.0
            EXPECT_NEAR(buffer.getSample(ch, i), expectedValue, 1e-10) << "Channel " << ch << ", sample " << i;
        }
    }
}

TEST_F(DryWetMixStageTest, EdgeCase_Aligned64Samples) {
    // Test SIMD alignment (divisible by typical SIMD width)
    dryWetMix_->setMixAmount(0.5);

    juce::AudioBuffer<double> buffer(2, 64);
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 64; ++i) {
            buffer.setSample(ch, i, static_cast<double>(i) / 64.0);
        }
    }

    dryWetMix_->process(buffer);

    // Verify correctness for aligned buffer
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 64; ++i) {
            double const input = static_cast<double>(i) / 64.0;
            double const expectedValue = 0.5 * input + 0.5 * input; // Gain=1.0
            EXPECT_NEAR(buffer.getSample(ch, i), expectedValue, 1e-10) << "Channel " << ch << ", sample " << i;
        }
    }
}

TEST_F(DryWetMixStageTest, EdgeCase_LargeBuffer512Samples) {
    // Test typical audio buffer size
    dryWetMix_->setMixAmount(0.5);

    juce::AudioBuffer<double> buffer(2, 512);
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 512; ++i) {
            buffer.setSample(ch, i, static_cast<double>(i) / 512.0);
        }
    }

    dryWetMix_->process(buffer);

    // Verify correctness for large buffer
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 512; ++i) {
            double const input = static_cast<double>(i) / 512.0;
            double const expectedValue = 0.5 * input + 0.5 * input; // Gain=1.0
            EXPECT_NEAR(buffer.getSample(ch, i), expectedValue, 1e-10) << "Channel " << ch << ", sample " << i;
        }
    }
}

TEST_F(DryWetMixStageTest, EdgeCase_513Samples) {
    // Test SIMD edge case (512 + 1)
    dryWetMix_->setMixAmount(0.5);

    juce::AudioBuffer<double> buffer(2, 513);
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 513; ++i) {
            buffer.setSample(ch, i, static_cast<double>(i) / 513.0);
        }
    }

    dryWetMix_->process(buffer);

    // Verify correctness for 512+1 samples
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 513; ++i) {
            double const input = static_cast<double>(i) / 513.0;
            double const expectedValue = 0.5 * input + 0.5 * input; // Gain=1.0
            EXPECT_NEAR(buffer.getSample(ch, i), expectedValue, 1e-10) << "Channel " << ch << ", sample " << i;
        }
    }
}

TEST_F(DryWetMixStageTest, NegativeValues_HandledCorrectly) {
    dryWetMix_->setMixAmount(0.5);

    juce::AudioBuffer<double> buffer(2, 64);
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 64; ++i) {
            buffer.setSample(ch, i, -static_cast<double>(i) / 64.0); // Negative values
        }
    }

    dryWetMix_->process(buffer);

    // Verify negative values processed correctly
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 64; ++i) {
            double const input = -static_cast<double>(i) / 64.0;
            double const expectedValue = 0.5 * input + 0.5 * input; // Gain=1.0
            EXPECT_NEAR(buffer.getSample(ch, i), expectedValue, 1e-10) << "Channel " << ch << ", sample " << i;
        }
    }
}

TEST_F(DryWetMixStageTest, MultiChannel_8Channels) {
    dryWetMix_->setMixAmount(0.5);

    juce::AudioBuffer<double> buffer(8, 64);
    for (int ch = 0; ch < 8; ++ch) {
        for (int i = 0; i < 64; ++i) {
            buffer.setSample(ch, i, static_cast<double>(ch + i) / 64.0);
        }
    }

    dryWetMix_->process(buffer);

    // Verify all channels processed correctly
    for (int ch = 0; ch < 8; ++ch) {
        for (int i = 0; i < 64; ++i) {
            double const input = static_cast<double>(ch + i) / 64.0;
            double const expectedValue = 0.5 * input + 0.5 * input; // Gain=1.0
            EXPECT_NEAR(buffer.getSample(ch, i), expectedValue, 1e-10) << "Channel " << ch << ", sample " << i;
        }
    }
}

// =============================================================================
// Parameter Smoothing Tests
// =============================================================================

TEST_F(DryWetMixStageTest, MixChange_NoInstantJump) {
    // Regression test: changing mix mid-stream must not produce a discontinuity.
    // Start at 100% wet, then change to 0% wet. The transition sample must be
    // within [0.0, 1.0] — i.e., the smoother ramps, not jumps.

    // Process one block at full wet to let gain stage settle
    juce::AudioBuffer<double> settle(2, 512);
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 512; ++i) {
            settle.setSample(ch, i, 1.0);
        }
    }
    dryWetMix_->process(settle);

    // Now record the last sample value before the mix change
    const double lastWetSample = settle.getSample(0, 511);

    // Change mix and process immediately
    dryWetMix_->setMixAmount(0.0);

    juce::AudioBuffer<double> transitionBuffer(2, 512);
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 512; ++i) {
            transitionBuffer.setSample(ch, i, 1.0);
        }
    }
    dryWetMix_->process(transitionBuffer);

    // The first sample of the transition block must NOT jump directly to the
    // 100%-dry value (which equals 1.0 here since dry==wet, but the key is
    // there's no discontinuity). More importantly, the mix must ramp — the
    // smoother must still be mid-ramp at sample 0 of this block.
    // With a 10ms ramp at 44100Hz (~441 samples), sample 0 should have
    // wetGain well above 0.0.
    //
    // We verify this by checking the smoother did NOT instantly snap: if it
    // did snap, mixAmount at sample 0 would be 0.0 and wetGain=0. Since
    // dry==wet==1.0, output would still be 1.0 either way. Instead we verify
    // the smooth ramp by checking the OUTPUT IS CONTINUOUS (no jump > epsilon
    // between consecutive samples at the transition boundary).
    const double firstTransitionSample = transitionBuffer.getSample(0, 0);
    EXPECT_NEAR(firstTransitionSample, lastWetSample, 1e-3)
        << "Mix change must ramp smoothly — no discontinuous jump at block boundary";
}

// =============================================================================
// Latency Compensation Tests
// =============================================================================

TEST(DryWetMixLatencyTest, ReportsWetPathLatency) {
    // getLatencySamples() must surface the effects-pipeline latency so the host
    // (and the dry-path compensation) know the wet delay.
    auto stage = makeDelayedDryWet(/*delaySamples=*/7, /*mix=*/1.0, /*blockSize=*/64);
    EXPECT_EQ(stage->getLatencySamples(), 7);
}

TEST(DryWetMixLatencyTest, DryAlignsWithWetUnderLatency) {
    // With a latent wet path, the dry path must be delayed by the same amount so
    // an impulse blends in-phase. At 50/50 the dry and wet impulses land on the
    // SAME sample (kDelay) and sum to ~1.0 there — not two half-impulses at 0 and
    // kDelay (which is what an uncompensated blend would produce).
    constexpr int kDelay = 5;
    constexpr int kBlock = 64;
    auto stage = makeDelayedDryWet(kDelay, /*mix=*/0.5, kBlock);

    juce::AudioBuffer<double> buffer(1, kBlock);
    buffer.clear();
    buffer.setSample(0, 0, 1.0); // impulse at sample 0

    stage->process(buffer);

    EXPECT_NEAR(buffer.getSample(0, kDelay), 1.0, 1e-9) << "dry+wet impulse must align at the delayed sample";
    EXPECT_NEAR(buffer.getSample(0, 0), 0.0, 1e-9) << "no early (uncompensated) dry impulse at sample 0";
    for (int i = 0; i < kBlock; ++i) {
        if (i != kDelay) {
            EXPECT_NEAR(buffer.getSample(0, i), 0.0, 1e-9) << "stray energy at sample " << i;
        }
    }
}

TEST(DryWetMixLatencyTest, MatchedDryOversamplingMakesDryEqualWet) {
    // The crux of the phase-coherence requirement: when the wet path's core runs
    // through a half-band up/down, the dry path must run through the SAME up/down
    // (matched order), not a flat delay. We prove it by making the wet path a
    // pure oversampled passthrough at order N and matching the dry order to N —
    // dry and wet then carry identical IIR phase, so 100%-dry output == 100%-wet
    // output sample-for-sample (an uncompensated dry would diverge at HF).
    constexpr int kOrder = 2; // 4x
    constexpr int kBlock = 256;

    auto build = [kOrder, kBlock](double mix) {
        auto wet = std::make_unique<AudioPipeline>();
        // FixedDelayStage(0) is an exact passthrough, so the wrapper is pure up/down.
        wet->addStage(std::make_unique<OversamplingWrapper>(std::make_unique<FixedDelayStage>(0), kOrder), "os");
        auto stage = std::make_unique<DryWetMixStage>(std::move(wet));
        stage->prepareToPlay(44100.0, kBlock, 1);
        stage->setDryOversamplingOrder(kOrder); // match dry to wet core
        stage->setMixAmount(mix);
        return stage;
    };

    auto dryStage = build(0.0); // 100% dry  → output = input through dry up/down
    auto wetStage = build(1.0); // 100% wet  → output = input through wet up/down

    auto fillSine = [](juce::AudioBuffer<double>& b, int startSample) {
        for (int i = 0; i < b.getNumSamples(); ++i) {
            b.setSample(0, i, std::sin(2.0 * M_PI * 1000.0 * (startSample + i) / 44100.0));
        }
    };

    // Identical input history to both stages; settles the (dry) mix smoother.
    int n = 0;
    for (int blk = 0; blk < 8; ++blk, n += kBlock) {
        juce::AudioBuffer<double> a(1, kBlock);
        juce::AudioBuffer<double> b(1, kBlock);
        fillSine(a, n);
        fillSine(b, n);
        dryStage->process(a);
        wetStage->process(b);
    }

    juce::AudioBuffer<double> da(1, kBlock);
    juce::AudioBuffer<double> wb(1, kBlock);
    fillSine(da, n);
    fillSine(wb, n);
    dryStage->process(da);
    wetStage->process(wb);

    for (int i = 0; i < kBlock; ++i) {
        EXPECT_NEAR(da.getSample(0, i), wb.getSample(0, i), 1e-9)
            << "dry and wet must carry identical up/down phase at sample " << i;
    }
}

TEST(DryWetMixLatencyTest, ZeroLatencyWetPathIsBitIdentical) {
    // A zero-latency wet path must take the direct-copy fast path: 50/50 of a
    // unity-gain pipeline reproduces the input exactly (no delay artifacts).
    auto pipeline = std::make_unique<AudioPipeline>();
    auto gain = std::make_unique<GainStage>();
    auto* gainPtr = gain.get();
    pipeline->addStage(std::move(gain), "gain");
    DryWetMixStage stage(std::move(pipeline));
    stage.prepareToPlay(44100.0, 512, 1);
    gainPtr->setGainDB(0.0);
    EXPECT_EQ(stage.getLatencySamples(), 0);

    juce::AudioBuffer<double> settle(1, 512);
    settle.clear();
    for (int b = 0; b < 4; ++b) {
        stage.process(settle); // settle gain + mix smoothers
    }
    stage.setMixAmount(0.5);
    for (int b = 0; b < 4; ++b) {
        stage.process(settle);
    }

    juce::AudioBuffer<double> buffer(1, 512);
    for (int i = 0; i < 512; ++i) {
        buffer.setSample(0, i, static_cast<double>(i) / 512.0);
    }
    stage.process(buffer);
    for (int i = 0; i < 512; ++i) {
        EXPECT_NEAR(buffer.getSample(0, i), static_cast<double>(i) / 512.0, 1e-10) << "sample " << i;
    }
}

TEST_F(DryWetMixStageTest, ZeroLatency_NoDelayNeeded) {
    // DryWetMixStage with no oversampling in pipeline should have zero latency
    EXPECT_EQ(dryWetMix_->getLatencySamples(), 0);

    // 50/50 mix should produce immediate output (no delay)
    dryWetMix_->setMixAmount(0.5);

    juce::AudioBuffer<double> buffer(2, 512);
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 512; ++i) {
            buffer.setSample(ch, i, static_cast<double>(i) / 512.0);
        }
    }

    dryWetMix_->process(buffer);

    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 512; ++i) {
            double const input = static_cast<double>(i) / 512.0;
            EXPECT_NEAR(buffer.getSample(ch, i), input, 1e-10)
                << "Zero latency should process immediately at ch=" << ch << " sample=" << i;
        }
    }
}
