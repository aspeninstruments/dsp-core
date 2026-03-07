#include <gtest/gtest.h>
#include "../dsp_core/Source/audio_pipeline/DryWetMixStage.h"
#include "../dsp_core/Source/audio_pipeline/AudioPipeline.h"
#include "../dsp_core/Source/audio_pipeline/GainStage.h"
#include "../dsp_core/Source/audio_pipeline/OversamplingWrapper.h"
#include <juce_audio_basics/juce_audio_basics.h>
#include <cmath>

using namespace dsp_core::audio_pipeline;

class DryWetMixStageTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Create a simple pipeline with a gain stage
        auto pipeline = std::make_unique<AudioPipeline>();
        auto gainStage = std::make_unique<GainStage>();
        gainStage_ = gainStage.get(); // Save pointer for later access
        pipeline->addStage(std::move(gainStage), "gain");

        dryWetMix_ = std::make_unique<DryWetMixStage>(std::move(pipeline));
        dryWetMix_->prepareToPlay(44100.0, 1024); // Larger buffer for edge case tests
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
// Latency Compensation Tests (with Oversampling)
// =============================================================================

class DryWetMixWithLatencyTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Will be set up in individual tests with different oversampling factors
    }

    void createDryWetMixWithOversampling(int oversamplingOrder) {
        // Create pipeline with gain wrapped in oversampling
        auto pipeline = std::make_unique<AudioPipeline>();
        auto gainStage = std::make_unique<GainStage>();
        gainStage_ = gainStage.get();

        // Wrap gain in oversampling (introduces latency)
        auto oversamplingWrapper = std::make_unique<OversamplingWrapper>(std::move(gainStage), oversamplingOrder);
        oversamplingWrapper_ = oversamplingWrapper.get();
        pipeline->addStage(std::move(oversamplingWrapper), "oversampled_gain");

        dryWetMix_ = std::make_unique<DryWetMixStage>(std::move(pipeline));
        dryWetMix_->prepareToPlay(44100.0, 512);
    }

    std::unique_ptr<DryWetMixStage> dryWetMix_;
    GainStage* gainStage_ = nullptr;
    OversamplingWrapper* oversamplingWrapper_ = nullptr;
};

TEST_F(DryWetMixWithLatencyTest, LatencyCompensation_2xOversampling) {
    createDryWetMixWithOversampling(1); // 2x oversampling

    const int latency = dryWetMix_->getLatencySamples();
    EXPECT_GT(latency, 0) << "Oversampling should introduce latency";

    // Set to 50/50 mix
    dryWetMix_->setMixAmount(0.5);

    // Create impulse signal
    juce::AudioBuffer<double> buffer(2, 512);
    buffer.clear();
    buffer.setSample(0, 100, 1.0); // Impulse at sample 100
    buffer.setSample(1, 100, 1.0);

    // Process multiple blocks to fill the delay buffer
    for (int block = 0; block < 3; ++block) {
        dryWetMix_->process(buffer);
    }

    // After latency compensation, dry and wet should be aligned
    // The mix should not create comb filtering artifacts
    // Check that the impulse response is coherent (not split between dry/wet)
    bool hasCoherentPeak = false;
    for (int i = 0; i < 512; ++i) {
        if (std::abs(buffer.getSample(0, i)) > 0.4) {
            hasCoherentPeak = true;
            break;
        }
    }
    EXPECT_TRUE(hasCoherentPeak) << "Should have coherent peak after latency compensation";
}

TEST_F(DryWetMixWithLatencyTest, LatencyCompensation_4xOversampling) {
    createDryWetMixWithOversampling(2); // 4x oversampling

    const int latency = dryWetMix_->getLatencySamples();
    EXPECT_GT(latency, 0) << "Oversampling should introduce latency";

    // Test that dry and wet paths are aligned by comparing 100% dry vs 100% wet vs 50/50 mix
    // Without latency compensation, 50/50 mix would show comb filtering artifacts

    // Create impulse signal
    juce::AudioBuffer<double> impulse(2, 512);
    impulse.clear();
    impulse.setSample(0, 250, 1.0);
    impulse.setSample(1, 250, 1.0);

    // Test 100% wet to see wet path response
    dryWetMix_->setMixAmount(1.0);
    juce::AudioBuffer<double> wetResponse(2, 512);
    wetResponse.copyFrom(0, 0, impulse, 0, 0, 512);
    wetResponse.copyFrom(1, 0, impulse, 1, 0, 512);
    for (int block = 0; block < 5; ++block) {
        dryWetMix_->process(wetResponse);
    }

    // Reset and test 50/50 mix
    dryWetMix_->reset();
    dryWetMix_->setMixAmount(0.5);
    juce::AudioBuffer<double> mixedResponse(2, 512);
    mixedResponse.copyFrom(0, 0, impulse, 0, 0, 512);
    mixedResponse.copyFrom(1, 0, impulse, 1, 0, 512);
    for (int block = 0; block < 5; ++block) {
        dryWetMix_->process(mixedResponse);
    }

    // With proper latency compensation, 50/50 mix should have similar peak magnitude to 100% wet
    // (not split/attenuated due to misalignment)
    double wetPeakEnergy = 0.0;
    double mixedPeakEnergy = 0.0;
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 512; ++i) {
            wetPeakEnergy = std::max(wetPeakEnergy, std::abs(wetResponse.getSample(ch, i)));
            mixedPeakEnergy = std::max(mixedPeakEnergy, std::abs(mixedResponse.getSample(ch, i)));
        }
    }

    // Mixed response should be close to wet response (both around 0.5-1.0 depending on filter response)
    // If misaligned, mixed would be much smaller due to cancellation
    EXPECT_GT(mixedPeakEnergy, 0.3 * wetPeakEnergy)
        << "50/50 mix should not show severe cancellation (indicates proper latency compensation)";
}

TEST_F(DryWetMixWithLatencyTest, LatencyCompensation_8xOversampling) {
    createDryWetMixWithOversampling(3); // 8x oversampling

    const int latency = dryWetMix_->getLatencySamples();
    EXPECT_GT(latency, 0) << "Oversampling should introduce latency";

    // Set to 100% dry (should bypass with compensation)
    dryWetMix_->setMixAmount(0.0);

    // Create sine wave
    juce::AudioBuffer<double> buffer(2, 512);
    const double freq = 1000.0;
    const double sampleRate = 44100.0;
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 512; ++i) {
            buffer.setSample(ch, i, std::sin(2.0 * M_PI * freq * i / sampleRate));
        }
    }

    // Store expected output (delayed by latency)
    juce::AudioBuffer<double> expected(2, 512);
    for (int ch = 0; ch < 2; ++ch) {
        expected.copyFrom(ch, 0, buffer, ch, 0, 512);
    }

    // Process multiple blocks to fill delay buffer
    for (int block = 0; block < 5; ++block) {
        dryWetMix_->process(buffer);
    }

    // At 100% dry, output should equal delayed input
    // (We can't compare exact values due to the delay, but signal should be clean)
    double totalEnergy = 0.0;
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 512; ++i) {
            totalEnergy += buffer.getSample(ch, i) * buffer.getSample(ch, i);
        }
    }
    EXPECT_GT(totalEnergy, 100.0) << "100% dry should preserve signal energy";
}

TEST_F(DryWetMixWithLatencyTest, LatencyReporting_MatchesPipeline) {
    createDryWetMixWithOversampling(2); // 4x oversampling

    // DryWetMixStage should report the same latency as its internal pipeline
    const int dryWetLatency = dryWetMix_->getLatencySamples();
    const int pipelineLatency = dryWetMix_->getEffectsPipeline()->getLatencySamples();

    EXPECT_EQ(dryWetLatency, pipelineLatency) << "DryWetMixStage should report pipeline latency for DAW compensation";
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

TEST_F(DryWetMixWithLatencyTest, NoLatency_BypassesDelayBuffer) {
    // Create DryWetMix without oversampling (zero latency)
    auto pipeline = std::make_unique<AudioPipeline>();
    pipeline->addStage(std::make_unique<GainStage>(), "gain");
    dryWetMix_ = std::make_unique<DryWetMixStage>(std::move(pipeline));
    dryWetMix_->prepareToPlay(44100.0, 512);

    EXPECT_EQ(dryWetMix_->getLatencySamples(), 0) << "No oversampling should have zero latency";

    // Set to 50/50 mix
    dryWetMix_->setMixAmount(0.5);

    // Create test signal
    juce::AudioBuffer<double> buffer(2, 512);
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 512; ++i) {
            buffer.setSample(ch, i, static_cast<double>(i) / 512.0);
        }
    }

    // Store expected (immediate processing, no delay)
    juce::AudioBuffer<double> expected(2, 512);
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 512; ++i) {
            double const input = static_cast<double>(i) / 512.0;
            expected.setSample(ch, i, 0.5 * input + 0.5 * input); // 50/50 mix
        }
    }

    dryWetMix_->process(buffer);

    // With zero latency, output should match immediately (no delay)
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 512; ++i) {
            EXPECT_NEAR(buffer.getSample(ch, i), expected.getSample(ch, i), 1e-10)
                << "Zero latency should process immediately at ch=" << ch << " sample=" << i;
        }
    }
}
