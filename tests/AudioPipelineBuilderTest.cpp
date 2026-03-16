#include <gtest/gtest.h>
#include <dsp_core/dsp_core.h>
#include <cmath>

using namespace dsp_core::audio_pipeline;

// Default constants for transfer function
constexpr int kDefaultTableSize = 16384;
constexpr double kMinSignalValue = -1.0;
constexpr double kMaxSignalValue = 1.0;

class AudioPipelineBuilderTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Initialize transfer function for waveshaping tests
        transferFunction_ = std::make_unique<dsp_core::LayeredTransferFunction>(
            kDefaultTableSize, kMinSignalValue, kMaxSignalValue);
    }

    std::unique_ptr<dsp_core::LayeredTransferFunction> transferFunction_;
};

// =============================================================================
// Basic Builder Tests
// =============================================================================

TEST_F(AudioPipelineBuilderTest, EmptyPipelineBuilds) {
    auto [pipeline, stages] = AudioPipelineBuilder().build();

    ASSERT_NE(pipeline, nullptr);
    EXPECT_EQ(stages.getDryWetMix(), nullptr); // No dry/wet wrapper
    EXPECT_NE(stages.getEffectsPipeline(), nullptr);
}

TEST_F(AudioPipelineBuilderTest, DryWetMixWraps) {
    auto [pipeline, stages] = AudioPipelineBuilder()
        .withDryWetMix()
        .build();

    ASSERT_NE(pipeline, nullptr);
    ASSERT_NE(stages.getDryWetMix(), nullptr);
    EXPECT_NE(stages.getEffectsPipeline(), nullptr);
}

// =============================================================================
// addStage Tests
// =============================================================================

TEST_F(AudioPipelineBuilderTest, AddSingleStage) {
    auto [pipeline, stages] = AudioPipelineBuilder()
        .addStage<GainStage>(StageTag::InputGain)
        .build();

    ASSERT_NE(pipeline, nullptr);
    auto* gain = stages.get<GainStage>(StageTag::InputGain);
    ASSERT_NE(gain, nullptr);
}

TEST_F(AudioPipelineBuilderTest, AddMultipleStages) {
    auto [pipeline, stages] = AudioPipelineBuilder()
        .addStage<GainStage>(StageTag::InputGain)
        .addStage<DCBlockingFilter>(StageTag::DCBlock)
        .addStage<GainStage>(StageTag::OutputGain)
        .build();

    ASSERT_NE(stages.get<GainStage>(StageTag::InputGain), nullptr);
    ASSERT_NE(stages.get<DCBlockingFilter>(StageTag::DCBlock), nullptr);
    ASSERT_NE(stages.get<GainStage>(StageTag::OutputGain), nullptr);

    // Verify they are different instances
    ASSERT_NE(stages.get<GainStage>(StageTag::InputGain),
              stages.get<GainStage>(StageTag::OutputGain));
}

TEST_F(AudioPipelineBuilderTest, StageHasMethod) {
    auto [pipeline, stages] = AudioPipelineBuilder()
        .addStage<GainStage>(StageTag::InputGain)
        .build();

    EXPECT_TRUE(stages.has(StageTag::InputGain));
    EXPECT_FALSE(stages.has(StageTag::OutputGain));
}

// =============================================================================
// Type Safety Tests
// =============================================================================

TEST_F(AudioPipelineBuilderTest, TypeMismatchReturnsNull) {
    auto [pipeline, stages] = AudioPipelineBuilder()
        .addStage<GainStage>(StageTag::InputGain)
        .build();

    // Wrong type should return nullptr
    EXPECT_EQ(stages.get<DCBlockingFilter>(StageTag::InputGain), nullptr);
}

TEST_F(AudioPipelineBuilderTest, MissingTagReturnsNull) {
    auto [pipeline, stages] = AudioPipelineBuilder()
        .addStage<GainStage>(StageTag::InputGain)
        .build();

    EXPECT_EQ(stages.get<GainStage>(StageTag::OutputGain), nullptr);
}

// =============================================================================
// addWrapped Tests
// =============================================================================

TEST_F(AudioPipelineBuilderTest, AddWrappedStage) {
    auto [pipeline, stages] = AudioPipelineBuilder()
        .addWrapped<OversamplingWrapper, WaveshapingStage>(
            StageTag::Waveshaper, *transferFunction_)
        .build();

    auto* oversampling = stages.get<OversamplingWrapper>(StageTag::Waveshaper);
    ASSERT_NE(oversampling, nullptr);
}

TEST_F(AudioPipelineBuilderTest, AddWrappedWithOuterArgs) {
    auto [pipeline, stages] = AudioPipelineBuilder()
        .addWrappedWithOuterArgs<OversamplingWrapper, WaveshapingStage>(
            StageTag::Waveshaper,
            std::make_tuple(2),  // oversamplingOrder = 2 (4x)
            *transferFunction_)
        .build();

    auto* oversampling = stages.get<OversamplingWrapper>(StageTag::Waveshaper);
    ASSERT_NE(oversampling, nullptr);
    EXPECT_EQ(oversampling->getOversamplingOrder(), 2);
}

// =============================================================================
// Full Pipeline Tests (Black Diamond configuration)
// =============================================================================

TEST_F(AudioPipelineBuilderTest, FullBlackDiamondPipeline) {
    auto [pipeline, stages] = AudioPipelineBuilder()
        .withDryWetMix()
        .addStage<GainStage>(StageTag::InputGain)
        .addWrappedWithOuterArgs<OversamplingWrapper, WaveshapingStage>(
            StageTag::Waveshaper,
            std::make_tuple(0),  // No oversampling initially
            *transferFunction_)
        .addStage<DCBlockingFilter>(StageTag::DCBlock)
        .addStage<GainStage>(StageTag::OutputGain)
        .build();

    // Verify all stages accessible
    ASSERT_NE(stages.get<GainStage>(StageTag::InputGain), nullptr);
    ASSERT_NE(stages.get<GainStage>(StageTag::OutputGain), nullptr);
    ASSERT_NE(stages.get<OversamplingWrapper>(StageTag::Waveshaper), nullptr);
    ASSERT_NE(stages.get<DCBlockingFilter>(StageTag::DCBlock), nullptr);
    ASSERT_NE(stages.getDryWetMix(), nullptr);
    ASSERT_NE(stages.getEffectsPipeline(), nullptr);
}

TEST_F(AudioPipelineBuilderTest, PipelineProcessesAudio) {
    auto [pipeline, stages] = AudioPipelineBuilder()
        .withDryWetMix()
        .addStage<GainStage>(StageTag::InputGain)
        .addStage<GainStage>(StageTag::OutputGain)
        .build();

    pipeline->prepareToPlay(44100.0, 512);

    // JUCE Gain defaults to 0.0 linear (silence), set to unity for passthrough
    auto* inputGain = stages.get<GainStage>(StageTag::InputGain);
    auto* outputGain = stages.get<GainStage>(StageTag::OutputGain);
    ASSERT_NE(inputGain, nullptr);
    ASSERT_NE(outputGain, nullptr);
    inputGain->setGainDB(0.0);
    outputGain->setGainDB(0.0);

    // Process a settle block so gain smoothers reach unity
    juce::AudioBuffer<double> settle(2, 512);
    settle.clear();
    pipeline->process(settle);

    juce::AudioBuffer<double> buffer(2, 512);
    buffer.clear();

    // Fill with test signal (sine wave)
    for (int i = 0; i < 512; ++i) {
        double const sample = std::sin(2.0 * M_PI * 440.0 * i / 44100.0);
        buffer.setSample(0, i, sample);
        buffer.setSample(1, i, sample);
    }

    pipeline->process(buffer);

    // Verify audio passed through (not silent)
    double maxSample = 0.0;
    for (int i = 0; i < 512; ++i) {
        maxSample = std::max(maxSample, std::abs(buffer.getSample(0, i)));
    }
    EXPECT_GT(maxSample, 0.0);
}

TEST_F(AudioPipelineBuilderTest, StageParametersCanBeModified) {
    auto [pipeline, stages] = AudioPipelineBuilder()
        .addStage<GainStage>(StageTag::InputGain)
        .build();

    pipeline->prepareToPlay(44100.0, 512);

    auto* gainStage = stages.get<GainStage>(StageTag::InputGain);
    ASSERT_NE(gainStage, nullptr);

    // Set gain to +6dB
    gainStage->setGainDB(6.0);

    // Process multiple buffers to allow smoothing to settle
    juce::AudioBuffer<double> buffer(2, 512);
    for (int block = 0; block < 10; ++block) {
        for (int i = 0; i < 512; ++i) {
            buffer.setSample(0, i, 0.5);
            buffer.setSample(1, i, 0.5);
        }
        pipeline->process(buffer);
    }

    // +6dB ≈ 2x gain, so 0.5 * 2 ≈ 1.0
    // After smoothing settles, should be close to expected
    EXPECT_GT(buffer.getSample(0, 511), 0.9);
}

TEST_F(AudioPipelineBuilderTest, DryWetMixCanBeModified) {
    auto [pipeline, stages] = AudioPipelineBuilder()
        .withDryWetMix()
        .addStage<GainStage>(StageTag::InputGain)
        .build();

    pipeline->prepareToPlay(44100.0, 512);

    auto* dryWet = stages.getDryWetMix();
    ASSERT_NE(dryWet, nullptr);

    // Set to 50% mix
    dryWet->setMixAmount(0.5);
    EXPECT_DOUBLE_EQ(dryWet->getMixAmount(), 0.5);

    // Set to fully dry
    dryWet->setMixAmount(0.0);
    EXPECT_DOUBLE_EQ(dryWet->getMixAmount(), 0.0);
}
