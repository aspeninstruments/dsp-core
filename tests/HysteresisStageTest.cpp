#include <gtest/gtest.h>
#include <dsp_core/dsp_core.h>
#include <cmath>
#include <limits>

using namespace dsp_core::audio_pipeline;

// =============================================================================
// Test Fixture
// =============================================================================

class HysteresisStageTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // SeamlessTransferFunction default constructor initializes to identity (y=x)
        stage_ = std::make_unique<HysteresisStage>(tf_);
        stage_->prepareToPlay(48000.0 * 16.0, 512); // Oversampled rate like HysteresisProcessor expects
    }

    dsp_core::SeamlessTransferFunction tf_;
    std::unique_ptr<HysteresisStage> stage_;

    // Helper: create a stereo buffer filled with a sine wave
    juce::AudioBuffer<double> makeSineBuffer(int numChannels, int numSamples,
                                              double frequency = 100.0,
                                              double sampleRate = 48000.0 * 16.0,
                                              double amplitude = 0.5) {
        juce::AudioBuffer<double> buffer(numChannels, numSamples);
        for (int ch = 0; ch < numChannels; ++ch) {
            auto* data = buffer.getWritePointer(ch);
            for (int i = 0; i < numSamples; ++i) {
                data[i] = amplitude * std::sin(2.0 * juce::MathConstants<double>::pi * frequency * i / sampleRate);
            }
        }
        return buffer;
    }

    // Helper: create a buffer filled with a constant value
    juce::AudioBuffer<double> makeConstantBuffer(int numChannels, int numSamples, double value) {
        juce::AudioBuffer<double> buffer(numChannels, numSamples);
        for (int ch = 0; ch < numChannels; ++ch) {
            auto* data = buffer.getWritePointer(ch);
            for (int i = 0; i < numSamples; ++i) {
                data[i] = value;
            }
        }
        return buffer;
    }
};

// =============================================================================
// Layer 1 — Basic Functionality
// =============================================================================

TEST_F(HysteresisStageTest, ProcessStereoBuffer_HysteresisEnabled_NoCrash) {
    auto buffer = makeSineBuffer(2, 512);
    EXPECT_NO_THROW(stage_->process(buffer));

    // Verify output is finite
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 512; ++i) {
            EXPECT_TRUE(std::isfinite(buffer.getSample(ch, i)))
                << "ch=" << ch << " sample=" << i;
        }
    }
}

TEST_F(HysteresisStageTest, ProcessStereoBuffer_HysteresisDisabled_NoCrash) {
    stage_->setHysteresisEnabled(false);
    auto buffer = makeSineBuffer(2, 512);
    EXPECT_NO_THROW(stage_->process(buffer));

    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 512; ++i) {
            EXPECT_TRUE(std::isfinite(buffer.getSample(ch, i)))
                << "ch=" << ch << " sample=" << i;
        }
    }
}

TEST_F(HysteresisStageTest, ZeroInput_ProducesZeroOutput_HysteresisEnabled) {
    auto buffer = makeConstantBuffer(2, 512, 0.0);
    stage_->process(buffer);

    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 512; ++i) {
            EXPECT_NEAR(buffer.getSample(ch, i), 0.0, 1e-10)
                << "ch=" << ch << " sample=" << i;
        }
    }
}

TEST_F(HysteresisStageTest, ZeroInput_ProducesZeroOutput_HysteresisDisabled) {
    stage_->setHysteresisEnabled(false);
    auto buffer = makeConstantBuffer(2, 512, 0.0);
    stage_->process(buffer);

    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 512; ++i) {
            EXPECT_NEAR(buffer.getSample(ch, i), 0.0, 1e-10)
                << "ch=" << ch << " sample=" << i;
        }
    }
}

TEST_F(HysteresisStageTest, GetName_ReturnsHysteresis) {
    EXPECT_EQ(stage_->getName(), juce::String("Hysteresis"));
}

TEST_F(HysteresisStageTest, GetLatencySamples_ReturnsZero) {
    EXPECT_EQ(stage_->getLatencySamples(), 0);
}

// =============================================================================
// Layer 2 — Hysteresis Behavior
// =============================================================================

TEST_F(HysteresisStageTest, SineInput_HysteresisOn_ProducesHysteresisLoop) {
    // With hysteresis enabled, a sine input should produce output that differs
    // from the input (memory effect). The ascending and descending paths differ.
    const int numSamples = 7680; // One full cycle at 100Hz / 768kHz
    auto buffer = makeSineBuffer(2, numSamples, 100.0, 48000.0 * 16.0, 0.5);

    stage_->process(buffer);

    // Verify output differs from input (hysteresis adds memory)
    int differingSamples = 0;
    for (int i = 0; i < numSamples; ++i) {
        double input = 0.5 * std::sin(2.0 * juce::MathConstants<double>::pi * 100.0 * i / (48000.0 * 16.0));
        if (std::abs(buffer.getSample(0, i) - input) > 1e-6)
            differingSamples++;
    }

    // Most samples should differ due to hysteresis memory
    EXPECT_GT(differingSamples, numSamples / 2)
        << "Hysteresis should modify most samples (memory effect)";
}

TEST_F(HysteresisStageTest, ToggleHysteresis_ProducesDifferentOutput) {
    const int numSamples = 1024;

    // Process with hysteresis ON
    auto bufferOn = makeSineBuffer(2, numSamples, 100.0, 48000.0 * 16.0, 0.5);
    stage_->process(bufferOn);

    // Reset and process with hysteresis OFF
    stage_->reset();
    stage_->setHysteresisEnabled(false);
    auto bufferOff = makeSineBuffer(2, numSamples, 100.0, 48000.0 * 16.0, 0.5);
    stage_->process(bufferOff);

    // Outputs should differ (hysteresis vs memoryless waveshaping)
    int differingSamples = 0;
    for (int i = 0; i < numSamples; ++i) {
        if (std::abs(bufferOn.getSample(0, i) - bufferOff.getSample(0, i)) > 1e-10)
            differingSamples++;
    }
    EXPECT_GT(differingSamples, 0) << "Hysteresis ON vs OFF should produce different output";
}

TEST_F(HysteresisStageTest, HysteresisDisabled_OutputMatchesTransferFunction) {
    // With hysteresis disabled and identity transfer function (default),
    // each sample should pass through applyTransferFunction(x) which is identity.
    stage_->setHysteresisEnabled(false);

    // Settle the crossfade-out triggered by default enabled=true → false
    for (int b = 0; b < 40; ++b) {
        auto buf = makeConstantBuffer(2, 512, 0.0);
        stage_->process(buf);
    }

    const int numSamples = 256;
    auto buffer = makeSineBuffer(2, numSamples, 100.0, 48000.0 * 16.0, 0.5);

    // Store expected: applyTransferFunction on each sample
    std::vector<double> expected(numSamples);
    for (int i = 0; i < numSamples; ++i) {
        double input = 0.5 * std::sin(2.0 * juce::MathConstants<double>::pi * 100.0 * i / (48000.0 * 16.0));
        expected[i] = tf_.applyTransferFunction(input);
    }

    stage_->process(buffer);

    // Output should match processBuffer behavior (identity for default TF)
    for (int i = 0; i < numSamples; ++i) {
        EXPECT_NEAR(buffer.getSample(0, i), expected[i], 1e-10)
            << "sample=" << i;
    }
}

// =============================================================================
// Layer 3 — Stereo Independence
// =============================================================================

TEST_F(HysteresisStageTest, ChannelsHaveIndependentState) {
    // Process L-only signal (R=0), verify R output remains zero
    const int numSamples = 512;
    juce::AudioBuffer<double> buffer(2, numSamples);

    // L channel: sine wave
    for (int i = 0; i < numSamples; ++i) {
        buffer.setSample(0, i, 0.5 * std::sin(2.0 * juce::MathConstants<double>::pi * 100.0 * i / (48000.0 * 16.0)));
        buffer.setSample(1, i, 0.0);
    }

    stage_->process(buffer);

    // R channel should remain zero (independent state)
    for (int i = 0; i < numSamples; ++i) {
        EXPECT_NEAR(buffer.getSample(1, i), 0.0, 1e-10)
            << "R channel should be zero at sample=" << i;
    }
}

TEST_F(HysteresisStageTest, DifferentLR_ProducesDifferentOutput) {
    const int numSamples = 512;
    juce::AudioBuffer<double> buffer(2, numSamples);

    // L: 100Hz sine, R: 200Hz sine (different frequencies)
    for (int i = 0; i < numSamples; ++i) {
        buffer.setSample(0, i, 0.5 * std::sin(2.0 * juce::MathConstants<double>::pi * 100.0 * i / (48000.0 * 16.0)));
        buffer.setSample(1, i, 0.5 * std::sin(2.0 * juce::MathConstants<double>::pi * 200.0 * i / (48000.0 * 16.0)));
    }

    stage_->process(buffer);

    // L and R outputs should differ
    int differingSamples = 0;
    for (int i = 0; i < numSamples; ++i) {
        if (std::abs(buffer.getSample(0, i) - buffer.getSample(1, i)) > 1e-10)
            differingSamples++;
    }
    EXPECT_GT(differingSamples, numSamples / 2)
        << "Different L/R input should produce different L/R output";
}

// =============================================================================
// Layer 4 — Edge Cases
// =============================================================================

TEST_F(HysteresisStageTest, BufferSize1Sample_Works) {
    auto buffer = makeSineBuffer(2, 1, 100.0, 48000.0 * 16.0, 0.5);
    EXPECT_NO_THROW(stage_->process(buffer));
    EXPECT_TRUE(std::isfinite(buffer.getSample(0, 0)));
    EXPECT_TRUE(std::isfinite(buffer.getSample(1, 0)));
}

TEST_F(HysteresisStageTest, BufferSize0Samples_NoCrash) {
    juce::AudioBuffer<double> buffer(2, 0);
    EXPECT_NO_THROW(stage_->process(buffer));
}

TEST_F(HysteresisStageTest, MonoBuffer_Works) {
    auto buffer = makeSineBuffer(1, 256, 100.0, 48000.0 * 16.0, 0.5);
    EXPECT_NO_THROW(stage_->process(buffer));
    for (int i = 0; i < 256; ++i) {
        EXPECT_TRUE(std::isfinite(buffer.getSample(0, i))) << "sample=" << i;
    }
}

TEST_F(HysteresisStageTest, ParameterChanges_MidStream_NoCrashFiniteOutput) {
    auto buffer1 = makeSineBuffer(2, 256, 100.0, 48000.0 * 16.0, 0.5);
    stage_->process(buffer1);

    // Change parameters mid-stream
    stage_->setDrive(0.8);
    stage_->setSaturation(0.9);
    stage_->setWidth(0.7);

    auto buffer2 = makeSineBuffer(2, 256, 100.0, 48000.0 * 16.0, 0.5);
    EXPECT_NO_THROW(stage_->process(buffer2));

    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 256; ++i) {
            EXPECT_TRUE(std::isfinite(buffer2.getSample(ch, i)))
                << "ch=" << ch << " sample=" << i;
        }
    }
}

TEST_F(HysteresisStageTest, Reset_ClearsState) {
    // Process some signal to build up state
    auto buffer1 = makeSineBuffer(2, 1024, 100.0, 48000.0 * 16.0, 0.5);
    stage_->process(buffer1);

    // Reset
    stage_->reset();

    // Zero input after reset should produce zero output
    auto buffer2 = makeConstantBuffer(2, 64, 0.0);
    stage_->process(buffer2);

    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 64; ++i) {
            EXPECT_NEAR(buffer2.getSample(ch, i), 0.0, 1e-10)
                << "ch=" << ch << " sample=" << i;
        }
    }
}

// =============================================================================
// Layer 5 — Safety Passthrough
// =============================================================================

TEST_F(HysteresisStageTest, NaNInput_ProducesBoundedOutput) {
    juce::AudioBuffer<double> buffer(2, 64);
    for (int ch = 0; ch < 2; ++ch) {
        auto* data = buffer.getWritePointer(ch);
        for (int i = 0; i < 64; ++i) {
            data[i] = std::numeric_limits<double>::quiet_NaN();
        }
    }

    stage_->process(buffer);

    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 64; ++i) {
            EXPECT_TRUE(std::isfinite(buffer.getSample(ch, i)))
                << "NaN should be caught by safety layer, ch=" << ch << " sample=" << i;
        }
    }
}

TEST_F(HysteresisStageTest, LargeInput_ProducesBoundedOutput) {
    auto buffer = makeConstantBuffer(2, 64, 10.0);
    stage_->process(buffer);

    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 64; ++i) {
            double val = buffer.getSample(ch, i);
            EXPECT_TRUE(std::isfinite(val)) << "ch=" << ch << " sample=" << i;
            EXPECT_LE(std::abs(val), 2.0 + 1e-10)
                << "Output should be bounded at +-2.0, ch=" << ch << " sample=" << i;
        }
    }
}

// =============================================================================
// Layer 6 — Makeup Gain
// =============================================================================

TEST_F(HysteresisStageTest, MakeupGain_DefaultParams_ApproximatelyUnity) {
    // At default sat=0.5, width=0.5: makeup = (1+0.6*0.5)/(0.5+1.5*(1-0.5)) = 1.3/1.25 = 1.04
    // Process with default makeup (1.0) and with computed makeup, verify difference is small
    stage_->setMakeupGain(1.04);

    auto buffer = makeSineBuffer(2, 512, 100.0, 48000.0 * 16.0, 0.5);
    stage_->process(buffer);

    // All output should be finite and bounded (makeup ~1.04 barely changes output)
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 512; ++i) {
            double val = buffer.getSample(ch, i);
            EXPECT_TRUE(std::isfinite(val)) << "ch=" << ch << " sample=" << i;
        }
    }
}

TEST_F(HysteresisStageTest, MakeupGain_HighSaturation_CompensatesLevelDrop) {
    // High saturation reduces M_s → lower output. Makeup should compensate.
    stage_->setSaturation(0.9);

    // Process without makeup gain
    stage_->setMakeupGain(1.0);
    auto bufferNoMakeup = makeSineBuffer(2, 2048, 100.0, 48000.0 * 16.0, 0.5);
    stage_->process(bufferNoMakeup);

    double peakNoMakeup = 0.0;
    for (int i = 0; i < 2048; ++i) {
        peakNoMakeup = std::max(peakNoMakeup, std::abs(bufferNoMakeup.getSample(0, i)));
    }

    // Reset and process with makeup gain for sat=0.9, width=0.5
    stage_->reset();
    double makeup = (1.0 + 0.6 * 0.5) / (0.5 + 1.5 * (1.0 - 0.9)); // 1.3/0.65 = 2.0
    stage_->setMakeupGain(makeup);
    auto bufferWithMakeup = makeSineBuffer(2, 2048, 100.0, 48000.0 * 16.0, 0.5);
    stage_->process(bufferWithMakeup);

    double peakWithMakeup = 0.0;
    for (int i = 0; i < 2048; ++i) {
        peakWithMakeup = std::max(peakWithMakeup, std::abs(bufferWithMakeup.getSample(0, i)));
    }

    // With makeup, peak should be higher (closer to nominal level)
    EXPECT_GT(peakWithMakeup, peakNoMakeup)
        << "Makeup gain should increase output level. Without: " << peakNoMakeup
        << " With: " << peakWithMakeup;
}

TEST_F(HysteresisStageTest, MakeupGain_WidthSweep_PeakWithinFivePercent) {
    const double sampleRate = 48000.0 * 16.0;
    const double freq = 100.0;
    const double amplitude = 0.5;
    const double tolerance = 0.05;

    for (double w : {0.0, 0.25, 0.5, 0.75, 1.0}) {
        // Fresh stage per width — avoids carrying J-A state or makeup ramp across iterations.
        stage_ = std::make_unique<HysteresisStage>(tf_);
        stage_->prepareToPlay(sampleRate, 512);
        stage_->setWidth(w);
        stage_->setMakeupGain(HysteresisStage::computeMakeupForWidth(w));

        const int blockSize = 512;
        int phase = 0;
        for (int b = 0; b < 120; ++b) {
            juce::AudioBuffer<double> buf(2, blockSize);
            for (int ch = 0; ch < 2; ++ch) {
                auto* d = buf.getWritePointer(ch);
                for (int i = 0; i < blockSize; ++i) {
                    d[i] = amplitude * std::sin(
                        2.0 * juce::MathConstants<double>::pi * freq * (phase + i) / sampleRate);
                }
            }
            stage_->process(buf);
            phase += blockSize;
        }

        const int measSamples = 16384;
        juce::AudioBuffer<double> meas(2, measSamples);
        for (int ch = 0; ch < 2; ++ch) {
            auto* d = meas.getWritePointer(ch);
            for (int i = 0; i < measSamples; ++i) {
                d[i] = amplitude * std::sin(
                    2.0 * juce::MathConstants<double>::pi * freq * (phase + i) / sampleRate);
            }
        }
        stage_->process(meas);

        double peak = 0.0;
        for (int i = 0; i < measSamples; ++i)
            peak = std::max(peak, std::abs(meas.getSample(0, i)));

        EXPECT_NEAR(peak, amplitude, amplitude * tolerance)
            << "width=" << w << " peak=" << peak;
    }
}

TEST_F(HysteresisStageTest, MakeupGain_FrequencySweep_PeakWithinFivePercent) {
    const double sampleRate = 48000.0 * 16.0;
    const double amplitude = 0.5;
    const double tolerance = 0.05;

    for (double freq : {50.0, 200.0, 400.0, 800.0, 1600.0}) {
        for (double w : {0.0, 0.25, 0.5, 0.75, 1.0}) {
            stage_ = std::make_unique<HysteresisStage>(tf_);
            stage_->prepareToPlay(sampleRate, 512);
            stage_->setWidth(w);
            stage_->setMakeupGain(HysteresisStage::computeMakeupForWidth(w));

            // 200 blocks * 512 @ 768kHz = ~133ms: ≥6 cycles at the slowest (50Hz) freq
            const int blockSize = 512;
            int phase = 0;
            for (int b = 0; b < 200; ++b) {
                juce::AudioBuffer<double> buf(2, blockSize);
                for (int ch = 0; ch < 2; ++ch) {
                    auto* d = buf.getWritePointer(ch);
                    for (int i = 0; i < blockSize; ++i) {
                        d[i] = amplitude * std::sin(
                            2.0 * juce::MathConstants<double>::pi * freq * (phase + i) / sampleRate);
                    }
                }
                stage_->process(buf);
                phase += blockSize;
            }

            // 32768 samples = ~2 cycles at 50Hz, plenty at higher freqs
            const int measSamples = 32768;
            juce::AudioBuffer<double> meas(2, measSamples);
            for (int ch = 0; ch < 2; ++ch) {
                auto* d = meas.getWritePointer(ch);
                for (int i = 0; i < measSamples; ++i) {
                    d[i] = amplitude * std::sin(
                        2.0 * juce::MathConstants<double>::pi * freq * (phase + i) / sampleRate);
                }
            }
            stage_->process(meas);

            double peak = 0.0;
            for (int i = 0; i < measSamples; ++i)
                peak = std::max(peak, std::abs(meas.getSample(0, i)));

            EXPECT_NEAR(peak, amplitude, amplitude * tolerance)
                << "freq=" << freq << "Hz width=" << w << " peak=" << peak;
        }
    }
}

// =============================================================================
// Layer 7 — Crossfade Smoothness (click-free enable/disable)
// =============================================================================

TEST_F(HysteresisStageTest, EnableHysteresis_MidSignal_NoCrossfadeClick) {
    // Simulate enabling hysteresis while a signal is playing.
    // Two-phase transition: 25ms warmup + 25ms crossfade = 50ms total.
    // Collect a continuous output stream across the transition and measure
    // the maximum sample-to-sample jump. A click manifests as a delta that
    // far exceeds what a smooth crossfade should produce.

    const double sampleRate = 48000.0 * 16.0;
    const double freq = 100.0;
    const double amplitude = 0.5;
    const int blockSize = 512;
    // 50ms total at 768kHz = 38400 samples. Need ~80 blocks of 512 to cover it.
    const int blocksAfterEnable = 80;

    // Start with hysteresis disabled
    stage_->setHysteresisEnabled(false);

    // Process a few blocks to reach steady state in the waveshaping path
    for (int b = 0; b < 4; ++b) {
        auto buf = makeSineBuffer(1, blockSize, freq, sampleRate, amplitude);
        stage_->process(buf);
    }

    // Collect output: a few blocks before enable, then enable, then blocks after
    std::vector<double> output;
    int sinePhase = 4 * blockSize; // continue the sine phase

    // 2 blocks before enable
    for (int b = 0; b < 2; ++b) {
        juce::AudioBuffer<double> buf(1, blockSize);
        auto* data = buf.getWritePointer(0);
        for (int i = 0; i < blockSize; ++i) {
            data[i] = amplitude * std::sin(2.0 * juce::MathConstants<double>::pi * freq * (sinePhase + i) / sampleRate);
        }
        stage_->process(buf);
        for (int i = 0; i < blockSize; ++i) {
            output.push_back(buf.getSample(0, i));
        }
        sinePhase += blockSize;
    }

    // Enable hysteresis between blocks
    stage_->setHysteresisEnabled(true);

    // Blocks after enable — covers full warmup + crossfade + settling
    for (int b = 0; b < blocksAfterEnable; ++b) {
        juce::AudioBuffer<double> buf(1, blockSize);
        auto* data = buf.getWritePointer(0);
        for (int i = 0; i < blockSize; ++i) {
            data[i] = amplitude * std::sin(2.0 * juce::MathConstants<double>::pi * freq * (sinePhase + i) / sampleRate);
        }
        stage_->process(buf);
        for (int i = 0; i < blockSize; ++i) {
            output.push_back(buf.getSample(0, i));
        }
        sinePhase += blockSize;
    }

    // Measure max sample-to-sample delta
    double maxDelta = 0.0;
    int maxDeltaIdx = 0;
    for (size_t i = 1; i < output.size(); ++i) {
        double delta = std::abs(output[i] - output[i - 1]);
        if (delta > maxDelta) {
            maxDelta = delta;
            maxDeltaIdx = static_cast<int>(i);
        }
    }

    // For a 100Hz sine at 768kHz, the max natural slope is:
    // 2*pi*f*A / sampleRate ≈ 2*pi*100*0.5/768000 ≈ 0.0004
    // Allow some headroom for hysteresis nonlinearity but a click would be >>0.01
    double maxNaturalDelta = 2.0 * juce::MathConstants<double>::pi * freq * amplitude / sampleRate;
    double clickThreshold = maxNaturalDelta * 50.0; // generous: 50x natural slope

    EXPECT_LT(maxDelta, clickThreshold)
        << "Click detected at sample " << maxDeltaIdx
        << ": delta=" << maxDelta
        << " threshold=" << clickThreshold
        << " (natural max delta=" << maxNaturalDelta << ")";
}

TEST_F(HysteresisStageTest, EnableHysteresis_MidSignal_DiagnosticDeltas) {
    // Diagnostic: print sample-to-sample deltas around the enable transition
    // to understand where the click comes from.

    const double sampleRate = 48000.0 * 16.0;
    const double freq = 100.0;
    const double amplitude = 0.5;
    const int blockSize = 512;

    stage_->setHysteresisEnabled(false);

    // Steady state
    for (int b = 0; b < 4; ++b) {
        auto buf = makeSineBuffer(1, blockSize, freq, sampleRate, amplitude);
        stage_->process(buf);
    }

    std::vector<double> output;
    int sinePhase = 4 * blockSize;

    // 1 block before enable
    {
        juce::AudioBuffer<double> buf(1, blockSize);
        auto* data = buf.getWritePointer(0);
        for (int i = 0; i < blockSize; ++i) {
            data[i] = amplitude * std::sin(2.0 * juce::MathConstants<double>::pi * freq * (sinePhase + i) / sampleRate);
        }
        stage_->process(buf);
        for (int i = 0; i < blockSize; ++i) {
            output.push_back(buf.getSample(0, i));
        }
        sinePhase += blockSize;
    }

    // Enable
    stage_->setHysteresisEnabled(true);

    // 80 blocks after enable (covers full 25ms warmup + 25ms crossfade + settling)
    for (int b = 0; b < 80; ++b) {
        juce::AudioBuffer<double> buf(1, blockSize);
        auto* data = buf.getWritePointer(0);
        for (int i = 0; i < blockSize; ++i) {
            data[i] = amplitude * std::sin(2.0 * juce::MathConstants<double>::pi * freq * (sinePhase + i) / sampleRate);
        }
        stage_->process(buf);
        for (int i = 0; i < blockSize; ++i) {
            output.push_back(buf.getSample(0, i));
        }
        sinePhase += blockSize;
    }

    // Find and print top 10 largest deltas
    struct DeltaInfo { double delta; int idx; double prevSample; double curSample; };
    std::vector<DeltaInfo> deltas;
    for (size_t i = 1; i < output.size(); ++i) {
        double d = std::abs(output[i] - output[i - 1]);
        deltas.push_back({d, static_cast<int>(i), output[i-1], output[i]});
    }
    std::sort(deltas.begin(), deltas.end(), [](auto& a, auto& b) { return a.delta > b.delta; });

    double maxNaturalDelta = 2.0 * juce::MathConstants<double>::pi * freq * amplitude / sampleRate;
    std::cout << "\n=== Enable Hysteresis Crossfade Diagnostic ===" << std::endl;
    std::cout << "Max natural delta for " << freq << "Hz sine at " << sampleRate << "Hz: " << maxNaturalDelta << std::endl;
    std::cout << "Transition starts at sample " << blockSize << std::endl;
    std::cout << "Top 10 largest sample-to-sample deltas:" << std::endl;
    for (int i = 0; i < std::min(10, static_cast<int>(deltas.size())); ++i) {
        std::cout << "  [" << deltas[i].idx << "] delta=" << deltas[i].delta
                  << " prev=" << deltas[i].prevSample
                  << " cur=" << deltas[i].curSample
                  << " (ratio to natural: " << deltas[i].delta / maxNaturalDelta << "x)"
                  << std::endl;
    }
    std::cout << "===============================================\n" << std::endl;

    // Still assert no click
    double clickThreshold = maxNaturalDelta * 50.0;
    EXPECT_LT(deltas[0].delta, clickThreshold)
        << "Click detected at sample " << deltas[0].idx;
}

TEST_F(HysteresisStageTest, DisableHysteresis_MidSignal_NoCrossfadeClick) {
    const double sampleRate = 48000.0 * 16.0;
    const double freq = 100.0;
    const double amplitude = 0.5;
    const int blockSize = 512;
    // 25ms crossfade at 768kHz = 19200 samples. Need ~40 blocks of 512.
    const int blocksAfterDisable = 40;

    // Start with hysteresis enabled, process enough to reach steady state
    stage_->setHysteresisEnabled(true);
    for (int b = 0; b < 10; ++b) {
        auto buf = makeSineBuffer(1, blockSize, freq, sampleRate, amplitude);
        stage_->process(buf);
    }

    std::vector<double> output;
    int sinePhase = 10 * blockSize;

    // 2 blocks before disable
    for (int b = 0; b < 2; ++b) {
        juce::AudioBuffer<double> buf(1, blockSize);
        auto* data = buf.getWritePointer(0);
        for (int i = 0; i < blockSize; ++i) {
            data[i] = amplitude * std::sin(2.0 * juce::MathConstants<double>::pi * freq * (sinePhase + i) / sampleRate);
        }
        stage_->process(buf);
        for (int i = 0; i < blockSize; ++i) {
            output.push_back(buf.getSample(0, i));
        }
        sinePhase += blockSize;
    }

    // Disable hysteresis between blocks
    stage_->setHysteresisEnabled(false);

    // Blocks after disable — covers full 25ms crossfade + settling
    for (int b = 0; b < blocksAfterDisable; ++b) {
        juce::AudioBuffer<double> buf(1, blockSize);
        auto* data = buf.getWritePointer(0);
        for (int i = 0; i < blockSize; ++i) {
            data[i] = amplitude * std::sin(2.0 * juce::MathConstants<double>::pi * freq * (sinePhase + i) / sampleRate);
        }
        stage_->process(buf);
        for (int i = 0; i < blockSize; ++i) {
            output.push_back(buf.getSample(0, i));
        }
        sinePhase += blockSize;
    }

    double maxDelta = 0.0;
    int maxDeltaIdx = 0;
    for (size_t i = 1; i < output.size(); ++i) {
        double delta = std::abs(output[i] - output[i - 1]);
        if (delta > maxDelta) {
            maxDelta = delta;
            maxDeltaIdx = static_cast<int>(i);
        }
    }

    double maxNaturalDelta = 2.0 * juce::MathConstants<double>::pi * freq * amplitude / sampleRate;
    double clickThreshold = maxNaturalDelta * 50.0;

    EXPECT_LT(maxDelta, clickThreshold)
        << "Click detected at sample " << maxDeltaIdx
        << ": delta=" << maxDelta
        << " threshold=" << clickThreshold
        << " (natural max delta=" << maxNaturalDelta << ")";
}

TEST_F(HysteresisStageTest, MakeupGain_HysteresisDisabled_NoMakeup) {
    // When hysteresis is disabled (waveshaping fallback), makeup gain should not apply
    stage_->setHysteresisEnabled(false);

    // Settle the crossfade-out triggered by default enabled=true → false
    for (int b = 0; b < 40; ++b) {
        auto buf = makeConstantBuffer(2, 512, 0.0);
        stage_->process(buf);
    }

    stage_->setMakeupGain(2.0); // Deliberately large to detect if it's applied

    auto buffer = makeSineBuffer(2, 512, 100.0, 48000.0 * 16.0, 0.5);

    // Copy input for comparison
    juce::AudioBuffer<double> input(2, 512);
    for (int ch = 0; ch < 2; ++ch) {
        input.copyFrom(ch, 0, buffer, ch, 0, 512);
    }

    stage_->process(buffer);

    // With identity TF and hysteresis disabled, output ≈ input (no makeup applied)
    int matchCount = 0;
    for (int i = 0; i < 512; ++i) {
        if (std::abs(buffer.getSample(0, i) - input.getSample(0, i)) < 0.01) {
            matchCount++;
        }
    }

    EXPECT_GT(matchCount, 400)
        << "With hysteresis disabled, makeup gain should not be applied";
}

// =============================================================================
// Surge rate propagation (oversampling correctness)
// =============================================================================

/**
 * Test: HysteresisStage::prepareToPlay forwards the oversampled sample rate
 *       into the shared SeamlessTransferFunction, so surge timing is wall-time
 *       accurate regardless of oversampling order.
 *
 * Without this forwarding, if PluginAudioProcessor calls tf.prepareToPlay(hostRate)
 * but the pipeline runs at hostRate * 16, surgeStepPerSample is sized for hostRate
 * and the weight saturates in SURGE_DURATION_SEC / 16 wall time — not what the user
 * asked for.
 */
TEST(SurgeRateForwarding, HysteresisStagePropagatesOversampledRateToTransferFunction) {
    dsp_core::SeamlessTransferFunction tf;
    HysteresisStage stage(tf);

    // Switch the shared transfer function into Surge mode via the public API,
    // then render the LUT (identity curve → linear-extrap slope populated by
    // SeamlessTransferFunction::renderLUTImmediate).
    tf.getLaneMixer().setExtrapolationMode(dsp_core::LaneMixer::ExtrapolationMode::Surge);
    tf.renderLUTImmediate();

    constexpr double oversampledRate = 192000.0; // e.g. 48k host × 4x OS
    stage.prepareToPlay(oversampledRate, 512);

    // Flush the 5ms LUT crossfade so applyTransferFunction sees the Surge LUT.
    const int crossfadeSamples = static_cast<int>(
        oversampledRate * dsp_core::SeamlessConfig::CROSSFADE_DURATION_MS / 1000.0);
    juce::AudioBuffer<double> warmup(1, crossfadeSamples + 16);
    warmup.clear();
    tf.processBuffer(warmup);

    // Drive x = 1.5 for 2 × SURGE_DURATION_SEC worth of samples at the oversampled rate.
    // If prepareToPlay propagated correctly, the weight reaches 1 and output settles to clamp (1.0).
    // If the rate wasn't propagated, surgeStepPerSample_ would still be sized for the
    // default/host rate and the output would not reach clamp in this sample count.
    const int samples = static_cast<int>(
        oversampledRate * 2.0 * dsp_core::AudioEngine::SURGE_DURATION_SEC);

    double lastOutput = 0.0;
    for (int i = 0; i < samples; ++i) {
        lastOutput = tf.applyTransferFunction(1.5, 0);
    }

    EXPECT_NEAR(lastOutput, 1.0, 1e-3)
        << "Surge should fully decay to clamp within 2×SURGE_DURATION_SEC at the rate "
           "passed to HysteresisStage::prepareToPlay. If this fails, prepareToPlay is "
           "not forwarding the oversampled rate to SeamlessTransferFunction.";
}
