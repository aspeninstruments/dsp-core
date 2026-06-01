#include <gtest/gtest.h>
#include "../dsp_core/Source/audio_pipeline/ImpulseResponseStrategy.h"
#include "../dsp_core/Source/audio_pipeline/ToneStage.h"
#include <juce_audio_formats/juce_audio_formats.h>

#include <cmath>
#include <vector>

namespace dsp_core::audio_pipeline::test {
namespace {

constexpr double kSampleRate = 48000.0;
constexpr int kBlockSize = 256;
constexpr int kNumChannels = 2;

// Build a synthetic IR file on disk for the strategy to load. Returns the
// File handle; caller is responsible for deletion.
juce::File writeImpulseWav(const juce::String& nameStem,
                            const std::vector<float>& impulse) {
    auto tmpDir = juce::File::getSpecialLocation(juce::File::tempDirectory);
    auto file = tmpDir.getNonexistentChildFile(nameStem, ".wav", false);

    juce::WavAudioFormat wav;
    auto fileStream = std::make_unique<juce::FileOutputStream>(file);
    EXPECT_TRUE(fileStream->openedOk());
    std::unique_ptr<juce::OutputStream> stream = std::move(fileStream);
    auto writer = wav.createWriterFor(stream,
        juce::AudioFormatWriterOptions{}
            .withSampleRate(kSampleRate)
            .withNumChannels(1)
            .withBitsPerSample(24));
    EXPECT_NE(writer, nullptr);

    juce::AudioBuffer<float> buf(1, static_cast<int>(impulse.size()));
    buf.clear();
    for (size_t i = 0; i < impulse.size(); ++i) {
        buf.setSample(0, static_cast<int>(i), impulse[i]);
    }
    EXPECT_TRUE(writer->writeFromAudioSampleBuffer(buf, 0, buf.getNumSamples()));
    writer.reset(); // closes the stream
    return file;
}

void fillSineBlock(juce::AudioBuffer<double>& buffer, double freqHz, double sr) {
    const double w = 2.0 * juce::MathConstants<double>::pi * freqHz / sr;
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
        for (int n = 0; n < buffer.getNumSamples(); ++n) {
            buffer.setSample(ch, n, std::sin(w * static_cast<double>(n)));
        }
    }
}

double maxAbs(const juce::AudioBuffer<double>& buffer) {
    double m = 0.0;
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
        for (int n = 0; n < buffer.getNumSamples(); ++n) {
            m = std::max(m, std::abs(buffer.getSample(ch, n)));
        }
    }
    return m;
}

TEST(ImpulseResponseStrategy, PassThroughWhenNoIRLoaded) {
    ImpulseResponseStrategy strategy;
    strategy.prepareToPlay(kSampleRate, kBlockSize, kNumChannels);

    juce::AudioBuffer<double> buffer(kNumChannels, kBlockSize);
    fillSineBlock(buffer, 1000.0, kSampleRate);
    juce::AudioBuffer<double> original;
    original.makeCopyOf(buffer);

    strategy.process(buffer);

    // Without an IR loaded the strategy must pass audio through unchanged so
    // the user hears raw signal while picking a file.
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
        for (int n = 0; n < buffer.getNumSamples(); ++n) {
            EXPECT_DOUBLE_EQ(buffer.getSample(ch, n), original.getSample(ch, n));
        }
    }
}

TEST(ImpulseResponseStrategy, IdentityIRApproximatesInput) {
    // Single unit impulse at t=0: convolution with input should equal the
    // input (within FFT/normalisation tolerance).
    auto file = writeImpulseWav("ir_identity_", {1.0f});

    ImpulseResponseStrategy strategy;
    strategy.prepareToPlay(kSampleRate, kBlockSize, kNumChannels);
    strategy.setImpulseResponseFile(file);

    // The Convolution loads asynchronously; pump a few blocks so the
    // background loader has a chance to publish the IR.
    juce::AudioBuffer<double> warmup(kNumChannels, kBlockSize);
    for (int i = 0; i < 50; ++i) {
        warmup.clear();
        fillSineBlock(warmup, 1000.0, kSampleRate);
        strategy.process(warmup);
        if (maxAbs(warmup) > 0.5) {
            break; // IR has taken effect
        }
        juce::Thread::sleep(10);
    }

    juce::AudioBuffer<double> buffer(kNumChannels, kBlockSize);
    fillSineBlock(buffer, 1000.0, kSampleRate);
    juce::AudioBuffer<double> input;
    input.makeCopyOf(buffer);

    strategy.process(buffer);

    // Output should at least be non-zero and roughly bounded by the input
    // peak — exact identity would require disabling Normalise=yes, which the
    // production code keeps on to prevent loud surprises with high-energy IRs.
    EXPECT_GT(maxAbs(buffer), 0.1);
    EXPECT_LE(maxAbs(buffer), 1.5 * maxAbs(input));

    file.deleteFile();
}

// Load a (coloring) IR and pump blocks until the convolver has published it,
// detected by the wet output diverging from the dry input. Returns true once
// active. Leaves mix at its default (1.0 = fully wet).
bool warmUpUntilIRActive(ImpulseResponseStrategy& strategy) {
    juce::AudioBuffer<double> probe(kNumChannels, kBlockSize);
    juce::AudioBuffer<double> dry(kNumChannels, kBlockSize);
    for (int i = 0; i < 100; ++i) {
        fillSineBlock(probe, 1000.0, kSampleRate);
        dry.makeCopyOf(probe);
        strategy.process(probe);
        double diff = 0.0;
        for (int ch = 0; ch < probe.getNumChannels(); ++ch) {
            for (int n = 0; n < probe.getNumSamples(); ++n) {
                diff = std::max(diff, std::abs(probe.getSample(ch, n) - dry.getSample(ch, n)));
            }
        }
        if (diff > 0.01) {
            return true;
        }
        juce::Thread::sleep(5);
    }
    return false;
}

TEST(ImpulseResponseStrategy, MixZeroIsDryPassthrough) {
    // A multi-tap IR that colors the signal — at mix=0 none of that color may
    // reach the output; it must equal the dry input bit-for-bit (the convolver
    // is zero-latency, so no dry delay compensation kicks in).
    auto file = writeImpulseWav("ir_mix0_", {1.0f, 0.5f, -0.3f, 0.2f});

    ImpulseResponseStrategy strategy;
    strategy.prepareToPlay(kSampleRate, kBlockSize, kNumChannels);
    strategy.setImpulseResponseFile(file);
    ASSERT_TRUE(warmUpUntilIRActive(strategy));

    // Settle the mix ramp to 0 (10 ms ≈ 480 samples; pump well past that).
    strategy.setMix(0.0);
    juce::AudioBuffer<double> settle(kNumChannels, kBlockSize);
    for (int i = 0; i < 10; ++i) {
        fillSineBlock(settle, 1000.0, kSampleRate);
        strategy.process(settle);
    }

    juce::AudioBuffer<double> buffer(kNumChannels, kBlockSize);
    fillSineBlock(buffer, 1000.0, kSampleRate);
    juce::AudioBuffer<double> input;
    input.makeCopyOf(buffer);
    strategy.process(buffer);

    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
        for (int n = 0; n < buffer.getNumSamples(); ++n) {
            EXPECT_DOUBLE_EQ(buffer.getSample(ch, n), input.getSample(ch, n));
        }
    }
    file.deleteFile();
}

TEST(ImpulseResponseStrategy, MixOneColorsSignal) {
    auto file = writeImpulseWav("ir_mix1_", {1.0f, 0.5f, -0.3f, 0.2f});

    ImpulseResponseStrategy strategy;
    strategy.prepareToPlay(kSampleRate, kBlockSize, kNumChannels);
    strategy.setImpulseResponseFile(file);
    ASSERT_TRUE(warmUpUntilIRActive(strategy)); // default mix = 1.0 (fully wet)

    juce::AudioBuffer<double> buffer(kNumChannels, kBlockSize);
    fillSineBlock(buffer, 1000.0, kSampleRate);
    juce::AudioBuffer<double> input;
    input.makeCopyOf(buffer);
    strategy.process(buffer);

    // Fully wet must diverge from the dry input (the IR shapes it) yet stay
    // bounded (Normalise=yes keeps levels sane).
    double diff = 0.0;
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
        for (int n = 0; n < buffer.getNumSamples(); ++n) {
            diff = std::max(diff, std::abs(buffer.getSample(ch, n) - input.getSample(ch, n)));
        }
    }
    EXPECT_GT(diff, 0.01);
    EXPECT_LE(maxAbs(buffer), 1.5 * maxAbs(input));
    file.deleteFile();
}

TEST(ToneStage, IRTypeRoutesToImpulseResponseStrategy) {
    ToneStage stage;
    stage.setEnabled(true);
    stage.prepareToPlay(kSampleRate, kBlockSize, kNumChannels);
    stage.setType(ToneStage::Type::ImpulseResponse);

    // No IR loaded: process() should pass through unchanged.
    juce::AudioBuffer<double> buffer(kNumChannels, kBlockSize);
    fillSineBlock(buffer, 440.0, kSampleRate);
    juce::AudioBuffer<double> input;
    input.makeCopyOf(buffer);

    stage.process(buffer);

    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
        for (int n = 0; n < buffer.getNumSamples(); ++n) {
            EXPECT_DOUBLE_EQ(buffer.getSample(ch, n), input.getSample(ch, n));
        }
    }
}

TEST(ToneStage, SwitchingToIRAndBackDoesNotCrash) {
    ToneStage stage;
    stage.setEnabled(true);
    stage.prepareToPlay(kSampleRate, kBlockSize, kNumChannels);

    juce::AudioBuffer<double> buffer(kNumChannels, kBlockSize);
    for (const auto type : {ToneStage::Type::Off,
                             ToneStage::Type::Lowpass12dB,
                             ToneStage::Type::ImpulseResponse,
                             ToneStage::Type::Bell,
                             ToneStage::Type::ImpulseResponse,
                             ToneStage::Type::Off}) {
        stage.setType(type);
        fillSineBlock(buffer, 440.0, kSampleRate);
        stage.process(buffer);
    }
    SUCCEED();
}

} // namespace
} // namespace dsp_core::audio_pipeline::test
