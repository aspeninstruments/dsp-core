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

// Makeup gain in dB for an IR given as a sample vector (mono), via the
// production analysis (makeupGainForIR). Deterministic — no async convolver.
double makeupDbFor(const std::vector<float>& ir, double sr) {
    juce::AudioBuffer<float> buf(1, static_cast<int>(ir.size()));
    for (int i = 0; i < static_cast<int>(ir.size()); ++i) {
        buf.setSample(0, i, ir[static_cast<size_t>(i)]);
    }
    return juce::Decibels::gainToDecibels(
        ImpulseResponseStrategy::makeupGainForIR(buf, sr));
}

// Pump steady-state sine blocks so a load-time makeup ramp (~20 ms) settles.
void settle(ImpulseResponseStrategy& strategy, int blocks, double freqHz = 1000.0) {
    juce::AudioBuffer<double> b(kNumChannels, kBlockSize);
    for (int i = 0; i < blocks; ++i) {
        fillSineBlock(b, freqHz, kSampleRate);
        strategy.process(b);
    }
}

// A Hann-windowed sinc lowpass IR — a genuinely band-limited "cabinet-like" IR
// long enough for the makeup analysis to resolve in-band content. A ~6 kHz
// cutoff spreads energy across [0, 6 kHz] so the in-band [80, 5 kHz] band is a
// proper, denser subset of the spectrum (in-band/broadband RMS ≈ 2), the same
// property that lands the shipped cabs at ~+12 dB of makeup.
std::vector<float> makeLowpassIR(double cutoffHz, int length, double sr) {
    std::vector<float> h(static_cast<size_t>(length), 0.0f);
    const double mid = (length - 1) / 2.0;
    const double fc = cutoffHz / sr; // normalised cutoff
    const double twoPi = 2.0 * juce::MathConstants<double>::pi;
    for (int n = 0; n < length; ++n) {
        const double x = n - mid;
        const double sinc = (std::abs(x) < 1e-9)
                                ? 2.0 * fc
                                : std::sin(twoPi * fc * x) / (juce::MathConstants<double>::pi * x);
        const double hann = 0.5 - 0.5 * std::cos(twoPi * n / (length - 1));
        h[static_cast<size_t>(n)] = static_cast<float>(sinc * hann);
    }
    return h;
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

    // A single-sample IR is too short for the makeup analysis to have any
    // in-band FFT bins, so it falls back to no compensation and the output sits
    // at JUCE's normalised level — non-zero and bounded by the input. Makeup for
    // real (multi-sample, band-limited) IRs is covered by MakeupRestoresInBandLevel.
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

// --- Makeup analysis (deterministic; tests the pure makeupGainForIR math, not
//     JUCE's async convolver, which does not reliably finish loading in a
//     headless gtest run). End-to-end level restoration is verified by ear in a
//     DAW per the change's verification steps. ---

TEST(ImpulseResponseStrategy, DISABLED_EndToEndRealCabWithMakeup) {
    // TEMP: load real cabs, let the async IR finish loading, measure gain WITH
    // makeup engaged. Expect the ~12 dB systematic drop gone (centered ~0 dB,
    // with the cab's natural ±few-dB voicing).
    const char* base = "/Users/johnjaniczek/Code/black-diamond-distortion/impulse-responses/";
    for (const juce::String name : {juce::String("Guitar Cab/Guitar Cab 1.wav"),
                                    juce::String("Guitar Cab/Guitar Cab 3.wav"),
                                    juce::String("Bass Cab/Bass Cab 2.wav")}) {
        juce::File f(juce::String(base) + name);
        if (!f.existsAsFile()) continue;
        ImpulseResponseStrategy strategy;
        strategy.prepareToPlay(kSampleRate, kBlockSize, kNumChannels);
        strategy.setImpulseResponseFile(f);
        juce::AudioBuffer<double> warm(kNumChannels, kBlockSize);
        for (int i = 0; i < 200; ++i) { fillSineBlock(warm, 1000.0, kSampleRate); strategy.process(warm); juce::Thread::sleep(10); }
        for (double hz : {110.0, 220.0, 440.0, 1000.0, 2000.0, 3000.0}) {
            settle(strategy, 110, hz);
            juce::AudioBuffer<double> b(kNumChannels, kBlockSize);
            fillSineBlock(b, hz, kSampleRate);
            const double in = maxAbs(b);
            strategy.process(b);
            (void) fprintf(stderr, "[E2E] %-20s %5.0fHz gain=%.3f (%+.1f dB)\n",
                f.getFileName().toRawUTF8(), hz, maxAbs(b) / in, 20.0 * std::log10(maxAbs(b) / in + 1e-20));
        }
    }
}

TEST(ImpulseResponseStrategy, MakeupRestoresBandLimitedIRLevel) {
    // A band-limited lowpass IR sits ~12 dB low under JUCE's Normalise=yes. The
    // makeup analysis must compute ~+12 dB to restore unity in-band loudness.
    // (Cross-checked against the live convolver: real cabs measure −8…−12 dB and
    //  this synthetic IR yields +12.0 dB.)
    const double db = makeupDbFor(makeLowpassIR(6000.0, 257, kSampleRate), kSampleRate);
    EXPECT_NEAR(db, 12.0, 1.0);
}

TEST(ImpulseResponseStrategy, MakeupClampedForPathologicalIR) {
    // An IR with no energy in [80, 5000] Hz (alternating taps = all energy at
    // Nyquist) would drive the makeup to infinity; it must clamp at +18 dB.
    std::vector<float> alternating(128);
    for (size_t i = 0; i < alternating.size(); ++i) {
        alternating[i] = (i % 2 == 0) ? 1.0f : -1.0f;
    }
    EXPECT_NEAR(makeupDbFor(alternating, kSampleRate), 18.0, 0.01);
}

TEST(ImpulseResponseStrategy, MakeupUnityForSilentIR) {
    // A silent IR carries no compensation (mirrors JUCE's own < 1e-8 guard).
    EXPECT_DOUBLE_EQ(makeupDbFor(std::vector<float>(256, 0.0f), kSampleRate), 0.0);
}

TEST(ImpulseResponseStrategy, MakeupAppliesToWetOnly) {
    // makeupGainForIR yields a non-unity makeup for this IR, stored synchronously
    // on load. At mix=0 the output must STILL be a bit-exact dry passthrough —
    // proving the makeup multiplies the wet path only, never the dry.
    auto file = writeImpulseWav("ir_wetonly_", makeLowpassIR(6000.0, 257, kSampleRate));

    ImpulseResponseStrategy strategy;
    strategy.prepareToPlay(kSampleRate, kBlockSize, kNumChannels);
    strategy.setImpulseResponseFile(file);
    ASSERT_TRUE(warmUpUntilIRActive(strategy));
    settle(strategy, 15); // makeup ramp settles to its (non-unity) target

    strategy.setMix(0.0);
    settle(strategy, 10); // mix ramp settles to exactly 0

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

TEST(ImpulseResponseStrategy, UnloadReturnsToPassthrough) {
    // After unloading the IR the strategy bypasses (and clears its makeup) so
    // audio passes through bit-exact again.
    auto file = writeImpulseWav("ir_unload_", makeLowpassIR(6000.0, 257, kSampleRate));

    ImpulseResponseStrategy strategy;
    strategy.prepareToPlay(kSampleRate, kBlockSize, kNumChannels);
    strategy.setImpulseResponseFile(file);
    ASSERT_TRUE(warmUpUntilIRActive(strategy));

    strategy.setImpulseResponseFile(juce::File{}); // unload

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
