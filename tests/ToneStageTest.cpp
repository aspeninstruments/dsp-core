#include "../dsp_core/Source/audio_pipeline/ToneStage.h"
#include <gtest/gtest.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <cmath>

using namespace dsp_core::audio_pipeline;

namespace {

constexpr double kSampleRate = 48000.0;
constexpr int kBlockSize = 512;

void fillSineBlock(juce::AudioBuffer<double>& buf, double freqHz, int sampleOffset,
                   double amp = 0.5) {
    for (int ch = 0; ch < buf.getNumChannels(); ++ch) {
        for (int i = 0; i < buf.getNumSamples(); ++i) {
            const int n = sampleOffset + i;
            buf.setSample(ch, i, amp * std::sin(2.0 * M_PI * freqHz * n / kSampleRate));
        }
    }
}

double maxAbs(const juce::AudioBuffer<double>& buf) {
    double m = 0.0;
    for (int ch = 0; ch < buf.getNumChannels(); ++ch) {
        for (int i = 0; i < buf.getNumSamples(); ++i) {
            m = std::max(m, std::abs(buf.getSample(ch, i)));
        }
    }
    return m;
}

double maxSampleToSampleDelta(const juce::AudioBuffer<double>& buf) {
    double m = 0.0;
    for (int ch = 0; ch < buf.getNumChannels(); ++ch) {
        for (int i = 1; i < buf.getNumSamples(); ++i) {
            m = std::max(m, std::abs(buf.getSample(ch, i) - buf.getSample(ch, i - 1)));
        }
    }
    return m;
}

} // namespace

// --------------------------------------------------------------------------
// Test 1 — Fat=0 must be a *true* bypass. Two parallel ToneStage instances,
// both Lowpass24dB with identical cutoff/resonance/input. Instance A
// explicitly calls setFat(0.0); instance B never touches Fat. Output must
// be bit-identical sample-for-sample — proves the bypass branch genuinely
// skips fat_.process(), because running the LowShelfStage at 0 dB would
// introduce ~1e-15 per-sample rounding and break EXPECT_DOUBLE_EQ.
// --------------------------------------------------------------------------
TEST(ToneStage, FatZero_IsTrueBypass) {
    ToneStage a;
    ToneStage b;

    a.setType(ToneStage::Type::Lowpass24dB);
    b.setType(ToneStage::Type::Lowpass24dB);
    a.setEnabled(true);
    b.setEnabled(true);
    a.setCutoffFrequency(2000.0);
    b.setCutoffFrequency(2000.0);
    a.setResonance(0.5);
    b.setResonance(0.5);
    a.setFat(0.0); // explicit zero
    // b: default Fat (never set) — must yield the same bytes as 0%

    a.prepareToPlay(kSampleRate, kBlockSize, 2);
    b.prepareToPlay(kSampleRate, kBlockSize, 2);

    juce::AudioBuffer<double> bufA(2, kBlockSize);
    juce::AudioBuffer<double> bufB(2, kBlockSize);

    for (int block = 0; block < 8; ++block) {
        fillSineBlock(bufA, 440.0, block * kBlockSize);
        fillSineBlock(bufB, 440.0, block * kBlockSize);
        a.process(bufA);
        b.process(bufB);
        for (int ch = 0; ch < 2; ++ch) {
            for (int i = 0; i < kBlockSize; ++i) {
                ASSERT_DOUBLE_EQ(bufA.getSample(ch, i), bufB.getSample(ch, i))
                    << "Block " << block << " ch " << ch << " sample " << i
                    << ": setFat(0) must be bit-identical to never-set Fat";
            }
        }
    }
}

// --------------------------------------------------------------------------
// Test 2 — Cycling Fat across the 0 boundary must not click or NaN. The
// 0→nonzero reset in process() is what keeps the LowShelfStage IIR state
// clean across bypass windows.
// --------------------------------------------------------------------------
TEST(ToneStage, FatTransitionDoesNotClick) {
    ToneStage s;
    s.setType(ToneStage::Type::Lowpass24dB);
    s.setEnabled(true);
    s.setCutoffFrequency(3000.0);
    s.setResonance(0.4);
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    juce::AudioBuffer<double> buf(1, kBlockSize);
    const double fatSequence[] = {0.0, 50.0, 0.0, 80.0, 0.0, 25.0};
    int sampleCount = 0;

    for (double fat : fatSequence) {
        s.setFat(fat);
        for (int block = 0; block < 4; ++block) {
            fillSineBlock(buf, 220.0, sampleCount);
            s.process(buf);
            // Finite and non-denormal
            for (int i = 0; i < kBlockSize; ++i) {
                const double v = buf.getSample(0, i);
                ASSERT_TRUE(v == 0.0 || std::isnormal(v))
                    << "Non-finite/denormal output at fat=" << fat
                    << " block=" << block << " sample=" << i << " v=" << v;
            }
            // No full-scale click: sample-to-sample delta stays modest. Input
            // is a 220 Hz sine at amp 0.5 (Nyquist=24kHz, ~218 samples/cycle),
            // so adjacent samples never differ by more than ~0.015 from the
            // signal itself; a filter transient that produces >0.5 delta is
            // unambiguously a click.
            const double delta = maxSampleToSampleDelta(buf);
            ASSERT_LT(delta, 0.5)
                << "Click detected at fat=" << fat << " block=" << block
                << " max-delta=" << delta;
            sampleCount += kBlockSize;
        }
    }
}

// --------------------------------------------------------------------------
// Test 3 — Type::Off must dominate. Even with Fat=100, the existing Off
// early-return must leave the buffer untouched.
// --------------------------------------------------------------------------
TEST(ToneStage, FatInactiveWhenTypeOff) {
    ToneStage s;
    s.setType(ToneStage::Type::Off);
    s.setEnabled(true);
    s.setFat(100.0);
    s.prepareToPlay(kSampleRate, kBlockSize, 2);

    juce::AudioBuffer<double> input(2, kBlockSize);
    juce::AudioBuffer<double> output(2, kBlockSize);
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < kBlockSize; ++i) {
            const double v = 0.3 * std::sin(2.0 * M_PI * 100.0 * i / kSampleRate);
            input.setSample(ch, i, v);
            output.setSample(ch, i, v);
        }
    }

    s.process(output);

    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < kBlockSize; ++i) {
            ASSERT_DOUBLE_EQ(input.getSample(ch, i), output.getSample(ch, i))
                << "Type=Off with Fat=100 must be transparent";
        }
    }
}

// --------------------------------------------------------------------------
// Test — Resonance sweep with Fat at default (0) preserves RMS within ±2 dB.
//
// End-to-end check that the LadderTPT bass-compensation fix actually fixes
// the user-reported volume drop through the full ToneStage. Earlier ToneStage
// behavior (without LadderTPT compensation) showed a ~-13 dB drop at max
// resonance with Fat untouched; the fix should keep RMS within ±2 dB across
// the whole UI-facing [0, 1] resonance range.
//
// Input: 100 Hz sine at amp 0.1 (well below the 1 kHz cutoff and small enough
// to stay in the linear-tanh regime where compensation is mathematically exact).
// --------------------------------------------------------------------------
TEST(ToneStage, BassCompensation_ResonanceSweepRmsPreserved_24dB) {
    constexpr int kTotalSamples = 9600; // 0.2s @ 48k
    constexpr double kSineHz = 100.0;
    constexpr double kAmp = 0.1;
    const double inputRms = kAmp / std::sqrt(2.0);

    for (double resNormalized : {0.0, 0.25, 0.5, 0.75, 1.0}) {
        ToneStage s;
        s.setType(ToneStage::Type::Lowpass24dB);
        s.setEnabled(true);
        s.setCutoffFrequency(1000.0);
        s.setResonance(resNormalized);
        // Fat at default (0) — must NOT be required to maintain passband level.
        s.prepareToPlay(kSampleRate, kBlockSize, 1);

        juce::AudioBuffer<double> buf(1, kTotalSamples);
        for (int i = 0; i < kTotalSamples; ++i) {
            buf.setSample(0, i, kAmp * std::sin(2.0 * M_PI * kSineHz * i / kSampleRate));
        }
        s.process(buf);

        // RMS over last 100 ms (4800 samples) to skip startup transient.
        double sumSq = 0.0;
        const int start = kTotalSamples - 4800;
        for (int i = start; i < kTotalSamples; ++i) {
            const double v = buf.getSample(0, i);
            sumSq += v * v;
        }
        const double outRms = std::sqrt(sumSq / (kTotalSamples - start));
        const double gainDb = 20.0 * std::log10(outRms / inputRms);
        EXPECT_NEAR(gainDb, 0.0, 2.0)
            << "ToneStage at resonance=" << resNormalized
            << " (Fat=0): output is " << gainDb
            << " dB. Compensation should keep this within ±2 dB across the sweep.";
    }
}

TEST(ToneStage, BassCompensation_ResonanceSweepRmsPreserved_12dB) {
    constexpr int kTotalSamples = 9600;
    constexpr double kSineHz = 100.0;
    constexpr double kAmp = 0.1;
    const double inputRms = kAmp / std::sqrt(2.0);

    for (double resNormalized : {0.0, 0.25, 0.5, 0.75, 1.0}) {
        ToneStage s;
        s.setType(ToneStage::Type::Lowpass12dB);
        s.setEnabled(true);
        s.setCutoffFrequency(1000.0);
        s.setResonance(resNormalized);
        s.prepareToPlay(kSampleRate, kBlockSize, 1);

        juce::AudioBuffer<double> buf(1, kTotalSamples);
        for (int i = 0; i < kTotalSamples; ++i) {
            buf.setSample(0, i, kAmp * std::sin(2.0 * M_PI * kSineHz * i / kSampleRate));
        }
        s.process(buf);

        double sumSq = 0.0;
        const int start = kTotalSamples - 4800;
        for (int i = start; i < kTotalSamples; ++i) {
            const double v = buf.getSample(0, i);
            sumSq += v * v;
        }
        const double outRms = std::sqrt(sumSq / (kTotalSamples - start));
        const double gainDb = 20.0 * std::log10(outRms / inputRms);
        EXPECT_NEAR(gainDb, 0.0, 2.0)
            << "12dB ToneStage at resonance=" << resNormalized
            << " (Fat=0): output is " << gainDb << " dB.";
    }
}

// --------------------------------------------------------------------------
// Test 4 — Switching LP topology with Fat engaged must not cause NaN/inf
// or a click. The fat_.reset() inside the t != lastType_ branch keeps the
// shelf IIR aligned with whichever LP runs next.
// --------------------------------------------------------------------------
TEST(ToneStage, FatResetsOnTypeChange) {
    ToneStage s;
    s.setType(ToneStage::Type::Lowpass24dB);
    s.setEnabled(true);
    s.setCutoffFrequency(1500.0);
    s.setResonance(0.6);
    s.setFat(80.0);
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    juce::AudioBuffer<double> buf(1, kBlockSize);
    int sampleCount = 0;

    // Build IIR state with loud signal in Lowpass24dB.
    for (int block = 0; block < 20; ++block) {
        fillSineBlock(buf, 80.0, sampleCount, 0.9);
        s.process(buf);
        sampleCount += kBlockSize;
    }
    ASSERT_TRUE(std::isfinite(maxAbs(buf)));

    // Switch topology mid-flight.
    s.setType(ToneStage::Type::Lowpass12dB);

    for (int block = 0; block < 20; ++block) {
        fillSineBlock(buf, 80.0, sampleCount, 0.9);
        s.process(buf);
        for (int i = 0; i < kBlockSize; ++i) {
            const double v = buf.getSample(0, i);
            ASSERT_TRUE(std::isfinite(v))
                << "Non-finite after LP-type switch at block " << block
                << " sample " << i << " v=" << v;
            ASSERT_LT(std::abs(v), 10.0)
                << "Runaway after LP-type switch at block " << block
                << " sample " << i << " v=" << v;
        }
        // No catastrophic discontinuity in the first post-switch block.
        const double delta = maxSampleToSampleDelta(buf);
        ASSERT_LT(delta, 1.0) << "Click at LP-type switch boundary, block " << block
                              << " delta=" << delta;
        sampleCount += kBlockSize;
    }
}

namespace {
// Bandpass-style narrowband RMS estimator: applies a quick goertzel-flavored
// RMS at the target frequency over the buffer's tail. Used to verify a shelf
// actually shapes the low band without depending on which specific samples.
double rmsAtFrequency(const juce::AudioBuffer<double>& buf, double freqHz,
                      int startSample, int endSample) {
    double sumI = 0.0;
    double sumQ = 0.0;
    int n = 0;
    for (int i = startSample; i < endSample; ++i) {
        const double phase = 2.0 * M_PI * freqHz * i / kSampleRate;
        const double v = buf.getSample(0, i);
        sumI += v * std::cos(phase);
        sumQ += v * std::sin(phase);
        ++n;
    }
    const double normI = sumI / n;
    const double normQ = sumQ / n;
    // Amplitude of the freqHz tone in the signal (peak = 2*sqrt(I^2+Q^2));
    // RMS = peak/sqrt(2).
    return std::sqrt(2.0) * std::sqrt(normI * normI + normQ * normQ);
}
} // namespace

// --------------------------------------------------------------------------
// LowShelf tests. The DSP itself is exercised by LowShelfStage's own tests;
// these pin the ToneStage strategy plumbing: that LowShelf is reachable
// through ToneStage, ignores LP-only params, and is click-free on activation
// or deactivation.
// --------------------------------------------------------------------------

TEST(ToneStage, LowShelf_AtZeroDb_IsApproxIdentity) {
    ToneStage s;
    s.setType(ToneStage::Type::LowShelf);
    s.setEnabled(true);
    s.setCutoffFrequency(200.0);
    s.setShelfGainDb(0.0);
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    constexpr int kTotalSamples = 9600; // 0.2s @ 48k
    constexpr double kSineHz = 100.0;
    constexpr double kAmp = 0.3;

    juce::AudioBuffer<double> buf(1, kTotalSamples);
    for (int i = 0; i < kTotalSamples; ++i) {
        buf.setSample(0, i, kAmp * std::sin(2.0 * M_PI * kSineHz * i / kSampleRate));
    }
    s.process(buf);

    // RBJ low shelf at 0 dB gain is mathematically identity, but accumulates
    // tiny floating-point error. Tail-window RMS should match the input RMS
    // to within ±0.1 dB.
    const double inputRms = kAmp / std::sqrt(2.0);
    const double outRms = rmsAtFrequency(buf, kSineHz, kTotalSamples - 4800, kTotalSamples);
    const double gainDb = 20.0 * std::log10(outRms / inputRms);
    EXPECT_NEAR(gainDb, 0.0, 0.1)
        << "LowShelf at 0 dB should be transparent — measured " << gainDb << " dB";
}

TEST(ToneStage, LowShelf_PositiveGain_BoostsBass) {
    ToneStage s;
    s.setType(ToneStage::Type::LowShelf);
    s.setEnabled(true);
    s.setCutoffFrequency(400.0); // Comfortably above 100 Hz probe
    s.setShelfGainDb(12.0);
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    constexpr int kTotalSamples = 9600;
    constexpr double kSineHz = 100.0;
    constexpr double kAmp = 0.1;

    juce::AudioBuffer<double> buf(1, kTotalSamples);
    for (int i = 0; i < kTotalSamples; ++i) {
        buf.setSample(0, i, kAmp * std::sin(2.0 * M_PI * kSineHz * i / kSampleRate));
    }
    s.process(buf);

    const double inputRms = kAmp / std::sqrt(2.0);
    const double outRms = rmsAtFrequency(buf, kSineHz, kTotalSamples - 4800, kTotalSamples);
    const double gainDb = 20.0 * std::log10(outRms / inputRms);
    // RBJ shelf reaches the full plateau gain well below the corner. 100 Hz
    // sits a comfortable octave+ below the 400 Hz corner, so the boost should
    // be within ~1 dB of the +12 dB target.
    EXPECT_NEAR(gainDb, 12.0, 1.5)
        << "LowShelf +12 dB at 100 Hz (corner 400 Hz) should boost ~12 dB; got " << gainDb;
}

TEST(ToneStage, LowShelf_NegativeGain_CutsBass) {
    ToneStage s;
    s.setType(ToneStage::Type::LowShelf);
    s.setEnabled(true);
    s.setCutoffFrequency(400.0);
    s.setShelfGainDb(-12.0);
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    constexpr int kTotalSamples = 9600;
    constexpr double kSineHz = 100.0;
    constexpr double kAmp = 0.1;

    juce::AudioBuffer<double> buf(1, kTotalSamples);
    for (int i = 0; i < kTotalSamples; ++i) {
        buf.setSample(0, i, kAmp * std::sin(2.0 * M_PI * kSineHz * i / kSampleRate));
    }
    s.process(buf);

    const double inputRms = kAmp / std::sqrt(2.0);
    const double outRms = rmsAtFrequency(buf, kSineHz, kTotalSamples - 4800, kTotalSamples);
    const double gainDb = 20.0 * std::log10(outRms / inputRms);
    EXPECT_NEAR(gainDb, -12.0, 1.5)
        << "LowShelf -12 dB at 100 Hz should cut ~12 dB; got " << gainDb;
}

TEST(ToneStage, LowShelf_IgnoresResonanceAndFat) {
    // Resonance and Fat are LP-only — LowShelf strategy must treat them as
    // no-ops. Compare two ToneStage instances driven identically except one
    // has Resonance=1.0 and Fat=100 set.
    ToneStage clean;
    ToneStage poisoned;

    auto configure = [](ToneStage& s) {
        s.setType(ToneStage::Type::LowShelf);
        s.setEnabled(true);
        s.setCutoffFrequency(300.0);
        s.setShelfGainDb(6.0);
        s.prepareToPlay(kSampleRate, kBlockSize, 1);
    };
    configure(clean);
    configure(poisoned);
    poisoned.setResonance(1.0);
    poisoned.setFat(100.0);

    juce::AudioBuffer<double> bufA(1, kBlockSize);
    juce::AudioBuffer<double> bufB(1, kBlockSize);
    constexpr double kAmp = 0.2;
    for (int i = 0; i < kBlockSize; ++i) {
        const double v = kAmp * std::sin(2.0 * M_PI * 120.0 * i / kSampleRate);
        bufA.setSample(0, i, v);
        bufB.setSample(0, i, v);
    }
    clean.process(bufA);
    poisoned.process(bufB);

    for (int i = 0; i < kBlockSize; ++i) {
        ASSERT_DOUBLE_EQ(bufA.getSample(0, i), bufB.getSample(0, i))
            << "LowShelf must ignore Resonance/Fat at sample " << i;
    }
}

TEST(ToneStage, LowShelf_NoClickOnActivation) {
    // Switch LP24 -> LowShelf mid-flight with loud signal; verify no
    // NaN/inf and no catastrophic sample-to-sample discontinuity in the
    // first post-switch block.
    ToneStage s;
    s.setType(ToneStage::Type::Lowpass24dB);
    s.setEnabled(true);
    s.setCutoffFrequency(1500.0);
    s.setResonance(0.6);
    s.setFat(80.0);
    s.setShelfGainDb(9.0); // priming LS while it's idle
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    juce::AudioBuffer<double> buf(1, kBlockSize);
    int sampleCount = 0;

    for (int block = 0; block < 20; ++block) {
        fillSineBlock(buf, 80.0, sampleCount, 0.9);
        s.process(buf);
        sampleCount += kBlockSize;
    }
    ASSERT_TRUE(std::isfinite(maxAbs(buf)));

    s.setType(ToneStage::Type::LowShelf);

    for (int block = 0; block < 20; ++block) {
        fillSineBlock(buf, 80.0, sampleCount, 0.9);
        s.process(buf);
        for (int i = 0; i < kBlockSize; ++i) {
            const double v = buf.getSample(0, i);
            ASSERT_TRUE(std::isfinite(v))
                << "Non-finite after LP->LowShelf switch at block " << block
                << " sample " << i << " v=" << v;
            ASSERT_LT(std::abs(v), 10.0)
                << "Runaway after LP->LowShelf switch at block " << block
                << " sample " << i << " v=" << v;
        }
        const double delta = maxSampleToSampleDelta(buf);
        ASSERT_LT(delta, 1.0) << "Click at LP->LowShelf switch boundary, block " << block
                              << " delta=" << delta;
        sampleCount += kBlockSize;
    }
}

TEST(ToneStage, LowShelf_NoClickOnDeactivation) {
    // Symmetric to activation: LowShelf -> LP12 mid-flight.
    ToneStage s;
    s.setType(ToneStage::Type::LowShelf);
    s.setEnabled(true);
    s.setCutoffFrequency(300.0);
    s.setShelfGainDb(12.0);
    s.setResonance(0.5); // priming LP while it's idle
    s.setFat(50.0);
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    juce::AudioBuffer<double> buf(1, kBlockSize);
    int sampleCount = 0;

    for (int block = 0; block < 20; ++block) {
        fillSineBlock(buf, 80.0, sampleCount, 0.9);
        s.process(buf);
        sampleCount += kBlockSize;
    }
    ASSERT_TRUE(std::isfinite(maxAbs(buf)));

    s.setType(ToneStage::Type::Lowpass12dB);

    for (int block = 0; block < 20; ++block) {
        fillSineBlock(buf, 80.0, sampleCount, 0.9);
        s.process(buf);
        for (int i = 0; i < kBlockSize; ++i) {
            const double v = buf.getSample(0, i);
            ASSERT_TRUE(std::isfinite(v))
                << "Non-finite after LowShelf->LP switch at block " << block
                << " sample " << i << " v=" << v;
            ASSERT_LT(std::abs(v), 10.0)
                << "Runaway after LowShelf->LP switch at block " << block
                << " sample " << i << " v=" << v;
        }
        const double delta = maxSampleToSampleDelta(buf);
        ASSERT_LT(delta, 1.0) << "Click at LowShelf->LP switch boundary, block " << block
                              << " delta=" << delta;
        sampleCount += kBlockSize;
    }
}
