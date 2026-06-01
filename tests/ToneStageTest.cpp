#include "../dsp_core/Source/audio_pipeline/ToneStage.h"
#include "../dsp_core/Source/audio_pipeline/ToneFrequencyResponse.h"
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

// --------------------------------------------------------------------------
// HighShelf tests. Mirror the LowShelf battery but probe ABOVE the corner so
// the shelf's boost/cut affects the measured tone (HS shapes the high band).
// --------------------------------------------------------------------------

TEST(ToneStage, HighShelf_AtZeroDb_IsApproxIdentity) {
    ToneStage s;
    s.setType(ToneStage::Type::HighShelf);
    s.setEnabled(true);
    s.setCutoffFrequency(2000.0);
    s.setShelfGainDb(0.0);
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    constexpr int kTotalSamples = 9600; // 0.2s @ 48k
    constexpr double kSineHz = 6000.0;
    constexpr double kAmp = 0.3;

    juce::AudioBuffer<double> buf(1, kTotalSamples);
    for (int i = 0; i < kTotalSamples; ++i) {
        buf.setSample(0, i, kAmp * std::sin(2.0 * M_PI * kSineHz * i / kSampleRate));
    }
    s.process(buf);

    const double inputRms = kAmp / std::sqrt(2.0);
    const double outRms = rmsAtFrequency(buf, kSineHz, kTotalSamples - 4800, kTotalSamples);
    const double gainDb = 20.0 * std::log10(outRms / inputRms);
    EXPECT_NEAR(gainDb, 0.0, 0.1)
        << "HighShelf at 0 dB should be transparent — measured " << gainDb << " dB";
}

TEST(ToneStage, HighShelf_PositiveGain_BoostsHighs) {
    ToneStage s;
    s.setType(ToneStage::Type::HighShelf);
    s.setEnabled(true);
    s.setCutoffFrequency(2000.0); // Probe sits comfortably above
    s.setShelfGainDb(12.0);
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    constexpr int kTotalSamples = 9600;
    constexpr double kSineHz = 8000.0;
    constexpr double kAmp = 0.1;

    juce::AudioBuffer<double> buf(1, kTotalSamples);
    for (int i = 0; i < kTotalSamples; ++i) {
        buf.setSample(0, i, kAmp * std::sin(2.0 * M_PI * kSineHz * i / kSampleRate));
    }
    s.process(buf);

    const double inputRms = kAmp / std::sqrt(2.0);
    const double outRms = rmsAtFrequency(buf, kSineHz, kTotalSamples - 4800, kTotalSamples);
    const double gainDb = 20.0 * std::log10(outRms / inputRms);
    // 8 kHz is two octaves above the 2 kHz corner — RBJ shelf reaches the full
    // plateau gain well beyond the corner.
    EXPECT_NEAR(gainDb, 12.0, 1.5)
        << "HighShelf +12 dB at 8 kHz (corner 2 kHz) should boost ~12 dB; got " << gainDb;
}

TEST(ToneStage, HighShelf_NegativeGain_CutsHighs) {
    ToneStage s;
    s.setType(ToneStage::Type::HighShelf);
    s.setEnabled(true);
    s.setCutoffFrequency(2000.0);
    s.setShelfGainDb(-12.0);
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    constexpr int kTotalSamples = 9600;
    constexpr double kSineHz = 8000.0;
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
        << "HighShelf -12 dB at 8 kHz should cut ~12 dB; got " << gainDb;
}

TEST(ToneStage, HighShelf_IgnoresResonanceAndFat) {
    ToneStage clean;
    ToneStage poisoned;

    auto configure = [](ToneStage& s) {
        s.setType(ToneStage::Type::HighShelf);
        s.setEnabled(true);
        s.setCutoffFrequency(3000.0);
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
        const double v = kAmp * std::sin(2.0 * M_PI * 5000.0 * i / kSampleRate);
        bufA.setSample(0, i, v);
        bufB.setSample(0, i, v);
    }
    clean.process(bufA);
    poisoned.process(bufB);

    for (int i = 0; i < kBlockSize; ++i) {
        ASSERT_DOUBLE_EQ(bufA.getSample(0, i), bufB.getSample(0, i))
            << "HighShelf must ignore Resonance/Fat at sample " << i;
    }
}

TEST(ToneStage, HighShelf_NoClickOnActivation) {
    // Switch LP24 -> HighShelf mid-flight with loud signal.
    ToneStage s;
    s.setType(ToneStage::Type::Lowpass24dB);
    s.setEnabled(true);
    s.setCutoffFrequency(1500.0);
    s.setResonance(0.6);
    s.setFat(80.0);
    s.setShelfGainDb(9.0); // priming HS while idle
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    juce::AudioBuffer<double> buf(1, kBlockSize);
    int sampleCount = 0;

    for (int block = 0; block < 20; ++block) {
        fillSineBlock(buf, 80.0, sampleCount, 0.9);
        s.process(buf);
        sampleCount += kBlockSize;
    }
    ASSERT_TRUE(std::isfinite(maxAbs(buf)));

    s.setType(ToneStage::Type::HighShelf);

    for (int block = 0; block < 20; ++block) {
        fillSineBlock(buf, 80.0, sampleCount, 0.9);
        s.process(buf);
        for (int i = 0; i < kBlockSize; ++i) {
            const double v = buf.getSample(0, i);
            ASSERT_TRUE(std::isfinite(v))
                << "Non-finite after LP->HighShelf switch at block " << block
                << " sample " << i << " v=" << v;
            ASSERT_LT(std::abs(v), 10.0)
                << "Runaway after LP->HighShelf switch at block " << block
                << " sample " << i << " v=" << v;
        }
        const double delta = maxSampleToSampleDelta(buf);
        ASSERT_LT(delta, 1.0) << "Click at LP->HighShelf switch boundary, block " << block
                              << " delta=" << delta;
        sampleCount += kBlockSize;
    }
}

TEST(ToneStage, HighShelf_NoClickOnShelfToShelfSwap) {
    // LowShelf -> HighShelf, both shelves share Tone_Cutoff and Tone_Gain;
    // verify the topology swap is graceful.
    ToneStage s;
    s.setType(ToneStage::Type::LowShelf);
    s.setEnabled(true);
    s.setCutoffFrequency(800.0);
    s.setShelfGainDb(12.0);
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    juce::AudioBuffer<double> buf(1, kBlockSize);
    int sampleCount = 0;

    for (int block = 0; block < 20; ++block) {
        fillSineBlock(buf, 200.0, sampleCount, 0.7);
        s.process(buf);
        sampleCount += kBlockSize;
    }
    ASSERT_TRUE(std::isfinite(maxAbs(buf)));

    s.setType(ToneStage::Type::HighShelf);

    for (int block = 0; block < 20; ++block) {
        fillSineBlock(buf, 200.0, sampleCount, 0.7);
        s.process(buf);
        for (int i = 0; i < kBlockSize; ++i) {
            const double v = buf.getSample(0, i);
            ASSERT_TRUE(std::isfinite(v))
                << "Non-finite after LS->HS switch at block " << block
                << " sample " << i << " v=" << v;
            ASSERT_LT(std::abs(v), 10.0)
                << "Runaway after LS->HS switch at block " << block
                << " sample " << i << " v=" << v;
        }
        const double delta = maxSampleToSampleDelta(buf);
        ASSERT_LT(delta, 1.0) << "Click at LS->HS switch boundary, block " << block
                              << " delta=" << delta;
        sampleCount += kBlockSize;
    }
}

// --------------------------------------------------------------------------
// Smile tests — combined LS + HS strategy with linked gain and LS-corner-as-
// ratio-of-HS-corner. The DSP per-shelf is already exercised by LowShelfStage
// and HighShelfStage tests; these pin the ToneStage strategy plumbing:
// frequency drives HS, gain drives both shelves, and the LS corner tracks the
// HS corner through the ratio.
// --------------------------------------------------------------------------

TEST(ToneStage, Smile_AtZeroDb_IsApproxIdentity) {
    ToneStage s;
    s.setType(ToneStage::Type::Smile);
    s.setEnabled(true);
    s.setCutoffFrequency(3000.0);
    s.setShelfGainDb(0.0);
    s.setLowShelfRatio(0.0625);
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    constexpr int kTotalSamples = 9600;
    constexpr double kSineHz = 100.0;
    constexpr double kAmp = 0.3;

    juce::AudioBuffer<double> buf(1, kTotalSamples);
    for (int i = 0; i < kTotalSamples; ++i) {
        buf.setSample(0, i, kAmp * std::sin(2.0 * M_PI * kSineHz * i / kSampleRate));
    }
    s.process(buf);

    const double inputRms = kAmp / std::sqrt(2.0);
    const double outRms = rmsAtFrequency(buf, kSineHz, kTotalSamples - 4800, kTotalSamples);
    const double gainDb = 20.0 * std::log10(outRms / inputRms);
    EXPECT_NEAR(gainDb, 0.0, 0.1) << "Smile at 0 dB should be transparent — measured " << gainDb << " dB";
}

TEST(ToneStage, Smile_PositiveGain_BoostsBothBands) {
    // hsHz=3000, ratio=0.0625 -> lsHz=187.5 Hz.
    // 60 Hz probe sits an octave+ below the LS corner -> low-band boost.
    // 12 kHz probe sits two octaves above the HS corner -> high-band boost.
    ToneStage s;
    s.setType(ToneStage::Type::Smile);
    s.setEnabled(true);
    s.setCutoffFrequency(3000.0);
    s.setShelfGainDb(12.0);
    s.setLowShelfRatio(0.0625);
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    auto measureGainDb = [&](double probeHz) {
        ToneStage local;
        local.setType(ToneStage::Type::Smile);
        local.setEnabled(true);
        local.setCutoffFrequency(3000.0);
        local.setShelfGainDb(12.0);
        local.setLowShelfRatio(0.0625);
        local.prepareToPlay(kSampleRate, kBlockSize, 1);

        constexpr int kTotalSamples = 9600;
        constexpr double kAmp = 0.1;
        juce::AudioBuffer<double> buf(1, kTotalSamples);
        for (int i = 0; i < kTotalSamples; ++i) {
            buf.setSample(0, i, kAmp * std::sin(2.0 * M_PI * probeHz * i / kSampleRate));
        }
        local.process(buf);
        const double inputRms = kAmp / std::sqrt(2.0);
        const double outRms = rmsAtFrequency(buf, probeHz, kTotalSamples - 4800, kTotalSamples);
        return 20.0 * std::log10(outRms / inputRms);
    };

    EXPECT_NEAR(measureGainDb(60.0), 12.0, 1.5) << "Smile +12 dB at 60 Hz should boost ~12 dB";
    EXPECT_NEAR(measureGainDb(12000.0), 12.0, 1.5) << "Smile +12 dB at 12 kHz should boost ~12 dB";
}

TEST(ToneStage, Smile_NegativeGain_CutsBothBands) {
    auto measureGainDb = [&](double probeHz) {
        ToneStage local;
        local.setType(ToneStage::Type::Smile);
        local.setEnabled(true);
        local.setCutoffFrequency(3000.0);
        local.setShelfGainDb(-12.0);
        local.setLowShelfRatio(0.0625);
        local.prepareToPlay(kSampleRate, kBlockSize, 1);

        constexpr int kTotalSamples = 9600;
        constexpr double kAmp = 0.1;
        juce::AudioBuffer<double> buf(1, kTotalSamples);
        for (int i = 0; i < kTotalSamples; ++i) {
            buf.setSample(0, i, kAmp * std::sin(2.0 * M_PI * probeHz * i / kSampleRate));
        }
        local.process(buf);
        const double inputRms = kAmp / std::sqrt(2.0);
        const double outRms = rmsAtFrequency(buf, probeHz, kTotalSamples - 4800, kTotalSamples);
        return 20.0 * std::log10(outRms / inputRms);
    };

    EXPECT_NEAR(measureGainDb(60.0), -12.0, 1.5) << "Smile -12 dB at 60 Hz should cut ~12 dB";
    EXPECT_NEAR(measureGainDb(12000.0), -12.0, 1.5) << "Smile -12 dB at 12 kHz should cut ~12 dB";
}

TEST(ToneStage, Smile_LowShelfRatioTracksHighShelfFrequency) {
    // Re-target HS to 1 kHz with ratio 0.125 -> LS corner at 125 Hz.
    // 60 Hz probe sits below the LS corner (still boosts).
    // 2 kHz probe sits above the HS corner (still boosts).
    auto measureGainDb = [&](double probeHz, double hsHz, double ratio) {
        ToneStage local;
        local.setType(ToneStage::Type::Smile);
        local.setEnabled(true);
        local.setCutoffFrequency(hsHz);
        local.setShelfGainDb(12.0);
        local.setLowShelfRatio(ratio);
        local.prepareToPlay(kSampleRate, kBlockSize, 1);

        constexpr int kTotalSamples = 9600;
        constexpr double kAmp = 0.1;
        juce::AudioBuffer<double> buf(1, kTotalSamples);
        for (int i = 0; i < kTotalSamples; ++i) {
            buf.setSample(0, i, kAmp * std::sin(2.0 * M_PI * probeHz * i / kSampleRate));
        }
        local.process(buf);
        const double inputRms = kAmp / std::sqrt(2.0);
        const double outRms = rmsAtFrequency(buf, probeHz, kTotalSamples - 4800, kTotalSamples);
        return 20.0 * std::log10(outRms / inputRms);
    };

    // With hsHz=1000, ratio=0.125, lsHz=125. 60 Hz (well below LS) -> ~+12 dB boost.
    EXPECT_GT(measureGainDb(60.0, 1000.0, 0.125), 9.0)
        << "Smile with hsHz=1000, ratio=0.125: 60 Hz probe should boost (LS corner is 125 Hz)";

    // With hsHz=1000, ratio=0.0156 (= 1/64), lsHz~15.6 Hz (effectively below
    // LowShelfStage's 20 Hz floor — LS becomes inert). 60 Hz probe now sits
    // ABOVE the LS corner and BELOW the HS corner -> should NOT boost.
    EXPECT_LT(measureGainDb(60.0, 1000.0, 0.0156), 3.0)
        << "Smile with hsHz=1000, ratio=0.0156: 60 Hz probe should NOT boost much (LS pushed below 20 Hz floor)";
}

TEST(ToneStage, Smile_IgnoresResonanceAndFat) {
    ToneStage clean;
    ToneStage poisoned;

    auto configure = [](ToneStage& s) {
        s.setType(ToneStage::Type::Smile);
        s.setEnabled(true);
        s.setCutoffFrequency(3000.0);
        s.setShelfGainDb(6.0);
        s.setLowShelfRatio(0.0625);
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
            << "Smile must ignore Resonance/Fat at sample " << i;
    }
}

TEST(ToneStage, Smile_NoClickOnActivation) {
    ToneStage s;
    s.setType(ToneStage::Type::Lowpass24dB);
    s.setEnabled(true);
    s.setCutoffFrequency(1500.0);
    s.setResonance(0.6);
    s.setFat(80.0);
    s.setShelfGainDb(9.0);   // primes Smile while idle
    s.setLowShelfRatio(0.0625);
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    juce::AudioBuffer<double> buf(1, kBlockSize);
    int sampleCount = 0;

    for (int block = 0; block < 20; ++block) {
        fillSineBlock(buf, 80.0, sampleCount, 0.9);
        s.process(buf);
        sampleCount += kBlockSize;
    }
    ASSERT_TRUE(std::isfinite(maxAbs(buf)));

    s.setType(ToneStage::Type::Smile);

    for (int block = 0; block < 20; ++block) {
        fillSineBlock(buf, 80.0, sampleCount, 0.9);
        s.process(buf);
        for (int i = 0; i < kBlockSize; ++i) {
            const double v = buf.getSample(0, i);
            ASSERT_TRUE(std::isfinite(v))
                << "Non-finite after LP->Smile switch at block " << block << " sample " << i << " v=" << v;
            ASSERT_LT(std::abs(v), 10.0)
                << "Runaway after LP->Smile switch at block " << block << " sample " << i << " v=" << v;
        }
        const double delta = maxSampleToSampleDelta(buf);
        ASSERT_LT(delta, 1.0) << "Click at LP->Smile switch boundary, block " << block << " delta=" << delta;
        sampleCount += kBlockSize;
    }
}

TEST(ToneStage, Smile_NoClickOnDeactivation) {
    ToneStage s;
    s.setType(ToneStage::Type::Smile);
    s.setEnabled(true);
    s.setCutoffFrequency(3000.0);
    s.setShelfGainDb(12.0);
    s.setLowShelfRatio(0.0625);
    s.setResonance(0.5);     // primes LP while idle
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
                << "Non-finite after Smile->LP switch at block " << block << " sample " << i << " v=" << v;
            ASSERT_LT(std::abs(v), 10.0)
                << "Runaway after Smile->LP switch at block " << block << " sample " << i << " v=" << v;
        }
        const double delta = maxSampleToSampleDelta(buf);
        ASSERT_LT(delta, 1.0) << "Click at Smile->LP switch boundary, block " << block << " delta=" << delta;
        sampleCount += kBlockSize;
    }
}

// --------------------------------------------------------------------------
// Bell (parametric peaking) tests. The BellStage DSP is exercised by its own
// tests; these pin the ToneStage strategy plumbing — that Bell is reachable
// through ToneStage, honours Q (the only Bell-specific param), ignores LP-only
// params, and is click-free on activation / deactivation.
// --------------------------------------------------------------------------

TEST(ToneStage, Bell_AtZeroDb_IsApproxIdentity) {
    ToneStage s;
    s.setType(ToneStage::Type::Bell);
    s.setEnabled(true);
    s.setCutoffFrequency(1000.0);
    s.setShelfGainDb(0.0);
    s.setQ(1.0);
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    constexpr int kTotalSamples = 9600;
    constexpr double kSineHz = 1000.0;
    constexpr double kAmp = 0.3;

    juce::AudioBuffer<double> buf(1, kTotalSamples);
    for (int i = 0; i < kTotalSamples; ++i) {
        buf.setSample(0, i, kAmp * std::sin(2.0 * M_PI * kSineHz * i / kSampleRate));
    }
    s.process(buf);

    const double inputRms = kAmp / std::sqrt(2.0);
    const double outRms = rmsAtFrequency(buf, kSineHz, kTotalSamples - 4800, kTotalSamples);
    const double gainDb = 20.0 * std::log10(outRms / inputRms);
    EXPECT_NEAR(gainDb, 0.0, 0.1)
        << "Bell at 0 dB should be transparent — measured " << gainDb << " dB";
}

TEST(ToneStage, Bell_PositiveGain_BoostsAtCentre) {
    ToneStage s;
    s.setType(ToneStage::Type::Bell);
    s.setEnabled(true);
    s.setCutoffFrequency(1000.0);
    s.setShelfGainDb(12.0);
    s.setQ(1.0);
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    constexpr int kTotalSamples = 9600;
    constexpr double kSineHz = 1000.0;
    constexpr double kAmp = 0.05; // headroom for 4x boost

    juce::AudioBuffer<double> buf(1, kTotalSamples);
    for (int i = 0; i < kTotalSamples; ++i) {
        buf.setSample(0, i, kAmp * std::sin(2.0 * M_PI * kSineHz * i / kSampleRate));
    }
    s.process(buf);

    const double inputRms = kAmp / std::sqrt(2.0);
    const double outRms = rmsAtFrequency(buf, kSineHz, kTotalSamples - 4800, kTotalSamples);
    const double gainDb = 20.0 * std::log10(outRms / inputRms);
    EXPECT_NEAR(gainDb, 12.0, 1.5)
        << "Bell +12 dB at 1 kHz centre should boost ~12 dB; got " << gainDb;
}

TEST(ToneStage, Bell_HigherQ_IsNarrower) {
    // Probe one octave above centre. At fixed +12 dB gain, the low-Q bell
    // should still meaningfully boost that probe; the high-Q bell should not.
    constexpr double centreHz = 1000.0;
    constexpr double probeHz = 2000.0;
    constexpr double gainDb = 12.0;
    constexpr int kTotalSamples = 9600;
    constexpr double kAmp = 0.05;

    auto measure = [&](double q) {
        ToneStage s;
        s.setType(ToneStage::Type::Bell);
        s.setEnabled(true);
        s.setCutoffFrequency(centreHz);
        s.setShelfGainDb(gainDb);
        s.setQ(q);
        s.prepareToPlay(kSampleRate, kBlockSize, 1);

        juce::AudioBuffer<double> buf(1, kTotalSamples);
        for (int i = 0; i < kTotalSamples; ++i) {
            buf.setSample(0, i, kAmp * std::sin(2.0 * M_PI * probeHz * i / kSampleRate));
        }
        s.process(buf);

        const double inputRms = kAmp / std::sqrt(2.0);
        const double outRms = rmsAtFrequency(buf, probeHz, kTotalSamples - 4800, kTotalSamples);
        return 20.0 * std::log10(outRms / inputRms);
    };

    const double wideBoostDb = measure(0.5);
    const double narrowBoostDb = measure(5.0);

    EXPECT_GT(wideBoostDb, narrowBoostDb + 3.0)
        << "Wide-Q bell should boost the off-centre probe by at least 3 dB more than narrow-Q. "
        << "Wide(Q=0.5): " << wideBoostDb << " dB, Narrow(Q=5): " << narrowBoostDb << " dB";
}

TEST(ToneStage, Bell_IgnoresResonanceAndFat) {
    // LP-only params (resonance, fat) must be no-ops on Bell. Two ToneStage
    // instances configured identically except one has the LP params hammered.
    ToneStage clean;
    ToneStage poisoned;

    auto configure = [](ToneStage& s) {
        s.setType(ToneStage::Type::Bell);
        s.setEnabled(true);
        s.setCutoffFrequency(1500.0);
        s.setShelfGainDb(6.0);
        s.setQ(1.5);
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
        const double v = kAmp * std::sin(2.0 * M_PI * 1500.0 * i / kSampleRate);
        bufA.setSample(0, i, v);
        bufB.setSample(0, i, v);
    }
    clean.process(bufA);
    poisoned.process(bufB);

    for (int i = 0; i < kBlockSize; ++i) {
        ASSERT_DOUBLE_EQ(bufA.getSample(0, i), bufB.getSample(0, i))
            << "Bell must ignore Resonance / Fat at sample " << i;
    }
}

TEST(ToneStage, Bell_NoClickOnActivation) {
    // Switch LP24 -> Bell mid-flight with loud signal; verify no NaN/inf and
    // no catastrophic sample-to-sample discontinuity in the first post-switch
    // block.
    ToneStage s;
    s.setType(ToneStage::Type::Lowpass24dB);
    s.setEnabled(true);
    s.setCutoffFrequency(1500.0);
    s.setResonance(0.6);
    s.setFat(80.0);
    s.setShelfGainDb(9.0); // priming Bell gain while idle
    s.setQ(2.0);           // priming Bell Q while idle
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    juce::AudioBuffer<double> buf(1, kBlockSize);
    int sampleCount = 0;

    for (int block = 0; block < 20; ++block) {
        fillSineBlock(buf, 80.0, sampleCount, 0.9);
        s.process(buf);
        sampleCount += kBlockSize;
    }
    ASSERT_TRUE(std::isfinite(maxAbs(buf)));

    s.setType(ToneStage::Type::Bell);

    for (int block = 0; block < 20; ++block) {
        fillSineBlock(buf, 80.0, sampleCount, 0.9);
        s.process(buf);
        for (int i = 0; i < kBlockSize; ++i) {
            const double v = buf.getSample(0, i);
            ASSERT_TRUE(std::isfinite(v))
                << "Non-finite after LP->Bell switch at block " << block << " sample " << i << " v=" << v;
            ASSERT_LT(std::abs(v), 10.0)
                << "Runaway after LP->Bell switch at block " << block << " sample " << i << " v=" << v;
        }
        const double delta = maxSampleToSampleDelta(buf);
        ASSERT_LT(delta, 1.0) << "Click at LP->Bell switch boundary, block " << block << " delta=" << delta;
        sampleCount += kBlockSize;
    }
}

TEST(ToneStage, Bell_NoClickOnDeactivation) {
    // Symmetric to activation: Bell -> LP12 mid-flight.
    ToneStage s;
    s.setType(ToneStage::Type::Bell);
    s.setEnabled(true);
    s.setCutoffFrequency(1000.0);
    s.setShelfGainDb(12.0);
    s.setQ(2.0);
    s.setResonance(0.5); // priming LP while idle
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
                << "Non-finite after Bell->LP switch at block " << block << " sample " << i << " v=" << v;
            ASSERT_LT(std::abs(v), 10.0)
                << "Runaway after Bell->LP switch at block " << block << " sample " << i << " v=" << v;
        }
        const double delta = maxSampleToSampleDelta(buf);
        ASSERT_LT(delta, 1.0) << "Click at Bell->LP switch boundary, block " << block << " delta=" << delta;
        sampleCount += kBlockSize;
    }
}

// --------------------------------------------------------------------------
// Hysteresis. HysteresisStrategy is now a pure marker — its process() is a
// no-op. Selecting Hysteresis as the tone type only signals the pipeline-level
// HysteresisStage (the saturator that wraps the waveshaper) to engage; the
// ToneStage itself produces no audio. These tests pin that contract.
// --------------------------------------------------------------------------

// With type == Hysteresis the stage is a sample-identical pass-through: the
// emphasis shelves were removed and the saturation lives elsewhere.
TEST(ToneStage, Hysteresis_IsPassThroughNoOp) {
    ToneStage s;
    s.setType(ToneStage::Type::Hysteresis);
    s.setEnabled(true);
    s.setCutoffFrequency(1500.0); // forwarded to the marker; must stay a no-op
    s.setResonance(0.6);
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    juce::AudioBuffer<double> input(1, kBlockSize);
    juce::AudioBuffer<double> output(1, kBlockSize);
    for (int i = 0; i < kBlockSize; ++i) {
        const double v = 0.3 * std::sin(2.0 * M_PI * 1000.0 * i / kSampleRate);
        input.setSample(0, i, v);
        output.setSample(0, i, v);
    }
    s.process(output);

    for (int i = 0; i < kBlockSize; ++i) {
        ASSERT_DOUBLE_EQ(input.getSample(0, i), output.getSample(0, i))
            << "Hysteresis tone type must be a strict pass-through (sample " << i << ")";
    }
}

// Switch LP24 -> Hysteresis mid-flight with a loud signal; verify no NaN/Inf
// and no runaway. After the switch the stage is a pass-through, so the only
// risk is stale state surviving the type transition.
TEST(ToneStage, Hysteresis_NoClickOnActivation) {
    ToneStage s;
    s.setType(ToneStage::Type::Lowpass24dB);
    s.setEnabled(true);
    s.setCutoffFrequency(1500.0);
    s.setResonance(0.6);
    s.setFat(80.0);
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    juce::AudioBuffer<double> buf(1, kBlockSize);
    int sampleCount = 0;

    for (int block = 0; block < 20; ++block) {
        fillSineBlock(buf, 80.0, sampleCount, 0.9);
        s.process(buf);
        sampleCount += kBlockSize;
    }
    ASSERT_TRUE(std::isfinite(maxAbs(buf)));

    s.setType(ToneStage::Type::Hysteresis);

    for (int block = 0; block < 20; ++block) {
        fillSineBlock(buf, 80.0, sampleCount, 0.9);
        s.process(buf);
        for (int i = 0; i < kBlockSize; ++i) {
            const double v = buf.getSample(0, i);
            ASSERT_TRUE(std::isfinite(v))
                << "Non-finite after LP->Hysteresis switch at block " << block << " sample " << i << " v=" << v;
            ASSERT_LT(std::abs(v), 10.0)
                << "Runaway after LP->Hysteresis switch at block " << block << " sample " << i << " v=" << v;
        }
        const double delta = maxSampleToSampleDelta(buf);
        ASSERT_LT(delta, 1.0)
            << "Click at LP->Hysteresis switch boundary, block " << block << " delta=" << delta;
        sampleCount += kBlockSize;
    }
}

// --------------------------------------------------------------------------
// Highpass — sibling of the Lowpass tests. Verifies the new Highpass12dB /
// Highpass24dB Tone types dispatch to the HPF strategy correctly and survive
// type switches without clicks.
// --------------------------------------------------------------------------

namespace {

// Drive a sine into ToneStage at the given Type/cutoff/resonance and measure
// settled RMS gain relative to input. Skips startup transient.
double toneStageSettledSineGain(ToneStage::Type type, double cutoffHz, double sineHz,
                                double resonanceNorm, double amp) {
    ToneStage s;
    s.setType(type);
    s.setEnabled(true);
    s.setCutoffFrequency(cutoffHz);
    s.setResonance(resonanceNorm);
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    constexpr int kBlocks = 24; // ~250 ms @ 48k, 512-sample blocks
    juce::AudioBuffer<double> buf(1, kBlockSize);
    int sampleCount = 0;
    // Warm-up — discard.
    for (int b = 0; b < kBlocks / 2; ++b) {
        fillSineBlock(buf, sineHz, sampleCount, amp);
        s.process(buf);
        sampleCount += kBlockSize;
    }
    // Measure across the latter half.
    double sumSq = 0.0;
    int count = 0;
    for (int b = 0; b < kBlocks / 2; ++b) {
        fillSineBlock(buf, sineHz, sampleCount, amp);
        s.process(buf);
        for (int i = 0; i < kBlockSize; ++i) {
            const double v = buf.getSample(0, i);
            sumSq += v * v;
            ++count;
        }
        sampleCount += kBlockSize;
    }
    const double rms = std::sqrt(sumSq / std::max(1, count));
    const double inputRms = amp / std::sqrt(2.0);
    return rms / inputRms;
}

} // namespace

TEST(ToneStage, Highpass24dB_PassesHfAttenuatesLf) {
    // Cutoff 1 kHz. 50 Hz (well below) should be strongly attenuated; 8 kHz
    // (well above) should pass at ≈unity (small-signal passband makeup).
    const double lfGain = toneStageSettledSineGain(ToneStage::Type::Highpass24dB,
                                                   /*cutoffHz=*/1000.0, /*sineHz=*/50.0,
                                                   /*resonanceNorm=*/0.0, /*amp=*/0.1);
    EXPECT_LT(lfGain, 0.01)
        << "24dB HPF passed 50 Hz at gain " << lfGain << " — expected <-40 dB";

    const double hfGain = toneStageSettledSineGain(ToneStage::Type::Highpass24dB,
                                                   /*cutoffHz=*/1000.0, /*sineHz=*/8000.0,
                                                   /*resonanceNorm=*/0.0, /*amp=*/0.1);
    EXPECT_NEAR(hfGain, 1.0, 0.05)
        << "24dB HPF HF gain at 8 kHz is " << hfGain << " (expected ≈1.0)";
}

TEST(ToneStage, Highpass12dB_PassesHfAttenuatesLf) {
    const double lfGain = toneStageSettledSineGain(ToneStage::Type::Highpass12dB,
                                                   /*cutoffHz=*/1000.0, /*sineHz=*/50.0,
                                                   /*resonanceNorm=*/0.0, /*amp=*/0.1);
    EXPECT_LT(lfGain, 0.05)
        << "12dB HPF passed 50 Hz at gain " << lfGain;

    const double hfGain = toneStageSettledSineGain(ToneStage::Type::Highpass12dB,
                                                   /*cutoffHz=*/1000.0, /*sineHz=*/8000.0,
                                                   /*resonanceNorm=*/0.0, /*amp=*/0.1);
    EXPECT_NEAR(hfGain, 1.0, 0.05)
        << "12dB HPF HF gain at 8 kHz is " << hfGain;
}

TEST(ToneStage, Highpass_ResonanceSweepHfRmsPreserved_24dB) {
    // Mirror of BassCompensation_ResonanceSweepRmsPreserved — across the
    // resonance range, the passband (HF) RMS should stay near unity. Pinning
    // this guards the (1+R) HF makeup. Tolerance ±0.7 dB to allow for the
    // slight pre-peak dip just below cutoff (3 kHz is above cutoff but the
    // resonance peak skirt still nudges it).
    for (double r : {0.0, 0.25, 0.5, 0.75}) {
        const double gain = toneStageSettledSineGain(ToneStage::Type::Highpass24dB,
                                                     /*cutoffHz=*/300.0, /*sineHz=*/8000.0,
                                                     /*resonanceNorm=*/r, /*amp=*/0.1);
        EXPECT_NEAR(gain, 1.0, 0.08)
            << "HF RMS gain at resonanceNorm=" << r << " is " << gain
            << " (≈" << 20.0 * std::log10(gain) << " dB)";
    }
}

TEST(ToneStage, Highpass_NoClickOnActivationFromLowpass) {
    // LP24 -> HP24 with same cutoff/resonance must transition cleanly. Both
    // strategies are kept primed via per-block forwarding, so the switch
    // boundary should not introduce a step discontinuity.
    ToneStage s;
    s.setType(ToneStage::Type::Lowpass24dB);
    s.setEnabled(true);
    s.setCutoffFrequency(1500.0);
    s.setResonance(0.4);
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    juce::AudioBuffer<double> buf(1, kBlockSize);
    int sampleCount = 0;
    for (int block = 0; block < 20; ++block) {
        fillSineBlock(buf, 800.0, sampleCount, 0.5);
        s.process(buf);
        sampleCount += kBlockSize;
    }
    ASSERT_TRUE(std::isfinite(maxAbs(buf)));

    s.setType(ToneStage::Type::Highpass24dB);

    for (int block = 0; block < 20; ++block) {
        fillSineBlock(buf, 800.0, sampleCount, 0.5);
        s.process(buf);
        for (int i = 0; i < kBlockSize; ++i) {
            const double v = buf.getSample(0, i);
            ASSERT_TRUE(std::isfinite(v))
                << "Non-finite after LP->HP switch at block " << block << " sample " << i;
            ASSERT_LT(std::abs(v), 10.0)
                << "Runaway after LP->HP switch at block " << block << " sample " << i << " v=" << v;
        }
        const double delta = maxSampleToSampleDelta(buf);
        ASSERT_LT(delta, 1.0)
            << "Click at LP24->HP24 boundary, block " << block << " delta=" << delta;
        sampleCount += kBlockSize;
    }
}

TEST(ToneStage, Highpass_FatSetterIsNoOp) {
    // HighpassStrategy ignores setFat (no Fat sub-stage). Setting Fat to
    // 100 % before and after activation must not change the HPF's output.
    ToneStage withFat;
    ToneStage withoutFat;
    withFat.setType(ToneStage::Type::Highpass24dB);
    withoutFat.setType(ToneStage::Type::Highpass24dB);
    withFat.setEnabled(true);
    withoutFat.setEnabled(true);
    for (auto* s : {&withFat, &withoutFat}) {
        s->setCutoffFrequency(500.0);
        s->setResonance(0.3);
        s->prepareToPlay(kSampleRate, kBlockSize, 1);
    }
    withFat.setFat(100.0);

    juce::AudioBuffer<double> a(1, kBlockSize);
    juce::AudioBuffer<double> b(1, kBlockSize);
    for (int block = 0; block < 8; ++block) {
        fillSineBlock(a, 1200.0, block * kBlockSize, 0.5);
        fillSineBlock(b, 1200.0, block * kBlockSize, 0.5);
        withFat.process(a);
        withoutFat.process(b);
    }
    for (int i = 0; i < kBlockSize; ++i) {
        EXPECT_DOUBLE_EQ(a.getSample(0, i), b.getSample(0, i))
            << "HPF output differs with Fat=100% vs untouched at sample " << i
            << " — setFat() should be a no-op on the HPF";
    }
}

// --------------------------------------------------------------------------
// Editor-driven recompute: handing a caller-supplied params struct writes
// the LUT and bumps the version, regardless of whether the audio thread has
// ever called process() or moved cachedXxx atomics. This is what lets the
// freq-response trace update while the host transport is stopped.
//
// Sanity check: a Lowpass24dB centred at 1 kHz should attenuate well below
// 0 dB by 10 kHz (Moog ladder, ~-24 dB/oct asymptote → roughly -48 dB at 10×
// the cutoff with R=0; we test only "loudly down" to keep the assertion
// loose against the ladder's exact analytical shape).
// --------------------------------------------------------------------------
TEST(ToneStage, RecomputeFrequencyResponseFromParams_WritesLut) {
    ToneStage stage;
    stage.prepareToPlay(kSampleRate, kBlockSize, 1);

    const auto* version = stage.getFrequencyResponseVersion();
    ASSERT_NE(version, nullptr);
    const uint64_t versionBefore = version->load(std::memory_order_acquire);

    ToneFrequencyResponseParams params;
    params.type = ToneStage::Type::Lowpass24dB;
    params.cutoffHz = 1000.0;
    params.resonanceNorm = 0.0;
    params.sampleRate = kSampleRate;
    stage.recomputeFrequencyResponse(params);

    EXPECT_GT(version->load(std::memory_order_acquire), versionBefore);

    // Pull the freshly published LUT and check a few sentinel samples:
    //  - at ~1 kHz (cutoff) the magnitude should be near 0 dB or down a little.
    //  - at ~10 kHz it should be well below 0 dB.
    const double* lut = stage.getFrequencyResponseLUT();
    ASSERT_NE(lut, nullptr);
    const int n = stage.getFrequencyResponseSize();
    const double logMin = std::log10(stage.getFrequencyResponseMinHz());
    const double logMax = std::log10(stage.getFrequencyResponseMaxHz());
    auto indexAtHz = [&](double hz) {
        const double t = (std::log10(hz) - logMin) / (logMax - logMin);
        return std::clamp(static_cast<int>(std::round(t * (n - 1))), 0, n - 1);
    };

    // Pin only what the test really cares about: the new public method
    // actually fills the LUT with a monotonically-rolling-off lowpass shape.
    // The exact dB at cutoff depends on the ladder topology (4 cascaded poles
    // each ~-3 dB at fc → ~-12 dB) which is exercised by other ToneStage
    // tests; here we only check the rolloff is present and proportional.
    const double dbAt100  = lut[indexAtHz(100.0)];
    const double dbAt1k   = lut[indexAtHz(1000.0)];
    const double dbAt10k  = lut[indexAtHz(10000.0)];
    EXPECT_NEAR(dbAt100, 0.0, 1.0) << "LP24 deep below cutoff must pass with ~0 dB";
    EXPECT_LT(dbAt10k, dbAt1k)     << "LP24 must roll off past cutoff";
    EXPECT_LT(dbAt10k, -30.0)      << "LP24 at 10× cutoff should be deeply attenuated";
}

TEST(ToneStage, RecomputeFrequencyResponse_OffTypeProducesUnityGain) {
    ToneStage stage;
    stage.prepareToPlay(kSampleRate, kBlockSize, 1);

    ToneFrequencyResponseParams params;
    params.type = ToneStage::Type::Off;
    params.sampleRate = kSampleRate;
    stage.recomputeFrequencyResponse(params);

    const double* lut = stage.getFrequencyResponseLUT();
    const int n = stage.getFrequencyResponseSize();
    ASSERT_NE(lut, nullptr);
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(lut[i], 0.0, 1e-9) << "Type::Off should publish a flat 0 dB response at i=" << i;
    }
}

// ── computeIRMagnitudeResponseDb ────────────────────────────────────────────

namespace {

std::array<double, ToneStage::kFrequencyResponseSize> makeLogFreqGrid() {
    std::array<double, ToneStage::kFrequencyResponseSize> grid{};
    const double logMin = std::log10(ToneStage::kFrequencyResponseMinHz);
    const double logMax = std::log10(ToneStage::kFrequencyResponseMaxHz);
    const double denom = static_cast<double>(ToneStage::kFrequencyResponseSize - 1);
    for (int i = 0; i < ToneStage::kFrequencyResponseSize; ++i) {
        const double t = static_cast<double>(i) / denom;
        grid[static_cast<size_t>(i)] = std::pow(10.0, logMin + t * (logMax - logMin));
    }
    return grid;
}

int indexOfFreq(const std::array<double, ToneStage::kFrequencyResponseSize>& grid, double hz) {
    const double logMin = std::log10(ToneStage::kFrequencyResponseMinHz);
    const double logMax = std::log10(ToneStage::kFrequencyResponseMaxHz);
    const double t = (std::log10(hz) - logMin) / (logMax - logMin);
    const auto last = static_cast<int>(grid.size()) - 1;
    return std::clamp(static_cast<int>(std::round(t * static_cast<double>(last))), 0, last);
}

} // namespace

TEST(ComputeIRMagnitudeResponseDb, DeltaIRProducesFlatPeakNormalizedResponse) {
    // Kronecker delta: H(f) = 1 for all f → after peak-normalization every bin
    // should sit at 0 dB (within the dB floor that the function applies).
    constexpr int kIRLen = 4096;
    std::vector<float> ir(kIRLen, 0.0f);
    ir[0] = 1.0f;

    const auto grid = makeLogFreqGrid();
    std::array<double, ToneStage::kFrequencyResponseSize> mags{};
    computeIRMagnitudeResponseDb(ir.data(), kIRLen, kSampleRate, grid.data(),
                                  mags.data(), static_cast<int>(grid.size()));

    for (int i = 0; i < static_cast<int>(mags.size()); ++i) {
        const double hz = grid[static_cast<size_t>(i)];
        if (hz >= kSampleRate * 0.5) {
            continue; // above Nyquist — function may floor these
        }
        EXPECT_NEAR(mags[static_cast<size_t>(i)], 0.0, 0.25)
            << "delta IR should be flat 0 dB after peak-normalization at i=" << i << " (" << hz << " Hz)";
    }
}

TEST(ComputeIRMagnitudeResponseDb, SineIRPeaksAtSineFrequency) {
    // A pure tone IR (1 kHz sine, integer number of periods) has its spectral
    // peak at 1 kHz. After peak-normalization 1 kHz should be 0 dB and other
    // bands should be well below.
    constexpr double kSineHz = 1000.0;
    constexpr int kIRLen = 16384;
    std::vector<float> ir(kIRLen);
    for (int i = 0; i < kIRLen; ++i) {
        ir[static_cast<size_t>(i)] =
            static_cast<float>(std::sin(2.0 * M_PI * kSineHz * i / kSampleRate));
    }

    const auto grid = makeLogFreqGrid();
    std::array<double, ToneStage::kFrequencyResponseSize> mags{};
    computeIRMagnitudeResponseDb(ir.data(), kIRLen, kSampleRate, grid.data(),
                                  mags.data(), static_cast<int>(grid.size()));

    // Peak normalization invariant.
    double peak = mags[0];
    for (double m : mags) {
        peak = std::max(peak, m);
    }
    EXPECT_NEAR(peak, 0.0, 1e-9);

    // 1 kHz should be at or very near the peak; bands far from 1 kHz should
    // be well below it. The 1/6-octave bandpower smoothing widens the peak
    // slightly so we check ratios at 100 Hz and 8 kHz (well off the bandwidth).
    const double dbAt1k = mags[static_cast<size_t>(indexOfFreq(grid, 1000.0))];
    const double dbAt100 = mags[static_cast<size_t>(indexOfFreq(grid, 100.0))];
    const double dbAt8k = mags[static_cast<size_t>(indexOfFreq(grid, 8000.0))];
    EXPECT_GT(dbAt1k, -1.0) << "1 kHz tone IR should put its peak at 1 kHz (got " << dbAt1k << " dB)";
    EXPECT_LT(dbAt100, -20.0) << "100 Hz should be far below the 1 kHz peak (got " << dbAt100 << " dB)";
    EXPECT_LT(dbAt8k, -20.0) << "8 kHz should be far below the 1 kHz peak (got " << dbAt8k << " dB)";
}

TEST(ComputeIRMagnitudeResponseDb, EmptyInputFallsBackToFlat) {
    // Null pointer / zero length: writer should still leave the caller's buffer
    // in a defined state (no NaN, no out-of-range writes).
    const auto grid = makeLogFreqGrid();
    std::array<double, ToneStage::kFrequencyResponseSize> mags{};
    mags.fill(123.4);

    computeIRMagnitudeResponseDb(nullptr, 0, kSampleRate, grid.data(), mags.data(),
                                  static_cast<int>(grid.size()));

    for (double m : mags) {
        EXPECT_FALSE(std::isnan(m)) << "computeIRMagnitudeResponseDb must not produce NaN on empty input";
        EXPECT_EQ(m, 0.0) << "empty input must zero-fill the output buffer";
    }
}

TEST(ToneStage, RecomputeFrequencyResponse_IRTypeFullyWetReproducesMagnitudes) {
    // Editor flow: caller pre-computes magnitudes via computeIRMagnitudeResponseDb
    // and hands a pointer in via params.irMagnitudesDb. At irMix=1 (fully wet) the
    // dry/wet blend reproduces the IR curve (round-tripped through linear domain,
    // hence NEAR not exact).
    ToneStage stage;
    stage.prepareToPlay(kSampleRate, kBlockSize, 1);

    const int n = stage.getFrequencyResponseSize();
    std::vector<double> source(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        source[static_cast<size_t>(i)] = -3.0 * std::sin(static_cast<double>(i) * 0.01);
    }

    ToneFrequencyResponseParams params;
    params.type = ToneStage::Type::ImpulseResponse;
    params.sampleRate = kSampleRate;
    params.irMagnitudesDb = source.data();
    params.irMix = 1.0; // fully wet
    params.irPathFingerprint = "/fake/path.wav";
    stage.recomputeFrequencyResponse(params);

    const double* lut = stage.getFrequencyResponseLUT();
    ASSERT_NE(lut, nullptr);
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(lut[i], source[static_cast<size_t>(i)], 1e-9)
            << "ImpulseResponse at full wet must reproduce caller-supplied magnitudes at i=" << i;
    }
}

TEST(ToneStage, RecomputeFrequencyResponse_IRTypeMixZeroIsFlat) {
    // irMix=0 (fully dry) — the dry path is unity, so the blended trace must be
    // flat 0 dB regardless of the IR's shape.
    ToneStage stage;
    stage.prepareToPlay(kSampleRate, kBlockSize, 1);

    const int n = stage.getFrequencyResponseSize();
    std::vector<double> source(static_cast<size_t>(n), -12.0); // arbitrary non-flat IR
    source[100] = -40.0;

    ToneFrequencyResponseParams params;
    params.type = ToneStage::Type::ImpulseResponse;
    params.sampleRate = kSampleRate;
    params.irMagnitudesDb = source.data();
    params.irMix = 0.0; // fully dry
    params.irPathFingerprint = "/fake/path.wav";
    stage.recomputeFrequencyResponse(params);

    const double* lut = stage.getFrequencyResponseLUT();
    ASSERT_NE(lut, nullptr);
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(lut[i], 0.0, 1e-9) << "ImpulseResponse at mix=0 must be flat 0 dB at i=" << i;
    }
}

TEST(ToneStage, RecomputeFrequencyResponse_IRTypeMixHalfFillsNotch) {
    // A parallel dry path fills the IR's notches. At mix=0.5 a deep -40 dB notch
    // should rise well above -40 dB (dry energy fills it), while a 0 dB bin (peak)
    // stays at 0 dB (dry + wet both unity there).
    ToneStage stage;
    stage.prepareToPlay(kSampleRate, kBlockSize, 1);

    const int n = stage.getFrequencyResponseSize();
    std::vector<double> source(static_cast<size_t>(n), 0.0); // 0 dB everywhere...
    source[200] = -40.0;                                     // ...except a deep notch

    ToneFrequencyResponseParams params;
    params.type = ToneStage::Type::ImpulseResponse;
    params.sampleRate = kSampleRate;
    params.irMagnitudesDb = source.data();
    params.irMix = 0.5;
    params.irPathFingerprint = "/fake/path.wav";
    stage.recomputeFrequencyResponse(params);

    const double* lut = stage.getFrequencyResponseLUT();
    ASSERT_NE(lut, nullptr);
    // 0 dB bins stay at 0 dB: 0.5*1 + 0.5*1 = 1.0 → 0 dB.
    EXPECT_NEAR(lut[0], 0.0, 1e-9);
    // The notch fills: 0.5*1 + 0.5*10^(-40/20) = 0.505 → ~-5.9 dB, far above -40.
    EXPECT_GT(lut[200], -10.0) << "mix=0.5 must fill the -40 dB notch toward the dry reference";
    EXPECT_LT(lut[200], 0.0) << "the notch is still partly present at mix=0.5";
}

TEST(ToneStage, RecomputeFrequencyResponse_IRTypeNullPointerFallsBackToFlat) {
    // No IR loaded: irMagnitudesDb stays null. LUT should fall back to flat
    // 0 dB rather than carrying stale data from a previous strategy.
    ToneStage stage;
    stage.prepareToPlay(kSampleRate, kBlockSize, 1);

    // First populate the LUT with a non-flat strategy so we can prove the
    // IR fallback actually overwrites it.
    ToneFrequencyResponseParams setup;
    setup.type = ToneStage::Type::LowShelf;
    setup.cutoffHz = 200.0;
    setup.shelfGainDb = 6.0;
    setup.sampleRate = kSampleRate;
    stage.recomputeFrequencyResponse(setup);

    ToneFrequencyResponseParams params;
    params.type = ToneStage::Type::ImpulseResponse;
    params.sampleRate = kSampleRate;
    params.irMagnitudesDb = nullptr;
    stage.recomputeFrequencyResponse(params);

    const double* lut = stage.getFrequencyResponseLUT();
    ASSERT_NE(lut, nullptr);
    const int n = stage.getFrequencyResponseSize();
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(lut[i], 0.0, 1e-9)
            << "ImpulseResponse with no IR must publish a flat 0 dB trace at i=" << i;
    }
}

