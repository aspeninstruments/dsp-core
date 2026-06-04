#include "../dsp_core/Source/audio_pipeline/LadderHighpassTPTStage.h"
#include <gtest/gtest.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <cmath>

using namespace dsp_core::audio_pipeline;

namespace {

constexpr double kSampleRate = 48000.0;
constexpr int kBlockSize = 256;

template <typename Stage>
void processSilence(Stage& s, int numBlocks, juce::AudioBuffer<double>& buf) {
    for (int n = 0; n < numBlocks; ++n) {
        buf.clear();
        s.process(buf);
    }
}

template <typename Stage>
double settledSineGain(double cutoffHz, double sineHz, double R, double amp,
                       int totalSamples = 9600, int tailSamples = 4800) {
    Stage s;
    s.setCutoffFrequency(cutoffHz);
    s.setResonance(R);
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    juce::AudioBuffer<double> buf(1, totalSamples);
    for (int i = 0; i < totalSamples; ++i) {
        buf.getWritePointer(0)[i] = amp * std::sin(2.0 * M_PI * sineHz * i / kSampleRate);
    }
    s.process(buf);
    const int start = std::max(0, totalSamples - tailSamples);
    double sumSq = 0.0;
    int count = 0;
    for (int i = start; i < totalSamples; ++i) {
        const double v = buf.getReadPointer(0)[i];
        sumSq += v * v;
        ++count;
    }
    const double rms = std::sqrt(sumSq / std::max(1, count));
    const double inputRms = amp / std::sqrt(2.0);
    return rms / inputRms;
}

template <typename Stage>
double atCutoffSinePeak(double cutoffHz, double R, double amp) {
    Stage s;
    s.setCutoffFrequency(cutoffHz);
    s.setResonance(R);
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    constexpr int kTotalSamples = 14400;
    juce::AudioBuffer<double> buf(1, kTotalSamples);
    for (int i = 0; i < kTotalSamples; ++i) {
        buf.getWritePointer(0)[i] = amp * std::sin(2.0 * M_PI * cutoffHz * i / kSampleRate);
    }
    s.process(buf);
    double peak = 0.0;
    const int start = kTotalSamples - 4800;
    for (int i = start; i < kTotalSamples; ++i) {
        peak = std::max(peak, std::abs(buf.getReadPointer(0)[i]));
    }
    return peak / amp;
}

template <typename Stage>
double impulseDecayRatio(double R, int numSilenceBlocks) {
    Stage s;
    s.setCutoffFrequency(1000.0);
    s.setResonance(R);
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    juce::AudioBuffer<double> buf(1, kBlockSize);
    buf.clear();
    buf.getWritePointer(0)[0] = 1.0;
    s.process(buf);

    double energyFirst = 0.0;
    for (int i = 0; i < kBlockSize; ++i) {
        const double v = buf.getReadPointer(0)[i];
        energyFirst += v * v;
    }

    for (int b = 0; b < numSilenceBlocks; ++b) {
        buf.clear();
        s.process(buf);
    }

    double energyLast = 0.0;
    for (int i = 0; i < kBlockSize; ++i) {
        const double v = buf.getReadPointer(0)[i];
        energyLast += v * v;
    }
    return energyLast / std::max(energyFirst, 1e-300);
}

} // namespace

// --------------------------------------------------------------------------
// API / clamping (mirrors LadderTPTStage). Self-osc onset is R=4 for the
// HP cascade too (4-pole HP at cutoff = +180° phase, paired with the -1
// feedback sign closes the loop with the same rotation as the LP case);
// kMaxResonance must sit strictly below 4.0.
// --------------------------------------------------------------------------
TEST(LadderHighpassTPTStage, CutoffAndResonanceClamping_24dB) {
    LadderHPTPT24dBStage s;
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    s.setCutoffFrequency(-100.0);
    EXPECT_GE(s.getCutoffFrequency(), 20.0);

    s.setCutoffFrequency(1.0e9);
    EXPECT_LE(s.getCutoffFrequency(), 0.45 * kSampleRate + 1e-9);

    s.setResonance(-1.0);
    EXPECT_DOUBLE_EQ(0.0, s.getResonance());

    s.setResonance(100.0);
    EXPECT_GT(s.getResonance(), 0.0);
    EXPECT_LT(s.getResonance(), 4.0);
}

TEST(LadderHighpassTPTStage, CutoffAndResonanceClamping_12dB) {
    LadderHPTPT12dBStage s;
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    s.setCutoffFrequency(-100.0);
    EXPECT_GE(s.getCutoffFrequency(), 20.0);
    s.setCutoffFrequency(1.0e9);
    EXPECT_LE(s.getCutoffFrequency(), 0.45 * kSampleRate + 1e-9);

    s.setResonance(-1.0);
    EXPECT_DOUBLE_EQ(0.0, s.getResonance());
    s.setResonance(100.0);
    EXPECT_GT(s.getResonance(), 0.0);
    EXPECT_LT(s.getResonance(), 4.0);
}

// --------------------------------------------------------------------------
// Smoothing time matches the LP cascade (same kSmoothingTimeSec). Catches
// future regressions that would reintroduce zipper noise on the HPF.
// --------------------------------------------------------------------------
TEST(LadderHighpassTPTStage, SmoothingTakesMillisecondsToReachTarget_24dB) {
    LadderHPTPT24dBStage s;
    s.setCutoffFrequency(1000.0);
    s.prepareToPlay(kSampleRate, 64, 1);

    s.setCutoffFrequency(5000.0);

    juce::AudioBuffer<double> tenSamples(1, 10);
    tenSamples.clear();
    s.process(tenSamples);
    const double afterTen = s.getCurrentSmoothedCutoff();
    EXPECT_GT(afterTen, 1000.0);
    EXPECT_LT(afterTen, 4000.0)
        << "HPF smoothing reached target too fast — got " << afterTen << " Hz at sample 10";

    juce::AudioBuffer<double> fiveHundredMore(1, 500);
    fiveHundredMore.clear();
    s.process(fiveHundredMore);
    EXPECT_NEAR(5000.0, s.getCurrentSmoothedCutoff(), 1.0);
}

// --------------------------------------------------------------------------
// DC rejection — the defining property of a highpass. With any cutoff above
// 20 Hz the steady-state response to a DC step should attenuate by >40 dB
// regardless of resonance (passband makeup affects HF, not DC).
// --------------------------------------------------------------------------
namespace {
template <typename Stage>
double dcSteadyStateGain(double cutoffHz, double R, double inputAmp) {
    Stage s;
    s.setCutoffFrequency(cutoffHz);
    s.setResonance(R);
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    constexpr int kSettleSamples = 9600; // ~200 ms — well past 7/wc + smoothing
    juce::AudioBuffer<double> buf(1, kSettleSamples);
    for (int i = 0; i < kSettleSamples; ++i) {
        buf.getWritePointer(0)[i] = inputAmp;
    }
    s.process(buf);
    return std::abs(buf.getReadPointer(0)[kSettleSamples - 1]) / inputAmp;
}
} // namespace

TEST(LadderHighpassTPTStage, DCAttenuatedAtAllResonances_24dB) {
    for (double R : {0.0, 1.0, 2.0, 3.0, 3.5}) {
        const double gain = dcSteadyStateGain<LadderHPTPT24dBStage>(1000.0, R, 0.1);
        EXPECT_LT(gain, 0.01)
            << "HPF at cutoff 1 kHz, R=" << R << " passed DC at gain " << gain
            << " — expected <-40 dB attenuation";
    }
}

TEST(LadderHighpassTPTStage, DCAttenuatedAtAllResonances_12dB) {
    for (double R : {0.0, 1.0, 2.0, 3.0, 3.5}) {
        const double gain = dcSteadyStateGain<LadderHPTPT12dBStage>(1000.0, R, 0.1);
        EXPECT_LT(gain, 0.05)
            << "12dB HPF at cutoff 1 kHz, R=" << R << " passed DC at gain " << gain;
    }
}

// --------------------------------------------------------------------------
// Passband (HF) makeup: small-signal HF gain stays at unity across
// resonance (analogue of the LPF bass compensation). The (1+R) pre-amp
// restores HF level lost to the negative feedback at high R.
// Tolerance ±0.5 dB to stay tight on the linear regime.
// --------------------------------------------------------------------------
TEST(LadderHighpassTPTStage, PassbandMakeup_HfGainUnity_SmallSignal_24dB) {
    // Cutoff 500 Hz, probe at 8 kHz (~4 octaves above cutoff — well into
    // the flat passband for both N=2 and N=4 cascades).
    for (double R : {0.0, 0.5, 1.0, 2.0, 3.0, 3.5}) {
        const double gain = settledSineGain<LadderHPTPT24dBStage>(500.0, 8000.0, R, 0.05);
        EXPECT_NEAR(gain, 1.0, 0.06)
            << "HF passband gain at R=" << R << " is " << gain
            << " (linear theory predicts 1/(1+R)=" << 1.0 / (1.0 + R)
            << " without compensation)";
    }
}

TEST(LadderHighpassTPTStage, PassbandMakeup_HfGainUnity_SmallSignal_12dB) {
    for (double R : {0.0, 0.5, 1.0, 2.0, 3.0, 3.5}) {
        const double gain = settledSineGain<LadderHPTPT12dBStage>(500.0, 8000.0, R, 0.05);
        EXPECT_NEAR(gain, 1.0, 0.06)
            << "12dB HF passband gain at R=" << R << " is " << gain;
    }
}

// --------------------------------------------------------------------------
// Resonance peak: at cutoff, R=3 should bloom meaningfully louder than R=0.
// Without that the resonance knob is doing nothing audible. (Same conservative
// 2× / 1.5× thresholds as the LP test — true peak gain is much larger.)
// --------------------------------------------------------------------------
TEST(LadderHighpassTPTStage, ResonancePeakBlooms_24dB) {
    const double peakR0 = atCutoffSinePeak<LadderHPTPT24dBStage>(1000.0, 0.0, 0.1);
    const double peakR3 = atCutoffSinePeak<LadderHPTPT24dBStage>(1000.0, 3.0, 0.1);
    EXPECT_GT(peakR3, peakR0 * 2.0)
        << "HPF resonance peak at R=3.0 (" << peakR3
        << ") is not meaningfully louder than at R=0 (" << peakR0 << ")";
}

TEST(LadderHighpassTPTStage, ResonancePeakBlooms_12dB) {
    const double peakR0 = atCutoffSinePeak<LadderHPTPT12dBStage>(1000.0, 0.0, 0.1);
    const double peakR3 = atCutoffSinePeak<LadderHPTPT12dBStage>(1000.0, 3.0, 0.1);
    EXPECT_GT(peakR3, peakR0 * 1.5)
        << "12dB HPF resonance peak at R=3.0 (" << peakR3
        << ") not meaningfully louder than at R=0 (" << peakR0 << ")";
}

// --------------------------------------------------------------------------
// Ringing decay — at mid-R (2.5), the filter must not self-oscillate.
// Mirrors the LP test; failure would indicate the feedback sign / loop-gain
// math is off.
// --------------------------------------------------------------------------
TEST(LadderHighpassTPTStage, RingingDecaysAtMidResonance_24dB) {
    const double ratio = impulseDecayRatio<LadderHPTPT24dBStage>(2.5, 100);
    EXPECT_LT(ratio, 0.01)
        << "HPF self-oscillating at R=2.5 (tail/impulse energy = " << ratio << ")";
}

TEST(LadderHighpassTPTStage, RingingDecaysAtMidResonance_12dB) {
    const double ratio = impulseDecayRatio<LadderHPTPT12dBStage>(2.5, 100);
    EXPECT_LT(ratio, 0.01)
        << "12dB HPF self-oscillating at R=2.5 (ratio = " << ratio << ")";
}

TEST(LadderHighpassTPTStage, RingingDecaysAtMaxResonance_24dB) {
    LadderHPTPT24dBStage probe;
    probe.setResonance(100.0);
    const double rMax = probe.getResonance();
    const double ratio = impulseDecayRatio<LadderHPTPT24dBStage>(rMax, 200);
    EXPECT_LT(ratio, 0.1)
        << "HPF not decaying at max R=" << rMax << " (ratio=" << ratio
        << ") — knob top is at/above self-osc threshold";
}

// --------------------------------------------------------------------------
// Newton stability at near-max R under a sweeping sine. Same guard as the
// LP version — catches divergence, runaway, or too-tight tolerance regressions.
// --------------------------------------------------------------------------
TEST(LadderHighpassTPTStage, NewtonStableAtNearSelfOscillation_24dB) {
    LadderHPTPT24dBStage s;
    s.setCutoffFrequency(1000.0);
    s.setResonance(100.0); // clamps to kMaxResonance
    const double rUsed = s.getResonance();
    EXPECT_GT(rUsed, 3.0);
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    juce::AudioBuffer<double> buf(1, kBlockSize);
    constexpr int kNumBlocks = 50;
    for (int b = 0; b < kNumBlocks; ++b) {
        for (int i = 0; i < kBlockSize; ++i) {
            const int n = b * kBlockSize + i;
            const double t = n / kSampleRate;
            const double freq = 200.0 + 4000.0 * (n / static_cast<double>(kNumBlocks * kBlockSize));
            buf.getWritePointer(0)[i] = 0.5 * std::sin(2.0 * M_PI * freq * t);
        }
        s.process(buf);
        for (int i = 0; i < kBlockSize; ++i) {
            const double v = buf.getReadPointer(0)[i];
            ASSERT_TRUE(std::isfinite(v))
                << "HPF non-finite output at block " << b << " sample " << i
                << " (R=" << rUsed << ") — Newton may have diverged";
            ASSERT_LT(std::abs(v), 10.0)
                << "HPF runaway at block " << b << " sample " << i
                << " val=" << v << " (R=" << rUsed << ")";
        }
    }
}

// --------------------------------------------------------------------------
// Choke at max (10x) stacks on the (1+R)≈4.5× pre-amp, driving the pre-tanh
// level to ~45× at full resonance. Newton must stay finite/bounded; the
// gain-compensated output (×1/10) stays small. Mirrors the LP choke guard.
// --------------------------------------------------------------------------
TEST(LadderHighpassTPTStage, NewtonStableWithMaxChoke_24dB) {
    LadderHPTPT24dBStage s;
    s.setCutoffFrequency(1000.0);
    s.setResonance(100.0); // clamps to kMaxResonance
    s.setChoke(100.0);     // clamps to kMaxChoke (10x)
    EXPECT_DOUBLE_EQ(s.getChoke(), 10.0);
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    juce::AudioBuffer<double> buf(1, kBlockSize);
    constexpr int kNumBlocks = 50;
    for (int b = 0; b < kNumBlocks; ++b) {
        for (int i = 0; i < kBlockSize; ++i) {
            const int n = b * kBlockSize + i;
            const double t = n / kSampleRate;
            const double freq = 200.0 + 4000.0 * (n / static_cast<double>(kNumBlocks * kBlockSize));
            buf.getWritePointer(0)[i] = 0.5 * std::sin(2.0 * M_PI * freq * t);
        }
        s.process(buf);
        for (int i = 0; i < kBlockSize; ++i) {
            const double v = buf.getReadPointer(0)[i];
            ASSERT_TRUE(std::isfinite(v))
                << "HPF non-finite output at block " << b << " sample " << i << " with max choke";
            ASSERT_LT(std::abs(v), 10.0)
                << "HPF runaway at block " << b << " sample " << i << " val=" << v << " with max choke";
        }
    }
}

// --------------------------------------------------------------------------
// Stability with hot input at max R: pre-amp (1+R)≈4.5× drives tanh deep
// into saturation. Must remain bounded.
// --------------------------------------------------------------------------
TEST(LadderHighpassTPTStage, StableAtHotSignalMaxR_24dB) {
    LadderHPTPT24dBStage s;
    s.setCutoffFrequency(500.0);
    s.setResonance(3.5);
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    constexpr int kBlocks = 100;
    juce::AudioBuffer<double> buf(1, kBlockSize);
    for (int b = 0; b < kBlocks; ++b) {
        for (int i = 0; i < kBlockSize; ++i) {
            buf.getWritePointer(0)[i] = (((b * kBlockSize + i) & 1) != 0) ? 1.0 : -1.0;
        }
        s.process(buf);
        for (int i = 0; i < kBlockSize; ++i) {
            const double v = buf.getReadPointer(0)[i];
            ASSERT_TRUE(std::isfinite(v))
                << "HPF non-finite at block " << b << " sample " << i << ": " << v;
            ASSERT_LT(std::abs(v), 6.0)
                << "HPF runaway at block " << b << " sample " << i << ": " << v;
        }
    }
}

// --------------------------------------------------------------------------
// Denormals must flush — same concern as the LPF: TPT integrator states
// decay asymptotically and would otherwise enter the denormal range during
// long silence after an impulse.
// --------------------------------------------------------------------------
TEST(LadderHighpassTPTStage, NoDenormalOutputAfterLongSilence_24dB) {
    LadderHPTPT24dBStage s;
    s.setCutoffFrequency(5000.0);
    s.setResonance(0.0);
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    juce::AudioBuffer<double> buf(1, kBlockSize);
    buf.clear();
    buf.getWritePointer(0)[0] = 1.0;
    s.process(buf);

    processSilence(s, 100, buf);

    for (int i = 0; i < buf.getNumSamples(); ++i) {
        const double v = buf.getReadPointer(0)[i];
        const bool ok = (v == 0.0) || std::isnormal(v);
        EXPECT_TRUE(ok) << "Sample " << i << " denormal: " << v;
        if (!ok) {
            break;
        }
    }
}
