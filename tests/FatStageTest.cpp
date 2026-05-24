#include "../dsp_core/Source/audio_pipeline/FatStage.h"
#include <gtest/gtest.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <cmath>

using namespace dsp_core::audio_pipeline;

namespace {

constexpr double kSampleRate = 48000.0;
constexpr int kBlockSize = 512;

// Process `numSamples` of a continuous-phase sine through the stage in
// kBlockSize chunks. Returns the RMS of the LAST `measSamples` of output
// (after `numSamples - measSamples` samples of warm-up).
double processSineAndMeasureRms(FatStage& s, double freqHz, int warmupSamples,
                                int measSamples) {
    juce::AudioBuffer<double> buf(1, kBlockSize);
    int total = warmupSamples + measSamples;
    int produced = 0;
    double sumSq = 0.0;
    int measCount = 0;
    while (produced < total) {
        const int n = std::min(kBlockSize, total - produced);
        buf.setSize(1, n, false, false, true);
        for (int i = 0; i < n; ++i) {
            const int sampleIdx = produced + i;
            buf.setSample(0, i,
                          std::sin(2.0 * M_PI * freqHz * sampleIdx / kSampleRate));
        }
        s.process(buf);
        for (int i = 0; i < n; ++i) {
            const int sampleIdx = produced + i;
            if (sampleIdx >= warmupSamples) {
                const double v = buf.getSample(0, i);
                sumSq += v * v;
                ++measCount;
            }
        }
        produced += n;
    }
    return std::sqrt(sumSq / std::max(1, measCount));
}

double measureFreqRms(double fatPercent, double freqHz) {
    FatStage s;
    s.setFat(fatPercent);
    s.prepareToPlay(kSampleRate, kBlockSize, 1);
    // 9600 samples = exactly 10 cycles at 50 Hz @ 48 kHz (clean RMS window).
    return processSineAndMeasureRms(s, freqHz, /*warmup*/ 9600, /*meas*/ 9600);
}

} // namespace

// --------------------------------------------------------------------------
// Test 1 — Fat=0 must be a true bypass.
// The shelf at 0 dB is mathematically the identity transfer function; in
// double precision the recursive biquad introduces only ~1e-15 rounding per
// sample. RMS-of-difference against the unmodified input should sit far
// below 1e-10.
// --------------------------------------------------------------------------
TEST(FatStage, FatZero_IsTransparent) {
    FatStage s;
    s.setFat(0.0);
    s.prepareToPlay(kSampleRate, kBlockSize, 2);

    juce::AudioBuffer<double> input(2, kBlockSize);
    juce::AudioBuffer<double> output(2, kBlockSize);

    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < kBlockSize; ++i) {
            double v = 0.3 * std::sin(2.0 * M_PI * 1000.0 * i / kSampleRate);
            v += 0.1; // DC
            if (i == 100) {
                v += 1.0; // impulse
            }
            input.setSample(ch, i, v);
            output.setSample(ch, i, v);
        }
    }

    s.process(output);

    double sumSqDiff = 0.0;
    int total = 0;
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < kBlockSize; ++i) {
            const double d = output.getSample(ch, i) - input.getSample(ch, i);
            sumSqDiff += d * d;
            ++total;
        }
    }
    const double rmsDiff = std::sqrt(sumSqDiff / std::max(1, total));
    EXPECT_LT(rmsDiff, 1e-10)
        << "FatStage with Fat=0 must be transparent within float noise. RMS diff: "
        << rmsDiff;
}

// --------------------------------------------------------------------------
// Test 2 — Fat=100 boosts lows, leaves highs alone.
// Shelf is hardcoded at 200 Hz, Q=1/sqrt(2), peak gain +6 dB. A 50 Hz sine
// sits well inside the shelf and should pick up at least +5 dB (allowing
// margin under the +6 dB plateau due to the corner-region rolloff). A 5 kHz
// sine sits far above the shelf and should change by less than 0.5 dB.
// --------------------------------------------------------------------------
TEST(FatStage, FatHundred_BoostsLowFrequenciesAndPreservesHighs) {
    const double rms50_at0   = measureFreqRms(0.0,   50.0);
    const double rms50_at100 = measureFreqRms(100.0, 50.0);
    const double rms5k_at0   = measureFreqRms(0.0,   5000.0);
    const double rms5k_at100 = measureFreqRms(100.0, 5000.0);

    ASSERT_GT(rms50_at0, 0.0);
    ASSERT_GT(rms5k_at0, 0.0);

    const double boost50_dB   = 20.0 * std::log10(rms50_at100 / rms50_at0);
    const double change5k_dB  = 20.0 * std::log10(rms5k_at100 / rms5k_at0);

    EXPECT_GE(boost50_dB, 5.0)
        << "Fat=100 should boost 50 Hz by at least +5 dB. Got: " << boost50_dB << " dB";
    EXPECT_LT(std::abs(change5k_dB), 0.5)
        << "Fat=100 should leave 5 kHz nearly unchanged (<0.5 dB). Got: "
        << change5k_dB << " dB";
}

// --------------------------------------------------------------------------
// Test 3 — Fat percent clamps to [0, 100].
// --------------------------------------------------------------------------
TEST(FatStage, FatPercentClamping) {
    FatStage s;
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    s.setFat(-50.0);
    EXPECT_DOUBLE_EQ(0.0, s.getFat());

    s.setFat(500.0);
    EXPECT_DOUBLE_EQ(100.0, s.getFat());

    s.setFat(42.5);
    EXPECT_DOUBLE_EQ(42.5, s.getFat());
}

// --------------------------------------------------------------------------
// Test 4 — No denormals / NaNs / infs under prolonged moderate program
// material. Mirrors LadderTPTStageTest.NoDenormalOutput*: with
// ScopedNoDenormals in process(), every output sample must be either
// exactly 0.0 or std::isnormal.
// --------------------------------------------------------------------------
TEST(FatStage, NoNaNUnderLongSignal) {
    FatStage s;
    s.setFat(50.0);
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    juce::AudioBuffer<double> buf(1, kBlockSize);
    juce::Random rng(42);

    constexpr int kTotalSamples = 10000;
    int produced = 0;
    while (produced < kTotalSamples) {
        const int n = std::min(kBlockSize, kTotalSamples - produced);
        buf.setSize(1, n, false, false, true);
        for (int i = 0; i < n; ++i) {
            const int sampleIdx = produced + i;
            const double t = sampleIdx / kSampleRate;
            const double sig =
                0.3 * std::sin(2.0 * M_PI * 220.0 * t) + 0.1 * (rng.nextDouble() * 2.0 - 1.0);
            buf.setSample(0, i, sig);
        }
        s.process(buf);
        for (int i = 0; i < n; ++i) {
            const double v = buf.getSample(0, i);
            const bool ok = (v == 0.0) || std::isnormal(v);
            ASSERT_TRUE(ok) << "Sample " << (produced + i)
                            << " is denormal/NaN/inf: " << v;
        }
        produced += n;
    }
}
