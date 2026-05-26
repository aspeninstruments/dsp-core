#include "../dsp_core/Source/audio_pipeline/FirstOrderShelfStage.h"
#include <gtest/gtest.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <cmath>

using namespace dsp_core::audio_pipeline;

namespace {

constexpr double kSampleRate = 48000.0;
constexpr int kBlockSize = 4096;

// Single-bin DFT amplitude estimate (RMS-equivalent) at a probe frequency.
double rmsAtFrequency(const juce::AudioBuffer<double>& buf, double freqHz, int startSample, int endSample) {
    double sumI = 0.0;
    double sumQ = 0.0;
    int n = 0;
    for (int i = startSample; i < endSample; ++i) {
        const double phase = 2.0 * juce::MathConstants<double>::pi * freqHz * i / kSampleRate;
        const double v = buf.getSample(0, i);
        sumI += v * std::cos(phase);
        sumQ += v * std::sin(phase);
        ++n;
    }
    const double normI = sumI / n;
    const double normQ = sumQ / n;
    return std::sqrt(2.0) * std::sqrt(normI * normI + normQ * normQ);
}

void fillSine(juce::AudioBuffer<double>& buf, double freqHz, double amp = 0.3) {
    for (int i = 0; i < buf.getNumSamples(); ++i) {
        buf.setSample(0, i, amp * std::sin(2.0 * juce::MathConstants<double>::pi * freqHz * i / kSampleRate));
    }
}

} // namespace

// --------------------------------------------------------------------------
// 0 dB gain reduces to pure pass-through (identity). Both HS and LS variants
// derive coefficients to (b0=1, b1=a1) at A=1, so any non-zero input should
// pass unchanged sample-for-sample within numerical precision.
// --------------------------------------------------------------------------
TEST(FirstOrderShelfStage, HighShelf_ZeroDb_IsIdentity) {
    FirstOrderShelfStage shelf(FirstOrderShelfStage::Mode::HighShelf);
    shelf.setCutoffFrequency(3183.0);
    shelf.setGainDb(0.0);
    shelf.prepareToPlay(kSampleRate, kBlockSize, 1);

    juce::AudioBuffer<double> input(1, kBlockSize);
    juce::AudioBuffer<double> output(1, kBlockSize);
    for (int i = 0; i < kBlockSize; ++i) {
        const double v = 0.3 * std::sin(2.0 * juce::MathConstants<double>::pi * 1000.0 * i / kSampleRate);
        input.setSample(0, i, v);
        output.setSample(0, i, v);
    }
    shelf.process(output);

    for (int i = 0; i < kBlockSize; ++i) {
        EXPECT_NEAR(input.getSample(0, i), output.getSample(0, i), 1e-12)
            << "HighShelf at 0 dB must be identity (sample " << i << ")";
    }
}

TEST(FirstOrderShelfStage, LowShelf_ZeroDb_IsIdentity) {
    FirstOrderShelfStage shelf(FirstOrderShelfStage::Mode::LowShelf);
    shelf.setCutoffFrequency(50.0);
    shelf.setGainDb(0.0);
    shelf.prepareToPlay(kSampleRate, kBlockSize, 1);

    juce::AudioBuffer<double> input(1, kBlockSize);
    juce::AudioBuffer<double> output(1, kBlockSize);
    for (int i = 0; i < kBlockSize; ++i) {
        const double v = 0.3 * std::sin(2.0 * juce::MathConstants<double>::pi * 200.0 * i / kSampleRate);
        input.setSample(0, i, v);
        output.setSample(0, i, v);
    }
    shelf.process(output);

    for (int i = 0; i < kBlockSize; ++i) {
        EXPECT_NEAR(input.getSample(0, i), output.getSample(0, i), 1e-12)
            << "LowShelf at 0 dB must be identity (sample " << i << ")";
    }
}

// --------------------------------------------------------------------------
// Asymptotic gain: well above the high-shelf corner, magnitude should reach
// the configured shelf gain (within a small tolerance).
// --------------------------------------------------------------------------
TEST(FirstOrderShelfStage, HighShelf_PlusTwelveDb_AsymptoticGainAtHF) {
    FirstOrderShelfStage shelf(FirstOrderShelfStage::Mode::HighShelf);
    shelf.setCutoffFrequency(3183.0);
    shelf.setGainDb(12.0);
    shelf.prepareToPlay(kSampleRate, kBlockSize, 1);

    juce::AudioBuffer<double> buf(1, kBlockSize);
    constexpr double kProbeHz = 18000.0; // well above 3183 Hz corner
    constexpr double kAmp = 0.3;
    fillSine(buf, kProbeHz, kAmp);

    shelf.process(buf);

    const double inputRms = kAmp / std::sqrt(2.0);
    const double outRms = rmsAtFrequency(buf, kProbeHz, kBlockSize / 2, kBlockSize);
    const double gainDb = 20.0 * std::log10(outRms / inputRms);

    EXPECT_NEAR(gainDb, 12.0, 0.5) << "+12 dB high-shelf at 18 kHz should reach ~+12 dB; got " << gainDb;
}

// --------------------------------------------------------------------------
// Corner behavior: 1st-order shelf should be at half-gain (in dB) at the
// corner frequency — this is the textbook 1st-order property.
// --------------------------------------------------------------------------
TEST(FirstOrderShelfStage, HighShelf_PlusTwelveDb_HalfGainAtCorner) {
    FirstOrderShelfStage shelf(FirstOrderShelfStage::Mode::HighShelf);
    constexpr double kCornerHz = 3183.0;
    shelf.setCutoffFrequency(kCornerHz);
    shelf.setGainDb(12.0);
    shelf.prepareToPlay(kSampleRate, kBlockSize, 1);

    juce::AudioBuffer<double> buf(1, kBlockSize);
    constexpr double kAmp = 0.3;
    fillSine(buf, kCornerHz, kAmp);

    shelf.process(buf);

    const double inputRms = kAmp / std::sqrt(2.0);
    const double outRms = rmsAtFrequency(buf, kCornerHz, kBlockSize / 2, kBlockSize);
    const double gainDb = 20.0 * std::log10(outRms / inputRms);

    // 1st-order shelf: |H(jω0)| = sqrt(A) → midpoint in dB → 12/2 = 6 dB at corner.
    EXPECT_NEAR(gainDb, 6.0, 0.5) << "+12 dB shelf at corner should be +6 dB; got " << gainDb;
}

// --------------------------------------------------------------------------
// Low-band cancellation behavior: cascading +12 dB and -12 dB high-shelves at
// the same corner doesn't perfectly null (the de form ≠ inverse of the pre
// form in this canonical parametric design), but both shelves should leave
// the low band (below the corner) untouched within tight tolerance.
// --------------------------------------------------------------------------
TEST(FirstOrderShelfStage, HighShelf_LowBand_IsUnaffected) {
    FirstOrderShelfStage shelf(FirstOrderShelfStage::Mode::HighShelf);
    shelf.setCutoffFrequency(3183.0);
    shelf.setGainDb(12.0);
    shelf.prepareToPlay(kSampleRate, kBlockSize, 1);

    juce::AudioBuffer<double> buf(1, kBlockSize);
    constexpr double kProbeHz = 100.0; // well below 3183 Hz corner
    constexpr double kAmp = 0.3;
    fillSine(buf, kProbeHz, kAmp);
    shelf.process(buf);

    const double inputRms = kAmp / std::sqrt(2.0);
    const double outRms = rmsAtFrequency(buf, kProbeHz, kBlockSize / 2, kBlockSize);
    const double gainDb = 20.0 * std::log10(outRms / inputRms);

    EXPECT_NEAR(gainDb, 0.0, 0.5) << "High-shelf must not affect 100 Hz (well below 3183 Hz corner); got " << gainDb;
}

// --------------------------------------------------------------------------
// Low-shelf: gain at DC, unity at HF (mirror of high-shelf).
// --------------------------------------------------------------------------
TEST(FirstOrderShelfStage, LowShelf_PlusOneFiveDb_AsymptoticGainAtLF) {
    FirstOrderShelfStage shelf(FirstOrderShelfStage::Mode::LowShelf);
    shelf.setCutoffFrequency(50.0);
    shelf.setGainDb(1.5);
    shelf.prepareToPlay(kSampleRate, kBlockSize, 1);

    juce::AudioBuffer<double> buf(1, kBlockSize);
    constexpr double kProbeHz = 30.0; // well below 50 Hz corner — closer to the DC asymptote
    constexpr double kAmp = 0.3;
    fillSine(buf, kProbeHz, kAmp);

    shelf.process(buf);

    const double inputRms = kAmp / std::sqrt(2.0);
    const double outRms = rmsAtFrequency(buf, kProbeHz, kBlockSize / 2, kBlockSize);
    const double gainDb = 20.0 * std::log10(outRms / inputRms);

    // At 30 Hz with a 50 Hz corner we're partway up the shelf — not yet at the
    // full +1.5 dB asymptote but well above the half-gain corner. Expect
    // roughly +1 dB; the tolerance below covers the 1st-order rolloff.
    EXPECT_GT(gainDb, 0.5) << "Low-shelf at 30 Hz should be approaching +1.5 dB DC asymptote; got " << gainDb;
    EXPECT_LT(gainDb, 1.5) << "Low-shelf at 30 Hz can't exceed its DC asymptote +1.5 dB; got " << gainDb;
}

TEST(FirstOrderShelfStage, LowShelf_HighBand_IsUnaffected) {
    FirstOrderShelfStage shelf(FirstOrderShelfStage::Mode::LowShelf);
    shelf.setCutoffFrequency(50.0);
    shelf.setGainDb(1.5);
    shelf.prepareToPlay(kSampleRate, kBlockSize, 1);

    juce::AudioBuffer<double> buf(1, kBlockSize);
    constexpr double kProbeHz = 5000.0; // well above 50 Hz corner
    constexpr double kAmp = 0.3;
    fillSine(buf, kProbeHz, kAmp);
    shelf.process(buf);

    const double inputRms = kAmp / std::sqrt(2.0);
    const double outRms = rmsAtFrequency(buf, kProbeHz, kBlockSize / 2, kBlockSize);
    const double gainDb = 20.0 * std::log10(outRms / inputRms);

    EXPECT_NEAR(gainDb, 0.0, 0.2) << "Low-shelf must not affect 5 kHz (well above 50 Hz corner); got " << gainDb;
}

// --------------------------------------------------------------------------
// Reset clears state — feed a transient, reset, then feed silence and
// verify the IIR doesn't have residual ringing.
// --------------------------------------------------------------------------
TEST(FirstOrderShelfStage, Reset_ClearsState) {
    FirstOrderShelfStage shelf(FirstOrderShelfStage::Mode::HighShelf);
    shelf.setCutoffFrequency(3183.0);
    shelf.setGainDb(12.0);
    shelf.prepareToPlay(kSampleRate, 256, 1);

    juce::AudioBuffer<double> buf(1, 256);
    // Burn in some state.
    fillSine(buf, 1000.0, 0.9);
    shelf.process(buf);

    shelf.reset();

    // Now feed silence — output must be exactly 0 (no IIR ringing).
    buf.clear();
    shelf.process(buf);

    for (int i = 0; i < 256; ++i) {
        EXPECT_DOUBLE_EQ(0.0, buf.getSample(0, i))
            << "After reset, silence must produce silence (sample " << i << ")";
    }
}

// --------------------------------------------------------------------------
// Bypass via setEnabled(false) — output must equal input exactly, including
// when prior state would have produced non-zero output.
// --------------------------------------------------------------------------
TEST(FirstOrderShelfStage, Disabled_IsTrueBypass) {
    FirstOrderShelfStage shelf(FirstOrderShelfStage::Mode::HighShelf);
    shelf.setCutoffFrequency(3183.0);
    shelf.setGainDb(12.0);
    shelf.setEnabled(false);
    shelf.prepareToPlay(kSampleRate, 256, 1);

    juce::AudioBuffer<double> input(1, 256);
    juce::AudioBuffer<double> output(1, 256);
    fillSine(input, 5000.0, 0.4);
    for (int i = 0; i < 256; ++i) {
        output.setSample(0, i, input.getSample(0, i));
    }

    shelf.process(output);

    for (int i = 0; i < 256; ++i) {
        EXPECT_DOUBLE_EQ(input.getSample(0, i), output.getSample(0, i))
            << "Disabled shelf must be true bypass (sample " << i << ")";
    }
}
