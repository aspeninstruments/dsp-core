#include <gtest/gtest.h>
#include "../dsp_core/Source/audio_pipeline/LowShelfStage.h"
#include <juce_audio_basics/juce_audio_basics.h>
#include <array>
#include <cmath>
#include <vector>

using namespace dsp_core::audio_pipeline;

namespace {

constexpr double kSampleRate = 44100.0;
constexpr int kBlockSize = 512;
constexpr int kNumChannels = 2;

void fillWithSine(juce::AudioBuffer<double>& buffer, double frequency, double sampleRate,
                  double amplitude, double phaseStart) {
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
        for (int i = 0; i < buffer.getNumSamples(); ++i) {
            const double phase = phaseStart + 2.0 * M_PI * frequency * i / sampleRate;
            buffer.setSample(ch, i, amplitude * std::sin(phase));
        }
    }
}

double measureRMS(const juce::AudioBuffer<double>& buffer) {
    double sumSq = 0.0;
    int total = 0;
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
        for (int i = 0; i < buffer.getNumSamples(); ++i) {
            const double s = buffer.getSample(ch, i);
            sumSq += s * s;
            ++total;
        }
    }
    return std::sqrt(sumSq / std::max(1, total));
}

double measurePeak(const juce::AudioBuffer<double>& buffer) {
    double peak = 0.0;
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
        for (int i = 0; i < buffer.getNumSamples(); ++i) {
            peak = std::max(peak, std::abs(buffer.getSample(ch, i)));
        }
    }
    return peak;
}

// Run a continuous-phase sine through the filter for `totalSamples` to drive
// it to steady state, then return the phase to continue the signal into the
// measurement block (avoids phase-discontinuity transients on the boundary).
double warmupSine(LowShelfStage& filter, double frequency, double sampleRate,
                  double amplitude, int numChannels, int totalSamples) {
    juce::AudioBuffer<double> buf(numChannels, kBlockSize);
    double phase = 0.0;
    int produced = 0;
    while (produced < totalSamples) {
        const int thisBlock = std::min(kBlockSize, totalSamples - produced);
        buf.setSize(numChannels, thisBlock, false, false, true);
        for (int ch = 0; ch < numChannels; ++ch) {
            for (int i = 0; i < thisBlock; ++i) {
                const double p = phase + 2.0 * M_PI * frequency * i / sampleRate;
                buf.setSample(ch, i, amplitude * std::sin(p));
            }
        }
        filter.process(buf);
        phase += 2.0 * M_PI * frequency * thisBlock / sampleRate;
        produced += thisBlock;
    }
    return phase;
}

} // namespace

class LowShelfStageTest : public ::testing::Test {
  protected:
    void SetUp() override {
        filter_ = std::make_unique<LowShelfStage>();
        filter_->prepareToPlay(kSampleRate, kBlockSize, kNumChannels);
    }

    std::unique_ptr<LowShelfStage> filter_;
};

// (1) Lifecycle: construct → prepare → process zero buffer doesn't crash and
// produces exact zeros.
TEST_F(LowShelfStageTest, ConstructionAndPrepareDoNotCrash) {
    juce::AudioBuffer<double> buffer(kNumChannels, kBlockSize);
    buffer.clear();
    filter_->process(buffer);

    for (int ch = 0; ch < kNumChannels; ++ch) {
        for (int i = 0; i < kBlockSize; ++i) {
            EXPECT_DOUBLE_EQ(buffer.getSample(ch, i), 0.0);
        }
    }
}

// (2) Disabled = bit-exact passthrough. Filter must early-return when disabled.
TEST_F(LowShelfStageTest, DisabledIsBitExactPassthrough) {
    filter_->setCutoffFrequency(500.0);
    filter_->setGainDb(6.0);
    filter_->setEnabled(false);
    EXPECT_FALSE(filter_->isEnabled());

    juce::AudioBuffer<double> buffer(kNumChannels, kBlockSize);
    fillWithSine(buffer, 1000.0, kSampleRate, 0.5, 0.0);

    juce::AudioBuffer<double> original(kNumChannels, kBlockSize);
    original.copyFrom(0, 0, buffer, 0, 0, kBlockSize);
    original.copyFrom(1, 0, buffer, 1, 0, kBlockSize);

    filter_->process(buffer);

    for (int ch = 0; ch < kNumChannels; ++ch) {
        for (int i = 0; i < kBlockSize; ++i) {
            EXPECT_DOUBLE_EQ(buffer.getSample(ch, i), original.getSample(ch, i))
                << "Disabled filter must bypass processing (ch=" << ch << ", i=" << i << ")";
        }
    }

    filter_->setEnabled(true);
    EXPECT_TRUE(filter_->isEnabled());
}

// (3) Unity gain (0 dB) shelf is approximately transparent. Allow a small
// tolerance because RBJ at exactly gain=1.0 still has finite-precision
// coefficient noise (the formula goes through several sqrt/trig steps).
TEST_F(LowShelfStageTest, UnityGainIsTransparent) {
    filter_->setCutoffFrequency(500.0);
    filter_->setGainDb(0.0);

    constexpr double frequency = 1000.0;
    constexpr double amplitude = 1.0;

    const double phase = warmupSine(*filter_, frequency, kSampleRate, amplitude, kNumChannels, 10000);

    juce::AudioBuffer<double> measurement(kNumChannels, kBlockSize);
    for (int ch = 0; ch < kNumChannels; ++ch) {
        for (int i = 0; i < kBlockSize; ++i) {
            const double p = phase + 2.0 * M_PI * frequency * i / kSampleRate;
            measurement.setSample(ch, i, amplitude * std::sin(p));
        }
    }
    const double rmsBefore = measureRMS(measurement);
    filter_->process(measurement);
    const double rmsAfter = measureRMS(measurement);

    const double ratio = rmsAfter / rmsBefore;
    EXPECT_NEAR(ratio, 1.0, 0.02)
        << "0 dB shelf should be near-transparent. Got ratio " << ratio;
}

// (4) Boost below cutoff: at +6 dB, a sine well below the shelf corner should
// see ~+6 dB gain (linear ratio ~1.995).
TEST_F(LowShelfStageTest, BoostBelowCutoff) {
    constexpr double cutoff = 1000.0;
    constexpr double gainDb = 6.0;
    constexpr double frequency = 100.0; // well below cutoff
    constexpr double amplitude = 0.3;   // headroom for the boost

    filter_->setCutoffFrequency(cutoff);
    filter_->setGainDb(gainDb);

    const double phase = warmupSine(*filter_, frequency, kSampleRate, amplitude, kNumChannels, 20000);

    juce::AudioBuffer<double> measurement(kNumChannels, kBlockSize);
    for (int ch = 0; ch < kNumChannels; ++ch) {
        for (int i = 0; i < kBlockSize; ++i) {
            const double p = phase + 2.0 * M_PI * frequency * i / kSampleRate;
            measurement.setSample(ch, i, amplitude * std::sin(p));
        }
    }
    const double rmsBefore = measureRMS(measurement);
    filter_->process(measurement);
    const double rmsAfter = measureRMS(measurement);

    const double ratio = rmsAfter / rmsBefore;
    const double expected = std::pow(10.0, gainDb / 20.0); // ~1.995
    EXPECT_NEAR(ratio, expected, 0.1)
        << "100 Hz at +6 dB low-shelf (1 kHz corner) should boost by ~6 dB. Got ratio " << ratio;
}

// (5) No effect above cutoff: at +12 dB low-shelf, a sine well ABOVE the
// corner should be approximately unchanged.
TEST_F(LowShelfStageTest, NoEffectAboveCutoff) {
    constexpr double cutoff = 200.0;
    constexpr double gainDb = 12.0;
    constexpr double frequency = 4000.0; // far above cutoff
    constexpr double amplitude = 0.5;

    filter_->setCutoffFrequency(cutoff);
    filter_->setGainDb(gainDb);

    const double phase = warmupSine(*filter_, frequency, kSampleRate, amplitude, kNumChannels, 10000);

    juce::AudioBuffer<double> measurement(kNumChannels, kBlockSize);
    for (int ch = 0; ch < kNumChannels; ++ch) {
        for (int i = 0; i < kBlockSize; ++i) {
            const double p = phase + 2.0 * M_PI * frequency * i / kSampleRate;
            measurement.setSample(ch, i, amplitude * std::sin(p));
        }
    }
    const double rmsBefore = measureRMS(measurement);
    filter_->process(measurement);
    const double rmsAfter = measureRMS(measurement);

    const double ratio = rmsAfter / rmsBefore;
    EXPECT_NEAR(ratio, 1.0, 0.05)
        << "4 kHz with low-shelf at 200 Hz / +12 dB should be nearly unchanged. Got ratio " << ratio;
}

// (6) Cut symmetric to boost: -6 dB should attenuate by 1/1.995 ~= 0.501.
TEST_F(LowShelfStageTest, CutSymmetricToBoost) {
    constexpr double cutoff = 1000.0;
    constexpr double gainDb = -6.0;
    constexpr double frequency = 100.0; // well below cutoff
    constexpr double amplitude = 0.5;

    filter_->setCutoffFrequency(cutoff);
    filter_->setGainDb(gainDb);

    const double phase = warmupSine(*filter_, frequency, kSampleRate, amplitude, kNumChannels, 20000);

    juce::AudioBuffer<double> measurement(kNumChannels, kBlockSize);
    for (int ch = 0; ch < kNumChannels; ++ch) {
        for (int i = 0; i < kBlockSize; ++i) {
            const double p = phase + 2.0 * M_PI * frequency * i / kSampleRate;
            measurement.setSample(ch, i, amplitude * std::sin(p));
        }
    }
    const double rmsBefore = measureRMS(measurement);
    filter_->process(measurement);
    const double rmsAfter = measureRMS(measurement);

    const double ratio = rmsAfter / rmsBefore;
    const double expected = std::pow(10.0, gainDb / 20.0); // ~0.501
    EXPECT_NEAR(ratio, expected, 0.05)
        << "100 Hz at -6 dB low-shelf (1 kHz corner) should cut by ~6 dB. Got ratio " << ratio;
}

// (7) Cutoff setter clamps to [20, 20000].
TEST_F(LowShelfStageTest, CutoffSetterClamps) {
    filter_->setCutoffFrequency(1000.0);
    EXPECT_DOUBLE_EQ(filter_->getCutoffFrequency(), 1000.0);

    filter_->setCutoffFrequency(20.0); // min
    EXPECT_DOUBLE_EQ(filter_->getCutoffFrequency(), 20.0);

    filter_->setCutoffFrequency(5.0); // below min
    EXPECT_DOUBLE_EQ(filter_->getCutoffFrequency(), 20.0);

    filter_->setCutoffFrequency(20000.0); // max
    EXPECT_DOUBLE_EQ(filter_->getCutoffFrequency(), 20000.0);

    filter_->setCutoffFrequency(30000.0); // above max
    EXPECT_DOUBLE_EQ(filter_->getCutoffFrequency(), 20000.0);
}

// (8) Gain setter clamps to [-24, +24].
TEST_F(LowShelfStageTest, GainSetterClamps) {
    filter_->setGainDb(6.0);
    EXPECT_DOUBLE_EQ(filter_->getGainDb(), 6.0);

    filter_->setGainDb(-24.0); // min
    EXPECT_DOUBLE_EQ(filter_->getGainDb(), -24.0);

    filter_->setGainDb(-50.0); // below min
    EXPECT_DOUBLE_EQ(filter_->getGainDb(), -24.0);

    filter_->setGainDb(24.0); // max
    EXPECT_DOUBLE_EQ(filter_->getGainDb(), 24.0);

    filter_->setGainDb(50.0); // above max
    EXPECT_DOUBLE_EQ(filter_->getGainDb(), 24.0);
}

// (9) Reset clears filter state. With heavy boost, an impulse leaves a tail.
// After driving with noise then resetting, an impulse must produce the same
// response as a freshly prepared filter.
TEST_F(LowShelfStageTest, ResetClearsState) {
    filter_->setCutoffFrequency(500.0);
    filter_->setGainDb(12.0);

    auto impulseTail = [&](LowShelfStage& f) {
        juce::AudioBuffer<double> buf(1, 1024);
        buf.clear();
        buf.setSample(0, 0, 1.0);
        f.process(buf);
        return measurePeak(buf);
    };

    // Tail from a freshly prepared filter.
    LowShelfStage fresh;
    fresh.prepareToPlay(kSampleRate, kBlockSize, 1);
    fresh.setCutoffFrequency(500.0);
    fresh.setGainDb(12.0);
    // reset() snaps the parameter smoothers to their targets, so the impulse runs
    // with converged coefficients (no first-block ramp) — the apples-to-apples
    // baseline for the reset-after-warmup case below.
    fresh.reset();
    const double freshPeak = impulseTail(fresh);

    // Run our filter, drive it with arbitrary signal, then reset and remeasure.
    filter_.reset();
    filter_ = std::make_unique<LowShelfStage>();
    filter_->prepareToPlay(kSampleRate, kBlockSize, 1);
    filter_->setCutoffFrequency(500.0);
    filter_->setGainDb(12.0);
    warmupSine(*filter_, 500.0, kSampleRate, 0.5, 1, 5000);
    filter_->reset();
    const double afterResetPeak = impulseTail(*filter_);

    EXPECT_NEAR(afterResetPeak, freshPeak, 1e-10)
        << "Reset must restore initial state. Fresh peak: " << freshPeak
        << ", after-reset peak: " << afterResetPeak;
}

// (10) Multi-channel: identical input on every channel produces identical
// output on every channel (per-channel filter state vectors stay in sync).
TEST_F(LowShelfStageTest, MultiChannelIdenticalForIdenticalInput) {
    filter_->setCutoffFrequency(1000.0);
    filter_->setGainDb(6.0);

    // Warmup with identical signal on both channels.
    const double phase = warmupSine(*filter_, 500.0, kSampleRate, 0.5, kNumChannels, 10000);

    juce::AudioBuffer<double> buf(kNumChannels, kBlockSize);
    for (int ch = 0; ch < kNumChannels; ++ch) {
        for (int i = 0; i < kBlockSize; ++i) {
            const double p = phase + 2.0 * M_PI * 500.0 * i / kSampleRate;
            buf.setSample(ch, i, 0.5 * std::sin(p));
        }
    }
    filter_->process(buf);

    for (int i = 0; i < kBlockSize; ++i) {
        EXPECT_DOUBLE_EQ(buf.getSample(0, i), buf.getSample(1, i))
            << "Channels must process identically for identical input (i=" << i << ")";
    }
}

// (11) Stability under cutoff sweep with full-scale white-noise input and
// +12 dB boost. Sweep cutoff across the full legal range; output must stay
// finite and within a sane magnitude bound.
TEST_F(LowShelfStageTest, StabilityUnderSweep) {
    filter_->setGainDb(12.0);

    juce::Random rng(0xBD512AB2u);
    const int totalSamples = 100000;
    const double cutoffStart = 20.0;
    const double cutoffEnd = 19000.0;
    const int numBlocks = totalSamples / kBlockSize;

    juce::AudioBuffer<double> buf(1, kBlockSize);

    for (int block = 0; block < numBlocks; ++block) {
        const double t = static_cast<double>(block) / std::max(1, numBlocks - 1);
        const double cutoff = cutoffStart + t * (cutoffEnd - cutoffStart);
        filter_->setCutoffFrequency(cutoff);

        for (int i = 0; i < kBlockSize; ++i) {
            buf.setSample(0, i, rng.nextDouble() * 2.0 - 1.0);
        }
        filter_->process(buf);

        for (int i = 0; i < kBlockSize; ++i) {
            const double s = buf.getSample(0, i);
            ASSERT_TRUE(std::isfinite(s))
                << "Non-finite sample at block " << block << ", i=" << i << ", cutoff=" << cutoff;
            // +12 dB boost on full-scale noise can hit ~4x in the boosted band.
            // Bound loosely at 8.0 to catch only real instability.
            ASSERT_LT(std::abs(s), 8.0)
                << "Output magnitude blew up (>|8|) at block " << block << ", cutoff=" << cutoff
                << ", sample=" << s;
        }
    }
}

// The in-place writeCoefficients() RBJ math must match juce::dsp's makeLowShelf
// bit-for-bit (Butterworth Q). Params set BEFORE prepareToPlay start the
// smoothers converged (no ramp), so the stage's steady-state output equals a
// reference JUCE low-shelf biquad sample-for-sample.
TEST_F(LowShelfStageTest, SteadyStateMatchesJuceReference) {
    constexpr double kButterworthQ = 0.7071067811865476; // 1 / sqrt(2)
    struct Case {
        double f, g;
    };
    const std::array<Case, 6> cases = {{
        {500.0, 12.0},
        {100.0, -12.0},
        {2000.0, 6.0},
        {8000.0, 24.0},
        {50.0, -24.0},
        {1000.0, 0.0},
    }};

    for (const auto& c : cases) {
        LowShelfStage stage;
        stage.setCutoffFrequency(c.f);
        stage.setGainDb(c.g);
        stage.prepareToPlay(kSampleRate, kBlockSize, 1); // smoothers start converged

        juce::dsp::IIR::Filter<double> ref;
        ref.coefficients = juce::dsp::IIR::Coefficients<double>::makeLowShelf(
            kSampleRate, c.f, kButterworthQ, juce::Decibels::decibelsToGain(c.g));
        ref.reset();

        constexpr int kN = 2048;
        juce::Random rng(0x5E11u);
        juce::AudioBuffer<double> buf(1, kN);
        std::vector<double> refOut(static_cast<size_t>(kN));
        for (int i = 0; i < kN; ++i) {
            const double x = rng.nextDouble() * 2.0 - 1.0;
            buf.setSample(0, i, x);
            refOut[static_cast<size_t>(i)] = ref.processSample(x);
        }

        stage.process(buf);

        for (int i = 0; i < kN; ++i) {
            ASSERT_NEAR(buf.getSample(0, i), refOut[static_cast<size_t>(i)], 1e-7)
                << "Coefficient mismatch vs JUCE at sample " << i << " (f=" << c.f << ", g=" << c.g
                << ")";
        }
    }
}

// A gain step ramps in over the smoothing window instead of switching instantly
// — the regression guard for the click/pop the smoothing fixes. Probe sits in
// the boosted band (well below the corner) and high enough for a stable
// short-window RMS.
TEST_F(LowShelfStageTest, GainStepIsSmoothedNotInstant) {
    constexpr double cutoff = 6000.0;
    constexpr double probe = 1500.0; // ~2 octaves below corner: near full shelf gain
    constexpr double amp = 0.25;

    LowShelfStage stage;
    stage.setCutoffFrequency(cutoff);
    stage.setGainDb(0.0);
    stage.prepareToPlay(kSampleRate, kBlockSize, 1);

    const double phase = warmupSine(stage, probe, kSampleRate, amp, 1, 8000);

    stage.setGainDb(24.0);

    constexpr int kN = 2048;
    juce::AudioBuffer<double> buf(1, kN);
    for (int i = 0; i < kN; ++i) {
        const double p = phase + 2.0 * M_PI * probe * i / kSampleRate;
        buf.setSample(0, i, amp * std::sin(p));
    }
    stage.process(buf);

    auto rms = [&](int start, int len) {
        double sumSq = 0.0;
        for (int i = start; i < start + len; ++i) {
            const double v = buf.getSample(0, i);
            sumSq += v * v;
        }
        return std::sqrt(sumSq / len);
    };

    const double earlyRms = rms(0, 32);
    const double lateRms = rms(kN - 512, 512);
    const double transparentRms = amp / std::sqrt(2.0);

    EXPECT_LT(earlyRms, 0.5 * lateRms)
        << "Gain step not smoothed (instant swap?): early RMS " << earlyRms << " vs late RMS "
        << lateRms;
    EXPECT_GT(lateRms, 2.0 * transparentRms)
        << "Late window should reflect the applied boost. lateRms=" << lateRms;
}
