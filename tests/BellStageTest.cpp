#include <gtest/gtest.h>
#include "../dsp_core/Source/audio_pipeline/BellStage.h"
#include <juce_audio_basics/juce_audio_basics.h>
#include <cmath>

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
double warmupSine(BellStage& filter, double frequency, double sampleRate,
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

// Measure the steady-state RMS-gain ratio (output / input) for a sine at
// `probeFreq` after warming the filter to steady state.
double measureGainAt(BellStage& filter, double probeFreq, double amplitude) {
    const double phase = warmupSine(filter, probeFreq, kSampleRate, amplitude, kNumChannels, 10000);

    juce::AudioBuffer<double> measurement(kNumChannels, kBlockSize);
    for (int ch = 0; ch < kNumChannels; ++ch) {
        for (int i = 0; i < kBlockSize; ++i) {
            const double p = phase + 2.0 * M_PI * probeFreq * i / kSampleRate;
            measurement.setSample(ch, i, amplitude * std::sin(p));
        }
    }
    const double rmsBefore = measureRMS(measurement);
    filter.process(measurement);
    const double rmsAfter = measureRMS(measurement);
    return rmsAfter / rmsBefore;
}

} // namespace

class BellStageTest : public ::testing::Test {
  protected:
    void SetUp() override {
        filter_ = std::make_unique<BellStage>();
        filter_->prepareToPlay(kSampleRate, kBlockSize, kNumChannels);
    }

    std::unique_ptr<BellStage> filter_;
};

// (1) Lifecycle: construct → prepare → process zero buffer doesn't crash and
// produces exact zeros.
TEST_F(BellStageTest, ConstructionAndPrepareDoNotCrash) {
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
TEST_F(BellStageTest, DisabledIsBitExactPassthrough) {
    filter_->setCutoffFrequency(1000.0);
    filter_->setGainDb(6.0);
    filter_->setQ(2.0);
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

// (3) Unity gain (0 dB) bell is approximately transparent regardless of Q.
TEST_F(BellStageTest, UnityGainIsTransparent) {
    filter_->setCutoffFrequency(1000.0);
    filter_->setGainDb(0.0);
    filter_->setQ(2.0);

    const double ratio = measureGainAt(*filter_, 1000.0, 1.0);
    EXPECT_NEAR(ratio, 1.0, 0.02)
        << "0 dB bell should be near-transparent. Got ratio " << ratio;
}

// (4) Boost at centre frequency: +12 dB at the centre should give ~+12 dB.
TEST_F(BellStageTest, BoostAtCentreFrequency) {
    constexpr double centre = 1000.0;
    constexpr double gainDb = 12.0;
    constexpr double amplitude = 0.2; // headroom for ~4x boost

    filter_->setCutoffFrequency(centre);
    filter_->setGainDb(gainDb);
    filter_->setQ(1.0);

    const double ratio = measureGainAt(*filter_, centre, amplitude);
    const double expected = std::pow(10.0, gainDb / 20.0); // ~3.98
    EXPECT_NEAR(ratio, expected, 0.2)
        << "1 kHz at +12 dB bell (centre 1 kHz, Q=1) should boost by ~12 dB. Got ratio " << ratio;
}

// (5) Cut symmetric to boost: -12 dB at centre should attenuate by ~1/3.98.
TEST_F(BellStageTest, CutAtCentreFrequency) {
    constexpr double centre = 1000.0;
    constexpr double gainDb = -12.0;
    constexpr double amplitude = 0.5;

    filter_->setCutoffFrequency(centre);
    filter_->setGainDb(gainDb);
    filter_->setQ(1.0);

    const double ratio = measureGainAt(*filter_, centre, amplitude);
    const double expected = std::pow(10.0, gainDb / 20.0); // ~0.251
    EXPECT_NEAR(ratio, expected, 0.05)
        << "1 kHz at -12 dB bell should cut by ~12 dB. Got ratio " << ratio;
}

// (6) Higher Q narrows bandwidth: at fixed +12 dB centre boost, an off-centre
// probe sees more boost with low Q than with high Q. This pins the Q control's
// audible effect — the whole reason Bell needs a Q knob in the first place.
TEST_F(BellStageTest, HigherQNarrowsBandwidth) {
    constexpr double centre = 1000.0;
    constexpr double gainDb = 12.0;
    constexpr double probeFreq = 2000.0; // one octave above centre
    constexpr double amplitude = 0.2;

    // Wide bell (low Q): probe at 2 kHz sees meaningful boost.
    BellStage wide;
    wide.prepareToPlay(kSampleRate, kBlockSize, kNumChannels);
    wide.setCutoffFrequency(centre);
    wide.setGainDb(gainDb);
    wide.setQ(0.5);
    const double wideRatio = measureGainAt(wide, probeFreq, amplitude);

    // Narrow bell (high Q): probe at 2 kHz sees much less boost.
    BellStage narrow;
    narrow.prepareToPlay(kSampleRate, kBlockSize, kNumChannels);
    narrow.setCutoffFrequency(centre);
    narrow.setGainDb(gainDb);
    narrow.setQ(5.0);
    const double narrowRatio = measureGainAt(narrow, probeFreq, amplitude);

    EXPECT_GT(wideRatio, narrowRatio)
        << "Lower Q should boost off-centre frequencies more than higher Q. "
        << "Wide(Q=0.5): " << wideRatio << ", Narrow(Q=5): " << narrowRatio;
    // Lower bound a meaningful separation — at 1 octave out with these Q values
    // the wide bell should give noticeably more boost than the narrow one.
    EXPECT_GT(wideRatio - narrowRatio, 0.5)
        << "Q effect too small to matter musically. Wide " << wideRatio
        << " vs narrow " << narrowRatio;
}

// (7) No effect far from centre: at +12 dB, Q=2, a probe two decades below
// the centre should be approximately unchanged.
TEST_F(BellStageTest, NoEffectFarFromCentre) {
    constexpr double centre = 5000.0;
    constexpr double gainDb = 12.0;
    constexpr double probeFreq = 50.0; // two decades below
    constexpr double amplitude = 0.5;

    filter_->setCutoffFrequency(centre);
    filter_->setGainDb(gainDb);
    filter_->setQ(2.0);

    const double ratio = measureGainAt(*filter_, probeFreq, amplitude);
    EXPECT_NEAR(ratio, 1.0, 0.1)
        << "50 Hz with bell at 5 kHz / Q=2 should be nearly unchanged. Got ratio " << ratio;
}

// (8) Centre-frequency setter clamps to [20, 20000].
TEST_F(BellStageTest, CutoffSetterClamps) {
    filter_->setCutoffFrequency(5000.0);
    EXPECT_DOUBLE_EQ(filter_->getCutoffFrequency(), 5000.0);

    filter_->setCutoffFrequency(20.0); // min
    EXPECT_DOUBLE_EQ(filter_->getCutoffFrequency(), 20.0);

    filter_->setCutoffFrequency(5.0); // below min
    EXPECT_DOUBLE_EQ(filter_->getCutoffFrequency(), 20.0);

    filter_->setCutoffFrequency(20000.0); // max
    EXPECT_DOUBLE_EQ(filter_->getCutoffFrequency(), 20000.0);

    filter_->setCutoffFrequency(30000.0); // above max
    EXPECT_DOUBLE_EQ(filter_->getCutoffFrequency(), 20000.0);
}

// (9) Gain setter clamps to [-24, +24].
TEST_F(BellStageTest, GainSetterClamps) {
    filter_->setGainDb(6.0);
    EXPECT_DOUBLE_EQ(filter_->getGainDb(), 6.0);

    filter_->setGainDb(-24.0);
    EXPECT_DOUBLE_EQ(filter_->getGainDb(), -24.0);

    filter_->setGainDb(-50.0);
    EXPECT_DOUBLE_EQ(filter_->getGainDb(), -24.0);

    filter_->setGainDb(24.0);
    EXPECT_DOUBLE_EQ(filter_->getGainDb(), 24.0);

    filter_->setGainDb(50.0);
    EXPECT_DOUBLE_EQ(filter_->getGainDb(), 24.0);
}

// (10) Q setter clamps to [0.1, 10].
TEST_F(BellStageTest, QSetterClamps) {
    filter_->setQ(1.0);
    EXPECT_DOUBLE_EQ(filter_->getQ(), 1.0);

    filter_->setQ(0.1);
    EXPECT_DOUBLE_EQ(filter_->getQ(), 0.1);

    filter_->setQ(0.001); // below min
    EXPECT_DOUBLE_EQ(filter_->getQ(), 0.1);

    filter_->setQ(10.0);
    EXPECT_DOUBLE_EQ(filter_->getQ(), 10.0);

    filter_->setQ(100.0); // above max
    EXPECT_DOUBLE_EQ(filter_->getQ(), 10.0);
}

// (11) Reset clears filter state. Same shape as the LowShelf/HighShelf reset
// test — after a fresh prepare + drive + reset, an impulse must produce the
// same tail peak as a freshly prepared filter.
TEST_F(BellStageTest, ResetClearsState) {
    filter_->setCutoffFrequency(1000.0);
    filter_->setGainDb(12.0);
    filter_->setQ(2.0);

    auto impulseTail = [&](BellStage& f) {
        juce::AudioBuffer<double> buf(1, 1024);
        buf.clear();
        buf.setSample(0, 0, 1.0);
        f.process(buf);
        return measurePeak(buf);
    };

    BellStage fresh;
    fresh.prepareToPlay(kSampleRate, kBlockSize, 1);
    fresh.setCutoffFrequency(1000.0);
    fresh.setGainDb(12.0);
    fresh.setQ(2.0);
    const double freshPeak = impulseTail(fresh);

    filter_.reset();
    filter_ = std::make_unique<BellStage>();
    filter_->prepareToPlay(kSampleRate, kBlockSize, 1);
    filter_->setCutoffFrequency(1000.0);
    filter_->setGainDb(12.0);
    filter_->setQ(2.0);
    warmupSine(*filter_, 1000.0, kSampleRate, 0.5, 1, 5000);
    filter_->reset();
    const double afterResetPeak = impulseTail(*filter_);

    EXPECT_NEAR(afterResetPeak, freshPeak, 1e-10)
        << "Reset must restore initial state. Fresh peak: " << freshPeak
        << ", after-reset peak: " << afterResetPeak;
}

// (12) Multi-channel: identical input on every channel produces identical
// output on every channel (per-channel filter state vectors stay in sync).
TEST_F(BellStageTest, MultiChannelIdenticalForIdenticalInput) {
    filter_->setCutoffFrequency(1000.0);
    filter_->setGainDb(6.0);
    filter_->setQ(1.0);

    const double phase = warmupSine(*filter_, 1000.0, kSampleRate, 0.5, kNumChannels, 10000);

    juce::AudioBuffer<double> buf(kNumChannels, kBlockSize);
    for (int ch = 0; ch < kNumChannels; ++ch) {
        for (int i = 0; i < kBlockSize; ++i) {
            const double p = phase + 2.0 * M_PI * 1000.0 * i / kSampleRate;
            buf.setSample(ch, i, 0.5 * std::sin(p));
        }
    }
    filter_->process(buf);

    for (int i = 0; i < kBlockSize; ++i) {
        EXPECT_DOUBLE_EQ(buf.getSample(0, i), buf.getSample(1, i))
            << "Channels must process identically for identical input (i=" << i << ")";
    }
}

// (13) Stability under centre-frequency sweep with full-scale white-noise input
// and +12 dB boost. High-Q surgical bell tests worst-case coefficient changes.
TEST_F(BellStageTest, StabilityUnderSweep) {
    filter_->setGainDb(12.0);
    filter_->setQ(5.0); // narrow, more potential for instability

    juce::Random rng(0xBE111234u);
    const int totalSamples = 100000;
    const double centreStart = 20.0;
    const double centreEnd = 19000.0;
    const int numBlocks = totalSamples / kBlockSize;

    juce::AudioBuffer<double> buf(1, kBlockSize);

    for (int block = 0; block < numBlocks; ++block) {
        const double t = static_cast<double>(block) / std::max(1, numBlocks - 1);
        const double centre = centreStart + t * (centreEnd - centreStart);
        filter_->setCutoffFrequency(centre);

        for (int i = 0; i < kBlockSize; ++i) {
            buf.setSample(0, i, rng.nextDouble() * 2.0 - 1.0);
        }
        filter_->process(buf);

        for (int i = 0; i < kBlockSize; ++i) {
            const double s = buf.getSample(0, i);
            ASSERT_TRUE(std::isfinite(s))
                << "Non-finite sample at block " << block << ", i=" << i << ", centre=" << centre;
            ASSERT_LT(std::abs(s), 8.0)
                << "Output magnitude blew up (>|8|) at block " << block << ", centre=" << centre
                << ", sample=" << s;
        }
    }
}
