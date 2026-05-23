#include <gtest/gtest.h>
#include "../dsp_core/Source/audio_pipeline/LowpassStage.h"
#include <juce_audio_basics/juce_audio_basics.h>
#include <cmath>

using namespace dsp_core::audio_pipeline;

namespace {

constexpr double kSampleRate = 44100.0;
constexpr int kBlockSize = 512;
constexpr int kNumChannels = 2;
constexpr double kButterworthQ = 0.7071067811865476; // 1 / sqrt(2)

void fillWithDC(juce::AudioBuffer<double>& buffer, double dcValue) {
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
        for (int i = 0; i < buffer.getNumSamples(); ++i) {
            buffer.setSample(ch, i, dcValue);
        }
    }
}

void fillWithSine(juce::AudioBuffer<double>& buffer, double frequency, double sampleRate,
                  double amplitude, double phaseStart) {
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
        for (int i = 0; i < buffer.getNumSamples(); ++i) {
            const double phase = phaseStart + 2.0 * M_PI * frequency * i / sampleRate;
            buffer.setSample(ch, i, amplitude * std::sin(phase));
        }
    }
}

double measureMean(const juce::AudioBuffer<double>& buffer) {
    double sum = 0.0;
    int total = 0;
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
        for (int i = 0; i < buffer.getNumSamples(); ++i) {
            sum += buffer.getSample(ch, i);
            ++total;
        }
    }
    return sum / std::max(1, total);
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
double warmupSine(LowpassStage& filter, double frequency, double sampleRate,
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

class LowpassStageTest : public ::testing::Test {
  protected:
    void SetUp() override {
        filter_ = std::make_unique<LowpassStage>();
        filter_->prepareToPlay(kSampleRate, kBlockSize, kNumChannels);
    }

    std::unique_ptr<LowpassStage> filter_;
};

// (a) Lifecycle: construct → prepare → reset → process zero buffer doesn't
// crash and produces exact zeros.
TEST_F(LowpassStageTest, ConstructionAndPrepareDoNotCrash) {
    filter_->reset();

    juce::AudioBuffer<double> buffer(kNumChannels, kBlockSize);
    buffer.clear();
    filter_->process(buffer);

    for (int ch = 0; ch < kNumChannels; ++ch) {
        for (int i = 0; i < kBlockSize; ++i) {
            EXPECT_DOUBLE_EQ(buffer.getSample(ch, i), 0.0);
        }
    }
}

// (b) Lowpass passes DC. Cutoff at 20kHz with DC=0.5 should converge to ~0.5.
// This is the key polarity test — proves the topology is LP, not HP.
TEST_F(LowpassStageTest, PassesDCThrough) {
    filter_->setCutoffFrequency(20000.0);
    filter_->setResonance(kButterworthQ);

    juce::AudioBuffer<double> buffer(kNumChannels, kBlockSize);

    // Steady-state on DC for a TPT lowpass is reached in a handful of samples.
    // Warmup ~5 blocks at 20 kHz cutoff is plenty.
    for (int block = 0; block < 5; ++block) {
        fillWithDC(buffer, 0.5);
        filter_->process(buffer);
    }

    fillWithDC(buffer, 0.5);
    filter_->process(buffer);

    EXPECT_NEAR(measureMean(buffer), 0.5, 0.01)
        << "Lowpass at 20 kHz must pass DC. If mean is near 0.0, the topology is wired as highpass.";
}

// (c) Disabled = bit-exact passthrough. Filter must early-return when disabled.
TEST_F(LowpassStageTest, DisabledIsBitExactPassthrough) {
    filter_->setCutoffFrequency(500.0);
    filter_->setResonance(2.0);
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

// (d) Sub-cutoff sine is preserved. 200 Hz signal at 2 kHz cutoff (one decade
// below) should pass through with > 95% amplitude under Butterworth Q.
// Butterworth math: |H(f/fc=0.1)| = 1 / sqrt(1 + 0.0001) ~= 0.99995.
TEST_F(LowpassStageTest, SubCutoffSinePreserved_Butterworth) {
    filter_->setCutoffFrequency(2000.0);
    filter_->setResonance(kButterworthQ);

    constexpr double frequency = 200.0;
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

    EXPECT_GT(rmsAfter / rmsBefore, 0.95)
        << "200 Hz sine should pass through 2 kHz LPF nearly unchanged (ratio: "
        << (rmsAfter / rmsBefore) << ")";
}

// (e) -3 dB at cutoff with Butterworth Q. The defining property of a 2nd-order
// Butterworth lowpass at fc: |H(fc)| = 1/sqrt(2) (-3.01 dB). RMS of a unit-
// amplitude sine is 1/sqrt(2), so filtered RMS ~= 1/sqrt(2) * 1/sqrt(2) = 0.5.
TEST_F(LowpassStageTest, MinusThreeDbAtCutoff_Butterworth) {
    constexpr double cutoff = 1000.0;
    filter_->setCutoffFrequency(cutoff);
    filter_->setResonance(kButterworthQ);

    constexpr double amplitude = 1.0;
    const double phase = warmupSine(*filter_, cutoff, kSampleRate, amplitude, kNumChannels, 20000);

    // ~100 periods at 1 kHz / 44.1 kHz to minimise fractional-period RMS error.
    const int measureSamples = 4410;
    juce::AudioBuffer<double> measurement(kNumChannels, measureSamples);
    for (int ch = 0; ch < kNumChannels; ++ch) {
        for (int i = 0; i < measureSamples; ++i) {
            const double p = phase + 2.0 * M_PI * cutoff * i / kSampleRate;
            measurement.setSample(ch, i, amplitude * std::sin(p));
        }
    }

    filter_->process(measurement);
    const double rms = measureRMS(measurement);

    // Expected: 1/sqrt(2) * 10^(-3.01/20) = 0.5. Tolerance ~0.03 = +/-0.5 dB.
    EXPECT_NEAR(rms, 0.5, 0.03)
        << "RMS at cutoff should be ~0.5 (i.e. -3 dB on a unit sine). Got " << rms;
}

// (f) Stopband attenuation. Two octaves above cutoff, 2nd-order rolloff is 24 dB.
// Butterworth exact magnitude: |H(4)| = 1/sqrt(1 + 4^4) = 1/sqrt(257) ~= 0.0624
// (~-24.1 dB).
TEST_F(LowpassStageTest, StopbandAttenuation_TwentyFourDbAtFourX_Butterworth) {
    constexpr double cutoff = 500.0;
    constexpr double signalHz = 2000.0; // 2 octaves above cutoff
    filter_->setCutoffFrequency(cutoff);
    filter_->setResonance(kButterworthQ);

    constexpr double amplitude = 1.0;
    const double phase = warmupSine(*filter_, signalHz, kSampleRate, amplitude, kNumChannels, 10000);

    const int measureSamples = 4410;
    juce::AudioBuffer<double> measurement(kNumChannels, measureSamples);
    for (int ch = 0; ch < kNumChannels; ++ch) {
        for (int i = 0; i < measureSamples; ++i) {
            const double p = phase + 2.0 * M_PI * signalHz * i / kSampleRate;
            measurement.setSample(ch, i, amplitude * std::sin(p));
        }
    }
    const double rmsBefore = measureRMS(measurement);

    filter_->process(measurement);
    const double rmsAfter = measureRMS(measurement);

    const double ratio = rmsAfter / rmsBefore;
    EXPECT_LT(ratio, 0.08) << "Two octaves above cutoff should yield >= ~22 dB attenuation (ratio: "
                           << ratio << ")";
    EXPECT_GT(ratio, 0.04) << "Attenuation excessive for 2nd-order Butterworth at 4x cutoff (ratio: "
                           << ratio << ")";
}

// (g) Resonance produces a measurable peak at cutoff. Q=4 should boost the
// cutoff frequency noticeably above the passband. We measure the ratio of
// in-band RMS at the cutoff frequency vs. a deep-passband reference.
TEST_F(LowpassStageTest, ResonancePeakAtCutoff_QFour) {
    constexpr double cutoff = 1000.0;
    constexpr double passbandRef = 100.0;
    constexpr double amplitude = 0.5; // Headroom for the resonance boost.
    constexpr double Q = 4.0;
    const int measureSamples = 4410;

    auto measureSteadyStateRMS = [&](double frequency) {
        filter_->reset();
        filter_->setCutoffFrequency(cutoff);
        filter_->setResonance(Q);
        const double phase = warmupSine(*filter_, frequency, kSampleRate, amplitude, kNumChannels, 20000);
        juce::AudioBuffer<double> buf(kNumChannels, measureSamples);
        for (int ch = 0; ch < kNumChannels; ++ch) {
            for (int i = 0; i < measureSamples; ++i) {
                const double p = phase + 2.0 * M_PI * frequency * i / kSampleRate;
                buf.setSample(ch, i, amplitude * std::sin(p));
            }
        }
        filter_->process(buf);
        return measureRMS(buf);
    };

    const double passbandRMS = measureSteadyStateRMS(passbandRef);
    const double cutoffRMS = measureSteadyStateRMS(cutoff);

    EXPECT_GT(cutoffRMS / passbandRMS, 2.0)
        << "Q=4 should boost cutoff above passband by at least ~6 dB. Got ratio "
        << (cutoffRMS / passbandRMS);
}

// (h) Cutoff setter clamps to [20, 0.45*fs]. At 44100 fs, max = 19845.
TEST_F(LowpassStageTest, CutoffSetterClamps) {
    filter_->setCutoffFrequency(1000.0);
    EXPECT_DOUBLE_EQ(filter_->getCutoffFrequency(), 1000.0);

    filter_->setCutoffFrequency(20.0); // min
    EXPECT_DOUBLE_EQ(filter_->getCutoffFrequency(), 20.0);

    filter_->setCutoffFrequency(5.0); // below min
    EXPECT_DOUBLE_EQ(filter_->getCutoffFrequency(), 20.0);

    const double maxCutoff = kSampleRate * 0.45;
    filter_->setCutoffFrequency(maxCutoff);
    EXPECT_DOUBLE_EQ(filter_->getCutoffFrequency(), maxCutoff);

    filter_->setCutoffFrequency(30000.0); // above max
    EXPECT_DOUBLE_EQ(filter_->getCutoffFrequency(), maxCutoff);
}

// (i) Resonance setter clamps to [0.5, 10.0].
TEST_F(LowpassStageTest, ResonanceSetterClamps) {
    filter_->setResonance(1.0);
    EXPECT_DOUBLE_EQ(filter_->getResonance(), 1.0);

    filter_->setResonance(0.5); // min
    EXPECT_DOUBLE_EQ(filter_->getResonance(), 0.5);

    filter_->setResonance(0.1); // below min
    EXPECT_DOUBLE_EQ(filter_->getResonance(), 0.5);

    filter_->setResonance(10.0); // max
    EXPECT_DOUBLE_EQ(filter_->getResonance(), 10.0);

    filter_->setResonance(100.0); // above max
    EXPECT_DOUBLE_EQ(filter_->getResonance(), 10.0);
}

// (j) Reset clears filter state. With high Q, an impulse leaves a ringing
// tail. After reset, the same impulse must produce an identical tail to the
// one produced from a freshly prepared filter.
TEST_F(LowpassStageTest, ResetClearsState) {
    filter_->setCutoffFrequency(500.0);
    filter_->setResonance(4.0);

    auto impulseTail = [&](LowpassStage& f) {
        juce::AudioBuffer<double> buf(1, 1024);
        buf.clear();
        buf.setSample(0, 0, 1.0);
        f.process(buf);
        return measurePeak(buf);
    };

    // Tail from a freshly prepared filter.
    LowpassStage fresh;
    fresh.prepareToPlay(kSampleRate, kBlockSize, 1);
    fresh.setCutoffFrequency(500.0);
    fresh.setResonance(4.0);
    const double freshPeak = impulseTail(fresh);

    // Run our filter, drive it with arbitrary signal, then reset and remeasure.
    filter_.reset();
    filter_ = std::make_unique<LowpassStage>();
    filter_->prepareToPlay(kSampleRate, kBlockSize, 1);
    filter_->setCutoffFrequency(500.0);
    filter_->setResonance(4.0);
    warmupSine(*filter_, 500.0, kSampleRate, 0.5, 1, 5000);
    filter_->reset();
    const double afterResetPeak = impulseTail(*filter_);

    EXPECT_NEAR(afterResetPeak, freshPeak, 1e-10)
        << "Reset must restore initial state. Fresh peak: " << freshPeak
        << ", after-reset peak: " << afterResetPeak;
}

// (k) Multi-channel: identical input on every channel produces identical output
// on every channel (per-channel state vectors are sized correctly and updated
// independently but symmetrically).
TEST_F(LowpassStageTest, MultiChannelIdenticalForIdenticalInput) {
    filter_->setCutoffFrequency(1000.0);
    filter_->setResonance(kButterworthQ);

    // Warmup with identical signal on both channels.
    const double phase = warmupSine(*filter_, 500.0, kSampleRate, 0.7, kNumChannels, 10000);

    juce::AudioBuffer<double> buf(kNumChannels, kBlockSize);
    for (int ch = 0; ch < kNumChannels; ++ch) {
        for (int i = 0; i < kBlockSize; ++i) {
            const double p = phase + 2.0 * M_PI * 500.0 * i / kSampleRate;
            buf.setSample(ch, i, 0.7 * std::sin(p));
        }
    }
    filter_->process(buf);

    for (int i = 0; i < kBlockSize; ++i) {
        EXPECT_DOUBLE_EQ(buf.getSample(0, i), buf.getSample(1, i))
            << "Channels must process identically for identical input (i=" << i << ")";
    }
}

// (l) Stability under cutoff sweep with full-scale white-noise input. Sweep
// cutoff across the full legal range while feeding random samples; the output
// must remain finite and within a sane magnitude bound.
TEST_F(LowpassStageTest, StabilityUnderCutoffSweep) {
    filter_->setResonance(2.0);

    juce::Random rng(0xBD512AB1u);
    const int totalSamples = 100000;
    const double cutoffStart = 20.0;
    const double cutoffEnd = 19000.0; // safely under 0.45 * 44100
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
            ASSERT_LT(std::abs(s), 4.0)
                << "Output magnitude blew up (>|4|) at block " << block << ", cutoff=" << cutoff
                << ", sample=" << s;
        }
    }
}

