#include <gtest/gtest.h>
#include "../dsp_core/Source/audio_pipeline/DCBlockingFilter.h"
#include <juce_audio_basics/juce_audio_basics.h>
#include <cmath>

using namespace dsp_core::audio_pipeline;

class DCBlockingFilterTest : public ::testing::Test {
  protected:
    void SetUp() override {
        filter_ = std::make_unique<DCBlockingFilter>();
        filter_->prepareToPlay(44100.0, 512);
    }

    std::unique_ptr<DCBlockingFilter> filter_;

    // Helper: Fill buffer with DC offset
    static void fillWithDC(juce::AudioBuffer<double>& buffer, double dcValue) {
        for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
            for (int i = 0; i < buffer.getNumSamples(); ++i) {
                buffer.setSample(ch, i, dcValue);
            }
        }
    }

    // Helper: Fill buffer with sine wave
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    static void fillWithSine(juce::AudioBuffer<double>& buffer, double frequency, double sampleRate,
                             double amplitude = 1.0, double dcOffset = 0.0) {
        for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
            for (int i = 0; i < buffer.getNumSamples(); ++i) {
                double const phase = 2.0 * M_PI * frequency * i / sampleRate;
                buffer.setSample(ch, i, amplitude * std::sin(phase) + dcOffset);
            }
        }
    }

    // Helper: Measure RMS of buffer
    static double measureRMS(const juce::AudioBuffer<double>& buffer) {
        double sumSquares = 0.0;
        int totalSamples = 0;
        for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
            for (int i = 0; i < buffer.getNumSamples(); ++i) {
                double const sample = buffer.getSample(ch, i);
                sumSquares += sample * sample;
                totalSamples++;
            }
        }
        return std::sqrt(sumSquares / totalSamples);
    }

    // Helper: Measure mean (DC) of buffer
    static double measureMean(const juce::AudioBuffer<double>& buffer) {
        double sum = 0.0;
        int totalSamples = 0;
        for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
            for (int i = 0; i < buffer.getNumSamples(); ++i) {
                sum += buffer.getSample(ch, i);
                totalSamples++;
            }
        }
        return sum / totalSamples;
    }
};

// Test 1: Remove pure DC offset
TEST_F(DCBlockingFilterTest, RemovesPureDC) {
    const double dcOffset = 0.5;
    // 5Hz filter needs ~200ms to converge (5 time constants)
    // At 44.1kHz, that's ~8800 samples
    const int warmupSamples = 10000;

    // Create buffer with pure DC
    juce::AudioBuffer<double> buffer(2, 512);
    fillWithDC(buffer, dcOffset);

    // Process multiple blocks to allow filter to converge
    const int numBlocks = (warmupSamples / 512) + 2;
    for (int block = 0; block < numBlocks; ++block) {
        fillWithDC(buffer, dcOffset);
        filter_->process(buffer);
    }

    // After convergence, DC should be significantly reduced
    // 5Hz HPF won't completely remove DC instantly, but should reduce it
    double const meanAfterFiltering = std::abs(measureMean(buffer));

    EXPECT_LT(meanAfterFiltering, 0.05) << "DC blocking filter should significantly reduce DC offset after convergence";

    // Verify each channel
    for (int ch = 0; ch < 2; ++ch) {
        double channelMean = 0.0;
        for (int i = 0; i < 512; ++i) {
            channelMean += buffer.getSample(ch, i);
        }
        channelMean = std::abs(channelMean) / 512;

        EXPECT_LT(channelMean, 0.05) << "Channel " << ch << " should have significantly reduced DC after filtering";
    }
}

// Test 2: Preserve 20Hz sine wave (passband)
TEST_F(DCBlockingFilterTest, Preserves20HzSine) {
    const double frequency = 20.0;
    const double sampleRate = 44100.0;
    const double amplitude = 1.0;
    const int warmupSamples = 10000;

    // Warmup filter with continuous sine wave
    juce::AudioBuffer<double> warmupBuffer(2, 512);
    double phase = 0.0;
    for (int block = 0; block < warmupSamples / 512; ++block) {
        for (int ch = 0; ch < 2; ++ch) {
            for (int i = 0; i < 512; ++i) {
                double const samplePhase = phase + (2.0 * M_PI * frequency * i / sampleRate);
                warmupBuffer.setSample(ch, i, amplitude * std::sin(samplePhase));
            }
        }
        filter_->process(warmupBuffer);
        phase += 2.0 * M_PI * frequency * 512 / sampleRate;
    }

    // Continue with same phase for test measurement
    juce::AudioBuffer<double> buffer(2, 512);
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 512; ++i) {
            double const samplePhase = phase + (2.0 * M_PI * frequency * i / sampleRate);
            buffer.setSample(ch, i, amplitude * std::sin(samplePhase));
        }
    }

    double const rmsBeforeFiltering = measureRMS(buffer);

    // Create copy for filtering (to preserve original for comparison)
    juce::AudioBuffer<double> filteredBuffer(2, 512);
    filteredBuffer.copyFrom(0, 0, buffer, 0, 0, 512);
    filteredBuffer.copyFrom(1, 0, buffer, 1, 0, 512);

    filter_->process(filteredBuffer);
    double const rmsAfterFiltering = measureRMS(filteredBuffer);

    // At 20Hz with 5Hz cutoff (4x ratio), 1st-order HPF has:
    // H(f) = f/fc / sqrt(1 + (f/fc)^2) = 4 / sqrt(17) ≈ 0.97
    // So we expect ~97% amplitude preservation at 20Hz
    double const attenuationRatio = rmsAfterFiltering / rmsBeforeFiltering;

    EXPECT_GT(attenuationRatio, 0.90) << "20Hz sine should pass through with minimal attenuation (ratio: "
                                      << attenuationRatio << ")";
}

// Test 3: Harmonic preservation - verify no even harmonic distortion
TEST_F(DCBlockingFilterTest, HarmonicPreservation) {
    const double fundamental = 100.0;
    const double sampleRate = 44100.0;
    const int numSamples = 4410; // 0.1 second for FFT analysis
    const int warmupSamples = 2000;

    // Create buffer with 100Hz sine (odd harmonic structure)
    juce::AudioBuffer<double> buffer(1, numSamples);
    for (int i = 0; i < numSamples; ++i) {
        double const phase = 2.0 * M_PI * fundamental * i / sampleRate;
        buffer.setSample(0, i, std::sin(phase));
    }

    // Warmup filter
    for (int block = 0; block < warmupSamples / 512; ++block) {
        juce::AudioBuffer<double> warmupBuffer(1, 512);
        for (int i = 0; i < 512; ++i) {
            double const phase = 2.0 * M_PI * fundamental * i / sampleRate;
            warmupBuffer.setSample(0, i, std::sin(phase));
        }
        filter_->process(warmupBuffer);
    }

    // Process test buffer in chunks
    const int blockSize = 512;
    for (int offset = 0; offset < numSamples; offset += blockSize) {
        int const samplesThisBlock = std::min(blockSize, numSamples - offset);
        juce::AudioBuffer<double> block(1, samplesThisBlock);

        for (int i = 0; i < samplesThisBlock; ++i) {
            block.setSample(0, i, buffer.getSample(0, offset + i));
        }

        filter_->process(block);

        for (int i = 0; i < samplesThisBlock; ++i) {
            buffer.setSample(0, offset + i, block.getSample(0, i));
        }
    }

    // Simple harmonic analysis: check for even harmonics (200Hz, 400Hz)
    // A pure sine should not introduce even harmonics
    // We'll check correlation with 200Hz sine (2nd harmonic)
    double evenHarmonicEnergy = 0.0;
    double fundamentalEnergy = 0.0;

    for (int i = 0; i < numSamples; ++i) {
        double const sample = buffer.getSample(0, i);

        // Correlate with 2nd harmonic (200Hz)
        double const evenPhase = 2.0 * M_PI * (2.0 * fundamental) * i / sampleRate;
        evenHarmonicEnergy += sample * std::sin(evenPhase);

        // Correlate with fundamental (100Hz)
        double const fundPhase = 2.0 * M_PI * fundamental * i / sampleRate;
        fundamentalEnergy += sample * std::sin(fundPhase);
    }

    evenHarmonicEnergy = std::abs(evenHarmonicEnergy);
    fundamentalEnergy = std::abs(fundamentalEnergy);

    // Even harmonic should be much smaller than fundamental
    double const harmonicRatio = evenHarmonicEnergy / fundamentalEnergy;

    EXPECT_LT(harmonicRatio, 0.01) << "DC blocking filter should not introduce even harmonics (ratio: " << harmonicRatio
                                   << ")";
}

// Test 4: Verify enabled/disabled state
TEST_F(DCBlockingFilterTest, EnabledDisabledState) {
    const double dcOffset = 0.5;

    // Create buffer with DC
    juce::AudioBuffer<double> buffer(2, 512);
    fillWithDC(buffer, dcOffset);

    // Store original
    juce::AudioBuffer<double> original(2, 512);
    original.copyFrom(0, 0, buffer, 0, 0, 512);
    original.copyFrom(1, 0, buffer, 1, 0, 512);

    // Disable filter
    filter_->setEnabled(false);
    EXPECT_FALSE(filter_->isEnabled());

    // Process should be bypassed
    filter_->process(buffer);

    // Buffer should be unchanged
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 512; ++i) {
            EXPECT_DOUBLE_EQ(buffer.getSample(ch, i), original.getSample(ch, i))
                << "Disabled filter should bypass processing";
        }
    }

    // Re-enable
    filter_->setEnabled(true);
    EXPECT_TRUE(filter_->isEnabled());
}

// Test 5: Cutoff frequency adjustment
TEST_F(DCBlockingFilterTest, CutoffFrequencyAdjustment) {
    // Set cutoff to 10Hz
    filter_->setCutoffFrequency(10.0);
    EXPECT_DOUBLE_EQ(filter_->getCutoffFrequency(), 10.0);

    // Set to edge cases
    filter_->setCutoffFrequency(1.0); // Min
    EXPECT_DOUBLE_EQ(filter_->getCutoffFrequency(), 1.0);

    filter_->setCutoffFrequency(20.0); // Max
    EXPECT_DOUBLE_EQ(filter_->getCutoffFrequency(), 20.0);

    // Test clamping (values outside 1-20Hz range)
    filter_->setCutoffFrequency(0.5); // Below min
    EXPECT_DOUBLE_EQ(filter_->getCutoffFrequency(), 1.0);

    filter_->setCutoffFrequency(50.0); // Above max
    EXPECT_DOUBLE_EQ(filter_->getCutoffFrequency(), 20.0);
}

// Test 6: Reset clears filter state
TEST_F(DCBlockingFilterTest, ResetClearsState) {
    const double dcOffset = 0.5;

    // Process some DC offset
    juce::AudioBuffer<double> buffer(2, 512);
    for (int block = 0; block < 5; ++block) {
        fillWithDC(buffer, dcOffset);
        filter_->process(buffer);
    }

    // Reset filter
    filter_->reset();

    // Process DC again - should take time to converge again
    fillWithDC(buffer, dcOffset);
    filter_->process(buffer);

    // Immediately after reset, filter hasn't converged yet
    // So output should still have significant DC (not fully removed)
    double const meanAfterReset = std::abs(measureMean(buffer));

    // Should be closer to input DC than fully converged state
    EXPECT_GT(meanAfterReset, 0.1) << "Reset filter should not immediately remove DC (needs convergence)";
}

// Test 7: Multi-channel processing
TEST_F(DCBlockingFilterTest, MultiChannelProcessing) {
    const int numChannels = 8;
    const double dcOffset = 0.3;
    const int warmupSamples = 10000;

    juce::AudioBuffer<double> buffer(numChannels, 512);

    // Warmup
    for (int block = 0; block < warmupSamples / 512; ++block) {
        fillWithDC(buffer, dcOffset);
        filter_->process(buffer);
    }

    // Test processing
    fillWithDC(buffer, dcOffset);
    filter_->process(buffer);

    // All channels should have DC significantly reduced
    for (int ch = 0; ch < numChannels; ++ch) {
        double channelMean = 0.0;
        for (int i = 0; i < 512; ++i) {
            channelMean += buffer.getSample(ch, i);
        }
        channelMean = std::abs(channelMean) / 512;

        EXPECT_LT(channelMean, 0.05) << "Channel " << ch << " should have DC significantly reduced";
    }
}

// Test 8: Sine wave with DC offset - remove DC, preserve AC
TEST_F(DCBlockingFilterTest, RemovesDCPreservesAC) {
    const double frequency = 440.0; // A4
    const double sampleRate = 44100.0;
    const double amplitude = 0.5;
    const double dcOffset = 0.3;
    const int warmupSamples = 10000;

    juce::AudioBuffer<double> buffer(2, 512);

    // Warmup filter with continuous waveform
    double phase = 0.0;
    for (int block = 0; block < warmupSamples / 512; ++block) {
        for (int ch = 0; ch < 2; ++ch) {
            for (int i = 0; i < 512; ++i) {
                double const samplePhase = phase + (2.0 * M_PI * frequency * i / sampleRate);
                buffer.setSample(ch, i, amplitude * std::sin(samplePhase) + dcOffset);
            }
        }
        filter_->process(buffer);
        phase += 2.0 * M_PI * frequency * 512 / sampleRate;
    }

    // Process test signal
    fillWithSine(buffer, frequency, sampleRate, amplitude, dcOffset);
    filter_->process(buffer);

    // Check that DC is significantly reduced
    double const mean = std::abs(measureMean(buffer));
    EXPECT_LT(mean, 0.1) << "DC component should be significantly reduced";

    // Check that AC is preserved (RMS should be close to amplitude/sqrt(2))
    double const rms = measureRMS(buffer);
    double const expectedRMS = amplitude / std::sqrt(2.0);

    // Allow some tolerance for filter transient and phase shift
    EXPECT_NEAR(rms, expectedRMS, 0.15) << "AC component should be preserved (RMS: " << rms
                                        << " expected: " << expectedRMS << ")";
}

// =========================================================================
// Transparency characterization tests
//
// User report: "when output is over 0dB, sounds fine with DC filter OFF
// but bad with DC filter ON". These tests probe the likely causes:
//   (a) peak amplification on hot symmetric signals
//   (b) envelope-tracking transients on signals with varying DC
//   (c) THD introduction by filter at high levels
// =========================================================================

namespace {

// Peak absolute value across a buffer
double measurePeak(const juce::AudioBuffer<double>& buffer) {
    double peak = 0.0;
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
        for (int i = 0; i < buffer.getNumSamples(); ++i) {
            peak = std::max(peak, std::abs(buffer.getSample(ch, i)));
        }
    }
    return peak;
}

// Warm up the filter by running a continuous signal generator for N samples.
// Generator signature: double(int sampleIndex) -> sample value.
template <typename Gen>
void warmupFilter(DCBlockingFilter& filter, Gen gen, int numChannels, int numSamples, int blockSize = 512) {
    juce::AudioBuffer<double> buffer(numChannels, blockSize);
    int produced = 0;
    while (produced < numSamples) {
        const int thisBlock = std::min(blockSize, numSamples - produced);
        buffer.setSize(numChannels, thisBlock, false, false, true);
        for (int ch = 0; ch < numChannels; ++ch) {
            for (int i = 0; i < thisBlock; ++i) {
                buffer.setSample(ch, i, gen(produced + i));
            }
        }
        filter.process(buffer);
        produced += thisBlock;
    }
}

} // namespace

// Test 9: Peak preservation on hot symmetric sine (no DC).
// A signal with no DC offset should pass through with peak amplitude preserved
// within tight tolerance. Any peak amplification indicates the filter is adding
// energy — making it non-transparent at high levels.
TEST_F(DCBlockingFilterTest, PeakPreservationOnHotSymmetricSine) {
    const double sampleRate = 44100.0;
    const double frequency = 100.0;
    const double amplitude = 1.5; // Hot: over 0dB

    // Warm up with continuous symmetric sine (no DC)
    warmupFilter(*filter_,
                 [=](int n) { return amplitude * std::sin(2.0 * M_PI * frequency * n / sampleRate); },
                 2, 20000);

    // Measurement buffer: one full period
    const int periodSamples = static_cast<int>(sampleRate / frequency);
    juce::AudioBuffer<double> buffer(2, periodSamples * 4);
    const int phaseOffset = 20000; // continue phase from warmup
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < buffer.getNumSamples(); ++i) {
            const double phase = 2.0 * M_PI * frequency * (phaseOffset + i) / sampleRate;
            buffer.setSample(ch, i, amplitude * std::sin(phase));
        }
    }

    const double inputPeak = measurePeak(buffer);
    filter_->process(buffer);
    const double outputPeak = measurePeak(buffer);

    // Transparent filter must not amplify peaks of a DC-free signal.
    // 0.5% tolerance for numerical error.
    EXPECT_LE(outputPeak, inputPeak * 1.005)
        << "Filter amplified peak (in: " << inputPeak << ", out: " << outputPeak << ")";
    EXPECT_GE(outputPeak, inputPeak * 0.995)
        << "Filter attenuated peak excessively (in: " << inputPeak << ", out: " << outputPeak << ")";
}

// Test 10: Envelope-change transient artifact.
// When a signal with a DC offset (like a waveshaped asymmetric signal) suddenly
// goes silent, the filter's tracked DC lags. Output overshoots to the opposite
// polarity of the tracked DC. This is the primary suspect for audible artifacts
// when the signal is hot and dynamics change fast.
TEST_F(DCBlockingFilterTest, EnvelopeDropDoesNotCreateLargeTransient) {
    const double sampleRate = 44100.0;
    const double frequency = 200.0;
    const double dcOffset = 0.4; // simulates hot asymmetric waveshaper output

    // Warm up with asymmetric signal (sine + DC offset)
    warmupFilter(*filter_,
                 [=](int n) {
                     return std::sin(2.0 * M_PI * frequency * n / sampleRate) + dcOffset;
                 },
                 1, 30000);

    // Now process a block of SILENCE (simulate envelope gate closing)
    const int silentSamples = 2048;
    juce::AudioBuffer<double> silent(1, silentSamples);
    silent.clear();

    filter_->process(silent);

    // The filter's tracked DC must not leak through as a large transient.
    // A transparent DC blocker can leak some residual DC while the tracker relaxes
    // to zero, but it should NOT approach the magnitude of the original DC.
    const double transientPeak = measurePeak(silent);

    // Threshold: the transient must be much smaller than the tracked DC itself.
    // If it approaches dcOffset, we have a clearly audible artifact.
    EXPECT_LT(transientPeak, dcOffset * 0.25)
        << "Envelope-drop transient is too large (peak: " << transientPeak
        << ", tracked DC was: " << dcOffset << "). This creates audible pops on hot signals.";
}

// Test 11: THD introduction at hot levels.
// A pure sine wave should remain pure after filtering. If THD scales with input
// amplitude, the filter is nonlinear at hot levels — which would explain why it
// sounds bad over 0dB but fine at lower levels.
TEST_F(DCBlockingFilterTest, THDIsLevelIndependent) {
    const double sampleRate = 44100.0;
    const double frequency = 440.0;
    const int numSamples = 8192;

    auto measureTHD = [&](double amplitude) {
        filter_->reset();
        warmupFilter(*filter_,
                     [=](int n) { return amplitude * std::sin(2.0 * M_PI * frequency * n / sampleRate); },
                     1, 10000);

        juce::AudioBuffer<double> buffer(1, numSamples);
        const int phaseOffset = 10000;
        for (int i = 0; i < numSamples; ++i) {
            const double phase = 2.0 * M_PI * frequency * (phaseOffset + i) / sampleRate;
            buffer.setSample(0, i, amplitude * std::sin(phase));
        }
        filter_->process(buffer);

        // Correlate with fundamental and 2nd/3rd harmonics
        double fund = 0.0;
        double h2 = 0.0;
        double h3 = 0.0;
        for (int i = 0; i < numSamples; ++i) {
            const double s = buffer.getSample(0, i);
            const double p1 = 2.0 * M_PI * frequency * (phaseOffset + i) / sampleRate;
            const double p2 = 2.0 * p1;
            const double p3 = 3.0 * p1;
            fund += s * std::sin(p1);
            h2 += s * std::sin(p2);
            h3 += s * std::sin(p3);
        }
        const double fundMag = std::abs(fund);
        const double harmMag = std::sqrt(h2 * h2 + h3 * h3);
        return harmMag / std::max(fundMag, 1e-12);
    };

    const double thdAtQuiet = measureTHD(0.1);
    const double thdAtHot = measureTHD(1.5);

    // THD introduced by a linear filter must not depend on input amplitude.
    // If these diverge by more than 10x, the filter is nonlinear at hot levels.
    EXPECT_NEAR(thdAtHot, thdAtQuiet, thdAtQuiet * 10.0 + 1e-6)
        << "THD at hot level (" << thdAtHot << ") differs from quiet level (" << thdAtQuiet
        << ") — filter is level-dependent, not transparent";
}

// Test 12: No peak amplification on complex harmonic signal (phase coherence).
// A filter that phase-shifts low frequencies more than high frequencies can
// reassemble harmonics into a waveform with higher peak than the input.
// This is a known cause of "intersample peak growth" through IIR filters.
TEST_F(DCBlockingFilterTest, PeakPreservationOnComplexHarmonicSignal) {
    const double sampleRate = 44100.0;
    const double f0 = 80.0; // fundamental in the range where phase shift matters
    const int numSamples = 16384;

    // Signal: fundamental + several harmonics, phased to peak near ±1.0
    auto gen = [=](int n) {
        const double t = static_cast<double>(n) / sampleRate;
        double v = std::sin(2.0 * M_PI * f0 * t);
        v += 0.6 * std::sin(2.0 * M_PI * 2.0 * f0 * t);
        v += 0.4 * std::sin(2.0 * M_PI * 3.0 * f0 * t);
        v += 0.25 * std::sin(2.0 * M_PI * 4.0 * f0 * t);
        return v;
    };

    // Warmup
    warmupFilter(*filter_, gen, 1, 20000);

    // Build input and keep a copy for comparison
    juce::AudioBuffer<double> input(1, numSamples);
    for (int i = 0; i < numSamples; ++i) {
        input.setSample(0, i, gen(20000 + i));
    }

    juce::AudioBuffer<double> filtered(1, numSamples);
    filtered.copyFrom(0, 0, input, 0, 0, numSamples);
    filter_->process(filtered);

    const double inputPeak = measurePeak(input);
    const double outputPeak = measurePeak(filtered);

    // Transparent filter: output peak must not exceed input peak meaningfully.
    // Small amount (~1%) is tolerable; more indicates phase-induced peak growth.
    EXPECT_LE(outputPeak, inputPeak * 1.02)
        << "Peak grew through filter (in: " << inputPeak << ", out: " << outputPeak
        << ") — phase shift is reassembling harmonics into a larger peak";
}
