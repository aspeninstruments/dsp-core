/**
 * SoftClippingComparisonTest.cpp
 *
 * Comparison of hard clipping vs soft clipping (SoftClippingSolver).
 * Uses FFT-based harmonic analysis to measure THD and high-frequency distortion.
 *
 * Goals:
 * 1. Verify soft clipper is transparent at low levels
 * 2. Measure distortion characteristics at overs
 * 3. Compare high-frequency harmonic energy (proxy for aliasing risk)
 */

#include <gtest/gtest.h>
#include "../dsp_core/Source/audio_pipeline/SoftClippingStage.h"
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace dsp_core::audio_pipeline;

namespace {

// Constants
constexpr double kPi = 3.14159265358979323846;
constexpr double kSampleRate = 48000.0;
constexpr int kFFTSize = 65536;
constexpr int kFundamentalBin = 1000; // Coherent bin for ~732.4 Hz

// Convert dB to linear amplitude
inline double dBToLinear(double dB) {
    return std::pow(10.0, dB / 20.0);
}

// Convert linear amplitude to dB
inline double linearToDB(double linear) {
    if (linear <= 0.0)
        return -200.0;
    return 20.0 * std::log10(linear);
}

/**
 * Hard clipper at 0 dBFS (±1.0)
 */
inline double hardClip(double x) {
    return juce::jlimit(-1.0, 1.0, x);
}

/**
 * Generate a coherent sine wave (no spectral leakage when analyzed with FFT)
 * @param N Number of samples (should equal FFT size)
 * @param binIndex Integer bin index for fundamental frequency
 * @param amplitude Peak amplitude
 * @return Vector of samples
 */
std::vector<double> generateCoherentSine(int N, int binIndex, double amplitude) {
    std::vector<double> signal(N);
    const double omega = 2.0 * kPi * binIndex / N;

    for (int i = 0; i < N; ++i) {
        signal[i] = amplitude * std::sin(omega * i);
    }

    return signal;
}

/**
 * Simple radix-2 DIT FFT (in-place)
 * Assumes N is a power of 2
 */
void fft(std::vector<std::complex<double>>& x) {
    const size_t N = x.size();
    if (N <= 1)
        return;

    // Bit-reversal permutation
    for (size_t i = 1, j = 0; i < N; ++i) {
        size_t bit = N >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j)
            std::swap(x[i], x[j]);
    }

    // Cooley-Tukey iterative FFT
    for (size_t len = 2; len <= N; len <<= 1) {
        double angle = -2.0 * kPi / len;
        std::complex<double> wlen(std::cos(angle), std::sin(angle));

        for (size_t i = 0; i < N; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (size_t j = 0; j < len / 2; ++j) {
                std::complex<double> u = x[i + j];
                std::complex<double> v = x[i + j + len / 2] * w;
                x[i + j] = u + v;
                x[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

/**
 * Compute magnitude spectrum from real signal
 * @param signal Input signal
 * @return Magnitude spectrum (first N/2+1 bins)
 */
std::vector<double> computeMagnitudeSpectrum(const std::vector<double>& signal) {
    const size_t N = signal.size();

    // Convert to complex
    std::vector<std::complex<double>> complex_signal(N);
    for (size_t i = 0; i < N; ++i) {
        complex_signal[i] = std::complex<double>(signal[i], 0.0);
    }

    fft(complex_signal);

    // Extract magnitudes (normalized)
    std::vector<double> mags(N / 2 + 1);
    for (size_t i = 0; i <= N / 2; ++i) {
        mags[i] = std::abs(complex_signal[i]) / N;
        if (i > 0 && i < N / 2) {
            mags[i] *= 2.0; // Account for negative frequencies
        }
    }

    return mags;
}

/**
 * Harmonic analysis results
 */
struct HarmonicAnalysis {
    double fundamentalMag = 0.0;      // Fundamental magnitude (linear)
    double thd = 0.0;                 // Total Harmonic Distortion (linear ratio)
    double thdDB = 0.0;               // THD in dB
    double oddTHD = 0.0;              // Odd-only THD
    double evenTHD = 0.0;             // Even-only THD
    double highFreqEnergy = 0.0;      // High-order harmonics (9th+) energy ratio
    double highFreqEnergyDB = 0.0;    // High-order energy in dB
    std::vector<double> harmonicMags; // Individual harmonic magnitudes (2nd through 15th)

    void print(const std::string& label) const {
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  " << label << ":\n";
        std::cout << "    Fundamental: " << std::setprecision(6) << fundamentalMag << " (" << std::setprecision(2)
                  << linearToDB(fundamentalMag) << " dBFS)\n";
        std::cout << "    THD: " << std::setprecision(6) << thd << " (" << std::setprecision(2) << thdDB << " dB)\n";
        std::cout << "    Odd THD: " << std::setprecision(6) << oddTHD << " (" << std::setprecision(2)
                  << linearToDB(oddTHD) << " dB)\n";
        std::cout << "    Even THD: " << std::setprecision(6) << evenTHD << " (" << std::setprecision(2)
                  << linearToDB(evenTHD) << " dB)\n";
        std::cout << "    High-freq (9th+): " << std::setprecision(6) << highFreqEnergy << " (" << std::setprecision(2)
                  << highFreqEnergyDB << " dB)\n";

        // Print individual harmonics
        std::cout << "    Harmonics (dBc): ";
        for (size_t i = 0; i < std::min(harmonicMags.size(), size_t(10)); ++i) {
            if (i > 0)
                std::cout << ", ";
            std::cout << "H" << (i + 2) << "=" << std::setprecision(1) << linearToDB(harmonicMags[i] / fundamentalMag);
        }
        std::cout << "\n";
    }
};

/**
 * Analyze harmonics from magnitude spectrum
 * @param mags Magnitude spectrum
 * @param fundamentalBin Bin index of fundamental frequency
 * @param maxHarmonic Maximum harmonic to analyze
 * @return HarmonicAnalysis structure
 */
HarmonicAnalysis analyzeHarmonics(const std::vector<double>& mags, int fundamentalBin, int maxHarmonic = 15) {
    HarmonicAnalysis result;

    result.fundamentalMag = mags[fundamentalBin];

    double oddSumSq = 0.0;
    double evenSumSq = 0.0;
    double highFreqSumSq = 0.0;

    for (int h = 2; h <= maxHarmonic; ++h) {
        int bin = h * fundamentalBin;
        if (bin >= static_cast<int>(mags.size()))
            break;

        double mag = mags[bin];
        result.harmonicMags.push_back(mag);

        double magSq = mag * mag;

        if (h % 2 == 0) {
            evenSumSq += magSq;
        } else {
            oddSumSq += magSq;
        }

        if (h >= 9) {
            highFreqSumSq += magSq;
        }
    }

    double totalSumSq = oddSumSq + evenSumSq;

    result.thd = std::sqrt(totalSumSq) / result.fundamentalMag;
    result.thdDB = linearToDB(result.thd);
    result.oddTHD = std::sqrt(oddSumSq) / result.fundamentalMag;
    result.evenTHD = std::sqrt(evenSumSq) / result.fundamentalMag;
    result.highFreqEnergy = std::sqrt(highFreqSumSq) / result.fundamentalMag;
    result.highFreqEnergyDB = linearToDB(result.highFreqEnergy);

    return result;
}

/**
 * Compute time-domain error metrics
 */
struct ErrorMetrics {
    double maxAbsError = 0.0;
    double rmsError = 0.0;

    void print(const std::string& label) const {
        std::cout << std::fixed << std::setprecision(10);
        std::cout << "  " << label << ":\n";
        std::cout << "    Max Abs Error: " << maxAbsError << "\n";
        std::cout << "    RMS Error: " << rmsError << "\n";
    }
};

ErrorMetrics computeError(const std::vector<double>& original, const std::vector<double>& processed) {
    ErrorMetrics result;
    double sumSq = 0.0;

    for (size_t i = 0; i < original.size(); ++i) {
        double err = std::abs(processed[i] - original[i]);
        result.maxAbsError = std::max(result.maxAbsError, err);
        sumSq += err * err;
    }

    result.rmsError = std::sqrt(sumSq / original.size());
    return result;
}

} // anonymous namespace

//==============================================================================
// Test Fixture
//==============================================================================

class SoftClippingComparisonTest : public ::testing::Test {
  protected:
    void SetUp() override {
        solver_ = std::make_unique<SoftClippingSolver>(kDefaultA);
    }

    static constexpr double kDefaultA = 0.95; // Default transition point

    std::unique_ptr<SoftClippingSolver> solver_;

    // Apply hard clipping to signal
    static std::vector<double> applyHardClip(const std::vector<double>& input) {
        std::vector<double> output(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = hardClip(input[i]);
        }
        return output;
    }

    // Apply soft clipping
    std::vector<double> applySoftClip(const std::vector<double>& input) {
        std::vector<double> output(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = solver_->process(input[i]);
        }
        return output;
    }

    // Run comparison test at a given input level
    void runComparisonAtLevel(double inputLevelDB, const std::string& levelLabel) {
        double amplitude = dBToLinear(inputLevelDB);

        std::cout << "\n========================================\n";
        std::cout << "Input Level: " << levelLabel << " (A = " << std::fixed << std::setprecision(6) << amplitude
                  << ")\n";
        std::cout << "========================================\n";

        // Generate coherent sine
        auto input = generateCoherentSine(kFFTSize, kFundamentalBin, amplitude);

        // Apply clippers
        auto hardClipped = applyHardClip(input);
        auto softClipped = applySoftClip(input);

        // Compute error metrics
        auto hardError = computeError(input, hardClipped);
        auto softError = computeError(input, softClipped);

        std::cout << "\nTime-Domain Error (vs original):\n";
        hardError.print("Hard Clip");
        softError.print("Soft Clip");

        // Compute magnitude spectra
        auto hardMags = computeMagnitudeSpectrum(hardClipped);
        auto softMags = computeMagnitudeSpectrum(softClipped);

        // Analyze harmonics
        auto hardAnalysis = analyzeHarmonics(hardMags, kFundamentalBin);
        auto softAnalysis = analyzeHarmonics(softMags, kFundamentalBin);

        std::cout << "\nHarmonic Analysis:\n";
        hardAnalysis.print("Hard Clip");
        softAnalysis.print("Soft Clip");

        // Store results for assertions
        lastHardAnalysis_ = hardAnalysis;
        lastSoftAnalysis_ = softAnalysis;
        lastHardError_ = hardError;
        lastSoftError_ = softError;
    }

    // Last analysis results for assertions
    HarmonicAnalysis lastHardAnalysis_;
    HarmonicAnalysis lastSoftAnalysis_;
    ErrorMetrics lastHardError_;
    ErrorMetrics lastSoftError_;
};

//==============================================================================
// Tests
//==============================================================================

// Test 1: Below knee, soft clipper should be perfectly transparent
TEST_F(SoftClippingComparisonTest, TransparentBelowKnee) {
    const double knee = solver_->getTransitionPoint();
    runComparisonAtLevel(linearToDB(knee * 0.9), "Below Knee (-0.9 * knee)");

    // Both should be perfectly linear (no clipping)
    EXPECT_LT(lastHardError_.maxAbsError, 1e-12) << "Hard clip should be linear below knee";
    EXPECT_LT(lastSoftError_.maxAbsError, 1e-12) << "Soft clip should be linear below knee";
    EXPECT_LT(lastHardAnalysis_.thd, 1e-8) << "Hard clip THD should be near zero below knee";
    EXPECT_LT(lastSoftAnalysis_.thd, 1e-8) << "Soft clip THD should be near zero below knee";
}

// Test 2: At knee level, soft clipper starts limiting but hard clip is still linear
TEST_F(SoftClippingComparisonTest, BehaviorAtKnee) {
    runComparisonAtLevel(-0.45, "-0.45 dBFS (near knee)");

    // Hard clip is still linear at this level (below 0 dBFS)
    EXPECT_LT(lastHardError_.maxAbsError, 1e-12) << "Hard clip should be linear at -0.45 dBFS";
    EXPECT_LT(lastHardAnalysis_.thd, 1e-8) << "Hard clip THD should be near zero at -0.45 dBFS";

    // Soft clip may show minimal distortion depending on exact knee
    // Even harmonics should remain near zero (odd symmetry preserved)
    EXPECT_LT(lastSoftAnalysis_.evenTHD, 1e-8) << "Soft clip should maintain odd symmetry";
}

// Test 3: At 0 dBFS, hard clip is still linear, soft clip is limiting
TEST_F(SoftClippingComparisonTest, BehaviorAtUnity) {
    runComparisonAtLevel(0.0, "0 dBFS");

    // Hard clip is linear at exactly 1.0 amplitude
    EXPECT_LT(lastHardError_.maxAbsError, 1e-12) << "Hard clip should be linear at 0 dBFS";
    EXPECT_LT(lastHardAnalysis_.thd, 1e-8) << "Hard clip THD should be near zero at 0 dBFS";

    // Soft clip introduces some distortion (knee is below 0 dBFS)
    EXPECT_GT(lastSoftAnalysis_.thd, 1e-6) << "Soft clip should show distortion at 0 dBFS (knee is at -0.45 dBFS)";

    // Even harmonics still near zero
    EXPECT_LT(lastSoftAnalysis_.evenTHD, 1e-6) << "Soft clip should maintain odd symmetry";
}

// Test 4: At +3 dBFS, compare distortion characteristics
TEST_F(SoftClippingComparisonTest, DistortionAtPlus3dBFS) {
    runComparisonAtLevel(3.0, "+3 dBFS");

    // Both should show measurable distortion
    EXPECT_GT(lastHardAnalysis_.thd, 1e-4) << "Hard clip should show distortion at +3 dBFS";
    EXPECT_GT(lastSoftAnalysis_.thd, 1e-4) << "Soft clip should show distortion at +3 dBFS";

    std::cout << "\nHigh-Frequency Comparison at +3 dBFS:\n";
    std::cout << "  Hard Clip HF: " << lastHardAnalysis_.highFreqEnergyDB << " dB\n";
    std::cout << "  Soft Clip HF: " << lastSoftAnalysis_.highFreqEnergyDB << " dB\n";
    std::cout << "  HF Reduction: " << (lastHardAnalysis_.highFreqEnergyDB - lastSoftAnalysis_.highFreqEnergyDB)
              << " dB\n";

    // Soft clipping should reduce HF energy compared to hard clipping
    EXPECT_LT(lastSoftAnalysis_.highFreqEnergy, lastHardAnalysis_.highFreqEnergy)
        << "Soft clip should have less high-frequency harmonic energy than hard clip at +3 dBFS";

    // Even harmonics should remain near zero
    EXPECT_LT(lastHardAnalysis_.evenTHD, 1e-8) << "Hard clip should preserve odd symmetry";
    EXPECT_LT(lastSoftAnalysis_.evenTHD, 1e-6) << "Soft clip should preserve odd symmetry";
}

// Test 5: At +6 dBFS (severe clipping), behavior converges
TEST_F(SoftClippingComparisonTest, DistortionAtPlus6dBFS) {
    runComparisonAtLevel(6.0, "+6 dBFS");

    std::cout << "\nHigh-Frequency Comparison at +6 dBFS:\n";
    std::cout << "  Hard Clip HF: " << lastHardAnalysis_.highFreqEnergyDB << " dB\n";
    std::cout << "  Soft Clip HF: " << lastSoftAnalysis_.highFreqEnergyDB << " dB\n";
    std::cout << "  HF Reduction: " << (lastHardAnalysis_.highFreqEnergyDB - lastSoftAnalysis_.highFreqEnergyDB)
              << " dB\n";

    // At extreme levels, soft clipping HF energy approaches hard clip
    // (within ~1 dB), as most of the signal is in the saturation region
    double hfDifference = std::abs(lastHardAnalysis_.highFreqEnergyDB - lastSoftAnalysis_.highFreqEnergyDB);
    EXPECT_LT(hfDifference, 1.0)
        << "At +6 dBFS, soft clip HF should be within 1 dB of hard clip (both are mostly saturating)";

    // Even harmonics should remain near zero (odd symmetry preserved)
    EXPECT_LT(lastHardAnalysis_.evenTHD, 1e-8) << "Hard clip should preserve odd symmetry";
    EXPECT_LT(lastSoftAnalysis_.evenTHD, 1e-6) << "Soft clip should preserve odd symmetry";
}

// Test 6: Odd symmetry verification
TEST_F(SoftClippingComparisonTest, OddSymmetryVerification) {
    std::cout << "\n\nOdd Symmetry Verification:\n";

    // Test that f(-x) = -f(x)
    std::vector<double> testValues = {0.0, 0.5, 0.9, 0.95, 0.99, 1.0, 1.5, 2.0};

    double maxError = 0.0;

    for (double x : testValues) {
        double posResult = solver_->process(x);
        double negResult = solver_->process(-x);
        double error = std::abs(posResult + negResult);
        maxError = std::max(maxError, error);
    }

    std::cout << "  Max symmetry error: " << maxError << "\n";

    EXPECT_LT(maxError, 1e-14) << "Soft clipper should be perfectly odd symmetric";
}

// Test 7: Derivative continuity at knee point
TEST_F(SoftClippingComparisonTest, DerivativeContinuityAtKnee) {
    std::cout << "\n\nDerivative Continuity at Knee:\n";

    const double epsilon = 1e-8;
    const double knee = solver_->getTransitionPoint();

    // Check derivative at x = a
    double yMinus = solver_->process(knee - epsilon);
    double yKnee = solver_->process(knee);
    double yPlus = solver_->process(knee + epsilon);

    double derivLeft = (yKnee - yMinus) / epsilon;
    double derivRight = (yPlus - yKnee) / epsilon;

    std::cout << "  At knee=" << knee << ":\n";
    std::cout << "    Left derivative: " << derivLeft << "\n";
    std::cout << "    Right derivative: " << derivRight << "\n";
    std::cout << "    Difference: " << std::abs(derivLeft - derivRight) << "\n";

    // Should have continuous derivatives (C1 continuity)
    EXPECT_NEAR(derivLeft, derivRight, 1e-4) << "Soft clipper should have continuous derivative at knee";

    // Derivative should be 1.0 in linear region (unity gain)
    EXPECT_NEAR(derivLeft, 1.0, 1e-4) << "Soft clipper should have derivative 1.0 in linear region";
}

// Test 8: Saturation point behavior
TEST_F(SoftClippingComparisonTest, SaturationPointBehavior) {
    std::cout << "\n\nSaturation Point Behavior:\n";

    const double saturationPoint = solver_->getSaturationPoint();
    std::cout << "  Saturation point (b): " << saturationPoint << " (" << linearToDB(saturationPoint) << " dBFS)\n";

    // At saturation point, output should be exactly 1.0
    double outputAtSat = solver_->process(saturationPoint);
    std::cout << "  Output at b: " << outputAtSat << "\n";
    EXPECT_DOUBLE_EQ(outputAtSat, 1.0) << "Output should be exactly 1.0 at saturation point";

    // Beyond saturation point, output should remain at 1.0
    double outputBeyond = solver_->process(saturationPoint * 1.5);
    EXPECT_DOUBLE_EQ(outputBeyond, 1.0) << "Output should remain at 1.0 beyond saturation";

    // At 0 dBFS (input = 1.0), output should be less than 1.0 (headroom creation)
    double outputAtUnity = solver_->process(1.0);
    std::cout << "  Output at 0 dBFS: " << outputAtUnity << " (" << linearToDB(outputAtUnity) << " dBFS)\n";
    std::cout << "  Headroom created: " << -linearToDB(outputAtUnity) << " dB\n";
    EXPECT_LT(outputAtUnity, 1.0) << "Output at 0 dBFS should be less than 1.0 (creates headroom)";
}

// Test 9: Parameter sweep for tuning
TEST_F(SoftClippingComparisonTest, ParameterSweep) {
    std::cout << "\n\n";
    std::cout << "============================================================\n";
    std::cout << "PARAMETER SWEEP (soft clip vs hard clip)\n";
    std::cout << "============================================================\n";

    const double testLevelDB = 3.0; // +3 dBFS input
    double amplitude = dBToLinear(testLevelDB);

    auto input = generateCoherentSine(kFFTSize, kFundamentalBin, amplitude);
    auto hardClipped = applyHardClip(input);
    auto hardMags = computeMagnitudeSpectrum(hardClipped);
    auto hardAnalysis = analyzeHarmonics(hardMags, kFundamentalBin);

    std::cout << "\nAt +3 dBFS input:\n";
    std::cout << "Hard Clip Reference - THD: " << std::setprecision(2) << hardAnalysis.thdDB
              << " dB, HF: " << hardAnalysis.highFreqEnergyDB << " dB\n\n";

    std::cout << std::setw(10) << "a param" << std::setw(12) << "Knee(dBFS)" << std::setw(12) << "Sat(dBFS)"
              << std::setw(12) << "THD(dB)" << std::setw(12) << "HF(dB)" << std::setw(12) << "HF Reduc.\n";
    std::cout << std::string(70, '-') << "\n";

    std::vector<double> aValues = {0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98};
    for (double a : aValues) {
        SoftClippingSolver solver(a);

        std::vector<double> clipped(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            clipped[i] = solver.process(input[i]);
        }

        auto mags = computeMagnitudeSpectrum(clipped);
        auto analysis = analyzeHarmonics(mags, kFundamentalBin);

        double kneeDB = linearToDB(a);
        double satDB = linearToDB(solver.getSaturationPoint());
        double hfReduction = hardAnalysis.highFreqEnergyDB - analysis.highFreqEnergyDB;

        std::cout << std::fixed << std::setprecision(4) << std::setw(10) << a << std::setprecision(2) << std::setw(12)
                  << kneeDB << std::setw(12) << satDB << std::setw(12) << analysis.thdDB << std::setw(12)
                  << analysis.highFreqEnergyDB << std::setw(12) << hfReduction << "\n";
    }
}

// Test 10: Multi-level comparison summary
TEST_F(SoftClippingComparisonTest, MultiLevelComparison) {
    std::cout << "\n\n";
    std::cout << "============================================================\n";
    std::cout << "SOFT CLIP vs HARD CLIP: MULTI-LEVEL COMPARISON\n";
    std::cout << "============================================================\n";

    std::vector<double> levels = {-6.0, -3.0, -0.5, 0.0, 1.0, 3.0, 6.0};

    std::cout << "\n"
              << std::setw(12) << "Level(dBFS)" << std::setw(15) << "Hard HF(dB)" << std::setw(15) << "Soft HF(dB)"
              << std::setw(15) << "HF Benefit\n";
    std::cout << std::string(57, '-') << "\n";

    for (double level : levels) {
        double amplitude = dBToLinear(level);
        auto input = generateCoherentSine(kFFTSize, kFundamentalBin, amplitude);

        auto hardClipped = applyHardClip(input);
        auto softClipped = applySoftClip(input);

        auto hardMags = computeMagnitudeSpectrum(hardClipped);
        auto softMags = computeMagnitudeSpectrum(softClipped);

        auto hardAnalysis = analyzeHarmonics(hardMags, kFundamentalBin);
        auto softAnalysis = analyzeHarmonics(softMags, kFundamentalBin);

        double benefit = hardAnalysis.highFreqEnergyDB - softAnalysis.highFreqEnergyDB;

        std::cout << std::fixed << std::setprecision(2) << std::setw(12) << level << std::setw(15)
                  << hardAnalysis.highFreqEnergyDB << std::setw(15) << softAnalysis.highFreqEnergyDB << std::setw(15)
                  << (benefit > 0 ? "+" : "") << benefit << "\n";
    }
}

// Test 11: Stage integration test - verify SoftClippingStage works correctly
TEST_F(SoftClippingComparisonTest, StageIntegrationTest) {
    SoftClippingStage stage;
    stage.prepareToPlay(kSampleRate, 512);

    // Create test buffer with +3 dB signal
    juce::AudioBuffer<double> buffer(2, 512);
    double amplitude = dBToLinear(3.0);
    double omega = 2.0 * kPi * 1000.0 / kSampleRate; // 1 kHz

    for (int i = 0; i < 512; ++i) {
        double sample = amplitude * std::sin(omega * i);
        buffer.setSample(0, i, sample);
        buffer.setSample(1, i, sample);
    }

    // Process
    stage.process(buffer);

    // Check output is limited
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 512; ++i) {
            double sample = buffer.getSample(ch, i);
            EXPECT_LE(std::abs(sample), 1.0) << "Soft clipper should limit output to ±1";
        }
    }

    // Check enabled/disabled
    stage.setEnabled(false);
    buffer.clear();
    for (int i = 0; i < 512; ++i) {
        double sample = amplitude * std::sin(omega * i);
        buffer.setSample(0, i, sample);
        buffer.setSample(1, i, sample);
    }

    stage.process(buffer);

    // When disabled, signal should pass through unchanged
    for (int i = 0; i < 512; ++i) {
        double expected = amplitude * std::sin(omega * i);
        EXPECT_DOUBLE_EQ(buffer.getSample(0, i), expected) << "Disabled stage should pass through unchanged";
    }
}
