#include <gtest/gtest.h>
#include <dsp_core/dsp_core.h>
#include <cmath>
#include <numeric>
#include <vector>

using dsp_core::HysteresisProcessor;

// =============================================================================
// Test Fixture
// =============================================================================

class HysteresisProcessorTest : public ::testing::Test {
  protected:
    void SetUp() override {
        proc_.prepareToPlay(sampleRate_);
    }

    // Generate one cycle of sine at given frequency
    std::vector<double> generateSine(double freq, double amplitude = 1.0) {
        int numSamples = static_cast<int>(sampleRate_ / freq);
        std::vector<double> signal(numSamples);
        for (int n = 0; n < numSamples; ++n)
            signal[n] = amplitude * std::sin(2.0 * juce::MathConstants<double>::pi * freq * n / sampleRate_);
        return signal;
    }

    // Process a full signal buffer and return output
    std::vector<double> processSignal(const std::vector<double>& input) {
        std::vector<double> output(input.size());
        for (size_t n = 0; n < input.size(); ++n)
            output[n] = proc_.process(input[n]);
        return output;
    }

    // Compute hysteresis loop area using shoelace formula (H vs M)
    double computeLoopArea(const std::vector<double>& H, const std::vector<double>& M) {
        double area = 0.0;
        int n = static_cast<int>(H.size());
        for (int i = 0; i < n; ++i) {
            int j = (i + 1) % n;
            area += H[i] * M[j] - H[j] * M[i];
        }
        return std::abs(area) / 2.0;
    }

    // Check that all samples are finite and bounded
    bool allFiniteAndBounded(const std::vector<double>& signal, double bound = 2.0) {
        for (double s : signal) {
            if (!std::isfinite(s) || std::abs(s) > bound)
                return false;
        }
        return true;
    }

    static constexpr double sampleRate_ = 48000.0 * 16.0;  // 768 kHz (oversampled)
    HysteresisProcessor proc_;
};

// =============================================================================
// Layer 1: Math Primitives
// =============================================================================

TEST_F(HysteresisProcessorTest, Langevin_ZeroInput_ReturnsZero) {
    // Standard Langevin: L(0) = 0 (odd symmetry at origin)
    EXPECT_DOUBLE_EQ(proc_.langevin(0.0), 0.0);
}

TEST_F(HysteresisProcessorTest, Langevin_LargePositive_ApproachesOne) {
    // L(x) → 1 as x → +∞
    double result = proc_.langevin(100.0);
    EXPECT_GE(result, 0.99);
    EXPECT_LE(result, 1.0);
}

TEST_F(HysteresisProcessorTest, Langevin_LargeNegative_ApproachesNegativeOne) {
    // L(x) → -1 as x → -∞ (odd symmetry)
    double result = proc_.langevin(-100.0);
    EXPECT_LE(result, -0.99);
    EXPECT_GE(result, -1.0);
}

TEST_F(HysteresisProcessorTest, Langevin_KnownValue_MatchesAnalytical) {
    // L(1.0) = coth(1) - 1 = 1/tanh(1) - 1 ≈ 0.3103
    double expected = 1.0 / std::tanh(1.0) - 1.0;
    EXPECT_NEAR(proc_.langevin(1.0), expected, 1e-10);
}

TEST_F(HysteresisProcessorTest, Langevin_OddSymmetry) {
    // L(-x) = -L(x) for various x
    for (double x : {0.1, 0.5, 1.0, 2.0, 5.0}) {
        EXPECT_NEAR(proc_.langevin(-x), -proc_.langevin(x), 1e-14)
            << "Odd symmetry violated at x=" << x;
    }
}

TEST_F(HysteresisProcessorTest, LangevinDeriv_AtZero_ReturnsOneThird) {
    // L'(0) = 1/3 (Taylor expansion)
    EXPECT_NEAR(proc_.langevinDeriv(0.0), 1.0 / 3.0, 1e-10);
}

TEST_F(HysteresisProcessorTest, LangevinDeriv_MatchesFiniteDifference) {
    // L'(x) should match (L(x+h) - L(x-h)) / (2h) within tolerance
    constexpr double h = 1e-6;
    for (double x : {0.5, 1.0, 2.0, 5.0}) {
        double analytical = proc_.langevinDeriv(x);
        double numerical = (proc_.langevin(x + h) - proc_.langevin(x - h)) / (2.0 * h);
        EXPECT_NEAR(analytical, numerical, 1e-6)
            << "Derivative mismatch at x=" << x;
    }
}

TEST_F(HysteresisProcessorTest, CustomNL_ReplacesLangevin) {
    // When we inject tanh as the NL, langevin should return tanh(Q_mapped)
    proc_.setNonlinearity([](double x) { return std::tanh(x); });

    // The custom NL replaces the entire Langevin function
    // langevin(Q) should now apply input mapping then tanh
    // Just verify it's different from standard Langevin
    double standardResult = 1.0 / std::tanh(1.0) - 1.0;  // L(1.0) ≈ 0.3103
    double customResult = proc_.langevin(1.0);
    // tanh-based result will differ from coth-1/x result
    EXPECT_NE(customResult, standardResult);
}

// =============================================================================
// Layer 2: ODE Function
// =============================================================================

TEST_F(HysteresisProcessorTest, DMdt_ReturnsFinite_ForAllParameterExtremes) {
    // dMdt should return finite values even at parameter boundaries
    for (double drive : {0.0, 0.5, 1.0}) {
        for (double sat : {0.0, 0.5, 1.0}) {
            for (double width : {0.0, 0.5, 1.0}) {
                proc_.setDrive(drive);
                proc_.setSaturation(sat);
                proc_.setWidth(width);
                double result = proc_.dMdt(0.5, 0.5, 1000.0);
                EXPECT_TRUE(std::isfinite(result))
                    << "dMdt not finite at drive=" << drive
                    << " sat=" << sat << " width=" << width;
            }
        }
    }
}

TEST_F(HysteresisProcessorTest, DMdt_SignFollowsFieldDirection) {
    // When field is increasing (H_d > 0) and M < M_anhysteretic,
    // dM/dt should generally be positive (magnetization increasing toward equilibrium)
    double result_positive_Hd = proc_.dMdt(0.0, 1.0, 10000.0);
    double result_negative_Hd = proc_.dMdt(0.0, 1.0, -10000.0);
    // They should differ in sign or magnitude
    EXPECT_NE(std::signbit(result_positive_Hd), std::signbit(result_negative_Hd));
}

// =============================================================================
// Layer 3: Solver Integration
// =============================================================================

TEST_F(HysteresisProcessorTest, RK4_ZeroInput_ReturnsZero) {
    // Processing silence should produce silence
    for (int i = 0; i < 100; ++i) {
        double out = proc_.process(0.0);
        EXPECT_NEAR(out, 0.0, 1e-10) << "Non-zero output at sample " << i;
    }
}

TEST_F(HysteresisProcessorTest, RK4_ConstantInput_ConvergesToSteadyState) {
    // A constant input should eventually settle
    double lastOutput = 0.0;
    for (int i = 0; i < 10000; ++i) {
        lastOutput = proc_.process(0.5);
    }
    // After many samples, output should be stable
    double nextOutput = proc_.process(0.5);
    EXPECT_NEAR(nextOutput, lastOutput, 1e-6);
    EXPECT_TRUE(std::isfinite(lastOutput));
}

TEST_F(HysteresisProcessorTest, SineInput_ProducesHysteresisLoop) {
    // Core hysteresis test: sine input should create a loop (different path up vs down)
    auto H = generateSine(100.0);
    auto M = processSignal(H);

    double area = computeLoopArea(H, M);
    EXPECT_GT(area, 0.0) << "Hysteresis loop has zero area — no memory effect";
    EXPECT_TRUE(allFiniteAndBounded(M)) << "Output contains non-finite or unbounded values";
}

TEST_F(HysteresisProcessorTest, CustomNL_ProducesDifferentLoop) {
    // Hard clip NL should produce different hysteresis shape than standard Langevin
    auto H = generateSine(100.0);

    // Standard Langevin
    auto M_standard = processSignal(H);
    double area_standard = computeLoopArea(H, M_standard);

    // Reset and use hard clip NL
    proc_.reset();
    proc_.setNonlinearity([](double x) {
        return std::max(-1.0, std::min(1.0, x * 3.0));  // hard clip at ±1/3
    });
    auto M_custom = processSignal(H);
    double area_custom = computeLoopArea(H, M_custom);

    // Both should produce valid loops but with different characteristics
    EXPECT_GT(area_standard, 0.0);
    EXPECT_GT(area_custom, 0.0);
    EXPECT_NE(area_standard, area_custom) << "Custom NL should produce different loop than standard";
}

TEST_F(HysteresisProcessorTest, GoldenTest_MatchesPythonReference) {
    // First 20 reference samples from generate_reference.py
    // Uses same normalized parameters and alpha-transform derivative
    static const double kReferenceM[] = {
        0.000000000000000000e+00,
        3.100352212472069248e-04,
        6.991803963515725315e-04,
        1.032774984958771203e-03,
        1.412482992802132666e-03,
        1.761528288336377613e-03,
        2.137858763332708967e-03,
        2.497711381241880628e-03,
        2.874105255052884257e-03,
        3.242100124941552693e-03,
        3.620503745947429870e-03,
        3.995104833193729478e-03,
        4.376611810808931149e-03,
        4.756927232181540691e-03,
        5.142144801541461210e-03,
        5.527650082785785443e-03,
        5.916908161629243212e-03,
        6.307288341513762886e-03,
        6.700758787364844565e-03,
        7.095818358000376136e-03,
    };

    // Generate same 100 Hz sine at 768 kHz
    double freq = 100.0;
    for (int n = 0; n < 20; ++n) {
        double H = std::sin(2.0 * juce::MathConstants<double>::pi * freq * n / sampleRate_);
        double M = proc_.process(H);
        EXPECT_NEAR(M, kReferenceM[n], 1e-8)
            << "Golden test mismatch at sample " << n
            << ": expected " << kReferenceM[n] << " got " << M;
    }
}

// =============================================================================
// Layer 4: Safety (core correctness)
// =============================================================================

TEST_F(HysteresisProcessorTest, NaNInput_ReturnsZero_ResetsState) {
    // Process some valid samples first to build up state
    for (int i = 0; i < 100; ++i)
        proc_.process(std::sin(0.01 * i));

    // Feed NaN
    double out = proc_.process(std::numeric_limits<double>::quiet_NaN());
    EXPECT_EQ(out, 0.0) << "NaN input should produce zero output";

    // After reset, zero input should produce zero
    out = proc_.process(0.0);
    EXPECT_NEAR(out, 0.0, 1e-10);
}

TEST_F(HysteresisProcessorTest, InfInput_ReturnsZero_ResetsState) {
    for (int i = 0; i < 100; ++i)
        proc_.process(std::sin(0.01 * i));

    double out = proc_.process(std::numeric_limits<double>::infinity());
    EXPECT_EQ(out, 0.0);

    out = proc_.process(0.0);
    EXPECT_NEAR(out, 0.0, 1e-10);
}

TEST_F(HysteresisProcessorTest, StepFunctionNL_BoundedOutput) {
    // Step function: extreme non-smooth NL
    proc_.setNonlinearity([](double x) {
        return (x >= 0.0) ? 1.0 : -1.0;
    });

    auto H = generateSine(100.0);
    auto M = processSignal(H);
    EXPECT_TRUE(allFiniteAndBounded(M, 2.0))
        << "Step function NL produced unbounded output";
}

TEST_F(HysteresisProcessorTest, NonMonotonicNL_BoundedOutput) {
    // sin(x) is non-monotonic — can cause ODE divergence
    proc_.setNonlinearity([](double x) {
        return std::sin(x * juce::MathConstants<double>::pi);
    });

    auto H = generateSine(100.0);
    auto M = processSignal(H);
    EXPECT_TRUE(allFiniteAndBounded(M, 2.0))
        << "Non-monotonic NL produced unbounded output";
}

TEST_F(HysteresisProcessorTest, AsymmetricNL_DCBlocked) {
    // Asymmetric NL: f(x) = x + 0.1 (biased)
    proc_.setNonlinearity([](double x) {
        return std::max(-1.0, std::min(1.0, x + 0.1));
    });

    // Process multiple cycles to let DC accumulate
    auto H = generateSine(100.0);
    std::vector<double> allOutput;
    for (int cycle = 0; cycle < 10; ++cycle) {
        auto M = processSignal(H);
        allOutput.insert(allOutput.end(), M.begin(), M.end());
    }

    // Check that output mean is reasonable (DC blocker should prevent runaway)
    double mean = std::accumulate(allOutput.begin(), allOutput.end(), 0.0) / allOutput.size();
    EXPECT_LT(std::abs(mean), 0.5) << "Asymmetric NL caused excessive DC bias: mean=" << mean;
    EXPECT_TRUE(allFiniteAndBounded(allOutput, 2.0));
}

TEST_F(HysteresisProcessorTest, ConstantNL_BoundedOutput) {
    // f(x) = 0.5 for all x (degenerate case)
    proc_.setNonlinearity([](double) { return 0.5; });

    auto H = generateSine(100.0);
    auto M = processSignal(H);
    EXPECT_TRUE(allFiniteAndBounded(M, 2.0))
        << "Constant NL produced unbounded output";
}

TEST_F(HysteresisProcessorTest, IdentityNL_BoundedOutput) {
    // f(x) = x (no distortion, just pass-through)
    proc_.setNonlinearity([](double x) { return x; });

    auto H = generateSine(100.0);
    auto M = processSignal(H);
    EXPECT_TRUE(allFiniteAndBounded(M, 2.0))
        << "Identity NL produced unbounded output";
}

TEST_F(HysteresisProcessorTest, ConsecutiveResets_MutesOutput) {
    // If solver resets > 10 times consecutively, output should be muted
    // Create a pathological NL that always produces NaN
    proc_.setNonlinearity([](double) {
        return std::numeric_limits<double>::quiet_NaN();
    });

    int zeroCount = 0;
    for (int i = 0; i < 200; ++i) {
        double out = proc_.process(std::sin(0.1 * i));
        if (out == 0.0) zeroCount++;
    }

    // Most output should be zero (muted)
    EXPECT_GT(zeroCount, 180) << "Should be mostly muted after consecutive resets";
}

TEST_F(HysteresisProcessorTest, OutputNeverExceedsBound) {
    // Sweep various NLs and verify output clamp
    auto H = generateSine(100.0, 5.0);  // large amplitude

    auto M = processSignal(H);
    for (size_t n = 0; n < M.size(); ++n) {
        EXPECT_LE(std::abs(M[n]), 2.0)
            << "Output exceeded ±2.0 at sample " << n << ": " << M[n];
    }
}

// =============================================================================
// Layer 5: Parameters & State
// =============================================================================

TEST_F(HysteresisProcessorTest, ParameterExtremes_ProduceFiniteOutput) {
    auto H = generateSine(100.0);

    for (double drive : {0.0, 1.0}) {
        for (double sat : {0.0, 1.0}) {
            for (double width : {0.0, 1.0}) {
                proc_.reset();
                proc_.setDrive(drive);
                proc_.setSaturation(sat);
                proc_.setWidth(width);

                auto M = processSignal(H);
                EXPECT_TRUE(allFiniteAndBounded(M))
                    << "Non-finite output at drive=" << drive
                    << " sat=" << sat << " width=" << width;
            }
        }
    }
}

TEST_F(HysteresisProcessorTest, Reset_ZerosAllState) {
    // Build up state
    for (int i = 0; i < 1000; ++i)
        proc_.process(std::sin(0.01 * i));

    proc_.reset();

    // After reset, zero input should produce zero output
    for (int i = 0; i < 10; ++i) {
        EXPECT_NEAR(proc_.process(0.0), 0.0, 1e-15)
            << "State not fully reset at sample " << i;
    }
}

TEST_F(HysteresisProcessorTest, DifferentSampleRates_SimilarLoopShape) {
    // Same frequency should produce similar hysteresis loops at different sample rates
    double freq = 100.0;

    // Process at 768 kHz (default)
    auto H1 = generateSine(freq);
    auto M1 = processSignal(H1);
    double area1 = computeLoopArea(H1, M1);

    // Process at 44100 * 16 = 705600 Hz
    HysteresisProcessor proc2;
    proc2.prepareToPlay(44100.0 * 16.0);
    int numSamples2 = static_cast<int>(44100.0 * 16.0 / freq);
    std::vector<double> H2(numSamples2), M2(numSamples2);
    for (int n = 0; n < numSamples2; ++n) {
        H2[n] = std::sin(2.0 * juce::MathConstants<double>::pi * freq * n / (44100.0 * 16.0));
        M2[n] = proc2.process(H2[n]);
    }
    double area2 = computeLoopArea(H2, M2);

    // Loop areas should be in the same ballpark (within 50%)
    EXPECT_GT(area1, 0.0);
    EXPECT_GT(area2, 0.0);
    double ratio = area1 / area2;
    EXPECT_GT(ratio, 0.5) << "Loop areas differ too much between sample rates";
    EXPECT_LT(ratio, 2.0) << "Loop areas differ too much between sample rates";
}

TEST_F(HysteresisProcessorTest, PrepareToPlay_UpdatesTimingConstants) {
    // Changing sample rate should update internal timing
    proc_.prepareToPlay(96000.0 * 16.0);

    // Should still produce valid output
    auto H = generateSine(100.0);
    // Regenerate for new sample rate
    int numSamples = static_cast<int>(96000.0 * 16.0 / 100.0);
    std::vector<double> signal(numSamples);
    for (int n = 0; n < numSamples; ++n)
        signal[n] = std::sin(2.0 * juce::MathConstants<double>::pi * 100.0 * n / (96000.0 * 16.0));

    auto M = processSignal(signal);
    EXPECT_TRUE(allFiniteAndBounded(M));
}

// =============================================================================
// Stress Tests
// =============================================================================

TEST_F(HysteresisProcessorTest, Stress_ImpulseInput_NoCrash) {
    // Sudden loud input after silence
    for (int i = 0; i < 100; ++i)
        proc_.process(0.0);

    // Impulse
    double out = proc_.process(1.0);
    EXPECT_TRUE(std::isfinite(out));
    EXPECT_LE(std::abs(out), 2.0);

    // Continue processing
    for (int i = 0; i < 1000; ++i) {
        out = proc_.process(0.0);
        EXPECT_TRUE(std::isfinite(out)) << "Non-finite after impulse at sample " << i;
    }
}

TEST_F(HysteresisProcessorTest, Stress_RapidParameterSweep_Stable) {
    // Change parameters every sample
    auto H = generateSine(100.0);
    for (size_t n = 0; n < H.size(); ++n) {
        double t = static_cast<double>(n) / H.size();
        proc_.setDrive(t);
        proc_.setSaturation(1.0 - t);
        proc_.setWidth(std::abs(std::sin(t * 10.0)));

        double out = proc_.process(H[n]);
        EXPECT_TRUE(std::isfinite(out))
            << "Non-finite output during parameter sweep at sample " << n;
        EXPECT_LE(std::abs(out), 2.0)
            << "Unbounded output during parameter sweep at sample " << n;
    }
}
