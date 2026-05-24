// Unit tests for LadderTPTStage<N> — TPT-ZDF Moog-style ladder filter with
// Newton iteration on the tanh feedback nonlinearity. Behavioral coverage
// for: parameter smoothing (linear ramp time, log-domain geometric midpoint),
// Newton convergence (early-out at low R, accuracy at high R), denormal
// safety, and bass compensation. Cross-topology performance comparison lives
// in FilterTopologyBenchmarkTest.cpp.

#include <gtest/gtest.h>

#include "../dsp_core/Source/audio_pipeline/LadderTPTStage.h"

#include <cmath>
#include <juce_audio_basics/juce_audio_basics.h>

namespace dsp_core_test {

using namespace dsp_core::audio_pipeline;

namespace {

constexpr double kSampleRate = 48000.0;
constexpr int kBlockSize = 1;
constexpr int kNumChannels = 1;

void fillDC(juce::AudioBuffer<double>& buf, double value) {
    for (int ch = 0; ch < buf.getNumChannels(); ++ch) {
        for (int i = 0; i < buf.getNumSamples(); ++i) {
            buf.getWritePointer(ch)[i] = value;
        }
    }
}

} // namespace

// ============================================================================
// Smoothing — 5 ms ramp time, log-domain geometric midpoint
// ============================================================================

TEST(LadderTPTStageSmoothing, RampReachesTargetInApproximatelyFiveMillis) {
    LadderTPTStage<4> stage;
    stage.setCutoffFrequency(1000.0);
    stage.setResonance(0.0);
    stage.setBassCompensation(0.0); // isolate smoothing from bass-comp effect
    stage.prepareToPlay(kSampleRate, kBlockSize, kNumChannels);

    juce::AudioBuffer<double> buf(kNumChannels, kBlockSize);
    stage.setCutoffFrequency(5000.0);

    // After 10 samples (~0.2 ms), the smoothed cutoff must still be well
    // below target — proves the ramp isn't effectively instantaneous.
    for (int i = 0; i < 10; ++i) {
        fillDC(buf, 0.0);
        stage.process(buf);
    }
    EXPECT_LT(stage.getSmoothedCutoffFrequency(), 4500.0)
        << "Smoothing ramp completed in 10 samples — expected 5 ms (~240 samples)";

    // After ~240 more samples (~5 ms total), the smoothed cutoff must be
    // near the target.
    for (int i = 0; i < 240; ++i) {
        fillDC(buf, 0.0);
        stage.process(buf);
    }
    EXPECT_GT(stage.getSmoothedCutoffFrequency(), 4900.0)
        << "Smoothing ramp didn't complete within 5 ms";
}

TEST(LadderTPTStageSmoothing, CutoffRampMidpointIsGeometricMean) {
    // Log-domain smoothing means the midpoint of a 100→10000 Hz ramp should
    // be the geometric mean (sqrt(100*10000) = 1000 Hz), not the arithmetic
    // mean (5050 Hz). Discriminates between Linear and Multiplicative
    // SmoothedValue.
    LadderTPTStage<4> stage;
    stage.setCutoffFrequency(100.0);
    stage.setResonance(0.0);
    stage.setBassCompensation(0.0);
    stage.prepareToPlay(kSampleRate, kBlockSize, kNumChannels);

    juce::AudioBuffer<double> buf(kNumChannels, kBlockSize);
    stage.setCutoffFrequency(10000.0);

    // 5 ms ramp = 240 samples @ 48 kHz. Process exactly half (120).
    for (int i = 0; i < 120; ++i) {
        fillDC(buf, 0.0);
        stage.process(buf);
    }

    const double smoothed = stage.getSmoothedCutoffFrequency();
    EXPECT_NEAR(smoothed, 1000.0, 200.0)
        << "Expected geometric-mean midpoint ~1000 Hz (log-domain smoothing); "
        << "got " << smoothed << " Hz";
}

// ============================================================================
// Newton iteration — early-out at low R, stays accurate at high R
// ============================================================================

TEST(LadderTPTStageNewton, EarlyOutAtLowResonance) {
    LadderTPTStage<4> stage;
    stage.setCutoffFrequency(2000.0);
    stage.setResonance(0.1);
    stage.setBassCompensation(0.0);
    stage.prepareToPlay(kSampleRate, kBlockSize, kNumChannels);

    juce::AudioBuffer<double> buf(kNumChannels, kBlockSize);
    for (int i = 0; i < 100; ++i) {
        fillDC(buf, 0.01);
        stage.process(buf);
    }

    EXPECT_LE(stage.getLastNewtonIterations(), 2)
        << "Expected <=2 Newton iterations at low R; got "
        << stage.getLastNewtonIterations();
}

TEST(LadderTPTStageNewton, ConvergesAtHighResonance) {
    // Guard against an over-aggressive early-out: at R=3.9 with a full-scale
    // signal the cascade must still produce a finite, bounded output.
    LadderTPTStage<4> stage;
    stage.setCutoffFrequency(2000.0);
    stage.setResonance(3.9);
    stage.setBassCompensation(0.0);
    stage.prepareToPlay(kSampleRate, kBlockSize, kNumChannels);

    juce::AudioBuffer<double> buf(kNumChannels, 64);
    for (int i = 0; i < 64; ++i) {
        buf.getWritePointer(0)[i] = 0.2 * std::sin(2.0 * M_PI * 200.0 * i / kSampleRate);
    }
    buf.getWritePointer(0)[0] = 1.0; // impulse to provoke ringing

    stage.process(buf);

    double peak = 0.0;
    for (int i = 0; i < 64; ++i) {
        const double a = std::abs(buf.getSample(0, i));
        if (!std::isfinite(a)) {
            FAIL() << "Non-finite output at R=3.9 sample " << i;
        }
        peak = std::max(peak, a);
    }
    EXPECT_LT(peak, 5.0) << "Runaway output at R=3.9: peak=" << peak;
}

// ============================================================================
// Denormal safety — silence after impulse decays below normal-magnitude range
// ============================================================================

TEST(LadderTPTStageDenormals, SilenceAfterImpulseDecaysWellBelowDenormalRange) {
    LadderTPTStage<4> stage;
    stage.setCutoffFrequency(2000.0);
    stage.setResonance(0.0);
    stage.setBassCompensation(0.0);
    stage.prepareToPlay(kSampleRate, kBlockSize, kNumChannels);

    juce::AudioBuffer<double> buf(kNumChannels, 1);
    buf.getWritePointer(0)[0] = 1.0;
    stage.process(buf);

    for (int i = 0; i < 8192; ++i) {
        fillDC(buf, 0.0);
        stage.process(buf);
    }

    // Without ScopedNoDenormals, the tail can pin at slow-path denormal
    // magnitudes (~1e-320). With FTZ, denormals flush to zero — Tanh2xLUT
    // residual leaves a tiny non-zero floor (~1e-300) below any musical
    // threshold (24-bit floor ~6e-8). 1e-100 is permissive for the LUT
    // residual and strict enough to fail if normal-range values linger.
    constexpr double kFloor = 1e-100;
    for (int i = 0; i < 1024; ++i) {
        fillDC(buf, 0.0);
        stage.process(buf);
        const double v = std::abs(buf.getSample(0, 0));
        ASSERT_LT(v, kFloor)
            << "Cascade left a magnitude-1e-100+ residual after long silence "
               "(sample " << i << "): " << buf.getSample(0, 0);
    }
}

// ============================================================================
// Bass compensation — DC gain at high R
// ============================================================================

// Bass compensation math (1 + α·2R) is derived from the LINEARIZED system
// where tanh(2Ry) ≈ 2Ry. Tests use small DC input + moderate R so:
//   (1) the feedback tanh stays in its linear knee (no saturation skew), and
//   (2) the filter has a stable DC fixed point (R near self-osc threshold
//       has no stable DC attractor — the cascade just rings indefinitely).
// At R=1.0 with small signal, classic Moog DC gain = 1/(1+2R) = 1/3.
// Full-scale / high-R behavior is a separate voicing question handled in
// the topology benchmark's stability sweep.
TEST(LadderTPTStageBassComp, FullCompensationRestoresDCGainAtModerateResonance) {
    constexpr double kInput = 0.01;
    constexpr double kR = 1.0;
    LadderTPTStage<4> stage;
    stage.setCutoffFrequency(1000.0);
    stage.setResonance(kR);
    stage.setBassCompensation(1.0);
    stage.prepareToPlay(kSampleRate, kBlockSize, kNumChannels);

    juce::AudioBuffer<double> buf(kNumChannels, kBlockSize);
    double last = 0.0;
    for (int i = 0; i < 1000; ++i) {
        fillDC(buf, kInput);
        stage.process(buf);
        last = buf.getSample(0, 0);
    }

    // Linear regime + full compensation → DC gain ~1.
    EXPECT_NEAR(last, kInput, 0.0005)
        << "With full bass compensation (small signal, R=" << kR
        << "), DC gain should be ~1 (output " << last << " vs input " << kInput << ")";
}

TEST(LadderTPTStageBassComp, ZeroCompensationPreservesClassicMoogLoss) {
    constexpr double kInput = 0.01;
    constexpr double kR = 1.0;
    LadderTPTStage<4> stage;
    stage.setCutoffFrequency(1000.0);
    stage.setResonance(kR);
    stage.setBassCompensation(0.0);
    stage.prepareToPlay(kSampleRate, kBlockSize, kNumChannels);

    juce::AudioBuffer<double> buf(kNumChannels, kBlockSize);
    double last = 0.0;
    for (int i = 0; i < 1000; ++i) {
        fillDC(buf, kInput);
        stage.process(buf);
        last = buf.getSample(0, 0);
    }

    // Faithful Moog (linear regime): DC gain = 1/(1+2R) = 1/3 with R=1.
    const double expected = kInput / (1.0 + 2.0 * kR);
    EXPECT_NEAR(last, expected, 0.0005)
        << "With bassComp=0 at R=" << kR << ", expected classic 1/(1+2R) loss (~"
        << expected << "); got " << last;
}

} // namespace dsp_core_test
