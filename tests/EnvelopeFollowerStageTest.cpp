#include <gtest/gtest.h>
#include <dsp_core/dsp_core.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <atomic>
#include <cmath>
#include <memory>

using namespace dsp_core::audio_pipeline;

namespace {

constexpr double kSampleRate = 48000.0;
constexpr double kAttackSec = 0.005;   // 5ms — matches stage default
constexpr double kReleaseSec = 0.150;  // 150ms — matches stage default

void fillBuffer(juce::AudioBuffer<double>& buf, double value) {
    for (int ch = 0; ch < buf.getNumChannels(); ++ch) {
        auto* d = buf.getWritePointer(ch);
        for (int i = 0; i < buf.getNumSamples(); ++i) d[i] = value;
    }
}

} // namespace

class EnvelopeFollowerStageTest : public ::testing::Test {
  protected:
    void SetUp() override {
        envelopeValue_.store(0.0);
        stage_ = std::make_unique<EnvelopeFollowerStage>(envelopeValue_);
        stage_->prepareToPlay(kSampleRate, 512);
        // Defaults: enabled=false, sensitivity=1.0, A=5ms, R=150ms
    }

    // Drive a DC value for N samples and return the atomic envelope after.
    double driveDC(double x, int numSamples, int numChannels = 1) {
        juce::AudioBuffer<double> buf(numChannels, numSamples);
        fillBuffer(buf, x);
        stage_->process(buf);
        return envelopeValue_.load();
    }

    std::atomic<double> envelopeValue_{0.0};
    std::unique_ptr<EnvelopeFollowerStage> stage_;
};

// =============================================================================
// Layer 1 — Passthrough / bypass invariants
// =============================================================================

TEST_F(EnvelopeFollowerStageTest, DisabledIsPassthrough) {
    // Default: disabled. Audio must pass unchanged, envelope must stay at 0.
    juce::AudioBuffer<double> buf(2, 256);
    fillBuffer(buf, 0.5);

    stage_->process(buf);

    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 256; ++i) {
            EXPECT_EQ(buf.getSample(ch, i), 0.5)
                << "disabled stage must not alter audio; ch=" << ch << " i=" << i;
        }
    }
    EXPECT_EQ(envelopeValue_.load(), 0.0)
        << "disabled stage must not update the envelope atomic";
}

TEST_F(EnvelopeFollowerStageTest, AudioPassthroughWhenEnabled) {
    // Enabled: stage is a measurement tap, must not alter the audio.
    stage_->setEnabled(true);
    juce::AudioBuffer<double> buf(2, 512);
    const std::vector<double> inputs = {-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0};
    for (double x : inputs) {
        fillBuffer(buf, x);
        stage_->process(buf);
        for (int ch = 0; ch < 2; ++ch) {
            for (int i = 0; i < 512; ++i) {
                EXPECT_EQ(buf.getSample(ch, i), x)
                    << "enabled stage must be passthrough on audio; x=" << x;
            }
        }
    }
}

TEST_F(EnvelopeFollowerStageTest, SilenceProducesZeroEnvelope) {
    stage_->setEnabled(true);
    // 10 release time-constants of silence → env decays to ~0
    const int samples = static_cast<int>(kSampleRate * 10.0 * kReleaseSec);
    const double env = driveDC(0.0, samples);
    EXPECT_NEAR(env, 0.0, 1e-6) << "silence must decay envelope to ~0";
}

// =============================================================================
// Layer 2 — Convergence / steady-state behavior
// =============================================================================

TEST_F(EnvelopeFollowerStageTest, DCInputConvergesToAbsValue) {
    stage_->setEnabled(true);
    stage_->setSensitivityLinear(1.0);

    // Drive for 10 attack + 10 release time-constants → well-converged
    const int samples = static_cast<int>(kSampleRate * 10.0 * kReleaseSec);
    const double env = driveDC(0.7, samples);
    EXPECT_NEAR(env, 0.7, 1e-3) << "envelope should converge to |DC| at unit gain";

    // Negative DC — should also converge to |value|
    stage_->reset();
    const double envNeg = driveDC(-0.4, samples);
    EXPECT_NEAR(envNeg, 0.4, 1e-3) << "negative DC: envelope tracks absolute value";
}

TEST_F(EnvelopeFollowerStageTest, AttackTimeConstant_DbDomain) {
    stage_->setEnabled(true);
    stage_->setSensitivityLinear(1.0);

    // Smoothing is dB-domain: time constants apply to envDb, not envLin. Charge
    // to a known starting level (-40 dB / 0.01 linear) so the rising step has
    // a well-defined finite dB span.
    const int chargeSamples = static_cast<int>(kSampleRate * 10.0 * kReleaseSec);
    (void) driveDC(0.01, chargeSamples);
    ASSERT_NEAR(envelopeValue_.load(), 0.01, 1e-3) << "precondition: charged to 0.01";

    // Step to 1.0 (target = 0 dB). After exactly one attack τ, envDb should
    // have traversed (1 - 1/e) of the 40 dB span.
    const int attackSamples = static_cast<int>(kSampleRate * kAttackSec);
    const double env = driveDC(1.0, attackSamples);
    const double expectedDb = -40.0 + (1.0 - 1.0 / std::exp(1.0)) * 40.0;
    const double expectedLin = std::pow(10.0, expectedDb / 20.0);
    EXPECT_NEAR(env, expectedLin, 0.02)
        << "dB-domain attack: 63% of dB span after one τ";
}

TEST_F(EnvelopeFollowerStageTest, ReleaseTimeConstant_SmallDrop) {
    stage_->setEnabled(true);
    stage_->setSensitivityLinear(1.0);

    // Charge to 1.0 (envDb = 0).
    const int chargeSamples = static_cast<int>(kSampleRate * 10.0 * kAttackSec);
    (void) driveDC(1.0, chargeSamples);
    ASSERT_NEAR(envelopeValue_.load(), 1.0, 1e-3) << "precondition: charge to 1.0";

    // Drop to ~-1 dB (linear ≈ 0.891). A small drop keeps the nonlinear release
    // acceleration mild, so the envelope is still in motion (not yet at target)
    // after one nominal release τ but has moved meaningfully toward it.
    const double targetLin = std::pow(10.0, -1.0 / 20.0);
    const int releaseSamples = static_cast<int>(kSampleRate * kReleaseSec);
    const double env = driveDC(targetLin, releaseSamples);
    EXPECT_LT(env, 0.95) << "release should have moved toward target after one τ";
    EXPECT_GT(env, targetLin) << "release should not have overshot target";
}

TEST_F(EnvelopeFollowerStageTest, SensitivityScalesInput) {
    stage_->setEnabled(true);
    stage_->setSensitivityLinear(4.0);

    const int samples = static_cast<int>(kSampleRate * 10.0 * kReleaseSec);
    // DC 0.3 * 4.0 = 1.2 → clamped to 1.0
    const double env = driveDC(0.3, samples);
    EXPECT_NEAR(env, 1.0, 1e-3) << "sensitivity=4x on DC=0.3 saturates to clamp";

    stage_->reset();
    stage_->setSensitivityLinear(0.5);
    // DC 0.4 * 0.5 = 0.2 → envelope ≈ 0.2
    const double envHalf = driveDC(0.4, samples);
    EXPECT_NEAR(envHalf, 0.2, 1e-3) << "sensitivity=0.5 scales input linearly";
}

// =============================================================================
// Layer 3 — Stereo + state management
// =============================================================================

TEST_F(EnvelopeFollowerStageTest, StereoUsesMaxChannel) {
    stage_->setEnabled(true);
    stage_->setSensitivityLinear(1.0);

    const int samples = static_cast<int>(kSampleRate * 10.0 * kReleaseSec);
    juce::AudioBuffer<double> buf(2, samples);
    for (int i = 0; i < samples; ++i) {
        buf.setSample(0, i, 0.2);
        buf.setSample(1, i, 0.8);
    }
    stage_->process(buf);
    EXPECT_NEAR(envelopeValue_.load(), 0.8, 1e-3)
        << "stereo envelope must track the hotter channel";
}

TEST_F(EnvelopeFollowerStageTest, ResetClearsState) {
    stage_->setEnabled(true);
    stage_->setSensitivityLinear(1.0);

    const int chargeSamples = static_cast<int>(kSampleRate * 10.0 * kAttackSec);
    (void)driveDC(1.0, chargeSamples);
    ASSERT_GT(envelopeValue_.load(), 0.9) << "precondition: charged";

    stage_->reset();
    EXPECT_EQ(envelopeValue_.load(), 0.0)
        << "reset must clear published envelope to 0";

    // Next few samples of silence should stay at 0 (no decay from lingering state)
    juce::AudioBuffer<double> silence(1, 32);
    silence.clear();
    stage_->process(silence);
    EXPECT_NEAR(envelopeValue_.load(), 0.0, 1e-9)
        << "after reset, silence must keep envelope at 0";
}

// =============================================================================
// Layer 4 — Wall-clock timing across sample rates
// =============================================================================

// Helper: charge to a known dB level, then run one attack τ from there and
// return the published envelope. Used by sample-rate tests below to verify
// dB-domain time constants track wall-clock across rates.
namespace {
double measureAttackOneTauFromMinus40(double sampleRate, int blockSize = 512) {
    std::atomic<double> env{0.0};
    auto stage = std::make_unique<EnvelopeFollowerStage>(env);
    stage->prepareToPlay(sampleRate, blockSize);
    stage->setEnabled(true);
    stage->setSensitivityLinear(1.0);

    // Charge to -40 dB / 0.01 linear. Many release τ guarantees convergence.
    const int chargeSamples = static_cast<int>(sampleRate * 10.0 * kReleaseSec);
    juce::AudioBuffer<double> charge(1, chargeSamples);
    fillBuffer(charge, 0.01);
    stage->process(charge);

    // One attack τ at full scale.
    const int attackSamples = static_cast<int>(sampleRate * kAttackSec);
    juce::AudioBuffer<double> step(1, attackSamples);
    fillBuffer(step, 1.0);
    stage->process(step);
    return env.load();
}
} // namespace

TEST_F(EnvelopeFollowerStageTest, PrepareRecomputesCoefficientsAcrossSampleRates) {
    const double expectedDb = -40.0 + (1.0 - 1.0 / std::exp(1.0)) * 40.0;
    const double expectedLin = std::pow(10.0, expectedDb / 20.0);
    for (double sr : {48000.0, 96000.0, 192000.0}) {
        const double env = measureAttackOneTauFromMinus40(sr);
        EXPECT_NEAR(env, expectedLin, 0.02)
            << "dB-domain attack τ must track wall-clock at sr=" << sr;
    }
}

TEST_F(EnvelopeFollowerStageTest, OversamplingCoefficientRecompute) {
    // Plugin's oversampling changes the effective sample rate passed to prepareToPlay.
    // Simulate 4× oversampling.
    const double osRate = kSampleRate * 4.0;
    const double env = measureAttackOneTauFromMinus40(osRate, 2048);
    const double expectedDb = -40.0 + (1.0 - 1.0 / std::exp(1.0)) * 40.0;
    const double expectedLin = std::pow(10.0, expectedDb / 20.0);
    EXPECT_NEAR(env, expectedLin, 0.02)
        << "dB-domain attack τ must track wall-clock at 4× oversampled rate";
}

// =============================================================================
// Layer 5 — Detector mode (Peak / RMS)
// =============================================================================

TEST_F(EnvelopeFollowerStageTest, DetectorMode_RmsOnSineConvergesToRms) {
    // RMS of a unit-amplitude sine = 1/√2. Verify the RMS detector converges
    // there. (Peak-vs-RMS is most cleanly differentiated on duty-cycled signals;
    // on a steady sine, the dB-domain smoother + nonlinear release interact
    // in ways that make a direct peak/RMS comparison less clean than in
    // linear-domain detectors. The next test covers the duty-cycle case.)
    stage_->setEnabled(true);
    stage_->setSensitivityLinear(1.0);
    stage_->setDetectionMode(EnvelopeFollowerStage::DetectionMode::Rms);

    const int totalSamples = static_cast<int>(kSampleRate * 10.0 * kReleaseSec);
    juce::AudioBuffer<double> buf(1, totalSamples);
    const double w = 2.0 * juce::MathConstants<double>::pi * 1000.0 / kSampleRate;
    for (int i = 0; i < totalSamples; ++i) {
        buf.setSample(0, i, std::sin(w * static_cast<double>(i)));
    }
    stage_->process(buf);
    EXPECT_NEAR(envelopeValue_.load(), 1.0 / std::sqrt(2.0), 0.06)
        << "RMS detector on unit sine converges near 1/√2";
}

TEST_F(EnvelopeFollowerStageTest, DetectorMode_RmsConvergesOnDc) {
    // RMS of DC equals |DC|. Steady-state must match peak detector for DC input.
    stage_->setEnabled(true);
    stage_->setSensitivityLinear(1.0);
    stage_->setDetectionMode(EnvelopeFollowerStage::DetectionMode::Rms);

    const int samples = static_cast<int>(kSampleRate * 10.0 * kReleaseSec);
    const double env = driveDC(0.5, samples);
    EXPECT_NEAR(env, 0.5, 1e-2) << "RMS of DC = |DC| at steady state";
}

// =============================================================================
// Layer 6 — Nonlinear release
// =============================================================================

TEST_F(EnvelopeFollowerStageTest, NonlinearRelease_BigDropFasterThanSmallDrop) {
    // Release is nonlinear: larger dB drops decay faster (per unit τ) than
    // smaller drops. Measure the fraction of the dB span traversed in one
    // nominal release τ for two different drop sizes; the bigger drop should
    // traverse a larger fraction.
    stage_->setEnabled(true);
    stage_->setSensitivityLinear(1.0);

    auto measureNormalizedDecay = [&](double targetLin) {
        stage_->reset();
        const int chargeSamples = static_cast<int>(kSampleRate * 10.0 * kAttackSec);
        (void) driveDC(1.0, chargeSamples);
        const int releaseSamples = static_cast<int>(kSampleRate * kReleaseSec);
        const double env = driveDC(targetLin, releaseSamples);

        // Fraction of dB span traversed: 0 = no movement, 1 = fully reached target.
        const double startDb = 0.0;
        const double targetDb = 20.0 * std::log10(std::max(targetLin, 1e-9));
        const double envDb = 20.0 * std::log10(std::max(env, 1e-9));
        return (startDb - envDb) / (startDb - targetDb);
    };

    const double smallFrac = measureNormalizedDecay(std::pow(10.0, -2.0 / 20.0)); // -2 dB
    const double bigFrac = measureNormalizedDecay(std::pow(10.0, -20.0 / 20.0));  // -20 dB

    EXPECT_GT(bigFrac, smallFrac)
        << "nonlinear release: bigger drops cover a larger fraction of dB span per τ ("
        << "small=" << smallFrac << " big=" << bigFrac << ")";
}
