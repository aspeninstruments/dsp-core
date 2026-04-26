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

TEST_F(EnvelopeFollowerStageTest, AttackTimeConstant) {
    stage_->setEnabled(true);
    stage_->setSensitivityLinear(1.0);

    // Step from 0 to 1.0; at t = attackSec, one-pole reaches 1 - 1/e ≈ 0.632
    const int attackSamples = static_cast<int>(kSampleRate * kAttackSec);
    const double env = driveDC(1.0, attackSamples);
    EXPECT_NEAR(env, 1.0 - 1.0 / std::exp(1.0), 0.02)
        << "attack should reach ~63.2% of target after one time-constant";
}

TEST_F(EnvelopeFollowerStageTest, ReleaseTimeConstant) {
    stage_->setEnabled(true);
    stage_->setSensitivityLinear(1.0);

    // Charge to ~1.0
    const int chargeSamples = static_cast<int>(kSampleRate * 10.0 * kAttackSec);
    (void)driveDC(1.0, chargeSamples);
    const double peak = envelopeValue_.load();
    ASSERT_NEAR(peak, 1.0, 1e-3) << "precondition: charge to 1.0";

    // Release with silence for exactly one release time-constant
    const int releaseSamples = static_cast<int>(kSampleRate * kReleaseSec);
    const double env = driveDC(0.0, releaseSamples);
    // env(t=tau) = peak * exp(-1) ≈ 0.368
    EXPECT_NEAR(env, peak / std::exp(1.0), 0.02)
        << "release should decay to ~36.8% after one time-constant";
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

TEST_F(EnvelopeFollowerStageTest, PrepareRecomputesCoefficientsAcrossSampleRates) {
    for (double sr : {48000.0, 96000.0, 192000.0}) {
        std::atomic<double> env{0.0};
        auto stage = std::make_unique<EnvelopeFollowerStage>(env);
        stage->prepareToPlay(sr, 512);
        stage->setEnabled(true);
        stage->setSensitivityLinear(1.0);

        const int attackSamples = static_cast<int>(sr * kAttackSec);
        juce::AudioBuffer<double> buf(1, attackSamples);
        fillBuffer(buf, 1.0);
        stage->process(buf);

        EXPECT_NEAR(env.load(), 1.0 - 1.0 / std::exp(1.0), 0.02)
            << "attack time must track wall-clock at sr=" << sr;
    }
}

TEST_F(EnvelopeFollowerStageTest, OversamplingCoefficientRecompute) {
    // Plugin's oversampling changes the effective sample rate passed to prepareToPlay.
    // Simulate 4× oversampling.
    const double osRate = kSampleRate * 4.0;
    std::atomic<double> env{0.0};
    auto stage = std::make_unique<EnvelopeFollowerStage>(env);
    stage->prepareToPlay(osRate, 2048);
    stage->setEnabled(true);
    stage->setSensitivityLinear(1.0);

    const int attackSamples = static_cast<int>(osRate * kAttackSec);
    juce::AudioBuffer<double> buf(1, attackSamples);
    fillBuffer(buf, 1.0);
    stage->process(buf);

    EXPECT_NEAR(env.load(), 1.0 - 1.0 / std::exp(1.0), 0.02)
        << "attack time must track wall-clock even at 4× oversampled rate";
}
