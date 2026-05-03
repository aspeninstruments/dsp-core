#include <gtest/gtest.h>
#include <dsp_core/dsp_core.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <cmath>
#include <memory>

using namespace dsp_core::audio_pipeline;

// =============================================================================
// Test Fixture
// =============================================================================

class SurgeStageTest : public ::testing::Test {
  protected:
    static constexpr double kSampleRate = 48000.0;
    static constexpr double kSurgeDurationSec = 0.001; // 1ms — matches default param

    void SetUp() override {
        stage_ = std::make_unique<SurgeStage>();
        stage_->prepareToPlay(kSampleRate, 512, 2);
        stage_->setSurgeDurationSec(kSurgeDurationSec);
        stage_->setEnabled(true);
    }

    // Apply stage to a single DC value for `numSamples` samples across `numChannels`,
    // returning the final output sample of channel 0.
    double driveDC(double x, int numSamples, int numChannels = 1) {
        juce::AudioBuffer<double> buf(numChannels, numSamples);
        for (int ch = 0; ch < numChannels; ++ch) {
            auto* data = buf.getWritePointer(ch);
            for (int i = 0; i < numSamples; ++i)
                data[i] = x;
        }
        stage_->process(buf);
        return buf.getSample(0, numSamples - 1);
    }

    std::unique_ptr<SurgeStage> stage_;
};

// =============================================================================
// Layer 1 — In-range passthrough
// =============================================================================

TEST_F(SurgeStageTest, InRangeIsIdentity) {
    juce::AudioBuffer<double> buf(2, 512);
    const std::vector<double> inputs = {-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0};
    for (double x : inputs) {
        for (int ch = 0; ch < 2; ++ch) {
            auto* d = buf.getWritePointer(ch);
            for (int i = 0; i < 512; ++i)
                d[i] = x;
        }
        stage_->process(buf);
        for (int ch = 0; ch < 2; ++ch) {
            for (int i = 0; i < 512; ++i) {
                EXPECT_NEAR(buf.getSample(ch, i), x, 1e-12)
                    << "in-range input must pass unchanged; x=" << x << " ch=" << ch << " i=" << i;
            }
        }
    }
}

// =============================================================================
// Layer 2 — Bypass
// =============================================================================

TEST_F(SurgeStageTest, BypassIsPassthroughEvenForOvershoot) {
    stage_->setEnabled(false);
    juce::AudioBuffer<double> buf(2, 256);
    for (int ch = 0; ch < 2; ++ch) {
        auto* d = buf.getWritePointer(ch);
        for (int i = 0; i < 256; ++i)
            d[i] = 1.5;
    }
    stage_->process(buf);
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 256; ++i) {
            EXPECT_EQ(buf.getSample(ch, i), 1.5) << "bypassed stage must not alter samples; ch=" << ch << " i=" << i;
        }
    }
}

// =============================================================================
// Layer 3 — Behavioral properties (ported from AudioEngineTest Surge_*)
// =============================================================================

TEST_F(SurgeStageTest, FreshOvershootReturnsLinearValue) {
    // One in-range settle sample, then a single overshoot.
    juce::AudioBuffer<double> warmup(1, 1);
    warmup.setSample(0, 0, 0.5);
    stage_->process(warmup);

    juce::AudioBuffer<double> hit(1, 1);
    hit.setSample(0, 0, 1.5);
    stage_->process(hit);

    // With a fresh weight, w advances by one step (≈ 1 / (48000 * 0.001) = 1/48).
    // shapedWeight(1/48) ≈ 1 − (1 − (1/48)²)^8 ≈ 3.47e-3, so effective_x ≈ 1.4983.
    EXPECT_NEAR(hit.getSample(0, 0), 1.5, 1e-2) << "first overshoot sample should be ~linear (x)";
}

TEST_F(SurgeStageTest, SaturatedOvershootSettlesToClamp) {
    const int totalSamples = static_cast<int>(kSampleRate * 2.0 * kSurgeDurationSec);
    const double finalOutput = driveDC(1.5, totalSamples);
    EXPECT_NEAR(finalOutput, 1.0, 1e-3) << "after 2× surge window, output must decay to clamp value";
}

TEST_F(SurgeStageTest, MidpointMatchesShapedBlendAndMonotonic) {
    const int halfSamples = static_cast<int>(kSampleRate * 0.5 * kSurgeDurationSec);
    juce::AudioBuffer<double> buf(1, halfSamples);
    auto* data = buf.getWritePointer(0);
    for (int i = 0; i < halfSamples; ++i)
        data[i] = 1.5;
    stage_->process(buf);

    // At phase ~0.5, shapedWeight ≈ 1 − (0.75)^8 ≈ 0.8999, so blend ≈
    //   (1 − 0.9) · 1.5 + 0.9 · 1.0 = 0.15 + 0.9 = 1.05
    EXPECT_NEAR(data[halfSamples - 1], 1.050, 1e-2) << "half-window output should match Kumaraswamy-shaped blend";

    // Monotonically non-increasing: weight grows → clamp weighting grows → output falls.
    for (int i = 1; i < halfSamples; ++i) {
        EXPECT_LE(data[i], data[i - 1] + 1e-12) << "trajectory must be monotonically decreasing; i=" << i;
    }
}

TEST_F(SurgeStageTest, DecaysBackDuringInRange) {
    // Charge to saturation.
    const int charge = static_cast<int>(kSampleRate * 0.060);
    (void)driveDC(1.5, charge);

    // Hold in-range for the same duration so weight decays back to 0.
    const int discharge = static_cast<int>(kSampleRate * 0.060);
    (void)driveDC(0.0, discharge);

    // Next overshoot must surge fresh.
    juce::AudioBuffer<double> buf(1, 1);
    buf.setSample(0, 0, 1.5);
    stage_->process(buf);
    EXPECT_NEAR(buf.getSample(0, 0), 1.5, 1e-2) << "after in-range period, surge must re-trigger with ~linear output";
}

TEST_F(SurgeStageTest, PositiveAndNegativeRailsAreIndependent) {
    // Saturate positive rail.
    const int charge = static_cast<int>(kSampleRate * 2.0 * kSurgeDurationSec);
    const double lastPos = driveDC(1.5, charge);
    ASSERT_NEAR(lastPos, 1.0, 1e-3);

    // First negative overshoot still sees a fresh negative weight → linear.
    juce::AudioBuffer<double> neg(1, 1);
    neg.setSample(0, 0, -1.5);
    stage_->process(neg);
    EXPECT_NEAR(neg.getSample(0, 0), -1.5, 1e-2) << "negative rail must surge independently of positive-rail history";
}

TEST_F(SurgeStageTest, StereoChannelsIndependent) {
    const int samples = static_cast<int>(kSampleRate * 2.0 * kSurgeDurationSec);

    // Left drives sustained overshoot, right stays in-range.
    juce::AudioBuffer<double> buf(2, samples);
    for (int i = 0; i < samples; ++i) {
        buf.setSample(0, i, 1.5);
        buf.setSample(1, i, 0.5);
    }
    stage_->process(buf);

    // Now hit right with a single overshoot — should surge fresh.
    juce::AudioBuffer<double> hit(2, 1);
    hit.setSample(0, 0, 1.5);
    hit.setSample(1, 0, 1.5);
    stage_->process(hit);
    EXPECT_NEAR(hit.getSample(1, 0), 1.5, 1e-2) << "right channel must surge freshly; L-channel charge must not leak";
    // Left must remain clamped (another overshoot at saturation stays clamped).
    EXPECT_NEAR(hit.getSample(0, 0), 1.0, 5e-3) << "left channel should remain at clamp";
}

// =============================================================================
// Layer 4 — Timing is wall-clock (replaces SurgeRateForwarding)
// =============================================================================

TEST_F(SurgeStageTest, DurationTracksWallTimeAcrossSampleRates) {
    for (double sr : {48000.0, 96000.0, 192000.0}) {
        auto stage = std::make_unique<SurgeStage>();
        stage->prepareToPlay(sr, 512, 2);
        stage->setSurgeDurationSec(kSurgeDurationSec);
        stage->setEnabled(true);

        const int samples = static_cast<int>(sr * 2.0 * kSurgeDurationSec);
        juce::AudioBuffer<double> buf(1, samples);
        auto* d = buf.getWritePointer(0);
        for (int i = 0; i < samples; ++i)
            d[i] = 1.5;
        stage->process(buf);

        EXPECT_NEAR(d[samples - 1], 1.0, 1e-3)
            << "at sr=" << sr << ", saturation should complete within 2× surge window";
    }
}

TEST_F(SurgeStageTest, PrepareToPlayReconfiguresStep) {
    // Same number of samples but different sample rates → different wall-clock
    // windows → different saturation state. Confirms step was recomputed.
    constexpr int samples = 48;

    // 48 samples @ 48kHz = 1ms = full surge window → saturated at clamp.
    stage_->prepareToPlay(48000.0, 512, 2);
    stage_->setSurgeDurationSec(kSurgeDurationSec);
    stage_->reset();
    const double at48k = driveDC(1.5, samples);
    EXPECT_NEAR(at48k, 1.0, 1e-2) << "48 samples @ 48kHz should fill the 1ms window";

    // 48 samples @ 96kHz = 0.5ms = half window → well above clamp, below linear.
    stage_->prepareToPlay(96000.0, 512, 2);
    stage_->setSurgeDurationSec(kSurgeDurationSec);
    stage_->reset();
    const double at96k = driveDC(1.5, samples);
    EXPECT_GT(at96k, at48k + 0.02) << "at 96kHz, step should be smaller, so less saturation after 48 samples";
    EXPECT_LT(at96k, 1.5);
}

// =============================================================================
// Layer 5 — Reset
// =============================================================================

TEST_F(SurgeStageTest, ResetClearsChargedWeights) {
    // Saturate.
    const int charge = static_cast<int>(kSampleRate * 2.0 * kSurgeDurationSec);
    (void)driveDC(1.5, charge);

    stage_->reset();

    // Next overshoot should be fresh (linear), not clamp.
    juce::AudioBuffer<double> buf(1, 1);
    buf.setSample(0, 0, 1.5);
    stage_->process(buf);
    EXPECT_NEAR(buf.getSample(0, 0), 1.5, 1e-2) << "reset() must clear surge weights";
}

// =============================================================================
// Layer 6 — Safety
// =============================================================================

TEST_F(SurgeStageTest, FiniteOutputForLargeInputs) {
    juce::AudioBuffer<double> buf(1, 256);
    auto* d = buf.getWritePointer(0);
    for (int i = 0; i < 256; ++i)
        d[i] = 1e6; // absurdly large
    stage_->process(buf);
    for (int i = 0; i < 256; ++i) {
        EXPECT_TRUE(std::isfinite(d[i])) << "i=" << i;
    }
}

// =============================================================================
// Integration — Surge + Hysteresis: the whole reason for the migration.
// =============================================================================
//
// With the old architecture (surge inside the LUT), the Jiles-Atherton ODE sees
// a constant H under sustained DC overshoot, so dH/dt ≈ 0 → dM/dt ≈ 0 → output
// sits flat while the LUT's surge ramps underneath it. Expected failure mode:
// "flat-top + late dropoff".
//
// With surge as an input-side pre-processor, H = effective_x is now time-varying
// during the surge ramp, so dH/dt ≠ 0 and M responds. Expected: transient "hump"
// — output briefly above its steady-state value right after the step, decaying
// as surge saturates.

TEST(SurgeHysteresisIntegration, DCOvershootShowsTransientHumpNotFlatTop) {
    const double sr = 48000.0 * 16.0; // 768 kHz — typical oversampled rate

    dsp_core::SeamlessTransferFunction tf;
    tf.getLaneMixer().setExtrapolationMode(dsp_core::LaneMixer::ExtrapolationMode::Linear);
    tf.renderLUTImmediate();

    SurgeStage surge;
    surge.prepareToPlay(sr, 512, 2);
    surge.setSurgeDurationSec(0.001);
    surge.setEnabled(true);

    HysteresisStage hyst(tf);
    hyst.prepareToPlay(sr, 512, 2);
    hyst.setHysteresisEnabled(true);
    hyst.setWidth(0.5);
    hyst.setMakeupGain(HysteresisStage::computeMakeupForWidth(0.5));
    hyst.requestWarmup();

    // Flush the 25ms warmup + 25ms crossfade-in with silence so we land in
    // CrossfadeState::Inactive before the step.
    {
        const int warmSamples = static_cast<int>(sr * 0.060); // 60ms
        juce::AudioBuffer<double> warm(1, warmSamples);
        warm.clear();
        surge.process(warm);
        hyst.process(warm);
    }

    // DC step to x = 1.5 for 5 ms.
    const int stepSamples = static_cast<int>(sr * 0.005);
    juce::AudioBuffer<double> buf(1, stepSamples);
    auto* data = buf.getWritePointer(0);
    for (int i = 0; i < stepSamples; ++i)
        data[i] = 1.5;

    surge.process(buf);
    hyst.process(buf);

    // Hump region: first 2ms. Steady-state region: last 1ms.
    const int humpEnd = static_cast<int>(sr * 0.002);
    double humpPeak = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < humpEnd; ++i) {
        humpPeak = std::max(humpPeak, data[i]);
    }

    const int ssStart = stepSamples - static_cast<int>(sr * 0.001);
    double ssSum = 0.0;
    for (int i = ssStart; i < stepSamples; ++i)
        ssSum += data[i];
    const double ss = ssSum / static_cast<double>(stepSamples - ssStart);

    // All outputs finite.
    for (int i = 0; i < stepSamples; ++i) {
        ASSERT_TRUE(std::isfinite(data[i])) << "non-finite output at i=" << i;
    }

    // The hump must rise meaningfully above the settled level. The exact margin
    // depends on the Langevin slope at H∈[1.0, 1.5]; empirically ≥ 5% of full
    // scale is a clear signal that dM/dt responded to the surge ramp.
    EXPECT_GT(humpPeak, ss + 0.05) << "surge+hysteresis should show a transient hump above steady state. "
                                   << "peak=" << humpPeak << " ss=" << ss
                                   << " — if these are ~equal, we have flat-top, which is the pre-migration bug.";
}
