#include "../dsp_core/Source/audio_pipeline/LadderTPTStage.h"
#include <gtest/gtest.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <cmath>

using namespace dsp_core::audio_pipeline;

namespace {

constexpr double kSampleRate = 48000.0;
constexpr int kBlockSize = 256;

template <typename Stage>
void clearBuffer(juce::AudioBuffer<double>& buf) {
    buf.clear();
}

template <typename Stage>
void processSilence(Stage& s, int numBlocks, juce::AudioBuffer<double>& buf) {
    for (int n = 0; n < numBlocks; ++n) {
        buf.clear();
        s.process(buf);
    }
}

} // namespace

// --------------------------------------------------------------------------
// Test 1 — Smoothing must take meaningfully many samples to reach a new
// cutoff target (i.e., we're not zipper-noise levels of fast).
//
// Original kSmoothingTimeSec = 0.0002 s (≈10 samples @ 48 kHz) — test FAILS
// because the ramp completes within the first 10-sample buffer.
//
// Phase 1 bumped to 0.001 s (≈48 samples); Phase 1.5 to 0.002 s (≈96 samples).
// The test uses thresholds that are robust to either: assert "not at target
// after 10 samples" (true for any smoothing time ≥ ~0.5 ms) and "at target
// after generously many samples" (500 samples = ~10 ms, safely past any
// reasonable smoothing window).
// --------------------------------------------------------------------------
TEST(LadderTPTStage, SmoothingTakesMillisecondsToReachTarget_24dB) {
    LadderTPT24dBStage s;
    s.setCutoffFrequency(1000.0);
    s.prepareToPlay(kSampleRate, 64, 1);

    // Request a new cutoff target ~5x higher to make ramp progress easy to see.
    s.setCutoffFrequency(5000.0);

    // Process 10 samples (well under 1 ms = 48 samples).
    juce::AudioBuffer<double> tenSamples(1, 10);
    tenSamples.clear();
    s.process(tenSamples);

    const double afterTen = s.getCurrentSmoothedCutoff();
    EXPECT_GT(afterTen, 1000.0)
        << "Smoothing should have advanced from initial 1000 Hz after 10 samples";
    EXPECT_LT(afterTen, 4000.0)
        << "Smoothing reached too close to target (5000 Hz) in 10 samples — "
           "smoothing time is too short. Got " << afterTen << " Hz";

    // Process 500 more samples (≈10 ms @ 48 kHz), well past any reasonable
    // smoothing window. Robust to future bumps of kSmoothingTimeSec.
    juce::AudioBuffer<double> fiveHundredMore(1, 500);
    fiveHundredMore.clear();
    s.process(fiveHundredMore);

    EXPECT_NEAR(5000.0, s.getCurrentSmoothedCutoff(), 1.0)
        << "Smoothing should have settled at target after ~10 ms";
}

TEST(LadderTPTStage, SmoothingTakesMillisecondsToReachTarget_12dB) {
    LadderTPT12dBStage s;
    s.setCutoffFrequency(1000.0);
    s.prepareToPlay(kSampleRate, 64, 1);

    s.setCutoffFrequency(5000.0);

    juce::AudioBuffer<double> tenSamples(1, 10);
    tenSamples.clear();
    s.process(tenSamples);

    const double afterTen = s.getCurrentSmoothedCutoff();
    EXPECT_GT(afterTen, 1000.0);
    EXPECT_LT(afterTen, 4000.0)
        << "12dB variant: ramp too fast. Got " << afterTen << " Hz at sample 10";

    juce::AudioBuffer<double> fiveHundredMore(1, 500);
    fiveHundredMore.clear();
    s.process(fiveHundredMore);

    EXPECT_NEAR(5000.0, s.getCurrentSmoothedCutoff(), 1.0);
}

// --------------------------------------------------------------------------
// Test 2 — State must flush to exactly 0 after long silence.
//
// Pre-Phase-1: no ScopedNoDenormals. The TPT integrator states decay
// asymptotically; once they enter denormal range (≈1e-308) the per-sample
// output stays nonzero but extremely small (≈ -3e-310 .. 3e-310). The test
// asserts EXACT 0.0, which fails for denormals.
//
// Post-Phase-1: ScopedNoDenormals flushes denormals to true 0 within the
// process() scope. Once states reach denormal range they snap to 0 and the
// loop produces clean digital silence.
// --------------------------------------------------------------------------
// What we're actually testing: ScopedNoDenormals (FTZ/FZ mode) prevents the
// per-sample loop from producing denormal *arithmetic results*. That's what
// kills perf on x86 (10-100x slowdown) and what JUCE's ScopedNoDenormals is
// designed to fix.
//
// We can't assert "output reaches exact 0" — with FTZ, the state can get
// stuck at the smallest *normal* value (~1e-307): the arithmetic that would
// decrement it produces a denormal which flushes to 0, then state += 0
// preserves the small-but-normal state. That's expected and harmless.
//
// What we CAN assert: no value in the output is in the denormal range
// (i.e., every value is either exactly 0 or `std::isnormal`).
//
// Pre-Phase-1: tail values are ~1e-323 (deep denormal range) — test FAILS.
// Post-Phase-1: tail values are 0 or ≥ ~2.225e-308 — test PASSES.
//
// Resonance is fixed at 0 so the test exercises pure exponential decay, not
// the long Q-driven resonant tail. Stability under high R is covered by
// FilterTopologyBenchmark.StabilityAcrossResonance.
TEST(LadderTPTStage, NoDenormalOutputAfterLongSilence_24dB) {
    LadderTPT24dBStage s;
    s.setCutoffFrequency(5000.0);
    s.setResonance(0.0);
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    juce::AudioBuffer<double> buf(1, kBlockSize);
    buf.clear();
    buf.getWritePointer(0)[0] = 1.0;
    s.process(buf);

    // 100 blocks × 256 = 25600 samples (≈0.53s @ 48k). Without FTZ, the state
    // would have decayed deep into the denormal range by now.
    processSilence(s, 100, buf);

    for (int i = 0; i < buf.getNumSamples(); ++i) {
        const double v = buf.getReadPointer(0)[i];
        const bool ok = (v == 0.0) || std::isnormal(v);
        EXPECT_TRUE(ok)
            << "Sample " << i << " is denormal: " << v
            << " — ScopedNoDenormals not in effect (or not working on this platform).";
        if (!ok) {
            break;
        }
    }
}

TEST(LadderTPTStage, NoDenormalOutputAfterLongSilence_12dB) {
    LadderTPT12dBStage s;
    s.setCutoffFrequency(5000.0);
    s.setResonance(0.0);
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    juce::AudioBuffer<double> buf(1, kBlockSize);
    buf.clear();
    buf.getWritePointer(0)[0] = 1.0;
    s.process(buf);

    processSilence(s, 100, buf);

    for (int i = 0; i < buf.getNumSamples(); ++i) {
        const double v = buf.getReadPointer(0)[i];
        const bool ok = (v == 0.0) || std::isnormal(v);
        EXPECT_TRUE(ok)
            << "12dB variant: sample " << i << " is denormal: " << v;
        if (!ok) {
            break;
        }
    }
}

// --------------------------------------------------------------------------
// Test 3 — Cutoff and resonance clamping. Pure API sanity, already correct;
// just a guard that future edits don't relax the bounds.
// --------------------------------------------------------------------------
TEST(LadderTPTStage, CutoffAndResonanceClamping_24dB) {
    LadderTPT24dBStage s;
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    s.setCutoffFrequency(-100.0);
    EXPECT_GE(s.getCutoffFrequency(), 20.0);

    s.setCutoffFrequency(1.0e9);
    EXPECT_LE(s.getCutoffFrequency(), 0.45 * kSampleRate + 1e-9);

    s.setResonance(-1.0);
    EXPECT_DOUBLE_EQ(0.0, s.getResonance());

    // Max resonance must be > 0 and strictly below the self-osc threshold (R=4
    // under the corrected tanh(R*y) feedback math). Concrete value is an impl
    // tuning choice; assert the safety contract.
    s.setResonance(100.0);
    EXPECT_GT(s.getResonance(), 0.0);
    EXPECT_LT(s.getResonance(), 4.0);
}

TEST(LadderTPTStage, CutoffAndResonanceClamping_12dB) {
    LadderTPT12dBStage s;
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    s.setCutoffFrequency(-100.0);
    EXPECT_GE(s.getCutoffFrequency(), 20.0);

    s.setCutoffFrequency(1.0e9);
    EXPECT_LE(s.getCutoffFrequency(), 0.45 * kSampleRate + 1e-9);

    s.setResonance(-1.0);
    EXPECT_DOUBLE_EQ(0.0, s.getResonance());

    s.setResonance(100.0);
    EXPECT_GT(s.getResonance(), 0.0);
    EXPECT_LT(s.getResonance(), 4.0);
}

// --------------------------------------------------------------------------
// Phase 3.5 — Ringing must decay below max resonance.
//
// The original feedback math used tanh(2*R*y), which makes the small-signal
// loop gain 2R. For a 4-pole ladder, self-oscillation occurs at loop gain 4,
// i.e. **R = 2.0**. That's only 50% on a 0-4 knob — well below the documented
// kMaxResonance and the user reported it as "self-oscillating around 54%".
//
// The fix drops the inner *2 from the feedback nonlinearity (per-stage tanh(2x)
// saturation is unchanged — that's character, not loop gain). Self-osc moves
// to R=4, and kMaxResonance drops to 3.9 so the knob top is safely below.
//
// Tests:
//   - At R=2.5 (62% knob, well above the OLD self-osc point), impulse energy
//     should decay to <1% in 100 blocks (~0.5 s @ 48k). Fails pre-fix (filter
//     sustains), passes post-fix.
//   - At R=3.9 (the new max), impulse energy still decays — looser threshold
//     (<10%) because resonance is screaming but it shouldn't be self-oscillating.
// --------------------------------------------------------------------------
namespace {
template <typename Stage>
double impulseDecayRatio(Stage& s, double R, int numSilenceBlocks) {
    s.setCutoffFrequency(1000.0);
    s.setResonance(R);
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    juce::AudioBuffer<double> buf(1, kBlockSize);
    buf.clear();
    buf.getWritePointer(0)[0] = 1.0;
    s.process(buf);

    double energyFirst = 0.0;
    for (int i = 0; i < kBlockSize; ++i) {
        const double v = buf.getReadPointer(0)[i];
        energyFirst += v * v;
    }

    for (int b = 0; b < numSilenceBlocks; ++b) {
        buf.clear();
        s.process(buf);
    }

    double energyLast = 0.0;
    for (int i = 0; i < kBlockSize; ++i) {
        const double v = buf.getReadPointer(0)[i];
        energyLast += v * v;
    }
    return energyLast / std::max(energyFirst, 1e-300);
}
} // namespace

TEST(LadderTPTStage, RingingDecaysAtMidResonance_24dB) {
    LadderTPT24dBStage s;
    const double ratio = impulseDecayRatio(s, 2.5, 100);
    EXPECT_LT(ratio, 0.01)
        << "Filter self-oscillating at R=2.5 (tail energy / impulse energy = " << ratio
        << "). Feedback path likely has hidden gain >1 at this R.";
}

TEST(LadderTPTStage, RingingDecaysAtMidResonance_12dB) {
    LadderTPT12dBStage s;
    const double ratio = impulseDecayRatio(s, 2.5, 100);
    EXPECT_LT(ratio, 0.01)
        << "12dB filter self-oscillating at R=2.5 (ratio = " << ratio << ")";
}

TEST(LadderTPTStage, RingingDecaysAtMaxResonance_24dB) {
    LadderTPT24dBStage s;
    // Drive to whatever the configured max is; we then read it back to log.
    s.setResonance(100.0);
    const double r = s.getResonance();
    const double ratio = impulseDecayRatio(s, r, 200);
    EXPECT_LT(ratio, 0.1)
        << "Filter not decaying at max R=" << r
        << " (ratio = " << ratio << "). Knob top is at/above self-osc threshold.";
}

// --------------------------------------------------------------------------
// Test (Phase 3) — Newton convergence stress test at near-max resonance.
//
// Phase 3 changes the Newton loop from "fixed 3 iterations" to "early-out on
// residual, cap at 4." At low R the linear ZDF guess is near-exact and we exit
// after 1 iter (saving CPU). At very high R the nonlinearity is steep and
// 3-4 iterations are needed for the residual to shrink below tolerance.
//
// This test runs a frequency-sweep through the resonant peak at R=3.95 (just
// below kMaxResonance=4.0) for ~0.5 seconds. Catches:
//   - Newton diverging at high R (would NaN / explode)
//   - Tolerance too loose, leaving large residual that audibly destabilizes
//   - Cap too low (3 would have been borderline at R=3.95)
// Pre- and post-Phase-3 both pass; this is a regression guard, not red→green.
// --------------------------------------------------------------------------
TEST(LadderTPTStage, NewtonStableAtNearSelfOscillation_24dB) {
    LadderTPT24dBStage s;
    s.setCutoffFrequency(1000.0);
    // Drive to whatever max the implementation enforces (clamped if higher).
    s.setResonance(100.0);
    const double rUsed = s.getResonance();
    EXPECT_GT(rUsed, 3.0) << "max R unexpectedly low: " << rUsed;
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    juce::AudioBuffer<double> buf(1, kBlockSize);

    // Linear frequency sweep 200 -> 4200 Hz over 50 blocks (~0.27 s @ 48k).
    // Sweeping through the resonant cutoff exercises Newton hardest.
    constexpr int kNumBlocks = 50;
    for (int b = 0; b < kNumBlocks; ++b) {
        for (int i = 0; i < kBlockSize; ++i) {
            const int n = b * kBlockSize + i;
            const double t = n / kSampleRate;
            const double freq = 200.0 + 4000.0 * (n / static_cast<double>(kNumBlocks * kBlockSize));
            buf.getWritePointer(0)[i] = 0.5 * std::sin(2.0 * M_PI * freq * t);
        }
        s.process(buf);
        for (int i = 0; i < kBlockSize; ++i) {
            const double v = buf.getReadPointer(0)[i];
            ASSERT_TRUE(std::isfinite(v))
                << "Non-finite output at block " << b << " sample " << i
                << " (R=" << rUsed << "). Newton may have diverged.";
            ASSERT_LT(std::abs(v), 10.0)
                << "Runaway at block " << b << " sample " << i
                << " val=" << v << " (R=" << rUsed << ").";
        }
    }
}
