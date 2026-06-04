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
// Phase 5 — Cutoff smoothing follows a log (multiplicative) curve, not linear.
//
// Pitch perception is logarithmic. Linear-in-Hz smoothing means a sweep from
// 100 Hz → 8000 Hz is already at 4050 Hz (≈6 octaves above start, out of 6.3
// total) at the midpoint — the bass region whips by in a flash, the top end
// crawls. Multiplicative smoothing makes each per-sample step a constant
// ratio: at the midpoint the value is sqrt(100 * 8000) ≈ 894 Hz, exactly
// half the musical distance.
//
// At ramp midpoint (24 samples into a 48-sample / 1 ms ramp at 48k):
//   Multiplicative midpoint = sqrt(100 * 8000)  ≈ 894 Hz
//   Linear midpoint         = (100 + 8000) / 2  = 4050 Hz
//
// The < 2000 threshold is the differentiator — Linear's 4050 fails, Multiplicative's
// 894 passes. EXPECT_NEAR tightens to confirm we're actually on the log curve,
// not just "below 2000 by coincidence."
// --------------------------------------------------------------------------
TEST(LadderTPTStage, CutoffSmoothingFollowsLogCurve_24dB) {
    LadderTPT24dBStage s;
    s.setCutoffFrequency(100.0);
    s.prepareToPlay(kSampleRate, 64, 1);

    s.setCutoffFrequency(8000.0);

    // Advance smoother by exactly half the ramp window so we compare midpoint
    // values. Computing this from kSmoothingTimeSec keeps the test robust to
    // future smoothing-time tuning.
    const int halfRampSamples =
        static_cast<int>(LadderTPT24dBStage::kSmoothingTimeSec * kSampleRate * 0.5);
    juce::AudioBuffer<double> halfRamp(1, halfRampSamples);
    halfRamp.clear();
    s.process(halfRamp);

    const double cutoff = s.getCurrentSmoothedCutoff();
    EXPECT_GT(cutoff, 400.0)
        << "Smoother didn't advance from 100 Hz: " << cutoff;
    EXPECT_LT(cutoff, 2000.0)
        << "Cutoff at midpoint is " << cutoff
        << " Hz — Linear smoothing would give ~4050. Expected ~894 (geometric mean).";
    EXPECT_NEAR(894.4, cutoff, 250.0)
        << "Cutoff at midpoint should be near sqrt(100*8000)≈894 Hz, got " << cutoff;
}

TEST(LadderTPTStage, CutoffSmoothingFollowsLogCurve_12dB) {
    LadderTPT12dBStage s;
    s.setCutoffFrequency(100.0);
    s.prepareToPlay(kSampleRate, 64, 1);

    s.setCutoffFrequency(8000.0);

    const int halfRampSamples =
        static_cast<int>(LadderTPT12dBStage::kSmoothingTimeSec * kSampleRate * 0.5);
    juce::AudioBuffer<double> halfRamp(1, halfRampSamples);
    halfRamp.clear();
    s.process(halfRamp);

    const double cutoff = s.getCurrentSmoothedCutoff();
    EXPECT_GT(cutoff, 400.0);
    EXPECT_LT(cutoff, 2000.0)
        << "12dB variant: midpoint " << cutoff << " Hz suggests Linear smoothing";
    EXPECT_NEAR(894.4, cutoff, 250.0);
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

// --------------------------------------------------------------------------
// Choke at max (10x) drives the pre-tanh level to ~45x at full resonance
// (bassMakeup 1+R = 4.5 at R=3.5, times choke 10). Newton must stay finite and
// bounded at that hot internal operating point; the gain-compensated output
// (x1/10) stays small. Regression guard for the Choke feedback drive.
// --------------------------------------------------------------------------
TEST(LadderTPTStage, NewtonStableWithMaxChoke_24dB) {
    LadderTPT24dBStage s;
    s.setCutoffFrequency(1000.0);
    s.setResonance(100.0); // clamped to kMaxResonance
    s.setChoke(100.0);     // clamped to kMaxChoke (10x)
    EXPECT_DOUBLE_EQ(s.getChoke(), 10.0);
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    juce::AudioBuffer<double> buf(1, kBlockSize);

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
                << "Non-finite output at block " << b << " sample " << i << " with max choke.";
            ASSERT_LT(std::abs(v), 10.0)
                << "Runaway at block " << b << " sample " << i << " val=" << v << " with max choke.";
        }
    }
}

// --------------------------------------------------------------------------
// Bass compensation tests (TDD for resonance-induced volume drop).
//
// Without compensation, the textbook 4-pole Moog ladder has small-signal
// DC gain 1/(1+R). At kToneMaxResonance (R=3.5) that's -13 dB — a very
// audible volume drop as the user sweeps resonance up. This test group
// pins down the desired post-compensation behavior: signals below cutoff
// pass through at unity gain across the full resonance range, while the
// resonance peak character is preserved.
//
// Prior history: commit ae8ba9a tried input-side `x *= (1 + 2R)` and was
// reverted (48854a3, "sounded worse") — the 2R coefficient over-compensated
// by 2× in the linear regime. The correct linear coefficient is (1 + R),
// derived from y_N = x_eff - tanh(R*y_N); small-signal y_N = x_eff/(1+R);
// pre-amp by (1+R) yields unity DC gain. These tests would have caught
// the previous error.
// --------------------------------------------------------------------------
namespace {

// Process N samples of a DC step at amp, return the steady-state output (last
// sample). DC settles in ~7/cutoff_rad seconds; for cutoff=1kHz that's ≈1ms,
// well under the 5000-sample (≈104ms) window. Resonance smoother (2ms) is
// also flushed.
template <typename Stage>
double dcSteadyStateGain(double cutoffHz, double R, double inputAmp) {
    Stage s;
    s.setCutoffFrequency(cutoffHz);
    s.setResonance(R);
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    constexpr int kSettleSamples = 5000;
    juce::AudioBuffer<double> buf(1, kSettleSamples);
    for (int i = 0; i < kSettleSamples; ++i) {
        buf.getWritePointer(0)[i] = inputAmp;
    }
    s.process(buf);
    return buf.getReadPointer(0)[kSettleSamples - 1] / inputAmp;
}

// RMS of a buffer over the last `tailSamples` samples (skips startup transient).
double tailRms(const juce::AudioBuffer<double>& buf, int tailSamples) {
    const int n = buf.getNumSamples();
    const int start = std::max(0, n - tailSamples);
    double sumSq = 0.0;
    int count = 0;
    for (int i = start; i < n; ++i) {
        const double v = buf.getReadPointer(0)[i];
        sumSq += v * v;
        ++count;
    }
    return std::sqrt(sumSq / std::max(1, count));
}

template <typename Stage>
double subCutoffSineGain(double cutoffHz, double sineHz, double R, double amp) {
    Stage s;
    s.setCutoffFrequency(cutoffHz);
    s.setResonance(R);
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    constexpr int kTotalSamples = 9600; // 0.2s @ 48k — well past smoothing + 7τ.
    juce::AudioBuffer<double> buf(1, kTotalSamples);
    for (int i = 0; i < kTotalSamples; ++i) {
        buf.getWritePointer(0)[i] = amp * std::sin(2.0 * M_PI * sineHz * i / kSampleRate);
    }
    s.process(buf);
    const double rms = tailRms(buf, 4800); // last 100ms
    const double inputRms = amp / std::sqrt(2.0);
    return rms / inputRms;
}

template <typename Stage>
double atCutoffSinePeak(double cutoffHz, double R, double amp) {
    Stage s;
    s.setCutoffFrequency(cutoffHz);
    s.setResonance(R);
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    constexpr int kTotalSamples = 14400; // 0.3s @ 48k — resonant tail can take longer.
    juce::AudioBuffer<double> buf(1, kTotalSamples);
    for (int i = 0; i < kTotalSamples; ++i) {
        buf.getWritePointer(0)[i] = amp * std::sin(2.0 * M_PI * cutoffHz * i / kSampleRate);
    }
    s.process(buf);
    // Peak amplitude in the steady-state tail.
    double peak = 0.0;
    const int start = kTotalSamples - 4800;
    for (int i = start; i < kTotalSamples; ++i) {
        peak = std::max(peak, std::abs(buf.getReadPointer(0)[i]));
    }
    return peak / amp;
}

} // namespace

// At small signal levels (amp ≤ 0.1, well below tanh saturation), DC gain
// should be ≈ 1.0 across the full resonance range. This is the primary
// correctness assertion — the bug the user reported is exactly this drop.
// Tolerance: ±0.5 dB (ratio 0.944 to 1.059).
TEST(LadderTPTStage, BassCompensation_DcGainUnity_SmallSignal_24dB) {
    for (double R : {0.0, 0.5, 1.0, 2.0, 3.0, 3.5}) {
        const double gain = dcSteadyStateGain<LadderTPT24dBStage>(1000.0, R, 0.1);
        EXPECT_NEAR(gain, 1.0, 0.06)
            << "DC gain at R=" << R << " is " << gain
            << " (linear theory predicts 1/(1+R) = " << 1.0 / (1.0 + R)
            << " without compensation). Expected ≈1.0 with (1+R) input makeup.";
    }
}

TEST(LadderTPTStage, BassCompensation_DcGainUnity_SmallSignal_12dB) {
    // For the 12dB (2-pole) variant the linear DC gain is also 1/(1+R) — the
    // feedback equation y_N = x - tanh(R*y_N) and cascade-DC-gain-of-1 both
    // generalize to N=2. So the same compensation suffices.
    for (double R : {0.0, 0.5, 1.0, 2.0, 3.0, 3.5}) {
        const double gain = dcSteadyStateGain<LadderTPT12dBStage>(1000.0, R, 0.1);
        EXPECT_NEAR(gain, 1.0, 0.06)
            << "12dB DC gain at R=" << R << " is " << gain;
    }
}

// Sub-cutoff sine RMS should be preserved across resonance. 100 Hz at amp
// 0.1 with 1 kHz cutoff is well below the corner (frequency response is
// near-flat in passband) and small enough to stay in the linear-tanh regime.
TEST(LadderTPTStage, BassCompensation_SubCutoffSineRmsPreserved_24dB) {
    for (double R : {0.0, 0.5, 1.0, 2.0, 3.0, 3.5}) {
        const double gain = subCutoffSineGain<LadderTPT24dBStage>(1000.0, 100.0, R, 0.1);
        EXPECT_NEAR(gain, 1.0, 0.122)
            << "Sub-cutoff RMS gain at R=" << R << " is " << gain
            << " (≈" << 20.0 * std::log10(gain) << " dB)";
    }
}

TEST(LadderTPTStage, BassCompensation_SubCutoffSineRmsPreserved_12dB) {
    for (double R : {0.0, 0.5, 1.0, 2.0, 3.0, 3.5}) {
        const double gain = subCutoffSineGain<LadderTPT12dBStage>(1000.0, 100.0, R, 0.1);
        EXPECT_NEAR(gain, 1.0, 0.122)
            << "12dB sub-cutoff RMS gain at R=" << R << " is " << gain;
    }
}

// Resonance peak must still bloom — compensation can't flatten the character
// that makes resonance musically useful. At cutoff frequency, peak amplitude
// at R=3.0 should be visibly larger than at R=0 (low cutoff resonance keeps
// the passband intact). At least 2× growth (≈+6 dB) is a conservative bound;
// real peak gain at R=3.0 is typically +10 dB or more.
TEST(LadderTPTStage, BassCompensation_ResonancePeakStillBlooms_24dB) {
    const double peakR0 = atCutoffSinePeak<LadderTPT24dBStage>(1000.0, 0.0, 0.1);
    const double peakR3 = atCutoffSinePeak<LadderTPT24dBStage>(1000.0, 3.0, 0.1);
    EXPECT_GT(peakR3, peakR0 * 2.0)
        << "Resonance peak at R=3.0 (" << peakR3
        << ") is not meaningfully louder than at R=0 (" << peakR0
        << "). Compensation appears to flatten the resonant character.";
}

TEST(LadderTPTStage, BassCompensation_ResonancePeakStillBlooms_12dB) {
    const double peakR0 = atCutoffSinePeak<LadderTPT12dBStage>(1000.0, 0.0, 0.1);
    const double peakR3 = atCutoffSinePeak<LadderTPT12dBStage>(1000.0, 3.0, 0.1);
    // 12dB variant has a lower-Q resonance peak; threshold is a bit looser.
    EXPECT_GT(peakR3, peakR0 * 1.5)
        << "12dB resonance peak at R=3.0 (" << peakR3
        << ") not meaningfully louder than at R=0 (" << peakR0 << ").";
}

// At R=0 the compensation factor (1+R) is 1.0 — a no-op. Any nonzero R should
// produce a different output than R=0 for the same input, but R=0 should
// produce numerically identical output to a hypothetical uncompensated R=0
// (which is the same code path, since 1+0=1). This is a sanity test that
// compensation truly degenerates to identity at R=0.
TEST(LadderTPTStage, BassCompensation_NoOpAtZeroResonance_24dB) {
    // Process the same input twice; with R=0 the result should match a known
    // pre-compensation reference. We don't have the reference handy in-test,
    // so instead verify the weaker property: at R=0, DC gain is exactly 1.0
    // (within numerical precision), and at R=0 the filter is identical to
    // a sequential cascade of 4 unity-DC-gain integrators — i.e., passband
    // is flat at unity, irrespective of the (1+R) multiplier.
    const double gain = dcSteadyStateGain<LadderTPT24dBStage>(1000.0, 0.0, 0.5);
    EXPECT_NEAR(gain, 1.0, 1e-6)
        << "At R=0 the compensation must be a true no-op. Got DC gain " << gain;
}

// Stability at hot signals + max R. Compensation pre-amps input by 4.5× at
// R=3.5 — this drives tanh deep into saturation but must remain stable
// (no NaN, bounded output). The user complained about volume drop, not
// instability; this guards that the fix doesn't trade one bug for another.
TEST(LadderTPTStage, BassCompensation_StableAtHotSignalMaxR_24dB) {
    LadderTPT24dBStage s;
    s.setCutoffFrequency(500.0);
    s.setResonance(3.5);
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    constexpr int kBlocks = 100; // ~0.5s
    juce::AudioBuffer<double> buf(1, kBlockSize);
    for (int b = 0; b < kBlocks; ++b) {
        // Alternating ±1 (rail-to-rail square wave): worst case for input
        // amplitude after the (1+R) pre-amp = ±4.5.
        for (int i = 0; i < kBlockSize; ++i) {
            buf.getWritePointer(0)[i] = (((b * kBlockSize + i) & 1) != 0) ? 1.0 : -1.0;
        }
        s.process(buf);
        for (int i = 0; i < kBlockSize; ++i) {
            const double v = buf.getReadPointer(0)[i];
            ASSERT_TRUE(std::isfinite(v))
                << "Non-finite output at block " << b << " sample " << i << ": " << v;
            ASSERT_LT(std::abs(v), 6.0)
                << "Runaway at block " << b << " sample " << i << ": " << v
                << " — pre-amped input is bounded ±4.5; output should not exceed ~5.";
        }
    }
}
