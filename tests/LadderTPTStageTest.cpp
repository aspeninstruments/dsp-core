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
// Test 1 — Smoothing must take ≥ ~1 ms to reach a new cutoff target.
//
// Pre-Phase-1, kSmoothingTimeSec = 0.0002 s (≈10 samples @ 48 kHz). This test
// will FAIL because the ramp completes within the first 10-sample buffer.
//
// Post-Phase-1, kSmoothingTimeSec = 0.001 s (≈48 samples @ 48 kHz). At sample
// 10 we should be ~20% through the ramp; at sample 70 we should be at target.
// --------------------------------------------------------------------------
TEST(LadderTPTStage, SmoothingTakesAtLeast1msToReachTarget_24dB) {
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
           "expected ramp time ≥ 1 ms (48 samples). Got " << afterTen << " Hz";

    // Process another 60 samples (total 70, well past the 48-sample ramp).
    juce::AudioBuffer<double> sixtyMore(1, 60);
    sixtyMore.clear();
    s.process(sixtyMore);

    EXPECT_NEAR(5000.0, s.getCurrentSmoothedCutoff(), 1.0)
        << "Smoothing should have settled at target after >1 ms";
}

TEST(LadderTPTStage, SmoothingTakesAtLeast1msToReachTarget_12dB) {
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

    juce::AudioBuffer<double> sixtyMore(1, 60);
    sixtyMore.clear();
    s.process(sixtyMore);

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

    s.setResonance(100.0);
    EXPECT_DOUBLE_EQ(4.0, s.getResonance());
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
    EXPECT_DOUBLE_EQ(4.0, s.getResonance());
}
