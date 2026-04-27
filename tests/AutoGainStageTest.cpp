#include <gtest/gtest.h>
#include <dsp_core/dsp_core.h>

#include <cmath>
#include <random>
#include <vector>

using dsp_core::audio_pipeline::AutoGainState;
using dsp_core::audio_pipeline::AutoRestoreStage;
using dsp_core::audio_pipeline::AutoSquashStage;

// =============================================================================
// Test Fixture
// =============================================================================

class AutoGainStageTest : public ::testing::Test {
  protected:
    void SetUp() override {
        state_.enabled.store(true, std::memory_order_release);
        squash_ = std::make_unique<AutoSquashStage>(state_);
        restore_ = std::make_unique<AutoRestoreStage>(state_);
        squash_->prepareToPlay(sampleRate_, blockSize_);
        restore_->prepareToPlay(sampleRate_, blockSize_);
    }

    void setEnabled(bool on) {
        state_.enabled.store(on, std::memory_order_release);
    }

    // Process one block through squash → (identity middle) → restore.
    // The input vector is split into blocks of at most blockSize_ samples.
    std::vector<double> runSquashRestore(const std::vector<double>& input, int numChannels = 1) {
        std::vector<double> output(input.size());
        int pos = 0;
        const int N = static_cast<int>(input.size());
        while (pos < N) {
            const int thisBlock = std::min(blockSize_, N - pos);
            juce::AudioBuffer<double> buf(numChannels, thisBlock);
            for (int ch = 0; ch < numChannels; ++ch)
                for (int n = 0; n < thisBlock; ++n)
                    buf.getWritePointer(ch)[n] = input[static_cast<size_t>(pos + n)];

            squash_->process(buf);
            // identity middle
            restore_->process(buf);

            for (int n = 0; n < thisBlock; ++n)
                output[static_cast<size_t>(pos + n)] = buf.getReadPointer(0)[n];
            pos += thisBlock;
        }
        return output;
    }

    // Process through squash only, returning internal (post-squash) signal and
    // recorded per-block gain trace.
    struct InternalTrace {
        std::vector<double> internalSignal;
        std::vector<double> gainPerSample;
    };

    InternalTrace runSquashOnly(const std::vector<double>& input) {
        InternalTrace out;
        out.internalSignal.resize(input.size());
        out.gainPerSample.resize(input.size());
        int pos = 0;
        const int N = static_cast<int>(input.size());
        while (pos < N) {
            const int thisBlock = std::min(blockSize_, N - pos);
            juce::AudioBuffer<double> buf(1, thisBlock);
            for (int n = 0; n < thisBlock; ++n)
                buf.getWritePointer(0)[n] = input[static_cast<size_t>(pos + n)];
            squash_->process(buf);
            const auto* gbuf = state_.gainHistory.getReadPointer(0);
            for (int n = 0; n < thisBlock; ++n) {
                out.internalSignal[static_cast<size_t>(pos + n)] = buf.getReadPointer(0)[n];
                out.gainPerSample[static_cast<size_t>(pos + n)] = gbuf[n];
            }
            pos += thisBlock;
        }
        return out;
    }

    std::vector<double> generateSine(double freq, double amplitude, int numSamples) {
        std::vector<double> s(static_cast<size_t>(numSamples));
        for (int n = 0; n < numSamples; ++n)
            s[static_cast<size_t>(n)] =
                amplitude * std::sin(2.0 * juce::MathConstants<double>::pi * freq * n / sampleRate_);
        return s;
    }

    static double peakAbs(const std::vector<double>& v, size_t start = 0) {
        double p = 0.0;
        for (size_t i = start; i < v.size(); ++i)
            p = std::max(p, std::abs(v[i]));
        return p;
    }

    static double rmsLevel(const std::vector<double>& v, size_t start = 0) {
        double sumSq = 0.0;
        const size_t count = v.size() - start;
        for (size_t i = start; i < v.size(); ++i)
            sumSq += v[i] * v[i];
        return count > 0 ? std::sqrt(sumSq / static_cast<double>(count)) : 0.0;
    }

    static constexpr double sampleRate_ = 48000.0;
    static constexpr int blockSize_ = 512;

    AutoGainState state_;
    std::unique_ptr<AutoSquashStage> squash_;
    std::unique_ptr<AutoRestoreStage> restore_;
};

// =============================================================================
// 1. Auto-off is identity (bit-exact within reasonable float tolerance)
// =============================================================================

TEST_F(AutoGainStageTest, AutoOff_IsIdentity) {
    setEnabled(false);
    state_.resetRuntime();

    // Random but deterministic
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> dist(-0.8, 0.8);
    std::vector<double> in(4096);
    for (auto& s : in)
        s = dist(rng);

    const auto out = runSquashRestore(in);

    for (size_t i = 0; i < in.size(); ++i) {
        EXPECT_NEAR(out[i], in[i], 1e-12) << "idx=" << i;
    }
}

// =============================================================================
// 2. Shared state wiring: restore reads what squash writes
// =============================================================================

TEST_F(AutoGainStageTest, SharedState_RestoreIsInverseOfSquash) {
    setEnabled(true);
    state_.resetRuntime();

    // Non-trivial input: a short burst of a sine.
    auto in = generateSine(1000.0, 0.1, 2048);
    const auto out = runSquashRestore(in);

    // With no intermediate processing, squash*restore == identity (the
    // inverse gain cancels exactly).
    for (size_t i = 0; i < in.size(); ++i) {
        EXPECT_NEAR(out[i], in[i], 1e-10) << "idx=" << i;
    }
}

// =============================================================================
// 3. Steady-state DC: constant input converges to target internally
// =============================================================================

TEST_F(AutoGainStageTest, SteadyState_DC_ConvergesToTarget) {
    setEnabled(true);
    state_.resetRuntime();
    // Pre-warm the enable crossfade so we measure steady-state squash gain,
    // not the fade-in transient.
    state_.enableMix = 1.0;

    const double amp = 0.1;
    std::vector<double> in(static_cast<size_t>(sampleRate_ * 1.0), amp); // 1 second of DC

    const auto trace = runSquashOnly(in);

    // After 5× release time constant, envelope should have fully converged.
    // Inspect the final ~100 ms.
    const size_t tailStart = in.size() - static_cast<size_t>(sampleRate_ * 0.1);
    const double tailPeak = peakAbs(trace.internalSignal, tailStart);

    EXPECT_NEAR(tailPeak, state_.k.targetPeak, 0.02) << "Internal signal should converge to target peak";
}

// =============================================================================
// 4. Sine at -12 dBFS: inside hits target, outside recovers original
// =============================================================================

TEST_F(AutoGainStageTest, Sine_MinusTwelveDB_InsideHitsTarget_OutsideRecoversLevel) {
    setEnabled(true);
    state_.resetRuntime();
    state_.enableMix = 1.0;

    const double amp = 0.25118864315; // -12 dBFS (sine peak — RMS = peak/√2)
    auto in = generateSine(1000.0, amp, static_cast<int>(sampleRate_ * 1.0));

    // Split the "through squash → through restore" into two runs so we can
    // measure the internal level separately. Use a fresh state for the end-to-
    // end run to match that test's assumption of pre-warmed steady state.
    //
    // Under the asymmetric RMS detector (5 ms attack / 100 ms release), the
    // converged behavior on a sustained sine is closer to peak normalization
    // than RMS normalization: meanSquare gets pulled toward each cycle's peak
    // power on attack, holds during dips on slow release. Net result: the
    // internal *peak* lands within a small margin of targetPeak, while the
    // gain trajectory itself stays smooth (no per-cycle re-attacks, which is
    // the ring-mod artifact this iteration was built to remove).
    const auto trace = runSquashOnly(in);
    const size_t tailStart = in.size() - static_cast<size_t>(sampleRate_ * 0.1);
    const double internalPeak = peakAbs(trace.internalSignal, tailStart);
    EXPECT_NEAR(internalPeak, state_.k.targetPeak, 0.10) << "Internal peak should land near target";

    // End-to-end — re-run with a fresh state.
    state_.resetRuntime();
    state_.enableMix = 1.0;
    const auto out = runSquashRestore(in);
    const double outPeak = peakAbs(out, tailStart);
    EXPECT_NEAR(outPeak, amp, amp * 0.05) << "Outer output should recover -12 dB level";
}

// =============================================================================
// 5. Silence — no NaN/Inf, output bit-zero
// =============================================================================

TEST_F(AutoGainStageTest, Silence_NoDivergence) {
    setEnabled(true);
    state_.resetRuntime();
    std::vector<double> in(4096, 0.0);

    const auto out = runSquashRestore(in);

    for (double s : out) {
        EXPECT_TRUE(std::isfinite(s));
        EXPECT_EQ(s, 0.0);
    }
}

// =============================================================================
// 6. Very quiet input respects the gain ceiling (no runaway)
// =============================================================================

TEST_F(AutoGainStageTest, VeryQuietInput_GainCeilingBoundsInternal) {
    setEnabled(true);
    state_.resetRuntime();
    state_.enableMix = 1.0;

    const double amp = 1e-5; // way below noise floor
    std::vector<double> in(static_cast<size_t>(sampleRate_ * 0.5), amp);

    const auto trace = runSquashOnly(in);

    // Internal signal cannot exceed input_amp * maxGain.
    const double ceiling = amp * state_.k.maxGainLinear;
    const double internalPeak = peakAbs(trace.internalSignal);
    EXPECT_LE(internalPeak, ceiling + 1e-9);
    // And ensure the gain itself is bounded.
    for (double g : trace.gainPerSample) {
        EXPECT_LE(g, state_.k.maxGainLinear + 1e-9);
        EXPECT_GE(g, 1.0 / state_.k.maxGainLinear - 1e-9);
        EXPECT_TRUE(std::isfinite(g));
    }
}

// =============================================================================
// 7. Step-up attack — internal overshoot bounded
// =============================================================================

TEST_F(AutoGainStageTest, StepUp_InternalOvershootBounded) {
    setEnabled(true);
    state_.resetRuntime();
    state_.enableMix = 1.0;

    // 100 ms at -40 dB (to seed envelope), then 500 ms at 0 dB.
    const int preSamples = static_cast<int>(sampleRate_ * 0.1);
    const int postSamples = static_cast<int>(sampleRate_ * 0.5);
    const double quietAmp = 0.01; // -40 dB
    const double loudAmp = 1.0;   // 0 dB

    std::vector<double> in;
    in.reserve(static_cast<size_t>(preSamples + postSamples));
    for (int i = 0; i < preSamples; ++i)
        in.push_back(quietAmp);
    for (int i = 0; i < postSamples; ++i)
        in.push_back(loudAmp);

    const auto trace = runSquashOnly(in);

    // The RMS detector takes ~5 ms (rmsAttackTau) to catch up on a sudden
    // -40 dB → 0 dB jump, during which the still-low meanSquare keeps the
    // gain high and the internal signal overshoots significantly. This is
    // the explicit trade made in this iteration: the waveshaper saturates
    // any short-lived overshoot, which sounds dramatically better than
    // chasing every peak with audio-rate gain modulation. Bound it at +24 dB
    // (~16×) just to catch true runaway, not to enforce an audibly clean
    // attack edge.
    const double overshootCap = state_.k.targetPeak * 16.0;
    const double internalPeak = peakAbs(trace.internalSignal, static_cast<size_t>(preSamples));
    EXPECT_LE(internalPeak, overshootCap) << "Attack overshoot too high — runaway, not just clipping";

    // After ~50 ms the internal RMS should be near target.
    const size_t settleStart = static_cast<size_t>(preSamples) + static_cast<size_t>(sampleRate_ * 0.05);
    const double settledRms = rmsLevel(trace.internalSignal, settleStart);
    EXPECT_NEAR(settledRms, state_.k.targetPeak, 0.05);
}

// =============================================================================
// 8. Step-down release — envelope decays with ~release time constant
// =============================================================================

TEST_F(AutoGainStageTest, StepDown_ReleaseDecaysMonotonically) {
    setEnabled(true);
    state_.resetRuntime();
    state_.enableMix = 1.0;

    // 200 ms loud (env converges), then silence of 1s so we can watch release.
    const int loudSamples = static_cast<int>(sampleRate_ * 0.2);
    const int silenceSamples = static_cast<int>(sampleRate_ * 1.0);
    std::vector<double> in;
    in.reserve(static_cast<size_t>(loudSamples + silenceSamples));
    for (int i = 0; i < loudSamples; ++i)
        in.push_back(0.5);
    for (int i = 0; i < silenceSamples; ++i)
        in.push_back(0.0);

    const auto trace = runSquashOnly(in);

    // After the hold window, envelope must decay. Sample at hold+1ms vs
    // hold+50ms and assert strictly lower.
    const size_t holdEnd = static_cast<size_t>(loudSamples) + static_cast<size_t>(sampleRate_ * 0.011);
    const size_t laterIdx = static_cast<size_t>(loudSamples) + static_cast<size_t>(sampleRate_ * 0.060);
    const double gEarly = trace.gainPerSample[holdEnd];
    const double gLate = trace.gainPerSample[laterIdx];
    // As envelope decays, gain rises toward the cap.
    EXPECT_GT(gLate, gEarly);
}

// =============================================================================
// 9. Toggle while playing — no discontinuity beyond a small slew cap
// =============================================================================

TEST_F(AutoGainStageTest, Toggle_MidBlock_NoDiscontinuity) {
    setEnabled(true);
    state_.resetRuntime();
    state_.enableMix = 1.0;

    // Steady sine; let state converge first.
    auto warmup = generateSine(1000.0, 0.3, static_cast<int>(sampleRate_ * 0.3));
    (void)runSquashRestore(warmup);

    // Now feed a continuous sine while flipping the toggle mid-way.
    auto tail = generateSine(1000.0, 0.3, static_cast<int>(sampleRate_ * 0.3));
    std::vector<double> out(tail.size());

    const int half = static_cast<int>(tail.size()) / 2;
    // First half: enabled; second half: disabled.
    int pos = 0;
    while (pos < static_cast<int>(tail.size())) {
        const int thisBlock = std::min(blockSize_, static_cast<int>(tail.size()) - pos);
        if (pos >= half)
            setEnabled(false);
        juce::AudioBuffer<double> buf(1, thisBlock);
        for (int n = 0; n < thisBlock; ++n)
            buf.getWritePointer(0)[n] = tail[static_cast<size_t>(pos + n)];
        squash_->process(buf);
        restore_->process(buf);
        for (int n = 0; n < thisBlock; ++n)
            out[static_cast<size_t>(pos + n)] = buf.getReadPointer(0)[n];
        pos += thisBlock;
    }

    // Max sample-to-sample delta on the output: bounded by the natural sine
    // gradient (2πf/fs ≈ 0.131 at 1 kHz, amp 0.3) plus a small allowance for
    // the enable crossfade. End-to-end linearity means no big steps.
    double maxDelta = 0.0;
    for (size_t i = 1; i < out.size(); ++i)
        maxDelta = std::max(maxDelta, std::abs(out[i] - out[i - 1]));
    // Natural delta per sample for 1 kHz, amp 0.3 at 48 kHz is about 0.04.
    // Allow 3× for toggle slack.
    EXPECT_LE(maxDelta, 0.12) << "Toggle produced an audible click";
}

// =============================================================================
// 10. End-to-end outer linearity across levels (identity middle)
// =============================================================================

TEST_F(AutoGainStageTest, EndToEnd_OuterLinearity_AcrossLevels) {
    const std::vector<double> levelsDB = {-6.0, -12.0, -24.0};
    for (double dB : levelsDB) {
        const double amp = juce::Decibels::decibelsToGain(dB);

        state_.resetRuntime();
        state_.enableMix = 1.0;
        setEnabled(true);

        auto in = generateSine(1000.0, amp, static_cast<int>(sampleRate_ * 0.6));
        const auto out = runSquashRestore(in);

        // Measure peak in the tail after convergence.
        const size_t tailStart = in.size() - static_cast<size_t>(sampleRate_ * 0.1);
        const double outPeak = peakAbs(out, tailStart);
        // Within 0.5 dB
        const double allowed = amp * 0.06;
        EXPECT_NEAR(outPeak, amp, allowed) << "level " << dB << " dB";
    }
}

// =============================================================================
// 11. Stereo-linked detection preserves stereo image
// =============================================================================

TEST_F(AutoGainStageTest, StereoLinked_SameGainAppliedToBothChannels) {
    setEnabled(true);
    state_.resetRuntime();
    state_.enableMix = 1.0;

    // Hard-panned: L loud, R quiet.
    const int N = static_cast<int>(sampleRate_ * 0.3);
    juce::AudioBuffer<double> buf(2, N);
    for (int n = 0; n < N; ++n) {
        const double t = n / sampleRate_;
        buf.getWritePointer(0)[n] = 0.5 * std::sin(2.0 * juce::MathConstants<double>::pi * 440.0 * t);
        buf.getWritePointer(1)[n] = 0.05 * std::sin(2.0 * juce::MathConstants<double>::pi * 440.0 * t);
    }

    // Snapshot pre-squash ratio.
    const double preRatio = 0.05 / 0.5;

    squash_->process(buf);

    // The same gain applied to both channels — the L/R ratio must be
    // preserved exactly at every sample where both are non-zero.
    for (int n = 0; n < N; ++n) {
        const double l = buf.getReadPointer(0)[n];
        const double r = buf.getReadPointer(1)[n];
        if (std::abs(l) > 1e-6) {
            const double ratio = r / l;
            EXPECT_NEAR(ratio, preRatio, 1e-9) << "n=" << n;
        }
    }
}

// =============================================================================
// 12. Prepare re-allocates to worst-case block size; subsequent smaller blocks
//     do not crash and produce a valid gain history.
// =============================================================================

TEST_F(AutoGainStageTest, VariableBlockSize_HandledSafely) {
    setEnabled(true);
    state_.resetRuntime();
    state_.enableMix = 1.0;

    // Varying block sizes from 1 to blockSize_.
    std::mt19937 rng(99);
    std::uniform_int_distribution<int> blockDist(1, blockSize_);
    std::vector<double> in = generateSine(500.0, 0.3, 8000);
    std::vector<double> out(in.size());

    int pos = 0;
    while (pos < static_cast<int>(in.size())) {
        int thisBlock = std::min(blockDist(rng), static_cast<int>(in.size()) - pos);
        juce::AudioBuffer<double> buf(1, thisBlock);
        for (int n = 0; n < thisBlock; ++n)
            buf.getWritePointer(0)[n] = in[static_cast<size_t>(pos + n)];
        squash_->process(buf);
        restore_->process(buf);
        for (int n = 0; n < thisBlock; ++n)
            out[static_cast<size_t>(pos + n)] = buf.getReadPointer(0)[n];
        pos += thisBlock;
    }

    // End-to-end must still be near identity (with a bit more tolerance for
    // the varying-block slop).
    const size_t tailStart = in.size() - static_cast<size_t>(sampleRate_ * 0.02);
    for (size_t i = tailStart; i < in.size(); ++i)
        EXPECT_NEAR(out[i], in[i], 1e-9);
}

// =============================================================================
// =============================================================================
// Speed/Smoothness characterization tests
// =============================================================================
//
// These tests measure the gain-trajectory smoothness ("zipper proxy") and the
// normalization speed (step-up settling, release time) as numbers we can
// assert against, so the auto-gain constants can be tuned without falling
// outside an explicit speed↔smoothness envelope. Each test prints its measured
// values via std::cout under a "[CHARACTERIZATION]" tag — run with `ctest -V`
// to read them.
//
// Reference numbers (sampleRate=48 kHz, default Constants in AutoGainState.h)
// captured 2026-04-27:
//
//                                | peak follower         | peak +              | RMS detector
//                                |  (releaseTau=120ms,   |  asymmetric gain    |  (5ms attack /
//                                |   no smoothing)       |  smoothing,         |  100ms release)
//                                |                       |  releaseTau=200ms   |  + asym gain smth
//   ----------------------------------------------------------------------------------------------
//   SteadyStateSine maxAbsDelta  |   0.00544             |   0.000381 (-93%)   |   ~0.0005
//   SteadyStateSine rmsDelta     |   0.000250            |   0.0000175 (-93%)  |   ~0.0001
//   StepUp settlingTimeMs        |   0.21                |   5.54              |   ~30–60 (RMS lag)
//   StepDown t90Ms               | 276.5                 | 460.1               |   ~370
//   BurstTrain maxAbsDelta       |   0.171               |   0.0925 (-46%)     |   ~0.003 (-97%)
//   BurstTrain rmsDelta          |   0.00442             |   0.00252 (-43%)    |   ~0.0004 (-91%)
//
// The RMS detector is the column that finally killed the audio-rate ring-mod
// artifact: per-cycle peak re-attacks no longer drive the gain. The trade is
// short-lived overshoot on big level steps (saturated by the waveshaper) and
// a slower normalization onset (~30–60 ms vs ~6 ms peak). Thresholds are set
// ~3× above the measured RMS-detector values to leave room for tuning.
// =============================================================================

namespace {
constexpr const char* kCharTag = "[CHARACTERIZATION] ";
}

// 13. Steady-state sine — gain should be (nearly) constant once converged.
// Per-sample |Δgain| is the most direct zipper proxy.
TEST_F(AutoGainStageTest, SteadyStateSine_GainRippleIsSmall) {
    setEnabled(true);
    state_.resetRuntime();
    state_.enableMix = 1.0;

    const double amp = 0.25118864315; // -12 dBFS
    auto in = generateSine(1000.0, amp, static_cast<int>(sampleRate_ * 0.5));
    const auto trace = runSquashOnly(in);

    // Inspect the last 200 ms, after envelope has converged (5×releaseTau).
    const size_t tailStart = trace.gainPerSample.size() - static_cast<size_t>(sampleRate_ * 0.2);
    double maxAbsDelta = 0.0;
    double sumSqDelta = 0.0;
    size_t count = 0;
    for (size_t i = tailStart + 1; i < trace.gainPerSample.size(); ++i) {
        const double d = trace.gainPerSample[i] - trace.gainPerSample[i - 1];
        maxAbsDelta = std::max(maxAbsDelta, std::abs(d));
        sumSqDelta += d * d;
        ++count;
    }
    const double rmsDelta = std::sqrt(sumSqDelta / static_cast<double>(count));

    std::cout << kCharTag << "SteadyStateSine_GainRipple maxAbsDeltaGain=" << maxAbsDelta
              << " rmsDeltaGain=" << rmsDelta << std::endl;

    EXPECT_TRUE(std::isfinite(maxAbsDelta));
    EXPECT_LT(maxAbsDelta, 0.001) << "Steady-tone gain ripple regressed.";
    // The asymmetric RMS power detector produces a small 2× signal-frequency
    // ripple in meanSquare (sample² folds), which the gain smoother further
    // attenuates. ~1e-4 RMS is comfortably below audibility on the gain
    // signal — modulating a 1 kHz tone by ±1e-4 is ≪ −80 dB sideband.
    EXPECT_LT(rmsDelta, 0.0002);
}

// 14. Step-up: silence → -24 dBFS sine. Time from onset until gain converges
// to within 5% of its steady-state value and stays there. This is "how fast
// does normalization kick in on a quiet onset" — the speed we don't want to
// lose. With instant attack today this should be ~sub-millisecond.
TEST_F(AutoGainStageTest, StepUp_NormalizationSettlingTime_Ms) {
    setEnabled(true);
    state_.resetRuntime();
    state_.enableMix = 1.0;

    const int silenceSamples = static_cast<int>(sampleRate_ * 0.05); // 50 ms silence
    const int sineSamples = static_cast<int>(sampleRate_ * 1.0);     // 1 s sine — RMS
                                                                     // detector needs more
                                                                     // time than peak to
                                                                     // reach steady state
    const double amp = 0.06309573445;                                // -24 dBFS

    std::vector<double> in;
    in.reserve(static_cast<size_t>(silenceSamples + sineSamples));
    for (int i = 0; i < silenceSamples; ++i)
        in.push_back(0.0);
    auto sine = generateSine(1000.0, amp, sineSamples);
    in.insert(in.end(), sine.begin(), sine.end());

    const auto trace = runSquashOnly(in);

    // Empirical converged gain over the last 50 ms — under the RMS detector,
    // the analytic formula would be targetPeak/(amp/√2), but sampling the
    // actual tail keeps the test agnostic to detector type.
    const size_t finalStart = trace.gainPerSample.size() - static_cast<size_t>(sampleRate_ * 0.05);
    double finalGain = 0.0;
    for (size_t i = finalStart; i < trace.gainPerSample.size(); ++i)
        finalGain += trace.gainPerSample[i];
    finalGain /= static_cast<double>(trace.gainPerSample.size() - finalStart);
    const double tol = 0.05 * finalGain;

    // Walk forward from the onset; settlingSample is the LAST sample where
    // |gain - finalGain| > tol. Add 1 → first sample inside the band; the
    // remainder of the buffer must stay inside.
    int lastOutOfBand = silenceSamples; // pre-onset state doesn't count
    for (size_t i = static_cast<size_t>(silenceSamples); i < trace.gainPerSample.size(); ++i) {
        if (std::abs(trace.gainPerSample[i] - finalGain) > tol) {
            lastOutOfBand = static_cast<int>(i);
        }
    }
    const int settlingSamples = (lastOutOfBand + 1) - silenceSamples;
    const double settlingMs = 1000.0 * settlingSamples / sampleRate_;

    std::cout << kCharTag << "StepUp_NormalizationSettle settlingTimeMs=" << settlingMs
              << " (samples=" << settlingSamples << ", finalGain=" << finalGain << ")" << std::endl;

    EXPECT_GE(settlingMs, 0.0);
    // RMS attack tau (5 ms) + asymmetric gain-rise smoothing (2 ms) + the
    // initial transition out of the noise-floor branch: settling is on the
    // order of tens of ms. 200 ms is a generous regression cap; perceptual
    // "fade-in" feel of normalization onset is not noticeable below ~50 ms.
    EXPECT_LT(settlingMs, 200.0) << "Step-up normalization got too slow to be useful.";
}

// 15. Step-down: loud → quiet. Time for gain to traverse 90% of the way from
// its initial converged value to its new converged value. This is the
// release-side normalization speed — the dominant axis we'll trade against
// smoothness.
TEST_F(AutoGainStageTest, StepDown_ReleaseTime_Ms) {
    setEnabled(true);
    state_.resetRuntime();
    state_.enableMix = 1.0;

    const int loudSamples = static_cast<int>(sampleRate_ * 0.3);  // 300 ms to converge
    const int quietSamples = static_cast<int>(sampleRate_ * 1.5); // 1.5 s to fully release
    const double loudAmp = 0.5;                                   // gain ≈ 1.8
    const double quietAmp = 0.1;                                  // gain ≈ 9.0

    std::vector<double> in;
    in.reserve(static_cast<size_t>(loudSamples + quietSamples));
    auto loudSine = generateSine(1000.0, loudAmp, loudSamples);
    auto quietSine = generateSine(1000.0, quietAmp, quietSamples);
    in.insert(in.end(), loudSine.begin(), loudSine.end());
    in.insert(in.end(), quietSine.begin(), quietSine.end());

    const auto trace = runSquashOnly(in);

    const double initialGain = trace.gainPerSample[static_cast<size_t>(loudSamples) - 1];
    // Converged gain in the very last 50 ms.
    const size_t finalStart = trace.gainPerSample.size() - static_cast<size_t>(sampleRate_ * 0.05);
    double finalGain = 0.0;
    for (size_t i = finalStart; i < trace.gainPerSample.size(); ++i)
        finalGain += trace.gainPerSample[i];
    finalGain /= static_cast<double>(trace.gainPerSample.size() - finalStart);

    const double threshold = initialGain + 0.9 * (finalGain - initialGain);

    int crossingSample = -1;
    for (size_t i = static_cast<size_t>(loudSamples); i < trace.gainPerSample.size(); ++i) {
        if (trace.gainPerSample[i] >= threshold) {
            crossingSample = static_cast<int>(i);
            break;
        }
    }
    ASSERT_GE(crossingSample, loudSamples) << "Gain never reached 90% of new steady-state";

    const int releaseSamples = crossingSample - loudSamples;
    const double t90Ms = 1000.0 * releaseSamples / sampleRate_;

    std::cout << kCharTag << "StepDown_ReleaseTime t90Ms=" << t90Ms << " (initialGain=" << initialGain
              << " finalGain=" << finalGain << ")" << std::endl;

    // Sanity-bracket the release: too fast → reverts to old zipper; too slow
    // → loses normalization usefulness on quiet sections.
    EXPECT_GT(t90Ms, 200.0) << "Release got faster — zipper risk.";
    EXPECT_LT(t90Ms, 700.0) << "Release got too slow to normalize quiet material.";
}

// 16. Burst train (program-material proxy): alternating 30ms loud / 30ms
// quiet sine bursts. Each loud→quiet edge invokes a release; each quiet→loud
// edge invokes the instant attack. Max |Δgain|/sample over the back half is
// the transient zipper proxy — what the user actually hears on percussive
// material.
TEST_F(AutoGainStageTest, BurstTrain_TransientZipperProxy) {
    setEnabled(true);
    state_.resetRuntime();
    state_.enableMix = 1.0;

    const int burstLen = static_cast<int>(sampleRate_ * 0.030);
    const int numBursts = 16; // ~960 ms total
    const double loudAmp = 0.5;
    const double quietAmp = 0.05;

    std::vector<double> in;
    in.reserve(static_cast<size_t>(burstLen * numBursts * 2));
    int phase = 0;
    for (int b = 0; b < numBursts; ++b) {
        const double amp = (b % 2 == 0) ? loudAmp : quietAmp;
        for (int n = 0; n < burstLen; ++n) {
            const double t = static_cast<double>(phase++) / sampleRate_;
            in.push_back(amp * std::sin(2.0 * juce::MathConstants<double>::pi * 1000.0 * t));
        }
    }

    const auto trace = runSquashOnly(in);

    // Use the second half so the system is past initial convergence.
    const size_t halfStart = trace.gainPerSample.size() / 2;
    double maxAbsDelta = 0.0;
    double sumSqDelta = 0.0;
    size_t count = 0;
    for (size_t i = halfStart + 1; i < trace.gainPerSample.size(); ++i) {
        const double d = trace.gainPerSample[i] - trace.gainPerSample[i - 1];
        maxAbsDelta = std::max(maxAbsDelta, std::abs(d));
        sumSqDelta += d * d;
        ++count;
    }
    const double rmsDelta = std::sqrt(sumSqDelta / static_cast<double>(count));

    std::cout << kCharTag << "BurstTrain_TransientZipper maxAbsDeltaGain=" << maxAbsDelta
              << " rmsDeltaGain=" << rmsDelta << std::endl;

    EXPECT_TRUE(std::isfinite(maxAbsDelta));
    // RMS detection collapsed this metric ~30× from the prior peak-follower
    // implementation — the per-cycle peak re-attacks that drove the audio-
    // rate ring-mod artifact are simply absent now. The threshold is set ~3×
    // above the measured value to catch real regressions.
    EXPECT_LT(maxAbsDelta, 0.01) << "Transient zipper regressed.";
    EXPECT_LT(rmsDelta, 0.002);
}
