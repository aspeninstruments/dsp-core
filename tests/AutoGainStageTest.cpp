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

    const double amp = 0.25118864315; // -12 dBFS
    auto in = generateSine(1000.0, amp, static_cast<int>(sampleRate_ * 1.0));

    // Split the "through squash → through restore" into two runs so we can
    // measure the internal peak separately. Use a fresh state for the end-to-
    // end run to match that test's assumption of pre-warmed steady state.
    const auto trace = runSquashOnly(in);
    const size_t tailStart = in.size() - static_cast<size_t>(sampleRate_ * 0.1);
    const double internalPeak = peakAbs(trace.internalSignal, tailStart);
    EXPECT_NEAR(internalPeak, state_.k.targetPeak, 0.02);

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

    // During attack the internal signal can temporarily exceed the target.
    // Cap the acceptable overshoot at +6 dB above target (~1.8). Beyond that
    // we are letting unsquashed transients through.
    const double overshootCap = state_.k.targetPeak * 2.0; // +6 dB
    const double internalPeak = peakAbs(trace.internalSignal, static_cast<size_t>(preSamples));
    EXPECT_LE(internalPeak, overshootCap) << "Attack overshoot too high";

    // After ~30 ms (6×attackTau), internal should be near target.
    const size_t settleStart = static_cast<size_t>(preSamples) + static_cast<size_t>(sampleRate_ * 0.05);
    const double settledPeak = peakAbs(trace.internalSignal, settleStart);
    EXPECT_NEAR(settledPeak, state_.k.targetPeak, 0.05);
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
