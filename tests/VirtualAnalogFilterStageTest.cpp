#include "../dsp_core/Source/audio_pipeline/VirtualAnalogFilterStage.h"
#include <gtest/gtest.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <cmath>
#include <random>

using namespace dsp_core::audio_pipeline;

namespace {

constexpr double kSampleRate = 48000.0;
constexpr int kBlockSize = 256;

} // namespace

// --------------------------------------------------------------------------
// Phase 5 — Cutoff smoothing follows a log (multiplicative) curve, not linear.
//
// Mirror of LadderTPTStage.CutoffSmoothingFollowsLogCurve_24dB on the modular
// RK4 wrapper. The smoother lives in this class too and was Linear pre-Phase-5.
//
// At ramp midpoint (24 samples into a 48-sample / 1 ms ramp at 48k):
//   Multiplicative midpoint = sqrt(100 * 8000)  ≈ 894 Hz
//   Linear midpoint         = (100 + 8000) / 2  = 4050 Hz
// --------------------------------------------------------------------------
TEST(VirtualAnalogFilterStage, CutoffSmoothingFollowsLogCurve_ModularRK4) {
    VirtualAnalogFilterStage s;
    s.setUniform(VirtualAnalogFilterStage::StageType::Ladder);
    s.setNumStages(4);
    s.setCutoffFrequency(100.0);
    s.prepareToPlay(kSampleRate, 64, 1);

    s.setCutoffFrequency(8000.0);

    // Compute half-ramp from the class's smoothing time so the test stays
    // robust to future tuning (VA wrapper currently uses 0.2 ms vs the
    // ladder's 2 ms — different windows, same midpoint identity).
    const int halfRampSamples =
        static_cast<int>(VirtualAnalogFilterStage::kSmoothingTimeSec * kSampleRate * 0.5);
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

// --------------------------------------------------------------------------
// Scaffolding — basic process smoke test. White noise in at R=2, no NaN/Inf
// out and peak bounded. Modeled on the benchmark's StabilityAcrossResonance
// pattern but tighter / faster.
// --------------------------------------------------------------------------
TEST(VirtualAnalogFilterStage, ProcessProducesFiniteOutput_ModularRK4) {
    VirtualAnalogFilterStage s;
    s.setUniform(VirtualAnalogFilterStage::StageType::Ladder);
    s.setNumStages(4);
    s.setCutoffFrequency(1000.0);
    s.setResonance(2.0);
    s.prepareToPlay(kSampleRate, kBlockSize, 2);

    std::mt19937 rng(0x1234u);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);

    juce::AudioBuffer<double> buf(2, kBlockSize);
    for (int b = 0; b < 50; ++b) {
        for (int ch = 0; ch < buf.getNumChannels(); ++ch) {
            for (int i = 0; i < buf.getNumSamples(); ++i) {
                buf.getWritePointer(ch)[i] = dist(rng);
            }
        }
        s.process(buf);

        for (int ch = 0; ch < buf.getNumChannels(); ++ch) {
            for (int i = 0; i < buf.getNumSamples(); ++i) {
                const double v = buf.getReadPointer(ch)[i];
                ASSERT_TRUE(std::isfinite(v)) << "block " << b << " ch " << ch << " i " << i;
                ASSERT_LT(std::abs(v), 5.0) << "runaway: " << v;
            }
        }
    }
}

// --------------------------------------------------------------------------
// Scaffolding — cutoff/resonance clamping. Pure API sanity, mirrors the
// LadderTPTStage clamping tests.
// --------------------------------------------------------------------------
TEST(VirtualAnalogFilterStage, CutoffAndResonanceClamping_ModularRK4) {
    VirtualAnalogFilterStage s;
    s.setUniform(VirtualAnalogFilterStage::StageType::Ladder);
    s.prepareToPlay(kSampleRate, kBlockSize, 1);

    s.setCutoffFrequency(-100.0);
    EXPECT_GE(s.getCutoffFrequency(), 20.0);

    s.setCutoffFrequency(1.0e9);
    EXPECT_LE(s.getCutoffFrequency(), 0.45 * kSampleRate + 1e-9);

    s.setResonance(-1.0);
    EXPECT_DOUBLE_EQ(0.0, s.getResonance());

    s.setResonance(100.0);
    EXPECT_GT(s.getResonance(), 0.0);
    EXPECT_LE(s.getResonance(), 4.0);
}
