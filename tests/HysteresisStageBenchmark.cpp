// HysteresisStage throughput benchmark (code-cleanse DSP-001).
//
// Measures the steady-state hysteresis path — the RK4 loop whose nonlinearity
// evaluations dominate plugin CPU at 16x oversampling. Used to compare the
// std::function nonlinearity dispatch against the direct
// SeamlessTransferFunction pointer path.
//
// MEASURED RESULT (2026-07, Apple Silicon, Release): replacing the
// std::function members with a direct SeamlessTransferFunction* call was
// ~1.9% SLOWER (6 interleaved A/B pairs, baseline median 78.1 vs 79.6
// ns/channel-sample; bit-identical output). The monomorphic std::function
// call site is effectively free next to the RK4 math, so the refactor was
// not landed. Re-measure here before re-attempting (especially on x86_64 —
// this null result is ARM-only evidence).
//
// Dev-only: not registered with ctest (timing output is meaningless on shared
// CI runners). Run locally, Release build:
//   ./build-release/modules/dsp-core/tests/hysteresis_stage_benchmark

#include <gtest/gtest.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <dsp_core/dsp_core.h>
#include <chrono>
#include <cmath>
#include <iostream>

using namespace dsp_core;
using namespace dsp_core::audio_pipeline;

namespace {

constexpr double kOversampledRate = 48000.0 * 16.0;
constexpr int kBlockSize = 512;
constexpr int kNumChannels = 2;
constexpr int kWarmupIterations = 100;
constexpr int kTimedIterations = 2000;

juce::AudioBuffer<double> makeSineBuffer() {
    juce::AudioBuffer<double> buffer(kNumChannels, kBlockSize);
    for (int ch = 0; ch < kNumChannels; ++ch) {
        auto* data = buffer.getWritePointer(ch);
        for (int i = 0; i < kBlockSize; ++i) {
            data[i] = 0.5 * std::sin(2.0 * juce::MathConstants<double>::pi * 100.0 * i / kOversampledRate);
        }
    }
    return buffer;
}

} // namespace

TEST(HysteresisStageBenchmark, SteadyStateThroughput) {
    // Default-constructed SeamlessTransferFunction is the identity LUT — the
    // dispatch cost under test is the same for any curve.
    SeamlessTransferFunction tf;
    HysteresisStage stage(tf);

    // Enable BEFORE prepareToPlay so previousEnabled_ initializes true and the
    // stage runs steady-state hysteresis with no warmup/crossfade phases.
    stage.setHysteresisEnabled(true);
    stage.prepareToPlay(kOversampledRate, kBlockSize, kNumChannels);

    const auto source = makeSineBuffer();
    juce::AudioBuffer<double> work(kNumChannels, kBlockSize);

    // Warm up caches, smoothers, and the DC blocker.
    for (int i = 0; i < kWarmupIterations; ++i) {
        work.makeCopyOf(source);
        stage.process(work);
    }

    double sink = 0.0; // Defeats dead-code elimination of the timed loop.
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < kTimedIterations; ++i) {
        work.makeCopyOf(source);
        stage.process(work);
        sink += work.getSample(0, kBlockSize - 1);
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;

    const auto totalNs = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
    const long long channelSamples = static_cast<long long>(kTimedIterations) * kBlockSize * kNumChannels;
    const double nsPerChannelSample = static_cast<double>(totalNs) / static_cast<double>(channelSamples);
    const double totalMs = static_cast<double>(totalNs) / 1.0e6;

    std::cout << "HysteresisStage steady-state: " << totalMs << " ms for " << kTimedIterations << " blocks of "
              << kBlockSize << "x" << kNumChannels << " (" << nsPerChannelSample << " ns/channel-sample, sink=" << sink
              << ")\n";

    EXPECT_TRUE(std::isfinite(sink));
}
