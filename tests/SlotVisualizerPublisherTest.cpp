#include <gtest/gtest.h>
#include <dsp_core/dsp_core.h>
#include <atomic>
#include <cmath>

using namespace dsp_core::audio_pipeline;

namespace {

constexpr double kSampleRate = 48000.0;
constexpr int kBlockSize = 512;

juce::AudioBuffer<double> makeBuffer(int numChannels, int numSamples) {
    juce::AudioBuffer<double> buf(numChannels, numSamples);
    buf.clear();
    return buf;
}

void fillSineBuffer(juce::AudioBuffer<double>& buf, double freqHz, double sampleRate, double amplitude = 1.0) {
    const int n = buf.getNumSamples();
    const double twoPi = 6.283185307179586;
    for (int ch = 0; ch < buf.getNumChannels(); ++ch) {
        auto* w = buf.getWritePointer(ch);
        for (int i = 0; i < n; ++i) {
            w[i] = amplitude * std::sin(twoPi * freqHz * static_cast<double>(i) / sampleRate);
        }
    }
}

} // namespace

// --------------------------------------------------------------------------
// Envelope ring + writeIndex behaviour
// --------------------------------------------------------------------------

TEST(SlotVisualizerPublisherEnvRing, WriteIndexWrapsViaBitmask) {
    SlotVisualizerPublisher pub;
    std::atomic<double> storage{0.0};
    EnvelopeFollowerStage env(storage);
    env.prepareToPlay(kSampleRate, kBlockSize, 1);
    env.setEnabled(true);
    env.setSensitivityLinear(1.0);
    env.setVisualizerPublisher(&pub);

    // Push many blocks so the ring wraps multiple times. With 512-entry ring
    // at 48 kHz / 300 ms target history, downsample ratio ≈ 28 → 18 ring
    // writes per 512-sample block → ~28 blocks to fill the ring.
    auto buf = makeBuffer(1, kBlockSize);
    fillSineBuffer(buf, 440.0, kSampleRate);
    const int kBlocks = 200; // > 3 full ring wraps
    for (int b = 0; b < kBlocks; ++b) {
        // Re-fill each block so the envelope is constantly excited.
        fillSineBuffer(buf, 440.0, kSampleRate);
        env.process(buf);
    }

    // writeIndex must remain inside [0, kEnvRingSize).
    const int idx = pub.envWriteIndex.load(std::memory_order_acquire);
    EXPECT_GE(idx, 0);
    EXPECT_LT(idx, SlotVisualizerPublisher::kEnvRingSize);

    // envVersion bumps once per block — should equal kBlocks.
    EXPECT_EQ(pub.envVersion.load(std::memory_order_acquire), static_cast<uint64_t>(kBlocks));
}

TEST(SlotVisualizerPublisherEnvRing, RingValuesAreInUnitRangeAndNonZeroForActiveSignal) {
    SlotVisualizerPublisher pub;
    std::atomic<double> storage{0.0};
    EnvelopeFollowerStage env(storage);
    env.prepareToPlay(kSampleRate, kBlockSize, 1);
    env.setEnabled(true);
    env.setSensitivityLinear(1.0);
    env.setAttackReleaseSec(0.001, 0.050);
    env.setVisualizerPublisher(&pub);

    auto buf = makeBuffer(1, kBlockSize);
    for (int b = 0; b < 50; ++b) {
        fillSineBuffer(buf, 440.0, kSampleRate, /*amplitude=*/0.5);
        env.process(buf);
    }

    // Every populated ring slot must satisfy 0 <= v <= 1; some slot must be
    // > 0 since the input is a sustained sine.
    bool sawNonZero = false;
    for (int i = 0; i < SlotVisualizerPublisher::kEnvRingSize; ++i) {
        const double v = pub.envRing[static_cast<size_t>(i)];
        EXPECT_GE(v, 0.0);
        EXPECT_LE(v, 1.0);
        if (v > 0.01) {
            sawNonZero = true;
        }
    }
    EXPECT_TRUE(sawNonZero) << "Envelope ring stayed at zero despite a sustained 0.5 sine input";
}

TEST(SlotVisualizerPublisherEnvRing, DisabledStageDoesNotWriteRing) {
    SlotVisualizerPublisher pub;
    std::atomic<double> storage{0.0};
    EnvelopeFollowerStage env(storage);
    env.prepareToPlay(kSampleRate, kBlockSize, 1);
    env.setEnabled(false); // explicit: don't run the detector
    env.setVisualizerPublisher(&pub);

    auto buf = makeBuffer(1, kBlockSize);
    fillSineBuffer(buf, 440.0, kSampleRate);
    for (int b = 0; b < 10; ++b) {
        env.process(buf);
    }

    // No ring entries should have been written; no version bumps either.
    EXPECT_EQ(pub.envWriteIndex.load(std::memory_order_acquire), 0);
    EXPECT_EQ(pub.envVersion.load(std::memory_order_acquire), 0ULL);
}

// --------------------------------------------------------------------------
// LFO shape-version + phase publishing
// --------------------------------------------------------------------------

TEST(SlotVisualizerPublisherLfo, FirstProcessBumpsShapeVersionExactlyOnce) {
    SlotVisualizerPublisher pub;
    std::atomic<double> storage{0.0};
    LfoStage lfo(storage);
    lfo.prepareToPlay(kSampleRate, kBlockSize, 1);
    lfo.setEnabled(true);
    lfo.setShape(LfoStage::Shape::Sin);
    lfo.setRateHz(0.5); // < 1 Hz so cyclePosition floor doesn't advance during a few blocks
    lfo.setVisualizerPublisher(&pub);

    auto buf = makeBuffer(1, kBlockSize);
    lfo.process(buf);
    const uint64_t afterFirst = pub.lfoShapeVersion.load(std::memory_order_acquire);
    EXPECT_EQ(afterFirst, 1ULL) << "Initial process() after attach should bump shapeVersion exactly once";

    // Subsequent processes with no shape/seed/cycle change must not bump.
    for (int b = 0; b < 5; ++b) {
        lfo.process(buf);
    }
    EXPECT_EQ(pub.lfoShapeVersion.load(std::memory_order_acquire), afterFirst);
}

TEST(SlotVisualizerPublisherLfo, ShapeChangeBumpsShapeVersion) {
    SlotVisualizerPublisher pub;
    std::atomic<double> storage{0.0};
    LfoStage lfo(storage);
    lfo.prepareToPlay(kSampleRate, kBlockSize, 1);
    lfo.setEnabled(true);
    lfo.setShape(LfoStage::Shape::Sin);
    lfo.setRateHz(0.5);
    lfo.setVisualizerPublisher(&pub);

    auto buf = makeBuffer(1, kBlockSize);
    lfo.process(buf);
    const uint64_t baseline = pub.lfoShapeVersion.load(std::memory_order_acquire);

    lfo.setShape(LfoStage::Shape::Tri);
    lfo.process(buf);
    EXPECT_EQ(pub.lfoShapeVersion.load(std::memory_order_acquire), baseline + 1);

    lfo.setShape(LfoStage::Shape::Saw);
    lfo.process(buf);
    EXPECT_EQ(pub.lfoShapeVersion.load(std::memory_order_acquire), baseline + 2);
}

TEST(SlotVisualizerPublisherLfo, RandomShapeBumpsOnceAtCycleBoundary) {
    SlotVisualizerPublisher pub;
    std::atomic<double> storage{0.0};
    LfoStage lfo(storage);
    lfo.prepareToPlay(kSampleRate, kBlockSize, 1);
    lfo.setEnabled(true);
    lfo.setShape(LfoStage::Shape::Random);
    // 1 cycle per second; one block = ~10 ms; 100 blocks ≈ 1.067 s, so cycle
    // floor should advance from 0 → 1 exactly once across the run.
    lfo.setRateHz(1.0);
    lfo.setVisualizerPublisher(&pub);

    auto buf = makeBuffer(1, kBlockSize);
    lfo.process(buf);
    const uint64_t initial = pub.lfoShapeVersion.load(std::memory_order_acquire);
    // Initial process bumps version once (first attach, shape latched).

    int bumps = 0;
    uint64_t previous = initial;
    for (int b = 0; b < 100; ++b) {
        lfo.process(buf);
        const uint64_t now = pub.lfoShapeVersion.load(std::memory_order_acquire);
        if (now != previous) {
            ++bumps;
            previous = now;
        }
    }
    // Exactly one cycle-boundary crossing in ~1 second at 1 Hz.
    EXPECT_EQ(bumps, 1);
    EXPECT_EQ(pub.lfoShapeVersion.load(std::memory_order_acquire), initial + 1);
}

TEST(SlotVisualizerPublisherLfo, PhaseAndCyclePositionPublished) {
    SlotVisualizerPublisher pub;
    std::atomic<double> storage{0.0};
    LfoStage lfo(storage);
    lfo.prepareToPlay(kSampleRate, kBlockSize, 1);
    lfo.setEnabled(true);
    lfo.setShape(LfoStage::Shape::Saw);
    lfo.setRateHz(2.0);
    lfo.setVisualizerPublisher(&pub);

    auto buf = makeBuffer(1, kBlockSize);
    // Run enough samples for ~1 full cycle (48000/2 = 24000 samples → ~47 blocks).
    for (int b = 0; b < 50; ++b) {
        lfo.process(buf);
    }
    const double phase = pub.lfoPhase01.load(std::memory_order_acquire);
    const double cyclePos = pub.lfoCyclePosition.load(std::memory_order_acquire);
    EXPECT_GE(phase, 0.0);
    EXPECT_LT(phase, 1.0);
    EXPECT_GT(cyclePos, 0.5); // > half a cycle has elapsed
}

// --------------------------------------------------------------------------
// ModulatorSlotStage type switch propagates activeKind + version bumps
// --------------------------------------------------------------------------

TEST(ModulatorSlotStageVisualizer, SetTypeUpdatesActiveKindAndBumpsVersions) {
    SlotVisualizerPublisher pub;
    std::atomic<double> storage{0.0};
    ModulatorSlotStage slot(storage);
    slot.setVisualizerPublisher(&pub);

    // Default type is Envelope (= 0).
    EXPECT_EQ(pub.activeKind.load(std::memory_order_acquire), 0);

    const uint64_t envBefore = pub.envVersion.load(std::memory_order_acquire);
    const uint64_t lfoBefore = pub.lfoShapeVersion.load(std::memory_order_acquire);
    slot.setType(ModulatorSlotStage::Type::Lfo);
    EXPECT_EQ(pub.activeKind.load(std::memory_order_acquire), 1);
    EXPECT_GT(pub.envVersion.load(std::memory_order_acquire), envBefore);
    EXPECT_GT(pub.lfoShapeVersion.load(std::memory_order_acquire), lfoBefore);

    // Idempotent setType — re-setting to the same type doesn't bump.
    const uint64_t envAfter = pub.envVersion.load(std::memory_order_acquire);
    const uint64_t lfoAfter = pub.lfoShapeVersion.load(std::memory_order_acquire);
    slot.setType(ModulatorSlotStage::Type::Lfo);
    EXPECT_EQ(pub.envVersion.load(std::memory_order_acquire), envAfter);
    EXPECT_EQ(pub.lfoShapeVersion.load(std::memory_order_acquire), lfoAfter);
}
