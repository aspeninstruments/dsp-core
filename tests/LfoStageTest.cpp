#include <gtest/gtest.h>
#include <dsp_core/dsp_core.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <vector>

using namespace dsp_core::audio_pipeline;

namespace {

constexpr double kSampleRate = 48000.0;
constexpr int kBlockSize = 512;

juce::AudioBuffer<double> makeBuffer(int numChannels, int numSamples) {
    juce::AudioBuffer<double> buf(numChannels, numSamples);
    buf.clear();
    return buf;
}

class LfoStageTest : public ::testing::Test {
  protected:
    void SetUp() override {
        stage_ = std::make_unique<LfoStage>(value_);
        stage_->prepareToPlay(kSampleRate, kBlockSize, 2);
        stage_->setEnabled(true);
    }

    std::atomic<double> value_{0.0};
    std::unique_ptr<LfoStage> stage_;
};

} // namespace

TEST_F(LfoStageTest, DisabledPublishesZero) {
    stage_->setEnabled(false);
    stage_->setRateHz(1.0);
    auto buf = makeBuffer(1, kBlockSize);
    stage_->process(buf);
    EXPECT_DOUBLE_EQ(value_.load(), 0.0);
}

TEST_F(LfoStageTest, SineShapeIsUnipolarInRange) {
    stage_->setShape(LfoStage::Shape::Sin);
    stage_->setUnits(LfoStage::Units::Hz);
    stage_->setRateHz(2.0);

    // Process several blocks and confirm published values stay in [0, 1].
    for (int i = 0; i < 200; ++i) {
        auto buf = makeBuffer(1, kBlockSize);
        stage_->process(buf);
        const double v = value_.load();
        EXPECT_GE(v, 0.0);
        EXPECT_LE(v, 1.0);
    }
}

TEST_F(LfoStageTest, SawShapeRamps) {
    stage_->setShape(LfoStage::Shape::Saw);
    stage_->setUnits(LfoStage::Units::Hz);
    stage_->setRateHz(0.1); // very slow so phase advances predictably

    // After a few blocks at 0.1 Hz, phase should still be very small → saw value small.
    auto buf = makeBuffer(1, kBlockSize);
    stage_->process(buf);
    const double after1Block = value_.load();
    EXPECT_GE(after1Block, 0.0);
    EXPECT_LT(after1Block, 0.05);
}

TEST_F(LfoStageTest, SquareShapeIsBinary) {
    stage_->setShape(LfoStage::Shape::Sq);
    stage_->setUnits(LfoStage::Units::Hz);
    stage_->setRateHz(1.0);

    for (int i = 0; i < 100; ++i) {
        auto buf = makeBuffer(1, kBlockSize);
        stage_->process(buf);
        const double v = value_.load();
        EXPECT_TRUE(v == 0.0 || v == 1.0) << "Square LFO produced non-binary value " << v;
    }
}

TEST_F(LfoStageTest, AudioBufferIsPassthrough) {
    stage_->setShape(LfoStage::Shape::Sin);
    stage_->setRateHz(1.0);

    auto buf = makeBuffer(2, kBlockSize);
    for (int ch = 0; ch < 2; ++ch) {
        auto* p = buf.getWritePointer(ch);
        for (int i = 0; i < kBlockSize; ++i) {
            p[i] = 0.42;
        }
    }
    stage_->process(buf);
    for (int ch = 0; ch < 2; ++ch) {
        const auto* p = buf.getReadPointer(ch);
        for (int i = 0; i < kBlockSize; ++i) {
            EXPECT_DOUBLE_EQ(p[i], 0.42);
        }
    }
}

TEST_F(LfoStageTest, BpmModeLocksToHostPpq) {
    stage_->setShape(LfoStage::Shape::Saw);
    stage_->setUnits(LfoStage::Units::BPM);
    stage_->setDivision(LfoStage::Division::Quarter);
    stage_->setFlavor(LfoStage::Flavor::Straight);
    stage_->setHostTransport(120.0, 0.0, /*isPlaying=*/true);

    auto buf = makeBuffer(1, kBlockSize);
    stage_->process(buf);
    const double atPpq0 = value_.load();
    // At ppq=0 the saw should be near 0; one 512-sample block at 120 BPM @ 48 kHz
    // advances ppq by ~0.021 beats and the published value is the LAST sample.
    EXPECT_LT(atPpq0, 0.05);

    stage_->setHostTransport(120.0, 0.5, /*isPlaying=*/true);
    stage_->process(buf);
    const double atHalfBeat = value_.load();
    // Quarter-note period: ppq=0.5 is mid-cycle for a 1/4-beat saw.
    EXPECT_NEAR(atHalfBeat, 0.5, 0.05);
}

TEST_F(LfoStageTest, PeriodInBeatsTable) {
    using D = LfoStage::Division;
    using F = LfoStage::Flavor;
    EXPECT_DOUBLE_EQ(LfoStage::periodInBeats(D::Quarter, F::Straight), 1.0);
    EXPECT_DOUBLE_EQ(LfoStage::periodInBeats(D::Eighth, F::Straight), 0.5);
    EXPECT_DOUBLE_EQ(LfoStage::periodInBeats(D::Sixteenth, F::Straight), 0.25);
    EXPECT_DOUBLE_EQ(LfoStage::periodInBeats(D::Bar, F::Straight), 4.0);
    EXPECT_DOUBLE_EQ(LfoStage::periodInBeats(D::Quarter, F::Triplet), 2.0 / 3.0);
    EXPECT_DOUBLE_EQ(LfoStage::periodInBeats(D::Quarter, F::Dotted), 1.5);
}

TEST_F(LfoStageTest, OffShapePublishesZero) {
    // Off doubles as the LFO's enable gate (paralleling ENV's Input = Off).
    // process() should short-circuit and publish 0 even with enabled_ = true
    // and a non-zero rate, so a slot whose Shape is Off contributes nothing
    // to the modulation sum.
    stage_->setShape(LfoStage::Shape::Off);
    stage_->setRateHz(5.0);
    auto buf = makeBuffer(1, kBlockSize);
    for (int i = 0; i < 10; ++i) {
        stage_->process(buf);
        EXPECT_DOUBLE_EQ(value_.load(), 0.0);
    }
}

TEST_F(LfoStageTest, RandomShapeIsUnipolarInRange) {
    stage_->setShape(LfoStage::Shape::Random);
    stage_->setUnits(LfoStage::Units::Hz);
    stage_->setRateHz(3.0);

    for (int i = 0; i < 400; ++i) {
        auto buf = makeBuffer(1, kBlockSize);
        stage_->process(buf);
        const double v = value_.load();
        EXPECT_GE(v, 0.0);
        EXPECT_LE(v, 1.0);
    }
}

TEST_F(LfoStageTest, RandomShapeDoesNotRepeatEachCycle) {
    stage_->setShape(LfoStage::Shape::Random);
    stage_->setUnits(LfoStage::Units::Hz);
    // 9.375 Hz @ 48 kHz / 512-sample blocks = exactly 10 blocks per cycle, so a
    // recorded block sits at the same intra-cycle phase one cycle apart.
    stage_->setRateHz(9.375);

    constexpr int kBlocksPerCycle = 10;
    auto recordCycle = [&]() {
        std::vector<double> values;
        for (int i = 0; i < kBlocksPerCycle; ++i) {
            auto buf = makeBuffer(1, kBlockSize);
            stage_->process(buf);
            values.push_back(value_.load());
        }
        return values;
    };

    const auto cycle1 = recordCycle();
    const auto cycle2 = recordCycle();

    // Same intra-cycle phase, different cycle: a genuinely random LFO must
    // produce a different waveform. The old phase-indexed Perlin sampling
    // retraced an identical pattern every cycle.
    bool differs = false;
    for (size_t i = 0; i < cycle1.size(); ++i) {
        if (std::fabs(cycle1[i] - cycle2[i]) > 1.0e-6) {
            differs = true;
            break;
        }
    }
    EXPECT_TRUE(differs) << "Random LFO repeated the exact same pattern across cycles";
}

TEST_F(LfoStageTest, RandomShapeReachesFullDepth) {
    stage_->setShape(LfoStage::Shape::Random);
    stage_->setUnits(LfoStage::Units::Hz);
    // ~0.7 cycles per 512-sample block: consecutive published values land in
    // decorrelated regions of noise space, so a few thousand blocks sample the
    // whole distribution including its strong excursions.
    stage_->setRateHz(65.0);

    double minValue = 1.0;
    double maxValue = 0.0;
    for (int i = 0; i < 4000; ++i) {
        auto buf = makeBuffer(1, kBlockSize);
        stage_->process(buf);
        const double v = value_.load();
        EXPECT_GE(v, 0.0);
        EXPECT_LE(v, 1.0);
        minValue = std::min(minValue, v);
        maxValue = std::max(maxValue, v);
    }

    // The raw 0.5 + 0.5*n mapping topped out around [0.25, 0.75] because
    // perlinNoise1D only spans ~±0.5. tanh soft-saturation must drive the LFO
    // to within striking distance of both rails.
    EXPECT_GT(maxValue, 0.95) << "Random LFO never approached the upper rail";
    EXPECT_LT(minValue, 0.05) << "Random LFO never approached the lower rail";
}

TEST_F(LfoStageTest, ResetClearsStorage) {
    stage_->setShape(LfoStage::Shape::Sin);
    stage_->setRateHz(5.0);
    auto buf = makeBuffer(1, kBlockSize);
    stage_->process(buf);
    EXPECT_GT(value_.load(), 0.0);

    stage_->reset();
    EXPECT_DOUBLE_EQ(value_.load(), 0.0);
}
