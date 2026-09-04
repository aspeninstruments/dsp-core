#include <gtest/gtest.h>
#include <dsp_core/dsp_core.h>
#include <juce_events/juce_events.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>

namespace dsp_core_test {

namespace {

std::vector<double> makeCurve(double gain, double phase) {
    std::vector<double> curve(static_cast<size_t>(dsp_core::LaneMixer::TABLE_SIZE));
    for (int i = 0; i < dsp_core::LaneMixer::TABLE_SIZE; ++i) {
        const double x = -1.0 + 2.0 * i / (dsp_core::LaneMixer::TABLE_SIZE - 1);
        curve[static_cast<size_t>(i)] = gain * std::tanh(3.0 * x + phase);
    }
    return curve;
}

} // namespace

/**
 * Races the message-thread preset-restore mutation path against the
 * BDD-LUTRenderer worker thread, reproducing the Reaper AU fast-preset-switch
 * crash: LaneMixer::fromValueTree frees/reallocates lane curveData vectors
 * while the worker's computeSum/computeScan iterates them.
 *
 * Pre-fix this is a use-after-free (reliable under ASan; intermittent crash
 * otherwise). Post-fix the structural mutators serialize with the worker via
 * the render-lock mutation guard, with no change to this test's code.
 */
class SeamlessConcurrencyTest : public ::testing::Test {
  protected:
    void SetUp() override {
        stf = std::make_unique<dsp_core::SeamlessTransferFunction>();
        stf->prepareToPlay(44100.0, 512);
        stf->startSeamlessUpdates();
    }

    void TearDown() override {
        stf->stopSeamlessUpdates();
        stf.reset();
    }

    juce::ScopedJuceInitialiser_GUI juceInit_; // MessageManager for Timer/AsyncUpdater + thread asserts
    std::unique_ptr<dsp_core::SeamlessTransferFunction> stf;

    juce::ValueTree buildPresetTree(int laneCount) {
        auto& mixer = stf->getLaneMixer();
        mixer.resetToDefaults();
        while (mixer.getActiveLaneCount() < laneCount) {
            mixer.addLane();
        }
        for (int i = 0; i < laneCount; ++i) {
            mixer.setLaneCurveData(i, makeCurve(0.5 + 0.5 * (i % 3), 0.1 * i));
        }
        return mixer.toValueTree();
    }
};

TEST_F(SeamlessConcurrencyTest, StructuralMutationDuringWorkerRender_NoUseAfterFree) {
    auto& mixer = stf->getLaneMixer();

    // Audio-consumer thread: rotates the triple buffer and completes
    // crossfades (checkForNewLUT clears newLUTReady) so the worker's
    // renderIfNeeded gates keep opening — without it the worker never enters
    // doRender and the race window never exists. This models the real host,
    // whose render callback keeps running through a preset load.
    std::atomic<bool> stop{false};
    std::thread audioConsumer([&] {
        juce::AudioBuffer<double> buffer(2, 512);
        while (!stop.load(std::memory_order_acquire)) {
            buffer.clear();
            stf->processBuffer(buffer);
            std::this_thread::yield();
        }
    });

    // Hammer the worker awake from another thread so it is mid-computeSum as
    // often as possible while the main thread frees/reallocates lane storage.
    auto* trigger = stf->getRenderTrigger();
    ASSERT_NE(trigger, nullptr);
    std::thread hammer([&] {
        while (!stop.load(std::memory_order_acquire)) {
            trigger->requestRender();
            std::this_thread::yield();
        }
    });

    // The structural mutation sequence of a real preset/.tfunc load:
    // resetToDefaults() frees every lane's curve vector (lane = Lane{}),
    // then addLane + setLaneCurveData reallocate fresh storage — while the
    // worker's doRender iterates exactly those vectors.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(1500);
    int iterations = 0;
    int lastLaneCount = 0;
    while (std::chrono::steady_clock::now() < deadline) {
        lastLaneCount = (iterations % 2 == 0) ? 60 : 5;
        mixer.resetToDefaults();
        while (mixer.getActiveLaneCount() < lastLaneCount) {
            mixer.addLane();
        }
        while (mixer.getActiveLaneCount() > lastLaneCount) {
            mixer.removeLane(mixer.getActiveLaneCount() - 1); // shrink_to_fit frees the vector
        }
        for (int lane = 0; lane < lastLaneCount; ++lane) {
            mixer.setLaneCurveData(lane, makeCurve(0.5 + 0.5 * (lane % 3), 0.1 * lane));
        }
        if (iterations % 4 == 0) {
            stf->renderLUTImmediate();
        }
        ++iterations;
    }

    stop.store(true, std::memory_order_release);
    hammer.join();
    audioConsumer.join();

    // Threshold accommodates sanitizer builds (~10x slowdown, and each
    // iteration serializes with worker renders under the interlock).
    EXPECT_GT(iterations, 3) << "loop should have made meaningful progress";
    EXPECT_EQ(mixer.getActiveLaneCount(), lastLaneCount);

    std::vector<double> sum(static_cast<size_t>(dsp_core::LaneMixer::TABLE_SIZE), 0.0);
    mixer.computeSum(sum.data(), dsp_core::LaneMixer::TABLE_SIZE);
    for (const double v : sum) {
        ASSERT_TRUE(std::isfinite(v));
    }
}

/**
 * Races the synchronous renderLUTImmediate path (preset load) against the
 * worker's own doRender: pre-fix both write the same worker-target LUT buffer
 * with no mutual exclusion (renderLUTImmediate takes neither renderMutex_ nor
 * the newLUTReady gate). Torn writes are a data race — TSan is the arbiter;
 * the finite-output assertions catch gross corruption everywhere.
 */
TEST_F(SeamlessConcurrencyTest, RenderLUTImmediate_DuringWorkerRender_NoTornLUT) {
    auto& mixer = stf->getLaneMixer();

    std::atomic<bool> stop{false};
    std::thread audioConsumer([&] { // keeps the worker's doRender gate open
        juce::AudioBuffer<double> buffer(2, 512);
        while (!stop.load(std::memory_order_acquire)) {
            buffer.clear();
            stf->processBuffer(buffer);
            std::this_thread::yield();
        }
    });

    auto* trigger = stf->getRenderTrigger();
    ASSERT_NE(trigger, nullptr);
    std::thread hammer([&] {
        while (!stop.load(std::memory_order_acquire)) {
            trigger->requestRender();
            std::this_thread::yield();
        }
    });

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(1000);
    double amplitude = 0.1;
    while (std::chrono::steady_clock::now() < deadline) {
        amplitude = (amplitude > 0.9) ? 0.1 : amplitude + 0.05;
        mixer.setLaneAmplitude(0, amplitude); // bumps mix version → worker renders
        stf->renderLUTImmediate();
    }

    stop.store(true, std::memory_order_release);
    hammer.join();
    audioConsumer.join();

    juce::AudioBuffer<double> buffer(2, 4096);
    for (int ch = 0; ch < 2; ++ch) {
        auto* data = buffer.getWritePointer(ch);
        for (int i = 0; i < 4096; ++i) {
            data[i] = 0.9 * std::sin(2.0 * juce::MathConstants<double>::pi * 440.0 * i / 44100.0);
        }
    }
    stf->processBuffer(buffer);
    for (int ch = 0; ch < 2; ++ch) {
        const auto* data = buffer.getReadPointer(ch);
        for (int i = 0; i < 4096; ++i) {
            ASSERT_TRUE(std::isfinite(data[i])) << "ch=" << ch << " i=" << i;
        }
    }
}

} // namespace dsp_core_test
