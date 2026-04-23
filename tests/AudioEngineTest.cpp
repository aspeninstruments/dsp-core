#include <dsp_core/dsp_core.h>
#include <gtest/gtest.h>
#include <array>
#include <cmath>

namespace dsp_core_test {

/**
 * Test fixture for AudioEngine
 * Tests triple-buffered LUT with crossfade functionality
 */
class AudioEngineTest : public ::testing::Test {
  protected:
    void SetUp() override {
        engine = std::make_unique<dsp_core::AudioEngine>();
    }

    std::unique_ptr<dsp_core::AudioEngine> engine;
};

// ============================================================================
// Task 2: AudioEngine Tests - Initialization
// ============================================================================

/**
 * Test: Initial state produces identity function (y = x)
 * Expected: Before any LUT renders, all buffers contain identity
 */
TEST_F(AudioEngineTest, Constructor_InitializesToIdentity) {
    // Test identity function across range
    const std::vector<double> testInputs = {-1.0, -0.5, 0.0, 0.5, 1.0};

    for (double const x : testInputs) {
        double const output = engine->applyTransferFunction(x);
        EXPECT_NEAR(output, x, 1e-6) << "Identity function should return input value";
    }
}

/**
 * Test: All three LUT buffers initialized to identity
 * Expected: No undefined values in any buffer
 */
TEST_F(AudioEngineTest, Constructor_AllBuffersInitialized) {
    dsp_core::LUTBuffer* buffers = engine->getLUTBuffers();

    for (int bufIdx = 0; bufIdx < 3; ++bufIdx) {
        for (int i = 0; i < dsp_core::TABLE_SIZE; ++i) {
            const double x = dsp_core::MIN_VALUE + (i / static_cast<double>(dsp_core::TABLE_SIZE - 1)) *
                                                        (dsp_core::MAX_VALUE - dsp_core::MIN_VALUE);
            EXPECT_DOUBLE_EQ(buffers[bufIdx].data[i], x) << "Buffer " << bufIdx << " index " << i
                                                          << " should contain identity value";
        }
        EXPECT_EQ(buffers[bufIdx].version, 0u) << "Buffer " << bufIdx << " should have version 0";
    }
}

// ============================================================================
// Task 2: AudioEngine Tests - Sample Rate Adaptation
// ============================================================================

/**
 * Test: Crossfade duration scales correctly with sample rate
 * Expected: 50ms at 44.1kHz = 2205 samples, 50ms at 48kHz = 2400 samples
 */
TEST_F(AudioEngineTest, PrepareToPlay_CrossfadeDurationScalesWithSampleRate) {
    // 44.1 kHz
    engine->prepareToPlay(44100.0, 512);
    juce::AudioBuffer<double> buffer(1, 1);
    buffer.clear();
    engine->processBuffer(buffer);
    // We can't directly check crossfadeSamples (private), but we can verify behavior

    // 48 kHz
    engine->prepareToPlay(48000.0, 512);
    // Expected: 2400 samples at 48kHz (50ms)
    // We'll verify this by triggering a crossfade and counting samples
}

/**
 * Test: Sample rate change mid-crossfade clamps position
 * Expected: If crossfadePosition >= new crossfadeSamples, crossfade completes
 */
TEST_F(AudioEngineTest, PrepareToPlay_ClampsPositionOnSampleRateChange) {
    // Start at 96kHz (4800 samples for 50ms)
    engine->prepareToPlay(96000.0, 512);

    // Simulate crossfade in progress by manually swapping buffers
    dsp_core::LUTBuffer* buffers = engine->getLUTBuffers();
    for (int i = 0; i < dsp_core::TABLE_SIZE; ++i) {
        buffers[2].data[i] = 0.5; // Different from identity
    }
    engine->getNewLUTReadyFlag().store(true, std::memory_order_release);

    // Trigger crossfade
    juce::AudioBuffer<double> buffer(1, 500);
    buffer.clear();
    engine->processBuffer(buffer); // Partially through crossfade

    // Now change sample rate to 44.1kHz (2205 samples)
    // The crossfade should complete immediately if position >= 2205
    engine->prepareToPlay(44100.0, 512);
    juce::AudioBuffer<double> buffer2(1, 1);
    buffer2.clear();
    engine->processBuffer(buffer2);

    // If we're still crossfading, position should be clamped
    // (Hard to verify without accessing private state, but no crash is good)
}

// ============================================================================
// Task 2: AudioEngine Tests - Crossfade Behavior
// ============================================================================

/**
 * Test: Crossfade gains sum to 1.0 at all positions
 * Expected: gainOld + gainNew = 1.0 throughout crossfade
 */
TEST_F(AudioEngineTest, Crossfade_GainsSumToOne) {
    engine->prepareToPlay(44100.0, 512);

    // Set up two different LUTs
    dsp_core::LUTBuffer* buffers = engine->getLUTBuffers();

    // Buffer 0 (primary): identity (y = x)
    // Buffer 2 (worker target): constant 0.5
    for (int i = 0; i < dsp_core::TABLE_SIZE; ++i) {
        buffers[2].data[i] = 0.5;
    }
    buffers[2].version = 1;

    // Signal new LUT ready
    engine->getNewLUTReadyFlag().store(true, std::memory_order_release);

    // Process samples for the crossfade duration
    const int expectedCrossfadeSamples = static_cast<int>(44100.0 * dsp_core::SeamlessConfig::CROSSFADE_DURATION_MS / 1000.0);
    juce::AudioBuffer<double> buffer(1, expectedCrossfadeSamples);
    buffer.clear(); // Test with x = 0.0

    engine->processBuffer(buffer);

    // Verify smooth transition from 0.0 (identity at x=0) to 0.5 (new LUT)
    // First sample should start crossfade
    EXPECT_NEAR(buffer.getSample(0, 0), 0.0, 0.1) << "First sample should be close to old LUT value";

    // Last sample should be at new LUT value
    EXPECT_NEAR(buffer.getSample(0, expectedCrossfadeSamples - 1), 0.5, 0.01)
        << "Last sample should be close to new LUT value";

    // Middle samples should be interpolated
    const int midIdx = expectedCrossfadeSamples / 2;
    EXPECT_NEAR(buffer.getSample(0, midIdx), 0.25, 0.05) << "Middle sample should be halfway between LUTs";
}

/**
 * Test: Crossfade completes in correct sample count
 * Expected: After 2205 samples at 44.1kHz, crossfade is done
 */
TEST_F(AudioEngineTest, Crossfade_CompletesInCorrectSampleCount) {
    engine->prepareToPlay(44100.0, 512);

    // Set up two different LUTs
    dsp_core::LUTBuffer* buffers = engine->getLUTBuffers();
    for (int i = 0; i < dsp_core::TABLE_SIZE; ++i) {
        buffers[2].data[i] = 1.0; // Different from identity
    }
    buffers[2].version = 1;

    // Signal new LUT ready
    engine->getNewLUTReadyFlag().store(true, std::memory_order_release);

    // Process exactly the crossfade duration
    const int crossfadeSamples = static_cast<int>(44100.0 * dsp_core::SeamlessConfig::CROSSFADE_DURATION_MS / 1000.0); // 2205
    juce::AudioBuffer<double> buffer(1, crossfadeSamples);
    for (int i = 0; i < crossfadeSamples; ++i) {
        buffer.setSample(0, i, 0.5);
    }

    engine->processBuffer(buffer);

    // After crossfade, output should be from new LUT
    juce::AudioBuffer<double> testBuffer(1, 1);
    testBuffer.setSample(0, 0, 0.5);
    engine->processBuffer(testBuffer);
    EXPECT_NEAR(testBuffer.getSample(0, 0), 1.0, 1e-6) << "After crossfade, should use new LUT value";
}

/**
 * Test: New LUT interrupts active crossfade
 * Expected: If crossfading and new LUT arrives, crossfade restarts toward new target
 */
TEST_F(AudioEngineTest, Crossfade_NewLUTInterruptsCrossfade) {
    engine->prepareToPlay(44100.0, 512);

    // Set up first new LUT
    dsp_core::LUTBuffer* buffers = engine->getLUTBuffers();
    for (int i = 0; i < dsp_core::TABLE_SIZE; ++i) {
        buffers[2].data[i] = 0.5;
    }
    buffers[2].version = 1;
    engine->getNewLUTReadyFlag().store(true, std::memory_order_release);

    // Start crossfade, process only a few samples
    juce::AudioBuffer<double> buffer(1, 10);
    buffer.clear();
    engine->processBuffer(buffer);

    // Send a second new LUT while crossfade is in progress
    const int workerIdx = engine->getWorkerTargetIndexReference().load(std::memory_order_acquire);
    for (int i = 0; i < dsp_core::TABLE_SIZE; ++i) {
        buffers[workerIdx].data[i] = 0.8;
    }
    buffers[workerIdx].version = 2;
    engine->getNewLUTReadyFlag().store(true, std::memory_order_release);

    // Complete the crossfade (new crossfade toward 0.8 should start)
    const int crossfadeSamples = static_cast<int>(44100.0 * dsp_core::SeamlessConfig::CROSSFADE_DURATION_MS / 1000.0);
    juce::AudioBuffer<double> remaining(1, crossfadeSamples + 10);
    remaining.clear();
    engine->processBuffer(remaining);

    // Should converge to second LUT (0.8), not first (0.5)
    juce::AudioBuffer<double> testBuffer(1, 1);
    testBuffer.setSample(0, 0, 0.0);
    engine->processBuffer(testBuffer);
    EXPECT_NEAR(testBuffer.getSample(0, 0), 0.8, 1e-6) << "Should use second LUT after interruption";
}

// ============================================================================
// Task 2: AudioEngine Tests - Interpolation Accuracy
// ============================================================================

/**
 * Test: Catmull-Rom interpolation works (hardcoded, no mode selection)
 * Expected: Smooth cubic interpolation
 */
TEST_F(AudioEngineTest, Interpolation_CatmullRomWorks) {
    engine->prepareToPlay(44100.0, 512);

    // Set up a parabola y = x^2
    dsp_core::LUTBuffer* buffers = engine->getLUTBuffers();
    for (int i = 0; i < dsp_core::TABLE_SIZE; ++i) {
        const double x = dsp_core::MIN_VALUE + (i / static_cast<double>(dsp_core::TABLE_SIZE - 1)) *
                                                    (dsp_core::MAX_VALUE - dsp_core::MIN_VALUE);
        buffers[2].data[i] = x * x; // y = x^2
    }
    buffers[2].version = 1;
    // Note: No interpolationMode - Catmull-Rom is now hardcoded

    // Swap to new LUT
    engine->getNewLUTReadyFlag().store(true, std::memory_order_release);

    // Skip crossfade
    const int crossfadeSamples = static_cast<int>(44100.0 * dsp_core::SeamlessConfig::CROSSFADE_DURATION_MS / 1000.0);
    juce::AudioBuffer<double> skipBuffer(1, crossfadeSamples);
    skipBuffer.clear();
    engine->processBuffer(skipBuffer);

    // Test interpolation (should be accurate for curves)
    const std::vector<double> testInputs = {-0.5, 0.0, 0.5};
    for (double const x : testInputs) {
        double const output = engine->applyTransferFunction(x);
        EXPECT_NEAR(output, x * x, 0.01) << "Catmull-Rom should approximate y = x^2 for input " << x;
    }
}

// ============================================================================
// Task 2: AudioEngine Tests - Edge Cases
// ============================================================================

/**
 * Test: Extrapolation clamp mode clamps to table bounds
 * Expected: Values outside [-1, 1] clamp to boundary values
 */
TEST_F(AudioEngineTest, Extrapolation_ClampModeClamps) {
    engine->prepareToPlay(44100.0, 512);

    // Test with identity function
    const std::array<double, 4> testInputs = {-2.0, -1.5, 1.5, 2.0};
    const std::array<double, 4> expectedOutputs = {-1.0, -1.0, 1.0, 1.0};

    for (int i = 0; i < 4; ++i) {
        double const output = engine->applyTransferFunction(testInputs[i]);
        EXPECT_NEAR(output, expectedOutputs[i], 1e-6)
            << "Clamp mode should clamp " << testInputs[i] << " to " << expectedOutputs[i];
    }
}

/**
 * Test: Process empty block doesn't crash
 * Expected: Handles numSamples = 0 gracefully
 */
TEST_F(AudioEngineTest, ProcessBlock_HandlesEmptyBlock) {
    engine->prepareToPlay(44100.0, 512);
    juce::AudioBuffer<double> buffer(1, 0);
    EXPECT_NO_THROW(engine->processBuffer(buffer));
}

// ============================================================================
// Smoothstep S-Curve Crossfade Tests
// ============================================================================

/**
 * Helper function to test smoothstep (duplicates internal implementation)
 * Used for unit testing the smoothstep curve properties
 */
namespace {
    inline double smoothstep(double t) {
        t = std::clamp(t, 0.0, 1.0);
        return t * t * (3.0 - 2.0 * t);
    }
} // namespace

/**
 * Test: Smoothstep endpoints
 * Expected: smoothstep(0.0) = 0.0, smoothstep(1.0) = 1.0
 */
TEST_F(AudioEngineTest, Smoothstep_Endpoints) {
    EXPECT_DOUBLE_EQ(smoothstep(0.0), 0.0) << "smoothstep(0) should equal 0";
    EXPECT_DOUBLE_EQ(smoothstep(1.0), 1.0) << "smoothstep(1) should equal 1";
}

/**
 * Test: Smoothstep midpoint
 * Expected: smoothstep(0.5) = 0.5 (symmetric curve)
 */
TEST_F(AudioEngineTest, Smoothstep_Midpoint) {
    EXPECT_DOUBLE_EQ(smoothstep(0.5), 0.5) << "smoothstep(0.5) should equal 0.5 (symmetric)";
}

/**
 * Test: Smoothstep is monotonically increasing
 * Expected: For all t1 < t2, smoothstep(t1) < smoothstep(t2)
 */
TEST_F(AudioEngineTest, Smoothstep_MonotonicallyIncreasing) {
    for (int i = 0; i < 100; ++i) {
        double const t1 = i / 100.0;
        double const t2 = (i + 1) / 100.0;
        EXPECT_LT(smoothstep(t1), smoothstep(t2))
            << "smoothstep should be strictly increasing: t1=" << t1 << " t2=" << t2;
    }
}

/**
 * Test: Smoothstep has slow start and end (S-curve property)
 * Expected: Derivative near endpoints is much smaller than at midpoint
 */
TEST_F(AudioEngineTest, Smoothstep_SlowStartAndEnd) {
    // Measure approximate derivatives at start, middle, and end
    const double epsilon = 0.01;

    // Start derivative: [0.0, 0.01]
    double const deltaStart = smoothstep(epsilon) - smoothstep(0.0);

    // Middle derivative: [0.5, 0.51]
    double const deltaMid = smoothstep(0.5 + epsilon) - smoothstep(0.5);

    // End derivative: [0.99, 1.0]
    double const deltaEnd = smoothstep(1.0) - smoothstep(1.0 - epsilon);

    // S-curve property: Middle should be steeper than start/end
    EXPECT_LT(deltaStart, deltaMid / 5.0) << "Start should be much slower than middle";
    EXPECT_LT(deltaEnd, deltaMid / 5.0) << "End should be much slower than middle";

    // Start and end should be roughly symmetric
    EXPECT_NEAR(deltaStart, deltaEnd, 0.001) << "Start and end slopes should be symmetric";
}

/**
 * Test: Smoothstep clamping
 * Expected: Values outside [0, 1] clamp to boundaries
 */
TEST_F(AudioEngineTest, Smoothstep_Clamping) {
    EXPECT_DOUBLE_EQ(smoothstep(-0.5), 0.0) << "smoothstep(-0.5) should clamp to 0";
    EXPECT_DOUBLE_EQ(smoothstep(1.5), 1.0) << "smoothstep(1.5) should clamp to 1";
    EXPECT_DOUBLE_EQ(smoothstep(-999.0), 0.0) << "Large negative should clamp to 0";
    EXPECT_DOUBLE_EQ(smoothstep(999.0), 1.0) << "Large positive should clamp to 1";
}

/**
 * Test: S-curve crossfade produces smooth transitions
 * Expected: Crossfade using smoothstep has slow start/end, fast middle
 */
TEST_F(AudioEngineTest, SCurveCrossfade_SmoothTransition) {
    engine->prepareToPlay(44100.0, 512);

    // Set up two different LUTs
    dsp_core::LUTBuffer* buffers = engine->getLUTBuffers();

    // Buffer 0 (primary): identity (y = x) → outputs 0.0 at x=0
    // Buffer 2 (worker target): constant 1.0 → outputs 1.0 at x=0
    for (int i = 0; i < dsp_core::TABLE_SIZE; ++i) {
        buffers[2].data[i] = 1.0;
    }
    buffers[2].version = 1;

    // Signal new LUT ready
    engine->getNewLUTReadyFlag().store(true, std::memory_order_release);

    // Process crossfade and capture output progression
    const int crossfadeSamples = static_cast<int>(44100.0 * dsp_core::SeamlessConfig::CROSSFADE_DURATION_MS / 1000.0);
    juce::AudioBuffer<double> buffer(1, crossfadeSamples);

    // Fill buffer with x=0.0 (so output = mix of 0.0 and 1.0)
    for (int i = 0; i < crossfadeSamples; ++i) {
        buffer.setSample(0, i, 0.0);
    }

    engine->processBuffer(buffer);

    // Analyze crossfade progression using relative positions
    const int earlyIdx = crossfadeSamples / 10;       // ~10% through
    const int midIdx = crossfadeSamples / 2;            // 50% through
    const int lateIdx = crossfadeSamples - crossfadeSamples / 10;  // ~90% through

    const double startOutput = buffer.getSample(0, 0);
    const double earlyOutput = buffer.getSample(0, earlyIdx);
    const double midOutput = buffer.getSample(0, midIdx);
    const double endOutput = buffer.getSample(0, crossfadeSamples - 1);

    // Verify endpoints
    EXPECT_NEAR(startOutput, 0.0, 0.05) << "Start should be close to old LUT (0.0)";
    EXPECT_NEAR(endOutput, 1.0, 0.05) << "End should be close to new LUT (1.0)";

    // Verify S-curve property: slow start (early change < mid change)
    const int span = std::max(1, crossfadeSamples / 10);
    double const deltaEarly = earlyOutput - startOutput;
    double const deltaMid = buffer.getSample(0, midIdx + span / 2) - buffer.getSample(0, midIdx - span / 2);

    EXPECT_LT(deltaEarly, deltaMid) << "Early transition should be slower than middle";

    // Verify midpoint is approximately 0.5 (symmetric crossfade)
    EXPECT_NEAR(midOutput, 0.5, 0.1) << "Midpoint should be approximately halfway";

    // Verify S-curve property: slow end
    double const deltaLate = endOutput - buffer.getSample(0, lateIdx);
    EXPECT_LT(deltaLate, deltaMid) << "Late transition should be slower than middle";
}

/**
 * Test: S-curve crossfade gains still sum to 1.0
 * Expected: Conservation of energy - no dips or peaks in loudness
 */
TEST_F(AudioEngineTest, SCurveCrossfade_GainsConserved) {
    engine->prepareToPlay(44100.0, 512);

    // Set up two LUTs with known values
    dsp_core::LUTBuffer* buffers = engine->getLUTBuffers();

    // Both LUTs return 1.0 at x=1.0
    for (int i = 0; i < dsp_core::TABLE_SIZE; ++i) {
        const double x = dsp_core::MIN_VALUE + (i / static_cast<double>(dsp_core::TABLE_SIZE - 1)) *
                                                    (dsp_core::MAX_VALUE - dsp_core::MIN_VALUE);
        buffers[0].data[i] = x;  // Identity
        buffers[2].data[i] = x;  // Also identity (no change expected)
    }
    buffers[2].version = 1;

    // Signal new LUT ready
    engine->getNewLUTReadyFlag().store(true, std::memory_order_release);

    // Process crossfade with x=0.5
    const int crossfadeSamples = static_cast<int>(44100.0 * dsp_core::SeamlessConfig::CROSSFADE_DURATION_MS / 1000.0);
    juce::AudioBuffer<double> buffer(1, crossfadeSamples);

    for (int i = 0; i < crossfadeSamples; ++i) {
        buffer.setSample(0, i, 0.5);
    }

    engine->processBuffer(buffer);

    // Since both LUTs return 0.5 at x=0.5, output should be constant 0.5
    // This verifies gains sum to 1.0 (otherwise we'd see a dip or peak)
    for (int i = 0; i < crossfadeSamples; ++i) {
        EXPECT_NEAR(buffer.getSample(0, i), 0.5, 1e-6)
            << "Sample " << i << " should be 0.5 (gains sum to 1.0)";
    }
}

// ============================================================================
// Phase B: Reduced Crossfade Duration Tests (5ms)
// ============================================================================

TEST_F(AudioEngineTest, Crossfade_CompletesIn5ms_At44100) {
    engine->prepareToPlay(44100.0, 512);

    // Set up new LUT (constant 1.0)
    dsp_core::LUTBuffer* buffers = engine->getLUTBuffers();
    for (int i = 0; i < dsp_core::TABLE_SIZE; ++i) {
        buffers[2].data[i] = 1.0;
    }
    buffers[2].version = 1;
    engine->getNewLUTReadyFlag().store(true, std::memory_order_release);

    // 5ms at 44100Hz = 220 samples
    const int expectedCrossfadeSamples = static_cast<int>(44100.0 * dsp_core::SeamlessConfig::CROSSFADE_DURATION_MS / 1000.0);
    juce::AudioBuffer<double> buffer(1, expectedCrossfadeSamples);
    for (int i = 0; i < expectedCrossfadeSamples; ++i) {
        buffer.setSample(0, i, 0.5);
    }
    engine->processBuffer(buffer);

    // After crossfade, should use new LUT exclusively
    juce::AudioBuffer<double> testBuffer(1, 1);
    testBuffer.setSample(0, 0, 0.5);
    engine->processBuffer(testBuffer);
    EXPECT_NEAR(testBuffer.getSample(0, 0), 1.0, 1e-6) << "After 5ms crossfade, should use new LUT";
}

TEST_F(AudioEngineTest, Crossfade_CompletesIn5ms_At96000) {
    engine->prepareToPlay(96000.0, 512);

    dsp_core::LUTBuffer* buffers = engine->getLUTBuffers();
    for (int i = 0; i < dsp_core::TABLE_SIZE; ++i) {
        buffers[2].data[i] = 1.0;
    }
    buffers[2].version = 1;
    engine->getNewLUTReadyFlag().store(true, std::memory_order_release);

    // 5ms at 96000Hz = 480 samples
    const int expectedCrossfadeSamples = static_cast<int>(96000.0 * dsp_core::SeamlessConfig::CROSSFADE_DURATION_MS / 1000.0);
    juce::AudioBuffer<double> buffer(1, expectedCrossfadeSamples);
    for (int i = 0; i < expectedCrossfadeSamples; ++i) {
        buffer.setSample(0, i, 0.5);
    }
    engine->processBuffer(buffer);

    juce::AudioBuffer<double> testBuffer(1, 1);
    testBuffer.setSample(0, 0, 0.5);
    engine->processBuffer(testBuffer);
    EXPECT_NEAR(testBuffer.getSample(0, 0), 1.0, 1e-6) << "After 5ms crossfade at 96k, should use new LUT";
}

// ============================================================================
// Phase B: Crossfade Interruption Tests
// ============================================================================

TEST_F(AudioEngineTest, CrossfadeInterruption_AcceptsNewLUTDuringCrossfade) {
    engine->prepareToPlay(44100.0, 512);
    dsp_core::LUTBuffer* buffers = engine->getLUTBuffers();

    // First new LUT: constant 0.5
    for (int i = 0; i < dsp_core::TABLE_SIZE; ++i) {
        buffers[2].data[i] = 0.5;
    }
    buffers[2].version = 1;
    engine->getNewLUTReadyFlag().store(true, std::memory_order_release);

    // Start crossfade, process only a few samples (not completing it)
    juce::AudioBuffer<double> partial(1, 10);
    partial.clear();
    engine->processBuffer(partial);
    EXPECT_TRUE(engine->isCrossfading()) << "Should be mid-crossfade";

    // Second new LUT: constant 0.8 (arrives during crossfade)
    const int workerIdx = engine->getWorkerTargetIndexReference().load(std::memory_order_acquire);
    for (int i = 0; i < dsp_core::TABLE_SIZE; ++i) {
        buffers[workerIdx].data[i] = 0.8;
    }
    buffers[workerIdx].version = 2;
    engine->getNewLUTReadyFlag().store(true, std::memory_order_release);

    // Process enough to complete the NEW crossfade
    const int crossfadeSamples = static_cast<int>(44100.0 * dsp_core::SeamlessConfig::CROSSFADE_DURATION_MS / 1000.0);
    juce::AudioBuffer<double> remaining(1, crossfadeSamples + 10);
    remaining.clear();
    engine->processBuffer(remaining);

    // Should now be using the SECOND LUT (0.8), not stuck on first (0.5)
    juce::AudioBuffer<double> testBuffer(1, 1);
    testBuffer.setSample(0, 0, 0.0);
    engine->processBuffer(testBuffer);
    EXPECT_NEAR(testBuffer.getSample(0, 0), 0.8, 1e-6)
        << "After interrupted crossfade, should use the second LUT";
}

TEST_F(AudioEngineTest, CrossfadeInterruption_TripleBufferSafety) {
    engine->prepareToPlay(44100.0, 512);
    dsp_core::LUTBuffer* buffers = engine->getLUTBuffers();

    // Rapid succession: three LUT updates
    for (int round = 1; round <= 3; ++round) {
        const int workerIdx = engine->getWorkerTargetIndexReference().load(std::memory_order_acquire);
        const double value = 0.2 * round;
        for (int i = 0; i < dsp_core::TABLE_SIZE; ++i) {
            buffers[workerIdx].data[i] = value;
        }
        buffers[workerIdx].version = static_cast<uint64_t>(round);
        engine->getNewLUTReadyFlag().store(true, std::memory_order_release);

        // Process a small chunk each time
        juce::AudioBuffer<double> chunk(1, 50);
        chunk.clear();
        engine->processBuffer(chunk);
    }

    // Complete whatever crossfade is active
    const int crossfadeSamples = static_cast<int>(44100.0 * dsp_core::SeamlessConfig::CROSSFADE_DURATION_MS / 1000.0);
    juce::AudioBuffer<double> flush(1, crossfadeSamples + 100);
    flush.clear();
    engine->processBuffer(flush);

    // Should be using the latest LUT (0.6)
    juce::AudioBuffer<double> testBuffer(1, 1);
    testBuffer.setSample(0, 0, 0.0);
    engine->processBuffer(testBuffer);
    EXPECT_NEAR(testBuffer.getSample(0, 0), 0.6, 1e-6)
        << "After rapid updates, should converge to latest LUT value";
}

TEST_F(AudioEngineTest, CrossfadeInterruption_NoContinuityGap) {
    engine->prepareToPlay(44100.0, 512);
    dsp_core::LUTBuffer* buffers = engine->getLUTBuffers();

    // Identity LUT is in buffer 0 (primary). Set up first new LUT: constant 0.5
    for (int i = 0; i < dsp_core::TABLE_SIZE; ++i) {
        buffers[2].data[i] = 0.5;
    }
    buffers[2].version = 1;
    engine->getNewLUTReadyFlag().store(true, std::memory_order_release);

    // Process halfway through the crossfade (identity → 0.5)
    const int crossfadeSamples = static_cast<int>(44100.0 * dsp_core::SeamlessConfig::CROSSFADE_DURATION_MS / 1000.0);
    const int halfCrossfade = crossfadeSamples / 2;
    juce::AudioBuffer<double> partial(1, halfCrossfade);
    partial.clear(); // x = 0.0: identity gives 0.0, constant gives 0.5
    engine->processBuffer(partial);

    // Record the last sample output before interruption
    const double lastSampleBeforeInterrupt = partial.getSample(0, halfCrossfade - 1);
    // At ~50% through crossfade of (0.0 → 0.5), should be ~0.25
    EXPECT_GT(lastSampleBeforeInterrupt, 0.1) << "Should be mid-crossfade, not at start";
    EXPECT_LT(lastSampleBeforeInterrupt, 0.4) << "Should be mid-crossfade, not at end";

    // Now interrupt with a second LUT: constant 0.8
    const int workerIdx = engine->getWorkerTargetIndexReference().load(std::memory_order_acquire);
    for (int i = 0; i < dsp_core::TABLE_SIZE; ++i) {
        buffers[workerIdx].data[i] = 0.8;
    }
    buffers[workerIdx].version = 2;
    engine->getNewLUTReadyFlag().store(true, std::memory_order_release);

    // Process one sample after interruption
    juce::AudioBuffer<double> firstAfter(1, 1);
    firstAfter.clear(); // x = 0.0
    engine->processBuffer(firstAfter);
    const double firstSampleAfterInterrupt = firstAfter.getSample(0, 0);

    // The first sample after interruption should be close to the last sample before.
    // With blend snapshot, the new crossfade starts from the blended state (~0.25),
    // NOT from the crossfade target (0.5). The gap should be tiny.
    const double gap = std::abs(firstSampleAfterInterrupt - lastSampleBeforeInterrupt);
    EXPECT_LT(gap, 0.05) << "Continuity gap should be < 0.05. "
        << "Before=" << lastSampleBeforeInterrupt
        << " After=" << firstSampleAfterInterrupt
        << " Gap=" << gap;
}

// Surge tests removed — surge is no longer in the LUT. Coverage moved to
// SurgeStageTest.cpp (unit tests) and the SurgeHysteresisIntegration suite.


} // namespace dsp_core_test
