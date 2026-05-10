#include <gtest/gtest.h>
#include "../dsp_core/Source/StereoHistoryBuffer.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>

class StereoHistoryBufferTest : public ::testing::Test {
  protected:
    static constexpr int kDefaultBufferSize = 4096;
};

//==============================================================================
// Basic Functionality Tests
//==============================================================================

TEST_F(StereoHistoryBufferTest, ConstructorInitializesCorrectSize) {
    StereoHistoryBuffer buffer(1024);
    EXPECT_EQ(buffer.getSize(), 1024);
}

TEST_F(StereoHistoryBufferTest, PushAndReadSingleSample) {
    StereoHistoryBuffer buffer(16);

    buffer.pushSample(0.5, -0.5);

    std::vector<double> x(1), y(1);
    buffer.getHistory(x.data(), y.data(), 1);

    EXPECT_DOUBLE_EQ(x[0], 0.5);
    EXPECT_DOUBLE_EQ(y[0], -0.5);
}

TEST_F(StereoHistoryBufferTest, PushMultipleSamplesAndRead) {
    StereoHistoryBuffer buffer(16);

    // Push 5 paired samples
    for (int i = 0; i < 5; ++i) {
        buffer.pushSample(static_cast<double>(i) * 0.1, static_cast<double>(i) * 0.2);
    }

    std::vector<double> x(5), y(5);
    buffer.getHistory(x.data(), y.data(), 5);

    // Should get samples in order (oldest to newest)
    for (int i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(x[i], static_cast<double>(i) * 0.1);
        EXPECT_DOUBLE_EQ(y[i], static_cast<double>(i) * 0.2);
    }
}

//==============================================================================
// X/Y Correspondence Tests - The Critical Property
//==============================================================================

TEST_F(StereoHistoryBufferTest, ReadReturnsCorrespondingSamples) {
    StereoHistoryBuffer buffer(1024);

    // Push 100 paired samples where Y = X * 2 (known relationship)
    for (int i = 0; i < 100; ++i) {
        double xVal = static_cast<double>(i) * 0.01;
        double yVal = xVal * 2.0;
        buffer.pushSample(xVal, yVal);
    }

    std::vector<double> x(50), y(50);
    buffer.getHistory(x.data(), y.data(), 50);

    // Every Y should be exactly 2x its corresponding X
    for (int i = 0; i < 50; ++i) {
        EXPECT_DOUBLE_EQ(y[i], x[i] * 2.0) << "Mismatch at index " << i << ": x=" << x[i] << ", y=" << y[i];
    }
}

TEST_F(StereoHistoryBufferTest, ReadReturnsCorrespondingSamplesAfterWrap) {
    StereoHistoryBuffer buffer(64); // Small buffer to force wrap

    // Push 100 samples (wraps around multiple times)
    for (int i = 0; i < 100; ++i) {
        double xVal = static_cast<double>(i);
        double yVal = static_cast<double>(i) + 1000.0; // Y = X + 1000
        buffer.pushSample(xVal, yVal);
    }

    std::vector<double> x(32), y(32);
    buffer.getHistory(x.data(), y.data(), 32);

    // Every Y should be X + 1000, regardless of wraparound
    for (int i = 0; i < 32; ++i) {
        EXPECT_DOUBLE_EQ(y[i], x[i] + 1000.0) << "Wraparound broke correspondence at index " << i;
    }
}

TEST_F(StereoHistoryBufferTest, CorrespondencePreservedWithSineCosineRelationship) {
    StereoHistoryBuffer buffer(2048);

    // Push sin/cos pairs - these have a known mathematical relationship
    for (int i = 0; i < 1000; ++i) {
        double phase = static_cast<double>(i) * 0.1;
        buffer.pushSample(std::sin(phase), std::cos(phase));
    }

    std::vector<double> x(500), y(500);
    buffer.getHistory(x.data(), y.data(), 500);

    // sin²(θ) + cos²(θ) = 1 for all corresponding pairs
    for (int i = 0; i < 500; ++i) {
        double sumOfSquares = x[i] * x[i] + y[i] * y[i];
        EXPECT_NEAR(sumOfSquares, 1.0, 1e-10)
            << "Pythagorean identity violated at index " << i << ": sin=" << x[i] << ", cos=" << y[i];
    }
}

//==============================================================================
// Block Push Tests (for audio buffer integration)
//==============================================================================

TEST_F(StereoHistoryBufferTest, PushSamplesBlockMaintainsCorrespondence) {
    StereoHistoryBuffer buffer(1024);

    // Simulate audio block: X and Y buffers with known relationship
    constexpr int blockSize = 256;
    std::vector<double> xBlock(blockSize), yBlock(blockSize);

    for (int i = 0; i < blockSize; ++i) {
        xBlock[i] = static_cast<double>(i);
        yBlock[i] = static_cast<double>(i) * 3.0; // Y = X * 3
    }

    buffer.pushSamples(xBlock.data(), yBlock.data(), blockSize);

    std::vector<double> xOut(blockSize), yOut(blockSize);
    buffer.getHistory(xOut.data(), yOut.data(), blockSize);

    for (int i = 0; i < blockSize; ++i) {
        EXPECT_DOUBLE_EQ(yOut[i], xOut[i] * 3.0) << "Block push broke correspondence at index " << i;
    }
}

TEST_F(StereoHistoryBufferTest, MultipleBlockPushesPreserveCorrespondence) {
    StereoHistoryBuffer buffer(2048);

    // Push multiple blocks
    constexpr int blockSize = 128;
    for (int block = 0; block < 10; ++block) {
        std::vector<double> xBlock(blockSize), yBlock(blockSize);
        for (int i = 0; i < blockSize; ++i) {
            int globalIdx = block * blockSize + i;
            xBlock[i] = static_cast<double>(globalIdx);
            yBlock[i] = -static_cast<double>(globalIdx); // Y = -X
        }
        buffer.pushSamples(xBlock.data(), yBlock.data(), blockSize);
    }

    std::vector<double> xOut(512), yOut(512);
    buffer.getHistory(xOut.data(), yOut.data(), 512);

    for (int i = 0; i < 512; ++i) {
        EXPECT_DOUBLE_EQ(yOut[i], -xOut[i]) << "Multi-block push broke correspondence at index " << i;
    }
}

//==============================================================================
// Thread Safety Tests
//==============================================================================

TEST_F(StereoHistoryBufferTest, ConcurrentWritesDontCorruptCorrespondence) {
    StereoHistoryBuffer buffer(kDefaultBufferSize);
    std::atomic<bool> stopFlag{false};
    std::atomic<int> correspondenceErrors{0};

    // Writer thread: continuously push samples with Y = X + 1000
    std::thread writer([&]() {
        int counter = 0;
        while (!stopFlag.load(std::memory_order_relaxed)) {
            double x = static_cast<double>(counter);
            double y = static_cast<double>(counter) + 1000.0;
            buffer.pushSample(x, y);
            counter++;
        }
    });

    // Reader thread: continuously read and verify correspondence
    std::thread reader([&]() {
        std::vector<double> x(256), y(256);
        while (!stopFlag.load(std::memory_order_relaxed)) {
            buffer.getHistory(x.data(), y.data(), 256);

            // Verify correspondence: every Y should be X + 1000
            for (int i = 0; i < 256; ++i) {
                if (std::abs(y[i] - (x[i] + 1000.0)) > 1e-10 && x[i] != 0.0) {
                    correspondenceErrors.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    });

    // Run for 100ms
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    stopFlag.store(true, std::memory_order_relaxed);

    writer.join();
    reader.join();

    EXPECT_EQ(correspondenceErrors.load(), 0) << "Concurrent access caused X/Y desynchronization";
}

TEST_F(StereoHistoryBufferTest, HighFrequencyWritesPreserveCorrespondence) {
    StereoHistoryBuffer buffer(kDefaultBufferSize);
    std::atomic<bool> stopFlag{false};
    std::atomic<int> correspondenceErrors{0};

    // Writer thread: push blocks at "audio rate" (simulating 512 samples at 48kHz)
    std::thread writer([&]() {
        constexpr int blockSize = 512;
        std::vector<double> xBlock(blockSize), yBlock(blockSize);
        int blockCounter = 0;

        while (!stopFlag.load(std::memory_order_relaxed)) {
            // Each block has Y = X * (blockCounter + 1)
            double multiplier = static_cast<double>(blockCounter % 100 + 1);
            for (int i = 0; i < blockSize; ++i) {
                xBlock[i] = static_cast<double>(i);
                yBlock[i] = static_cast<double>(i) * multiplier;
            }
            buffer.pushSamples(xBlock.data(), yBlock.data(), blockSize);
            blockCounter++;
        }
    });

    // Reader thread: check that within each read, all samples have consistent relationship
    std::thread reader([&]() {
        std::vector<double> x(256), y(256);
        while (!stopFlag.load(std::memory_order_relaxed)) {
            buffer.getHistory(x.data(), y.data(), 256);

            // Find the multiplier from first non-zero sample
            double multiplier = 0.0;
            for (int i = 0; i < 256; ++i) {
                if (std::abs(x[i]) > 1e-10) {
                    multiplier = y[i] / x[i];
                    break;
                }
            }

            // All samples should have the same multiplier (within tolerance for block boundaries)
            // This is a weaker test but catches gross desynchronization
            if (multiplier > 0.0) {
                int badSamples = 0;
                for (int i = 0; i < 256; ++i) {
                    if (std::abs(x[i]) > 1e-10) {
                        double actualMultiplier = y[i] / x[i];
                        // Allow some tolerance for samples crossing block boundaries
                        if (std::abs(actualMultiplier - multiplier) > 0.5 && actualMultiplier > 0.5) {
                            badSamples++;
                        }
                    }
                }
                // If more than 10% of samples are bad, there's a real problem
                if (badSamples > 25) {
                    correspondenceErrors.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    stopFlag.store(true, std::memory_order_relaxed);

    writer.join();
    reader.join();

    EXPECT_EQ(correspondenceErrors.load(), 0) << "High-frequency writes caused X/Y desynchronization";
}

//==============================================================================
// Edge Cases
//==============================================================================

TEST_F(StereoHistoryBufferTest, ReadMoreThanWrittenReturnsZeros) {
    StereoHistoryBuffer buffer(1024);

    // Push only 10 samples
    for (int i = 0; i < 10; ++i) {
        buffer.pushSample(1.0, 2.0);
    }

    // Try to read 100
    std::vector<double> x(100), y(100);
    buffer.getHistory(x.data(), y.data(), 100);

    // First 90 should be zeros (buffer initialized to zero)
    for (int i = 0; i < 90; ++i) {
        EXPECT_DOUBLE_EQ(x[i], 0.0);
        EXPECT_DOUBLE_EQ(y[i], 0.0);
    }

    // Last 10 should be our values
    for (int i = 90; i < 100; ++i) {
        EXPECT_DOUBLE_EQ(x[i], 1.0);
        EXPECT_DOUBLE_EQ(y[i], 2.0);
    }
}

TEST_F(StereoHistoryBufferTest, ClearResetsBuffer) {
    StereoHistoryBuffer buffer(64);

    // Fill with data
    for (int i = 0; i < 100; ++i) {
        buffer.pushSample(1.0, 2.0);
    }

    buffer.clear();

    std::vector<double> x(64), y(64);
    buffer.getHistory(x.data(), y.data(), 64);

    for (int i = 0; i < 64; ++i) {
        EXPECT_DOUBLE_EQ(x[i], 0.0);
        EXPECT_DOUBLE_EQ(y[i], 0.0);
    }
}

TEST_F(StereoHistoryBufferTest, ExactBufferSizeRead) {
    StereoHistoryBuffer buffer(64);

    // Fill exactly
    for (int i = 0; i < 64; ++i) {
        buffer.pushSample(static_cast<double>(i), static_cast<double>(i) * 2.0);
    }

    std::vector<double> x(64), y(64);
    buffer.getHistory(x.data(), y.data(), 64);

    for (int i = 0; i < 64; ++i) {
        EXPECT_DOUBLE_EQ(y[i], x[i] * 2.0);
    }
}

//==============================================================================
// Regression Test: The Original Bug
//==============================================================================

TEST_F(StereoHistoryBufferTest, SimulatedVisualizationReadNeverShowsPhaseShift) {
    // This test simulates the actual usage pattern that was causing glitches:
    // - Audio thread pushes samples at ~48kHz
    // - UI thread reads at ~20Hz
    // - If X and Y get desynchronized, a sine wave pair appears as an ellipse

    StereoHistoryBuffer buffer(8192);
    std::atomic<bool> stopFlag{false};
    std::atomic<int> ellipseDetected{0};

    // Audio thread: push in-phase sine waves
    std::thread audioThread([&]() {
        int sampleIndex = 0;
        constexpr int blockSize = 512;
        std::vector<double> xBlock(blockSize), yBlock(blockSize);

        while (!stopFlag.load(std::memory_order_relaxed)) {
            for (int i = 0; i < blockSize; ++i) {
                double phase = static_cast<double>(sampleIndex + i) * 0.01;
                // Both channels are identical sine waves - should plot as a diagonal line
                xBlock[i] = std::sin(phase);
                yBlock[i] = std::sin(phase);
            }
            buffer.pushSamples(xBlock.data(), yBlock.data(), blockSize);
            sampleIndex += blockSize;

            // Simulate audio callback timing (~10ms at 48kHz with 512 samples)
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });

    // UI thread: read and check for phase shift
    std::thread uiThread([&]() {
        std::vector<double> x(2048), y(2048);

        while (!stopFlag.load(std::memory_order_relaxed)) {
            buffer.getHistory(x.data(), y.data(), 2048);

            // If X == Y (in-phase), all points should be on the diagonal
            // If desynchronized, we'd see deviation from diagonal (ellipse)
            double maxDeviation = 0.0;
            for (int i = 0; i < 2048; ++i) {
                double deviation = std::abs(x[i] - y[i]);
                maxDeviation = std::max(maxDeviation, deviation);
            }

            // Any significant deviation indicates desynchronization
            if (maxDeviation > 1e-10) {
                ellipseDetected.fetch_add(1, std::memory_order_relaxed);
            }

            // Simulate 20Hz UI refresh
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    });

    // Run for 500ms
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    stopFlag.store(true, std::memory_order_relaxed);

    audioThread.join();
    uiThread.join();

    EXPECT_EQ(ellipseDetected.load(), 0) << "Phase shift detected - X and Y became desynchronized!";
}
