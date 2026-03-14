#pragma once

#include <algorithm>
#include <atomic>
#include <cstring>
#include <vector>

/**
 * @brief Thread-safe stereo ring buffer that guarantees X/Y sample correspondence.
 *
 * This buffer solves the synchronization problem where separate X and Y history
 * buffers can become desynchronized when read from the UI thread while the audio
 * thread is writing. Uses a seqlock pattern to detect when reads overlap with writes.
 *
 * Thread Safety:
 * - pushSample/pushSamples: Called from audio thread only (not thread-safe for multiple writers)
 * - getHistory: Can be called from any thread while audio thread is writing
 * - clear: Should only be called when audio is stopped
 *
 * Synchronization Strategy (Seqlock):
 * - Writer increments sequence to odd before writing, even after writing
 * - Reader checks sequence before and after reading
 * - If sequence changed or was odd, reader retries with cached fallback
 * - This guarantees readers always see a consistent snapshot
 */
class StereoHistoryBuffer {
  public:
    explicit StereoHistoryBuffer(int historySize);

    /**
     * @brief Push a single stereo sample pair.
     * @param xSample The X channel (sidechain) sample
     * @param ySample The Y channel (main input) sample
     * @note Called from audio thread only.
     */
    void pushSample(double xSample, double ySample);

    /**
     * @brief Push a block of stereo samples.
     * @param xSamples Pointer to X channel samples
     * @param ySamples Pointer to Y channel samples
     * @param numSamples Number of samples to push
     * @note Called from audio thread only. xSamples[i] corresponds to ySamples[i].
     */
    void pushSamples(const double* xSamples, const double* ySamples, int numSamples);

    /**
     * @brief Read the most recent samples from both channels atomically.
     * @param outX Output buffer for X channel samples
     * @param outY Output buffer for Y channel samples
     * @param numSamples Number of samples to read
     * @note Thread-safe. Guarantees outX[i] and outY[i] are corresponding samples.
     *       Samples are returned oldest-to-newest order.
     */
    void getHistory(double* outX, double* outY, int numSamples) const;

    /**
     * @brief Get the buffer size.
     */
    int getSize() const {
        return size_;
    }

    /**
     * @brief Get the total number of samples written since construction or last clear().
     * @note Thread-safe. Useful for checking if enough data has accumulated for analysis.
     */
    int64_t getTotalSamplesWritten() const {
        return totalSamplesWritten_.load(std::memory_order_acquire);
    }

    /**
     * @brief Clear both buffers to zero.
     * @note Should only be called when audio is stopped.
     */
    void clear();

  private:
    void copyFromPosition(double* outX, double* outY, int numSamples, int startPos) const;

    std::vector<double> xBuffer_;
    std::vector<double> yBuffer_;
    std::atomic<uint64_t> sequence_{0};  // Seqlock: odd = write in progress, even = stable
    std::atomic<int> writePos_{0};
    std::atomic<int64_t> totalSamplesWritten_{0};
    int size_;

    // Cached last-good read for fallback when write is in progress
    mutable std::vector<double> cachedX_;
    mutable std::vector<double> cachedY_;
    mutable std::atomic<int> cachedSize_{0};
};
