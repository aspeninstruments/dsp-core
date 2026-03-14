#include "StereoHistoryBuffer.h"

StereoHistoryBuffer::StereoHistoryBuffer(int historySize)
    : xBuffer_(static_cast<size_t>(historySize), 0.0),
      yBuffer_(static_cast<size_t>(historySize), 0.0),
      sequence_(0),
      writePos_(0),
      size_(historySize),
      cachedX_(static_cast<size_t>(historySize), 0.0),
      cachedY_(static_cast<size_t>(historySize), 0.0),
      cachedSize_(0) {}

void StereoHistoryBuffer::pushSample(double xSample, double ySample) {
    // Begin write: increment sequence to odd (signals write in progress)
    uint64_t seq = sequence_.load(std::memory_order_relaxed);
    sequence_.store(seq + 1, std::memory_order_release);

    // Load current position
    int pos = writePos_.load(std::memory_order_relaxed);

    // Write both samples at the same position
    xBuffer_[static_cast<size_t>(pos)] = xSample;
    yBuffer_[static_cast<size_t>(pos)] = ySample;

    // Advance write position
    int nextPos = (pos + 1) % size_;
    writePos_.store(nextPos, std::memory_order_relaxed);

    // Track total samples for readiness checks
    totalSamplesWritten_.fetch_add(1, std::memory_order_relaxed);

    // End write: increment sequence to even (signals write complete)
    sequence_.store(seq + 2, std::memory_order_release);
}

void StereoHistoryBuffer::pushSamples(const double* xSamples, const double* ySamples, int numSamples) {
    // Begin write: increment sequence to odd
    uint64_t seq = sequence_.load(std::memory_order_relaxed);
    sequence_.store(seq + 1, std::memory_order_release);

    // Load current position
    int pos = writePos_.load(std::memory_order_relaxed);

    // Write all samples
    for (int i = 0; i < numSamples; ++i) {
        xBuffer_[static_cast<size_t>(pos)] = xSamples[i];
        yBuffer_[static_cast<size_t>(pos)] = ySamples[i];
        pos = (pos + 1) % size_;
    }

    // Update write position
    writePos_.store(pos, std::memory_order_relaxed);

    // Track total samples for readiness checks
    totalSamplesWritten_.fetch_add(static_cast<int64_t>(numSamples), std::memory_order_relaxed);

    // End write: increment sequence to even
    sequence_.store(seq + 2, std::memory_order_release);
}

void StereoHistoryBuffer::copyFromPosition(double* outX, double* outY, int numSamples, int startPos) const {
    // Copy X buffer
    int firstPart = std::min(size_ - startPos, numSamples);
    int secondPart = numSamples - firstPart;

    std::memcpy(outX, xBuffer_.data() + startPos, static_cast<size_t>(firstPart) * sizeof(double));
    if (secondPart > 0) {
        std::memcpy(outX + firstPart, xBuffer_.data(), static_cast<size_t>(secondPart) * sizeof(double));
    }

    // Copy Y buffer from the SAME positions
    std::memcpy(outY, yBuffer_.data() + startPos, static_cast<size_t>(firstPart) * sizeof(double));
    if (secondPart > 0) {
        std::memcpy(outY + firstPart, yBuffer_.data(), static_cast<size_t>(secondPart) * sizeof(double));
    }
}

void StereoHistoryBuffer::getHistory(double* outX, double* outY, int numSamples) const {
    constexpr int kMaxRetries = 3;

    for (int attempt = 0; attempt < kMaxRetries; ++attempt) {
        // Read sequence before reading data
        uint64_t seqBefore = sequence_.load(std::memory_order_acquire);

        // If write is in progress (odd sequence), use cache or retry
        if ((seqBefore & 1) != 0) {
            // Writer is active, use cached data if available
            if (cachedSize_.load(std::memory_order_acquire) >= numSamples) {
                std::memcpy(outX, cachedX_.data(), static_cast<size_t>(numSamples) * sizeof(double));
                std::memcpy(outY, cachedY_.data(), static_cast<size_t>(numSamples) * sizeof(double));
                return;
            }
            // No valid cache, retry after brief pause
            continue;
        }

        // Read write position and calculate start
        int currentWritePos = writePos_.load(std::memory_order_acquire);
        int start = (currentWritePos - numSamples + size_) % size_;

        // Copy data
        copyFromPosition(outX, outY, numSamples, start);

        // Check sequence after reading - if it changed, data may be inconsistent
        uint64_t seqAfter = sequence_.load(std::memory_order_acquire);

        if (seqBefore == seqAfter) {
            // Success! Update cache for future fallback
            if (numSamples <= size_) {
                std::memcpy(const_cast<double*>(cachedX_.data()), outX,
                            static_cast<size_t>(numSamples) * sizeof(double));
                std::memcpy(const_cast<double*>(cachedY_.data()), outY,
                            static_cast<size_t>(numSamples) * sizeof(double));
                cachedSize_.store(numSamples, std::memory_order_release);
            }
            return;
        }

        // Sequence changed during read - retry
    }

    // All retries exhausted, use cache if available
    int cached = cachedSize_.load(std::memory_order_acquire);
    if (cached > 0) {
        int toCopy = std::min(cached, numSamples);
        std::memcpy(outX, cachedX_.data(), static_cast<size_t>(toCopy) * sizeof(double));
        std::memcpy(outY, cachedY_.data(), static_cast<size_t>(toCopy) * sizeof(double));
        // Zero-fill remainder if cache is smaller
        if (toCopy < numSamples) {
            std::memset(outX + toCopy, 0, static_cast<size_t>(numSamples - toCopy) * sizeof(double));
            std::memset(outY + toCopy, 0, static_cast<size_t>(numSamples - toCopy) * sizeof(double));
        }
    } else {
        // No cache, return zeros
        std::memset(outX, 0, static_cast<size_t>(numSamples) * sizeof(double));
        std::memset(outY, 0, static_cast<size_t>(numSamples) * sizeof(double));
    }
}

void StereoHistoryBuffer::clear() {
    // Mark write in progress
    uint64_t seq = sequence_.load(std::memory_order_relaxed);
    sequence_.store(seq + 1, std::memory_order_release);

    std::fill(xBuffer_.begin(), xBuffer_.end(), 0.0);
    std::fill(yBuffer_.begin(), yBuffer_.end(), 0.0);
    writePos_.store(0, std::memory_order_relaxed);
    totalSamplesWritten_.store(0, std::memory_order_relaxed);

    // Clear cache too
    std::fill(cachedX_.begin(), cachedX_.end(), 0.0);
    std::fill(cachedY_.begin(), cachedY_.end(), 0.0);
    cachedSize_.store(0, std::memory_order_relaxed);

    // Mark write complete
    sequence_.store(seq + 2, std::memory_order_release);
}
