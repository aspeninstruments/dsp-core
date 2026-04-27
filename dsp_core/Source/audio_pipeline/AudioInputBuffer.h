#pragma once

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_core/juce_core.h>
#include <atomic>
#include <vector>

namespace dsp_core::audio_pipeline {

/**
 * Lock-free circular buffer for storing recent audio input samples.
 *
 * The audio thread writes samples continuously, and the UI thread
 * reads samples on-demand for visualization (peak detection).
 *
 * Thread safety:
 * - Audio thread: Writes samples (single writer)
 * - UI thread: Reads samples (single reader)
 * - Uses juce::AbstractFifo for lock-free coordination
 *
 * Design:
 * - Stores the most recent N samples per channel
 * - When buffer fills, oldest samples are overwritten
 * - UI thread reads available samples without blocking audio thread
 */
class AudioInputBuffer {
  public:
    /**
     * Construct with buffer size in samples per channel.
     *
     * Typical usage: 1 second of audio at max sample rate
     * Example: 48000 samples * 2 channels = 96000 samples total
     */
    explicit AudioInputBuffer(int bufferSizePerChannel = 48000)
        : bufferSize_(bufferSizePerChannel), fifo_(bufferSizePerChannel), numChannels_(0) {}

    /**
     * Prepare buffer for a specific channel count.
     * Called from prepareToPlay on audio thread.
     */
    void prepareToPlay(int numChannels) {
        numChannels_ = numChannels;

        // Allocate storage for each channel
        channelBuffers_.clear();
        channelBuffers_.resize(numChannels);
        for (auto& channelBuffer : channelBuffers_) {
            channelBuffer.resize(bufferSize_);
        }

        // Reset FIFO
        fifo_.reset();
    }

    /**
     * Write samples from audio thread (non-blocking).
     * Overwrites oldest samples when buffer is full.
     */
    void writeSamples(const juce::AudioBuffer<double>& buffer) {
        const int numSamples = buffer.getNumSamples();
        const int numChans = juce::jmin(buffer.getNumChannels(), numChannels_);

        // Get write region from FIFO
        int start1, size1, start2, size2;
        fifo_.prepareToWrite(numSamples, start1, size1, start2, size2);

        // Write to circular buffer (wraps around)
        for (int ch = 0; ch < numChans; ++ch) {
            const auto* src = buffer.getReadPointer(ch);
            auto& dest = channelBuffers_[ch];

            // Write first region
            for (int i = 0; i < size1; ++i) {
                dest[(start1 + i) % bufferSize_] = src[i];
            }

            // Write second region (if wraps around)
            for (int i = 0; i < size2; ++i) {
                dest[(start2 + i) % bufferSize_] = src[size1 + i];
            }
        }

        fifo_.finishedWrite(size1 + size2);
    }

    /**
     * Read available samples from UI thread and compute peak.
     * Returns peak amplitude across all channels and all available samples.
     * Non-blocking - returns 0.0 if no samples available.
     */
    double readAndComputePeak() {
        int numReady = fifo_.getNumReady();
        if (numReady == 0) {
            return 0.0;
        }

        // Get read region from FIFO
        int start1, size1, start2, size2;
        fifo_.prepareToRead(numReady, start1, size1, start2, size2);

        double peak = 0.0;

        // Compute peak across all channels and both regions
        for (int ch = 0; ch < numChannels_; ++ch) {
            const auto& src = channelBuffers_[ch];

            // First region
            for (int i = 0; i < size1; ++i) {
                peak = std::max(peak, std::abs(src[(start1 + i) % bufferSize_]));
            }

            // Second region (if wraps around)
            for (int i = 0; i < size2; ++i) {
                peak = std::max(peak, std::abs(src[(start2 + i) % bufferSize_]));
            }
        }

        fifo_.finishedRead(size1 + size2);

        return peak;
    }

    /**
     * Reset buffer (clears all samples).
     * Called from reset() on audio thread.
     */
    void reset() {
        fifo_.reset();
        for (auto& channelBuffer : channelBuffers_) {
            std::fill(channelBuffer.begin(), channelBuffer.end(), 0.0);
        }
    }

  private:
    const int bufferSize_;    // Size per channel
    juce::AbstractFifo fifo_; // Lock-free read/write coordination
    int numChannels_;
    std::vector<std::vector<double>> channelBuffers_; // One vector per channel
};

} // namespace dsp_core::audio_pipeline
