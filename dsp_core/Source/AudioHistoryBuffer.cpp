#include "AudioHistoryBuffer.h"

#include <cstring>

AudioHistoryBuffer::AudioHistoryBuffer(int historySize) : buffer(historySize, 0.0), size(historySize) {}

void AudioHistoryBuffer::pushSamples(const double* samples, int numSamples) {
    for (int i = 0; i < numSamples; ++i) {
        buffer[writePos] = samples[i];
        writePos = (writePos + 1) % size;
    }
}

void AudioHistoryBuffer::getHistory(double* outSamples, int numSamples) const {
    jassert(numSamples <= size);

    const int start = (writePos - numSamples + size) % size;
    const int firstPart = juce::jmin(size - start, numSamples);
    const int secondPart = numSamples - firstPart;

    std::memcpy(outSamples, buffer.data() + start, firstPart * sizeof(double));
    if (secondPart > 0) {
        std::memcpy(outSamples + firstPart, buffer.data(), secondPart * sizeof(double));
    }
}
