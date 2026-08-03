#pragma once

#include "AudioProcessingStage.h"
#include <atomic>
#include <algorithm>

namespace dsp_core::audio_pipeline {

/**
 * Shared publish channel for SignalRangeTap — owned by the processor so the
 * atomics outlive both the pipeline and any editor reading them.
 *
 * Thread safety: audio thread stores with memory_order_release, UI thread
 * loads with memory_order_acquire. blockCount is a per-process() heartbeat so
 * the UI can detect a stopped transport (values frozen = stale).
 */
struct SignalRangeState {
    std::atomic<double> minValue{0.0};
    std::atomic<double> maxValue{0.0};
    std::atomic<uint32_t> blockCount{0};
};

/**
 * Passthrough measurement tap: signed, independently-tracked min/max of the
 * signal at its insertion point. Audio is never modified.
 *
 * Extremes are held over a ~100 ms window (sized against the prepared sample
 * rate, so inside an oversampled group the window is correct at the
 * oversampled rate). The hold guarantees a 60 Hz UI reader never misses a
 * transient between frames without needing a read-and-reset exchange.
 *
 * Thread safety:
 * - process: audio thread only; publishes via release stores on state_
 * - prepareToPlay / reset: UI/prepare thread, between process() calls
 * - readers: acquire loads on the shared SignalRangeState
 */
class SignalRangeTap : public AudioProcessingStage {
  public:
    explicit SignalRangeTap(SignalRangeState& state) : state_(state) {}

    void prepareToPlay(double sampleRate, int samplesPerBlock, int numChannels) override {
        juce::ignoreUnused(samplesPerBlock, numChannels);
        holdWindowSamples_ = std::max(1, static_cast<int>(kHoldSeconds * sampleRate));
        holdValid_ = false;
    }

    void process(juce::AudioBuffer<double>& buffer) override {
        const int numChannels = buffer.getNumChannels();
        const int numSamples = buffer.getNumSamples();
        if (numChannels == 0 || numSamples == 0) {
            return;
        }

        auto blockRange = buffer.findMinMax(0, 0, numSamples);
        for (int ch = 1; ch < numChannels; ++ch) {
            blockRange = blockRange.getUnionWith(buffer.findMinMax(ch, 0, numSamples));
        }

        if (!holdValid_ || holdSamplesRemaining_ <= 0) {
            holdMin_ = blockRange.getStart();
            holdMax_ = blockRange.getEnd();
            holdSamplesRemaining_ = holdWindowSamples_;
            holdValid_ = true;
        } else {
            holdMin_ = std::min(holdMin_, blockRange.getStart());
            holdMax_ = std::max(holdMax_, blockRange.getEnd());
        }
        holdSamplesRemaining_ -= numSamples;

        state_.minValue.store(holdMin_, std::memory_order_release);
        state_.maxValue.store(holdMax_, std::memory_order_release);
        state_.blockCount.fetch_add(1, std::memory_order_release);
    }

    void reset() override {
        holdValid_ = false;
        holdSamplesRemaining_ = 0;
        holdMin_ = 0.0;
        holdMax_ = 0.0;
        state_.minValue.store(0.0, std::memory_order_release);
        state_.maxValue.store(0.0, std::memory_order_release);
        // blockCount stays monotonic — it's a heartbeat, not a value.
    }

    juce::String getName() const override {
        return "SignalRangeTap";
    }

  private:
    static constexpr double kHoldSeconds = 0.1;

    SignalRangeState& state_;
    double holdMin_ = 0.0;
    double holdMax_ = 0.0;
    int holdSamplesRemaining_ = 0;
    int holdWindowSamples_ = 4800;
    bool holdValid_ = false;
};

} // namespace dsp_core::audio_pipeline
