#pragma once

#include "AudioProcessingStage.h"
#include "EnvelopeFollowerStage.h"
#include "LfoStage.h"
#include <atomic>

namespace dsp_core::audio_pipeline {

/**
 * One modulator "slot" — owns both an envelope follower and an LFO and
 * dispatches process() to whichever the slot type currently selects. The
 * inactive sub-stage gets reset and the slot publishes 0 for it, so a swap
 * doesn't leave a stale value glued into the modulation sum.
 *
 * Both sub-stages publish into the same slotStorage_ atomic, so callers see
 * a single [0, 1] value per slot regardless of type.
 *
 * Thread safety: type_ is atomic; sub-stages handle their own thread safety.
 */
class ModulatorSlotStage : public AudioProcessingStage {
  public:
    enum class Type : int { Envelope = 0, Lfo = 1 };

    explicit ModulatorSlotStage(std::atomic<double>& slotStorage)
        : slotStorage_(slotStorage),
          envStage_(slotStorage),
          lfoStage_(slotStorage) {}

    void prepareToPlay(double sampleRate, int samplesPerBlock) override {
        envStage_.prepareToPlay(sampleRate, samplesPerBlock);
        lfoStage_.prepareToPlay(sampleRate, samplesPerBlock);
    }

    void process(juce::AudioBuffer<double>& buffer) override {
        const auto t = static_cast<Type>(type_.load(std::memory_order_acquire));
        if (t == Type::Envelope) {
            envStage_.process(buffer);
        } else {
            lfoStage_.process(buffer);
        }
    }

    void reset() override {
        envStage_.reset();
        lfoStage_.reset();
        slotStorage_.store(0.0, std::memory_order_release);
    }

    juce::String getName() const override { return "ModulatorSlot"; }

    void setType(Type t) {
        const auto previous = static_cast<Type>(type_.exchange(static_cast<int>(t), std::memory_order_acq_rel));
        if (previous != t) {
            // Make sure the now-inactive stage stops contributing on the next
            // process() and clear stale published value.
            if (t == Type::Envelope) {
                lfoStage_.setEnabled(false);
            } else {
                envStage_.setEnabled(false);
            }
            slotStorage_.store(0.0, std::memory_order_release);
        }
    }

    Type getType() const { return static_cast<Type>(type_.load(std::memory_order_acquire)); }

    EnvelopeFollowerStage& envelope() { return envStage_; }
    LfoStage& lfo() { return lfoStage_; }

  private:
    std::atomic<double>& slotStorage_;
    EnvelopeFollowerStage envStage_;
    LfoStage lfoStage_;
    std::atomic<int> type_{static_cast<int>(Type::Envelope)};
};

} // namespace dsp_core::audio_pipeline
