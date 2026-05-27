#pragma once

#include "AudioProcessingStage.h"
#include "EnvelopeFollowerStage.h"
#include "LfoStage.h"
#include "SlotVisualizerPublisher.h"
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
        : slotStorage_(slotStorage), envStage_(slotStorage), lfoStage_(slotStorage) {}

    void prepareToPlay(double sampleRate, int samplesPerBlock, int numChannels) override {
        envStage_.prepareToPlay(sampleRate, samplesPerBlock, numChannels);
        lfoStage_.prepareToPlay(sampleRate, samplesPerBlock, numChannels);
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

    juce::String getName() const override {
        return "ModulatorSlot";
    }

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
            if (visualizerPublisher_ != nullptr) {
                // Reflect the new type for the UI's trace selection, and bump
                // both version counters so the visualizer rebuilds (don't
                // briefly show a stale envelope ring or LFO shape from the
                // previous type).
                visualizerPublisher_->activeKind.store(static_cast<int>(t), std::memory_order_release);
                visualizerPublisher_->envVersion.fetch_add(1, std::memory_order_release);
                visualizerPublisher_->lfoShapeVersion.fetch_add(1, std::memory_order_release);
            }
        }
    }

    Type getType() const {
        return static_cast<Type>(type_.load(std::memory_order_acquire));
    }

    EnvelopeFollowerStage& envelope() {
        return envStage_;
    }
    LfoStage& lfo() {
        return lfoStage_;
    }

    /// Attach (or detach, with nullptr) the lock-free visualizer publish channel
    /// for this slot. Forwards the same pointer to both sub-stages so whichever
    /// is active publishes into the shared channel. Safe to call before or
    /// between prepareToPlay/process calls; not safe to call concurrently with
    /// process().
    void setVisualizerPublisher(SlotVisualizerPublisher* pub) {
        visualizerPublisher_ = pub;
        envStage_.setVisualizerPublisher(pub);
        lfoStage_.setVisualizerPublisher(pub);
        if (pub != nullptr) {
            pub->activeKind.store(type_.load(std::memory_order_acquire), std::memory_order_release);
        }
    }

  private:
    std::atomic<double>& slotStorage_;
    EnvelopeFollowerStage envStage_;
    LfoStage lfoStage_;
    std::atomic<int> type_{static_cast<int>(Type::Envelope)};
    SlotVisualizerPublisher* visualizerPublisher_{nullptr};
};

} // namespace dsp_core::audio_pipeline
