#pragma once

#include "AudioProcessingStage.h"
#include "AudioPipeline.h"
#include <juce_dsp/juce_dsp.h>
#include <array>
#include <atomic>

namespace dsp_core::audio_pipeline {

/**
 * Wraps a pipeline or stage with oversampling.
 *
 * The wrapped pipeline processes at a higher sample rate,
 * then downsampled back to the original rate.
 *
 * Example:
 *   auto innerPipeline = std::make_unique<AudioPipeline>();
 *   innerPipeline->addStage(makeDryWet(makeWaveshaper()));
 *
 *   auto wrapped = std::make_unique<OversamplingWrapper>(
 *       std::move(innerPipeline),
 *       3  // 8x oversampling (2^3)
 *   );
 *
 *   mainPipeline.addStage(std::move(wrapped));
 *
 * RUNTIME ORDER CHANGES (preset switches):
 *   setOversamplingOrder() only stores a pending target — the switch itself
 *   runs on the AUDIO thread inside process() as an in-place dual-render
 *   flip: the block is rendered through the old chain, the wrapped stage is
 *   re-prepared at the new rate (allocation-free by contract — see the
 *   FlipAllocationAudit test), the same input is rendered through the new
 *   chain, and the two passes are smoothstep-crossfaded. No suspension, no
 *   silence, no message-thread mutation of live DSP state (that mutation was
 *   the AU preset-switch crash). Flips are gated to blocks of at least
 *   kMinFlipBlockSamples so the crossfade has room, with a deferred-block
 *   timeout so tiny-buffer hosts still flip.
 */
class OversamplingWrapper : public AudioProcessingStage {
  public:
    /**
     * @param wrappedStage Stage to process at oversampled rate
     * @param oversamplingOrder 0=1x, 1=2x, 2=4x, 3=8x, 4=16x
     */
    OversamplingWrapper(std::unique_ptr<AudioProcessingStage> wrappedStage,
                        int oversamplingOrder = 3 // 8x default
    );

    void prepareToPlay(double sampleRate, int samplesPerBlock, int numChannels) override;
    void process(juce::AudioBuffer<double>& buffer) override;
    void reset() override;
    juce::String getName() const override;

    /** Latency of the TARGET order (pending flips included) — the value the
     *  host must be told right after a preset load, before the flip runs. */
    int getLatencySamples() const override;

    /**
     * Request an oversampling-order change. Callable from the message thread
     * while audio runs: stores the target atomically; the audio thread
     * performs the actual switch as a crossfaded flip in process().
     */
    void setOversamplingOrder(int order);

    /** The order the audio path is currently rendering with. */
    int getOversamplingOrder() const {
        return currentOrder_.load(std::memory_order_acquire);
    }

    /** The requested order (equals getOversamplingOrder() when no flip is
     *  pending). Message-thread readable. */
    int getTargetOrder() const {
        return targetOrder_.load(std::memory_order_acquire);
    }

    /**
     * AUDIO THREAD ONLY. Pin the target order this block must use. The
     * processor reads ONE shared atomic at the top of processBlock and pushes
     * the latched value into both the wet wrapper and the dry-side wrapper so
     * they flip in the same block (zero dry/wet skew even if a message-thread
     * store lands mid-block). When never called (standalone use, tests),
     * process() falls back to this wrapper's own atomic target.
     */
    void setTargetOrderForBlock(int order) {
        externalTargetOrder_ = order;
    }

    // Factory helpers for common oversampling factors
    static constexpr int orderForNone() {
        return 0;
    }
    static constexpr int orderFor2x() {
        return 1;
    }
    static constexpr int orderFor4x() {
        return 2;
    }
    static constexpr int orderFor8x() {
        return 3;
    }
    static constexpr int orderFor16x() {
        return 4;
    }

  private:
    /** One up→wrapped→down pass with the given order's oversampler. */
    void renderWithOrder(juce::AudioBuffer<double>& buffer, int order);

    /** The audio-thread switch: dual render + smoothstep crossfade. */
    void performFlip(juce::AudioBuffer<double>& buffer, int oldOrder, int newOrder);

    /** Re-prepare the wrapped stage for the given order (allocation-free on
     *  the flip path by contract). */
    void prepareWrappedStageForOrder(int order);

    std::unique_ptr<AudioProcessingStage> wrappedStage_;

    // Pre-allocated oversamplers (1x, 2x, 4x, 8x, 16x)
    std::array<std::unique_ptr<juce::dsp::Oversampling<double>>, 5> oversamplers_;

    // Pre-allocated channel pointers array (avoid std::vector allocation per process call)
    std::array<double*, 8> channelPointers_{}; // Max 8 channels (7.1 surround)

    // Order state. currentOrder_ is what the audio path renders with;
    // targetOrder_ is the requested order (message thread writes, audio
    // thread consumes at block start). externalTargetOrder_ is the per-block
    // latch pushed by the processor (audio-thread-confined; -1 = unset).
    std::atomic<int> currentOrder_;
    std::atomic<int> targetOrder_;
    int externalTargetOrder_ = -1;

    // Flip gate: blocks shorter than this defer the flip (the crossfade needs
    // room) — but never indefinitely.
    static constexpr int kMinFlipBlockSamples = 128;
    static constexpr int kMaxDeferredFlipBlocks = 8;
    int deferredFlipBlocks_ = 0;

    // Input snapshot for the flip's second render pass (sized in prepareToPlay).
    juce::AudioBuffer<double> flipScratch_;
    std::array<double*, 8> flipChannelPointers_{};

    double sampleRate_ = 44100.0;
    int maxBlockSize_ = 512;
    int oversamplerChannels_ = 2; // matches the channel count the constructor pre-creates with
};

} // namespace dsp_core::audio_pipeline
