#pragma once

#include <array>
#include <atomic>
#include <cstdint>

namespace dsp_core::audio_pipeline {

/**
 * Lock-free publish channel from a ModulatorSlot's audio-side stages to the
 * UI thread. One per slot, owned at processor scope so raw pointers handed
 * to the visualizer outlive both the editor and the pipeline.
 *
 * Envelope side: a small power-of-two ring of recent linear envelope values,
 * written at a downsampled rate by EnvelopeFollowerStage::process so the UI
 * gets a scrolling time-domain trace at sensible resolution without paying
 * for per-sample atomic stores. The downsample ratio is chosen in
 * EnvelopeFollowerStage::prepareToPlay against the actual sample rate so the
 * ring covers ~kEnvTargetHistorySec regardless of host SR.
 *
 * LFO side: scalar phase + unwrapped cycle position published once per block,
 * plus a shape-version counter the visualizer uses to decide when to rebuild
 * its cached one-cycle path. cyclePosition feeds the Random shape's Perlin
 * lookup, so the visualizer can render a per-cycle snapshot that swaps when
 * the audio thread enters a new cycle.
 *
 * Thread safety: writer is the audio thread; reader is the message thread.
 * Ring synchronization rides on envWriteIndex's release/acquire pairing;
 * the ring slots themselves are plain doubles. All other atomics are
 * single-writer scalars with release/acquire ordering.
 */
struct SlotVisualizerPublisher {
    // Power-of-two so the writer can wrap via bitmask. The 2048-entry ring
    // + 1.2 s target history gives roughly one UI sample per ~28 audio samples
    // at 48 kHz — fine enough to track sub-millisecond transients without
    // aliasing the Attack/Release knob's lower bound (~0.5 ms), and slow
    // enough that the on-screen scroll reads as "what just happened" rather
    // than blowing past the eye. Ring is 16 KB per slot — trivial overhead.
    static constexpr int kEnvRingSize = 2048;
    static_assert((kEnvRingSize & (kEnvRingSize - 1)) == 0, "kEnvRingSize must be a power of two");
    static constexpr int kEnvRingMask = kEnvRingSize - 1;
    static constexpr double kEnvTargetHistorySec = 1.200;

    std::array<double, kEnvRingSize> envRing{};
    std::atomic<int> envWriteIndex{0};
    std::atomic<uint64_t> envVersion{0};

    // LFO phase + unwrapped cycle position. Phase drives the UI's indicator
    // sub-segment; cyclePosition's floor change is what the editor watches
    // to know "Random shape stepped to a new cycle, time to redraw the
    // snapshot." Shape / seed are NOT published here — the editor reads
    // them directly from LfoStage's atomic params so visualization stays
    // live when the audio thread is idle (host transport stopped).
    std::atomic<double> lfoPhase01{0.0};
    std::atomic<double> lfoCyclePosition{0.0};

    // 0 = envelope, 1 = lfo. Mirrors ModulatorSlotStage::Type. Written by
    // ModulatorSlotStage::setType so the UI can pick which trace to bind.
    std::atomic<int> activeKind{0};
};

} // namespace dsp_core::audio_pipeline
