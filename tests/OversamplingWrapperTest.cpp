#include <gtest/gtest.h>
#include "../dsp_core/Source/audio_pipeline/OversamplingWrapper.h"
#include "../dsp_core/Source/audio_pipeline/AudioPipeline.h"
#include "../dsp_core/Source/audio_pipeline/BiasStage.h"
#include "../dsp_core/Source/audio_pipeline/HysteresisStage.h"
#include "../dsp_core/Source/audio_pipeline/SurgeStage.h"
#include "../dsp_core/Source/SeamlessTransferFunction.h"
#include <juce_audio_basics/juce_audio_basics.h>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <new>
#include <vector>

// ---------------------------------------------------------------------------
// Global allocation hooks for FlipAllocationAudit. Pass-through unless armed,
// and armed only on the arming thread (thread_local) so background threads
// left over from other tests in this binary can't contaminate the count.
// ---------------------------------------------------------------------------
namespace osw_alloc_audit {
thread_local bool armedOnThisThread = false;
std::atomic<int> flaggedAllocations{0};

inline void note() {
    if (armedOnThisThread) {
        flaggedAllocations.fetch_add(1, std::memory_order_relaxed);
    }
}
} // namespace osw_alloc_audit

#if !defined(_WIN32)
void* operator new(std::size_t size) {
    osw_alloc_audit::note();
    if (void* p = std::malloc(size > 0 ? size : 1)) {
        return p;
    }
    throw std::bad_alloc{};
}

void* operator new[](std::size_t size) {
    osw_alloc_audit::note();
    if (void* p = std::malloc(size > 0 ? size : 1)) {
        return p;
    }
    throw std::bad_alloc{};
}

void operator delete(void* p) noexcept {
    std::free(p);
}
void operator delete[](void* p) noexcept {
    std::free(p);
}
void operator delete(void* p, std::size_t) noexcept {
    std::free(p);
}
void operator delete[](void* p, std::size_t) noexcept {
    std::free(p);
}
#endif

using namespace dsp_core::audio_pipeline;

namespace {

// Trivial passthrough stage — leaves the buffer untouched. Lets us isolate
// OversamplingWrapper's up/down sampling and channel-count handling from any
// downstream DSP behaviour.
class PassthroughStage : public AudioProcessingStage {
  public:
    void prepareToPlay(double /*sampleRate*/, int /*samplesPerBlock*/, int /*numChannels*/) override {}
    void process(juce::AudioBuffer<double>& /*buffer*/) override {}
    void reset() override {}
    juce::String getName() const override {
        return "Passthrough";
    }
};

void fillSine(juce::AudioBuffer<double>& buffer, double frequencyHz, double sampleRate, double amplitude = 0.5) {
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
        auto* data = buffer.getWritePointer(ch);
        for (int i = 0; i < buffer.getNumSamples(); ++i) {
            data[i] = amplitude * std::sin(2.0 * juce::MathConstants<double>::pi * frequencyHz * i / sampleRate);
        }
    }
}

double bufferPeak(const juce::AudioBuffer<double>& buffer) {
    double peak = 0.0;
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
        const auto* data = buffer.getReadPointer(ch);
        for (int i = 0; i < buffer.getNumSamples(); ++i) {
            peak = std::max(peak, std::abs(data[i]));
        }
    }
    return peak;
}

bool bufferIsFinite(const juce::AudioBuffer<double>& buffer) {
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
        const auto* data = buffer.getReadPointer(ch);
        for (int i = 0; i < buffer.getNumSamples(); ++i) {
            if (!std::isfinite(data[i])) {
                return false;
            }
        }
    }
    return true;
}

} // namespace

TEST(OversamplingWrapperTest, MonoPrepareThenProcess) {
    auto wrapper = std::make_unique<OversamplingWrapper>(std::make_unique<PassthroughStage>(), 3); // 8x

    constexpr double sampleRate = 48000.0;
    constexpr int blockSize = 512;
    wrapper->prepareToPlay(sampleRate, blockSize, 1);

    juce::AudioBuffer<double> buffer(1, blockSize);
    fillSine(buffer, 1000.0, sampleRate);
    const double inputPeak = bufferPeak(buffer);

    wrapper->process(buffer);

    EXPECT_TRUE(bufferIsFinite(buffer));
    EXPECT_GT(bufferPeak(buffer), 0.1) << "1 kHz sine should pass through 8x oversampling without being silenced";
    EXPECT_LT(bufferPeak(buffer), inputPeak * 1.5) << "passthrough should not amplify";
}

TEST(OversamplingWrapperTest, StereoPrepareThenProcess) {
    auto wrapper = std::make_unique<OversamplingWrapper>(std::make_unique<PassthroughStage>(), 3);

    constexpr double sampleRate = 48000.0;
    constexpr int blockSize = 512;
    wrapper->prepareToPlay(sampleRate, blockSize, 2);

    juce::AudioBuffer<double> buffer(2, blockSize);
    fillSine(buffer, 1000.0, sampleRate);

    wrapper->process(buffer);

    EXPECT_TRUE(bufferIsFinite(buffer));
    EXPECT_GT(bufferPeak(buffer), 0.1);
}

// Regression test for the bug this change fixes: OversamplingWrapper used to
// hardcode 2 channels at construction, so calling prepareToPlay then handing
// it a mismatched-channel-count buffer asserted in Debug and read OOB in
// Release. After the fix, prepareToPlay reconstructs the oversamplers when the
// channel count changes and subsequent process() calls must remain safe.
TEST(OversamplingWrapperTest, ChannelCountChangeRebuildsOversamplers) {
    auto wrapper = std::make_unique<OversamplingWrapper>(std::make_unique<PassthroughStage>(), 3);

    constexpr double sampleRate = 48000.0;
    constexpr int blockSize = 512;

    wrapper->prepareToPlay(sampleRate, blockSize, 1);
    {
        juce::AudioBuffer<double> mono(1, blockSize);
        fillSine(mono, 1000.0, sampleRate);
        wrapper->process(mono);
        EXPECT_TRUE(bufferIsFinite(mono));
    }

    wrapper->prepareToPlay(sampleRate, blockSize, 2);
    {
        juce::AudioBuffer<double> stereo(2, blockSize);
        fillSine(stereo, 1000.0, sampleRate);
        wrapper->process(stereo);
        EXPECT_TRUE(bufferIsFinite(stereo));
        EXPECT_GT(bufferPeak(stereo), 0.1);
    }
}

// ===========================================================================
// Oversampling-flip tests (preset-switch crash fix).
//
// A preset switch may change the oversampling order while audio is running.
// The wrapper must transition seamlessly on the audio thread: no pops (bounded
// per-sample delta), no silence (windowed RMS holds), no allocations, and a
// deferral gate for very small blocks. Pre-fix, setOversamplingOrder hard-
// switched the active oversampler mid-stream (stale IIR state → discontinuity)
// and re-prepared the wrapped stage from the message thread (the crash).
// ===========================================================================

namespace {

constexpr double kSineHz = 440.0;
constexpr double kSineAmp = 0.5;
constexpr double kFlipSampleRate = 44100.0;
constexpr int kFlipBlockSize = 512;
// Natural per-sample slew of the test sine ≈ amp * 2π * f / sr ≈ 0.0313.
// 0.1 allows ~3x margin for blend-phase interaction; a hard switch through a
// zero-state IIR or a latency jump lands well above it.
constexpr double kMaxDeltaThreshold = 0.1;

// Feeds a phase-continuous sine block-by-block and tracks the maximum
// sample-to-sample delta (including across block boundaries) per channel.
struct ContinuityProbe {
    long globalSample = 0;
    std::array<double, 2> lastSample{0.0, 0.0};
    bool hasLast = false;
    double maxDelta = 0.0;

    void fillNextBlock(juce::AudioBuffer<double>& buffer) {
        for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
            auto* data = buffer.getWritePointer(ch);
            for (int i = 0; i < buffer.getNumSamples(); ++i) {
                data[i] = kSineAmp * std::sin(2.0 * juce::MathConstants<double>::pi * kSineHz *
                                              static_cast<double>(globalSample + i) / kFlipSampleRate);
            }
        }
        globalSample += buffer.getNumSamples();
    }

    void observe(const juce::AudioBuffer<double>& buffer) {
        for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
            const auto* data = buffer.getReadPointer(ch);
            double prev = hasLast ? lastSample[ch] : data[0];
            for (int i = 0; i < buffer.getNumSamples(); ++i) {
                ASSERT_TRUE(std::isfinite(data[i])) << "ch=" << ch << " i=" << i;
                maxDelta = std::max(maxDelta, std::abs(data[i] - prev));
                prev = data[i];
            }
            lastSample[ch] = prev;
        }
        hasLast = true;
    }
};

// RMS over consecutive complete 10ms windows of channel 0.
void collectWindowedRms(const juce::AudioBuffer<double>& buffer, std::vector<double>& stream) {
    const auto* data = buffer.getReadPointer(0);
    stream.insert(stream.end(), data, data + buffer.getNumSamples());
}

void assertRmsWithin1dB(const std::vector<double>& stream, const char* context) {
    const int window = static_cast<int>(kFlipSampleRate * 0.010); // 441
    const double steadyRms = kSineAmp / std::sqrt(2.0);
    const double hi = steadyRms * 1.122; // +1 dB
    const double lo = steadyRms / 1.122; // -1 dB
    for (size_t start = 0; start + static_cast<size_t>(window) <= stream.size();
         start += static_cast<size_t>(window)) {
        double acc = 0.0;
        for (int i = 0; i < window; ++i) {
            acc += stream[start + static_cast<size_t>(i)] * stream[start + static_cast<size_t>(i)];
        }
        const double rms = std::sqrt(acc / window);
        ASSERT_GE(rms, lo) << context << ": silence gap in window starting at sample " << start;
        ASSERT_LE(rms, hi) << context << ": overshoot in window starting at sample " << start;
    }
}

} // namespace

TEST(OversamplingWrapperTest, FlipContinuity_SteadySine) {
    for (int from = 0; from <= 4; ++from) {
        for (int to = 0; to <= 4; ++to) {
            if (from == to) {
                continue;
            }
            auto wrapper = std::make_unique<OversamplingWrapper>(std::make_unique<PassthroughStage>(), from);
            wrapper->prepareToPlay(kFlipSampleRate, kFlipBlockSize, 2);

            ContinuityProbe probe;
            juce::AudioBuffer<double> buffer(2, kFlipBlockSize);

            // Warmup: let the initial ring-in from silence settle (not under test).
            for (int b = 0; b < 4; ++b) {
                probe.fillNextBlock(buffer);
                wrapper->process(buffer);
            }

            std::vector<double> rmsStream;
            probe.hasLast = false; // deltas measured from here on
            probe.maxDelta = 0.0;

            for (int b = 0; b < 2; ++b) { // steady pre-flip
                probe.fillNextBlock(buffer);
                wrapper->process(buffer);
                probe.observe(buffer);
                collectWindowedRms(buffer, rmsStream);
            }

            wrapper->setOversamplingOrder(to);

            for (int b = 0; b < 4; ++b) { // flip + settle
                probe.fillNextBlock(buffer);
                wrapper->process(buffer);
                probe.observe(buffer);
                collectWindowedRms(buffer, rmsStream);
            }

            EXPECT_EQ(wrapper->getOversamplingOrder(), to) << from << "->" << to;
            EXPECT_LE(probe.maxDelta, kMaxDeltaThreshold)
                << "pop while flipping oversampling " << from << "->" << to;
            assertRmsWithin1dB(rmsStream, "FlipContinuity");
        }
    }
}

TEST(OversamplingWrapperTest, FlipCoalescing_RapidStepping) {
    auto wrapper = std::make_unique<OversamplingWrapper>(std::make_unique<PassthroughStage>(), 0);
    wrapper->prepareToPlay(kFlipSampleRate, kFlipBlockSize, 2);

    ContinuityProbe probe;
    juce::AudioBuffer<double> buffer(2, kFlipBlockSize);

    for (int b = 0; b < 4; ++b) { // warmup
        probe.fillNextBlock(buffer);
        wrapper->process(buffer);
    }
    probe.hasLast = false;
    probe.maxDelta = 0.0;

    // Deterministic pseudo-random order per block (LCG, fixed seed).
    unsigned int state = 12345u;
    int lastRequested = wrapper->getOversamplingOrder();
    for (int b = 0; b < 200; ++b) {
        state = state * 1664525u + 1013904223u;
        lastRequested = static_cast<int>(state % 5u);
        wrapper->setOversamplingOrder(lastRequested);
        probe.fillNextBlock(buffer);
        wrapper->process(buffer);
        probe.observe(buffer);
    }

    // Settle: no new requests; the pending order must coalesce to the latest.
    for (int b = 0; b < 10; ++b) {
        probe.fillNextBlock(buffer);
        wrapper->process(buffer);
        probe.observe(buffer);
    }

    EXPECT_EQ(wrapper->getOversamplingOrder(), lastRequested);
    EXPECT_LE(probe.maxDelta, kMaxDeltaThreshold) << "pop during rapid order stepping";
}

TEST(OversamplingWrapperTest, FlipSmallBlockGate) {
    auto wrapper = std::make_unique<OversamplingWrapper>(std::make_unique<PassthroughStage>(), 0);
    wrapper->prepareToPlay(kFlipSampleRate, kFlipBlockSize, 2);

    juce::AudioBuffer<double> tiny(2, 32);

    // Scenario 1: sub-128-sample blocks defer the flip...
    wrapper->setOversamplingOrder(2);
    for (int b = 0; b < 4; ++b) {
        fillSine(tiny, kSineHz, kFlipSampleRate);
        wrapper->process(tiny);
        EXPECT_TRUE(bufferIsFinite(tiny));
        EXPECT_EQ(wrapper->getOversamplingOrder(), 0)
            << "flip must be deferred on tiny blocks (block " << b << ")";
    }
    // ...but the timeout fallback flips regardless after ~8 deferred blocks.
    for (int b = 0; b < 8; ++b) {
        fillSine(tiny, kSineHz, kFlipSampleRate);
        wrapper->process(tiny);
        EXPECT_TRUE(bufferIsFinite(tiny));
    }
    EXPECT_EQ(wrapper->getOversamplingOrder(), 2) << "timeout fallback must eventually flip";

    // Scenario 2: a >=128-sample block flips immediately.
    wrapper->setOversamplingOrder(1);
    juce::AudioBuffer<double> block(2, 256);
    fillSine(block, kSineHz, kFlipSampleRate);
    wrapper->process(block);
    EXPECT_TRUE(bufferIsFinite(block));
    EXPECT_EQ(wrapper->getOversamplingOrder(), 1);
}

TEST(OversamplingWrapperTest, FlipLatencyReport) {
    for (const double sampleRate : {44100.0, 48000.0, 96000.0}) {
        // Reference latencies from freshly-prepared wrappers.
        std::array<int, 5> fresh{};
        for (int order = 0; order <= 4; ++order) {
            OversamplingWrapper reference(std::make_unique<PassthroughStage>(), order);
            reference.prepareToPlay(sampleRate, kFlipBlockSize, 2);
            fresh[order] = reference.getLatencySamples();
        }
        for (int order = 1; order <= 4; ++order) {
            EXPECT_LE(std::abs(fresh[order] - fresh[order - 1]), 4)
                << "order-to-order latency delta too large at sr " << sampleRate;
        }

        auto wrapper = std::make_unique<OversamplingWrapper>(std::make_unique<PassthroughStage>(), 0);
        wrapper->prepareToPlay(sampleRate, kFlipBlockSize, 2);

        juce::AudioBuffer<double> buffer(2, kFlipBlockSize);
        for (int order = 1; order <= 4; ++order) {
            wrapper->setOversamplingOrder(order);
            // The report must reflect the TARGET order immediately (the host
            // reads latency on the message thread right after a preset load).
            EXPECT_EQ(wrapper->getLatencySamples(), fresh[order]) << "sr " << sampleRate;
            for (int b = 0; b < 2; ++b) { // let the flip execute
                fillSine(buffer, kSineHz, sampleRate);
                wrapper->process(buffer);
            }
            EXPECT_EQ(wrapper->getOversamplingOrder(), order);
            EXPECT_EQ(wrapper->getLatencySamples(), fresh[order]) << "post-flip, sr " << sampleRate;
        }
    }
}

TEST(OversamplingWrapperTest, FlipAllocationAudit) {
#if defined(_WIN32)
    GTEST_SKIP() << "global operator new replacement is unreliable against the MSVC CRT";
#else
    // Wrap the real oversampled nonlinear group (Bias -> Surge -> Hysteresis)
    // so the audit covers the exact prepareToPlay call graph the audio-thread
    // flip re-runs — including HysteresisStage's std::function nonlinearity
    // binding, the one suspected allocation.
    dsp_core::SeamlessTransferFunction tf;
    tf.prepareToPlay(kFlipSampleRate, kFlipBlockSize);
    tf.renderLUTImmediate();

    auto pipeline = std::make_unique<AudioPipeline>();
    pipeline->addStage(std::make_unique<BiasStage>());
    pipeline->addStage(std::make_unique<SurgeStage>());
    pipeline->addStage(std::make_unique<HysteresisStage>(tf));

    auto wrapper = std::make_unique<OversamplingWrapper>(std::move(pipeline), 1);
    wrapper->prepareToPlay(kFlipSampleRate, kFlipBlockSize, 2);

    juce::AudioBuffer<double> buffer(2, kFlipBlockSize);
    for (int b = 0; b < 4; ++b) { // warmup, outside the audit
        fillSine(buffer, kSineHz, kFlipSampleRate);
        wrapper->process(buffer);
    }

    wrapper->setOversamplingOrder(3);

    fillSine(buffer, kSineHz, kFlipSampleRate);
    osw_alloc_audit::flaggedAllocations.store(0, std::memory_order_relaxed);
    osw_alloc_audit::armedOnThisThread = true;
    wrapper->process(buffer); // the flip block
    osw_alloc_audit::armedOnThisThread = false;

    EXPECT_EQ(wrapper->getOversamplingOrder(), 3);
    EXPECT_EQ(osw_alloc_audit::flaggedAllocations.load(std::memory_order_relaxed), 0)
        << "the audio-thread flip must be allocation-free";
    EXPECT_TRUE(bufferIsFinite(buffer));
#endif
}
