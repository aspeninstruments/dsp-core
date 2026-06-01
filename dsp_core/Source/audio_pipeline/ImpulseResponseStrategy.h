#pragma once

#include "ToneFilterStrategy.h"
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_dsp/juce_dsp.h>
#include <atomic>

namespace dsp_core::audio_pipeline {

/**
 * Impulse-response tone-filter strategy: a partitioned-FFT convolver fed by a
 * user-selected .wav file (cabinet sim, space capture, etc.). Wraps
 * juce::dsp::Convolution, which handles async load, sample-rate conversion,
 * and lock-free hot-swap of the IR buffer on the audio thread.
 *
 * Double-precision wrinkle: juce::dsp::Convolution operates on float, but the
 * pipeline carries juce::AudioBuffer<double>. We keep a pre-allocated float
 * scratch buffer (sized in prepareToPlay) and copy down → convolve → copy up
 * per block. No allocation on the audio thread.
 *
 * All ToneFilterStrategy parameter setters (frequency/resonance/gain/Q/etc.)
 * are no-ops — the IR file itself is the cabinet/space character, configured
 * out-of-band via setImpulseResponseFile().
 *
 * Mix: setMix() blends the convolved (wet) signal with the dry input. 1.0 =
 * 100% wet (the convolver's full output), 0.0 = dry passthrough. The blend is
 * phase-coherent: juce::dsp::Convolution is a zero-latency engine here, but we
 * query getLatency() and delay the dry path by that many samples if it is ever
 * non-zero (e.g. a future fixed-latency mode) so the parallel sum doesn't comb.
 */
class ImpulseResponseStrategy : public ToneFilterStrategy {
  public:
    ImpulseResponseStrategy()
        : convolution_(juce::dsp::Convolution::NonUniform{kHeadBlockSize}) {}

    void prepareToPlay(double sampleRate, int samplesPerBlock, int numChannels) override {
        prepared_ = false;
        currentSampleRate_ = sampleRate;
        currentBlockSize_ = samplesPerBlock;
        currentNumChannels_ = juce::jmax(1, numChannels);

        juce::dsp::ProcessSpec spec{};
        spec.sampleRate = sampleRate;
        spec.maximumBlockSize = static_cast<juce::uint32>(samplesPerBlock);
        spec.numChannels = static_cast<juce::uint32>(currentNumChannels_);
        convolution_.prepare(spec);

        scratch_.setSize(currentNumChannels_, samplesPerBlock, false, false, true);
        scratch_.clear();

        // Mix smoothing: ~10 ms ramp so dialing/modulating the blend is click-free.
        mixSmoothed_.reset(sampleRate, 0.01);
        mixSmoothed_.setCurrentAndTargetValue(mix_.load(std::memory_order_acquire));

        // Phase-coherent dry path: if the convolver reports a latency, delay the
        // dry signal by the same amount before blending. NonUniform is zero-
        // latency, so in practice this stays 0 and the direct-blend path runs.
        irLatencySamples_ = juce::jmax(0, convolution_.getLatency());
        dryDelay_.prepare(spec);
        dryDelay_.setMaximumDelayInSamples(juce::jmax(1, irLatencySamples_));
        dryDelay_.setDelay(static_cast<double>(irLatencySamples_));
        dryDelay_.reset();

        prepared_ = true;

        // If a file was queued before prepareToPlay (e.g. restored from state),
        // load it now that the convolution is sized.
        if (queuedFile_ != juce::File{}) {
            loadFileNow(queuedFile_);
        }
    }

    void process(juce::AudioBuffer<double>& buffer) override {
        if (!prepared_ || !hasImpulseResponse_.load(std::memory_order_acquire)) {
            // No IR loaded — pass audio through unchanged so the user hears
            // raw signal while picking a file, rather than silence. Mix is
            // irrelevant here (the "wet" convolver has nothing to contribute).
            return;
        }

        const int numChannels = juce::jmin(buffer.getNumChannels(), scratch_.getNumChannels());
        const int numSamples = buffer.getNumSamples();
        if (numChannels <= 0 || numSamples <= 0) {
            return;
        }

        mixSmoothed_.setTargetValue(mix_.load(std::memory_order_acquire));

        // Down-cast to float (dry copy for the convolver). The buffer still
        // holds the dry signal until the up-cast/blend loop overwrites it.
        for (int ch = 0; ch < numChannels; ++ch) {
            const auto* src = buffer.getReadPointer(ch);
            auto* dst = scratch_.getWritePointer(ch);
            for (int i = 0; i < numSamples; ++i) {
                dst[i] = static_cast<float>(src[i]);
            }
        }

        juce::dsp::AudioBlock<float> block(scratch_.getArrayOfWritePointers(),
                                           static_cast<size_t>(numChannels),
                                           static_cast<size_t>(numSamples));
        juce::dsp::ProcessContextReplacing<float> context(block);
        convolution_.process(context);

        // Up-cast wet back to double and blend with the (latency-matched) dry.
        // The smoothed mix is advanced once per frame and applied to all
        // channels so the L/R blend stays phase-aligned.
        for (int i = 0; i < numSamples; ++i) {
            const double g = mixSmoothed_.getNextValue();
            for (int ch = 0; ch < numChannels; ++ch) {
                const double dry = buffer.getReadPointer(ch)[i];
                double delayedDry = dry;
                if (irLatencySamples_ > 0) {
                    dryDelay_.pushSample(ch, dry);
                    delayedDry = dryDelay_.popSample(ch);
                }
                const double wet = static_cast<double>(scratch_.getReadPointer(ch)[i]);
                buffer.getWritePointer(ch)[i] = g * wet + (1.0 - g) * delayedDry;
            }
        }
    }

    void reset() override {
        convolution_.reset();
        dryDelay_.reset();
        mixSmoothed_.setCurrentAndTargetValue(mix_.load(std::memory_order_acquire));
    }

    juce::String getName() const override {
        return "ImpulseResponse";
    }

    // ToneFilterStrategy universal setters — IR is configured via the file
    // itself, not via the tone-row dials.
    void setFrequency(double /*frequencyHz*/) override {}
    void setResonance(double /*zeroToOne*/) override {}
    void setShelfGainDb(double /*gainDb*/) override {}
    void setFat(double /*percent*/) override {}
    void setLowShelfRatio(double /*ratio*/) override {}
    void setQ(double /*q*/) override {}

    /** Dry↔wet blend, [0, 1]. 1.0 = 100% wet (full convolver output), 0.0 =
     *  dry passthrough. Thread-safe; the audio thread smooths toward it. */
    void setMix(double zeroToOne) {
        mix_.store(juce::jlimit(0.0, 1.0, zeroToOne), std::memory_order_release);
    }

    /** Load an IR from disk. Must be called from the message thread.
     *  juce::dsp::Convolution::loadImpulseResponse loads on a background thread
     *  and the audio thread picks up the new buffer atomically — safe to call
     *  while audio is running. If prepareToPlay hasn't been called yet, the
     *  file is queued and loaded then. */
    void setImpulseResponseFile(const juce::File& file) {
        if (file == juce::File{}) {
            queuedFile_ = juce::File{};
            hasImpulseResponse_.store(false, std::memory_order_release);
            // No way to "unload" Convolution short of feeding it a unit impulse;
            // the audio path checks hasImpulseResponse_ and bypasses instead.
            return;
        }
        if (!prepared_) {
            queuedFile_ = file;
            return;
        }
        loadFileNow(file);
    }

  private:
    // Partitioned-convolution head size. 512-sample head limits the worst-case
    // audio-thread FFT cost while staying small enough that the latency the
    // convolver adds at typical block sizes is negligible vs. the IR length.
    static constexpr int kHeadBlockSize = 512;

    void loadFileNow(const juce::File& file) {
        convolution_.loadImpulseResponse(file,
                                          juce::dsp::Convolution::Stereo::yes,
                                          juce::dsp::Convolution::Trim::yes,
                                          0, // size: 0 → use original length
                                          juce::dsp::Convolution::Normalise::yes);
        queuedFile_ = juce::File{};
        hasImpulseResponse_.store(true, std::memory_order_release);
    }

    juce::dsp::Convolution convolution_;
    juce::AudioBuffer<float> scratch_;
    double currentSampleRate_ = 0.0;
    int currentBlockSize_ = 0;
    int currentNumChannels_ = 0;
    bool prepared_ = false;
    juce::File queuedFile_;
    std::atomic<bool> hasImpulseResponse_{false};

    // Dry↔wet blend. mix_ is the UI/automation target; mixSmoothed_ ramps the
    // audio thread toward it (click-free). Default 1.0 = 100% wet.
    std::atomic<double> mix_{1.0};
    juce::LinearSmoothedValue<double> mixSmoothed_{1.0};

    // Dry-path latency compensation for a phase-coherent parallel blend. 0 for
    // the zero-latency NonUniform convolver; the delay line is only walked when
    // irLatencySamples_ > 0.
    int irLatencySamples_ = 0;
    juce::dsp::DelayLine<double, juce::dsp::DelayLineInterpolationTypes::None> dryDelay_{1};
};

} // namespace dsp_core::audio_pipeline
