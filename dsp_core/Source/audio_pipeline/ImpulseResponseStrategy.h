#pragma once

#include "ToneFilterStrategy.h"
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_dsp/juce_dsp.h>
#include <atomic>
#include <cmath>
#include <memory>
#include <vector>

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
        : convolution_(juce::dsp::Convolution::NonUniform{kHeadBlockSize}) {
        formatManager_.registerBasicFormats(); // for offline makeup-gain analysis
    }

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

        // Makeup smoothing: ~20 ms (slower than mix) so a per-IR level jump at
        // load is gentle.
        makeupSmoothed_.reset(sampleRate, 0.02);
        makeupSmoothed_.setCurrentAndTargetValue(makeupLinear_.load(std::memory_order_acquire));

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
            // loadFileNow has computed the IR's makeup; land on it immediately so
            // a state-restored IR doesn't audibly ramp from unity at startup.
            makeupSmoothed_.setCurrentAndTargetValue(makeupLinear_.load(std::memory_order_acquire));
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
        makeupSmoothed_.setTargetValue(makeupLinear_.load(std::memory_order_acquire));

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
            // Makeup is advanced once per sample (same cadence as the mix) and
            // applied to the wet path only — so mix=0 stays a bit-exact dry
            // passthrough and makeup means "the IR's level," not the dry's.
            const double makeup = makeupSmoothed_.getNextValue();
            for (int ch = 0; ch < numChannels; ++ch) {
                const double dry = buffer.getReadPointer(ch)[i];
                double delayedDry = dry;
                if (irLatencySamples_ > 0) {
                    dryDelay_.pushSample(ch, dry);
                    delayedDry = dryDelay_.popSample(ch);
                }
                const double wet = makeup * static_cast<double>(scratch_.getReadPointer(ch)[i]);
                buffer.getWritePointer(ch)[i] = g * wet + (1.0 - g) * delayedDry;
            }
        }
    }

    void reset() override {
        convolution_.reset();
        dryDelay_.reset();
        mixSmoothed_.setCurrentAndTargetValue(mix_.load(std::memory_order_acquire));
        makeupSmoothed_.setCurrentAndTargetValue(makeupLinear_.load(std::memory_order_acquire));
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
            makeupLinear_.store(1.0, std::memory_order_release); // no stale makeup
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

    /** Makeup gain (linear) that restores ~unity in-band loudness for an IR
     *  buffer. JUCE's Normalise::yes pins every IR to L2 = 0.125 (−18 dB
     *  broadband), so the convolver's in-band gain is 0.125 × (inBandRms /
     *  broadbandRms) of the IR's magnitude response — a scale-invariant spectral
     *  ratio read straight off the samples (shape in Hz is sample-rate-
     *  independent; trim is phase-only for magnitude). Pure + static so it is
     *  unit-testable without the async convolver. Analyses the channel with the
     *  most energy (JUCE's normalisation reference). */
    static double makeupGainForIR(const juce::AudioBuffer<float>& buf, double sampleRate) {
        const int numCh = buf.getNumChannels();
        const int len = buf.getNumSamples();
        if (numCh <= 0 || len <= 0 || sampleRate <= 0.0) {
            return 1.0;
        }

        // Reference channel = the one JUCE normalises against (max energy).
        int refCh = 0;
        double bestSumSq = -1.0;
        for (int ch = 0; ch < numCh; ++ch) {
            double sumSq = 0.0;
            const float* p = buf.getReadPointer(ch);
            for (int n = 0; n < len; ++n) {
                sumSq += static_cast<double>(p[n]) * static_cast<double>(p[n]);
            }
            if (sumSq > bestSumSq) {
                bestSumSq = sumSq;
                refCh = ch;
            }
        }
        if (bestSumSq < 1e-8) {
            return 1.0; // silent IR — mirror JUCE's own guard
        }

        // Zero-padded magnitude spectrum of the reference channel.
        int order = 1;
        while ((1 << order) < len) {
            ++order;
        }
        order = juce::jlimit(1, kMaxFftOrder, order);
        const int fftSize = 1 << order;
        std::vector<float> fft(static_cast<size_t>(fftSize) * 2, 0.0f);
        const float* src = buf.getReadPointer(refCh);
        const int copyN = juce::jmin(len, fftSize);
        for (int n = 0; n < copyN; ++n) {
            fft[static_cast<size_t>(n)] = src[n];
        }
        juce::dsp::FFT transform(order);
        transform.performFrequencyOnlyForwardTransform(fft.data());

        // One-sided bins [0, fftSize/2]; bin k ↔ k · sampleRate / fftSize.
        const int nyqBin = fftSize / 2;
        const double hzPerBin = sampleRate / static_cast<double>(fftSize);
        const int loBin = juce::jlimit(0, nyqBin,
            static_cast<int>(std::ceil(kInBandLoHz / hzPerBin)));
        const int hiBin = juce::jlimit(0, nyqBin,
            static_cast<int>(std::floor(kInBandHiHz / hzPerBin)));

        double broadSumSq = 0.0;
        for (int k = 0; k <= nyqBin; ++k) {
            const double m = static_cast<double>(fft[static_cast<size_t>(k)]);
            broadSumSq += m * m;
        }
        double bandSumSq = 0.0;
        int bandN = 0;
        for (int k = loBin; k <= hiBin; ++k) {
            const double m = static_cast<double>(fft[static_cast<size_t>(k)]);
            bandSumSq += m * m;
            ++bandN;
        }
        if (bandN <= 0 || broadSumSq < 1e-20) {
            return 1.0;
        }

        const double broadMeanSq = broadSumSq / static_cast<double>(nyqBin + 1);
        const double bandMeanSq = bandSumSq / static_cast<double>(bandN);
        if (broadMeanSq < 1e-30) {
            return 1.0;
        }

        // Convolver in-band gain after JUCE's fixed 0.125 broadband pin.
        const double measuredInBandGain = 0.125 * std::sqrt(bandMeanSq / broadMeanSq);
        if (measuredInBandGain < 1e-9) {
            return juce::Decibels::decibelsToGain(kMaxMakeupDb); // pathological → clamp
        }

        constexpr double kTargetInBandRms = 1.0; // unity in-band loudness
        const double makeupDb = juce::jlimit(kMinMakeupDb, kMaxMakeupDb,
            juce::Decibels::gainToDecibels(kTargetInBandRms / measuredInBandGain));
        return juce::Decibels::decibelsToGain(makeupDb);
    }

  private:
    // Partitioned-convolution head size. 512-sample head limits the worst-case
    // audio-thread FFT cost while staying small enough that the latency the
    // convolver adds at typical block sizes is negligible vs. the IR length.
    static constexpr int kHeadBlockSize = 512;

    // Auto loudness-match analysis bounds. The makeup gain targets unity loudness
    // across the musical band [kInBandLoHz, kInBandHiHz], clamped to
    // [kMinMakeupDb, kMaxMakeupDb] (+18 dB == the flat/white-IR ceiling that just
    // restores broadband unity; −6 dB pulls down an unusually hot/bright IR).
    static constexpr double kInBandLoHz = 80.0;
    static constexpr double kInBandHiHz = 5000.0;
    static constexpr double kMinMakeupDb = -6.0;
    static constexpr double kMaxMakeupDb = 18.0;
    static constexpr int kMaxFftOrder = 16;                 // ≤ 65536-pt analysis
    static constexpr juce::int64 kMaxAnalysisSamples = 1 << 16;

    void loadFileNow(const juce::File& file) {
        convolution_.loadImpulseResponse(file,
                                          juce::dsp::Convolution::Stereo::yes,
                                          juce::dsp::Convolution::Trim::yes,
                                          0, // size: 0 → use original length
                                          juce::dsp::Convolution::Normalise::yes);
        // JUCE's Normalise::yes pins every IR to a fixed −18 dB broadband energy,
        // which leaves the musical band ~12 dB low. Restore ~unity in-band loudness
        // with a per-IR makeup gain measured from the file's own spectrum.
        makeupLinear_.store(computeMakeupLinear(file), std::memory_order_release);
        queuedFile_ = juce::File{};
        hasImpulseResponse_.store(true, std::memory_order_release);
    }

    /** Read an IR file into a buffer and compute its makeup gain. Runs on the
     *  message thread (called from loadFileNow); allocation is fine here. The
     *  analysis lives in the pure, testable makeupGainForIR(). */
    double computeMakeupLinear(const juce::File& file) {
        std::unique_ptr<juce::AudioFormatReader> reader(formatManager_.createReaderFor(file));
        if (reader == nullptr || reader->lengthInSamples <= 0 || reader->sampleRate <= 0.0) {
            return 1.0; // unreadable → no compensation
        }

        const int numCh = juce::jmax(1, static_cast<int>(reader->numChannels));
        const int len = static_cast<int>(
            juce::jmin<juce::int64>(reader->lengthInSamples, kMaxAnalysisSamples));
        if (len <= 0) {
            return 1.0;
        }

        juce::AudioBuffer<float> buf(numCh, len);
        buf.clear();
        reader->read(&buf, 0, len, 0, true, true);
        return makeupGainForIR(buf, reader->sampleRate);
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

    // Auto loudness-match makeup. makeupLinear_ is the message-thread target
    // (linear gain, computed per-IR at load); makeupSmoothed_ ramps the audio
    // thread toward it. Applied to the wet path only. Default 1.0 (no IR loaded).
    std::atomic<double> makeupLinear_{1.0};
    juce::LinearSmoothedValue<double> makeupSmoothed_{1.0};
    juce::AudioFormatManager formatManager_; // offline IR analysis (message thread)

    // Dry-path latency compensation for a phase-coherent parallel blend. 0 for
    // the zero-latency NonUniform convolver; the delay line is only walked when
    // irLatencySamples_ > 0.
    int irLatencySamples_ = 0;
    juce::dsp::DelayLine<double, juce::dsp::DelayLineInterpolationTypes::None> dryDelay_{1};
};

} // namespace dsp_core::audio_pipeline
