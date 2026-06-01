#pragma once

#include "ToneStage.h"
#include <juce_core/juce_core.h>
#include <juce_dsp/juce_dsp.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <vector>

namespace dsp_core::audio_pipeline {

/**
 * Per-strategy frequency-response evaluator for ToneStage. Pure functions of
 * the parameter snapshot — no internal state, no stage references. Computed
 * on the UI thread inside ToneStage::recomputeFrequencyResponse() whenever a
 * tone parameter changes; result is published via the visualizer LUT.
 *
 * Magnitude sources per strategy:
 *  - Lowpass 12/24 dB: analytical linear small-signal response of the Moog
 *    ladder, cascaded with the Fat low-shelf when Fat > 0. The audio path is
 *    nonlinear (tanh in feedback) — the linear response is what the small
 *    signal sees and is the canonical visualization for the filter shape.
 *  - LowShelf / HighShelf / Bell: juce::dsp::IIR::Coefficients RBJ biquads
 *    (built once per recompute, evaluated at every requested frequency).
 *  - Smile: dB sum of LowShelf and HighShelf cascaded.
 *  - Hysteresis: flat (0 dB) — the saturator is fundamentally nonlinear and
 *    runs at the pipeline-level HysteresisStage, so the tone curve is identity.
 */
struct ToneFrequencyResponseParams {
    ToneStage::Type type = ToneStage::Type::Off;
    double cutoffHz = 3000.0;
    double resonanceNorm = 0.0; // [0, 1] — the pre-mapping UI value
    double shelfGainDb = 0.0;
    double fatPercent = 0.0;
    double lowShelfRatio = 0.0625;
    double bellQ = 1.0;
    double irMix = 1.0; // [0, 1] dry↔wet blend — ImpulseResponse type only
    double sampleRate = 48000.0;

    // ImpulseResponse type only. Caller (editor) computes the FFT-based
    // magnitude trace once per IR file and passes a borrowed pointer in.
    // irPathFingerprint participates in operator== so a file change triggers
    // recompute via the existing param-fingerprint short-circuit; the raw
    // pointer alone wouldn't (the cache buffer is reused, so its address is
    // stable across IR changes).
    const double* irMagnitudesDb = nullptr;
    juce::String irPathFingerprint;

    // Member-wise equality so the editor can fingerprint and skip recompute
    // when nothing changed since the last timer tick.
    bool operator==(const ToneFrequencyResponseParams&) const = default;
};

namespace tone_response_detail {

constexpr double kPi = 3.14159265358979323846;

// Matches ToneStage's UI→ladder resonance mapping (see ToneStage.cpp).
constexpr double kToneMaxResonance = 3.5;

// Fat: 0%→0 dB, 100%→+6 dB low shelf at 200 Hz, Butterworth Q.
constexpr double kFatShelfHz = 200.0;
constexpr double kFatShelfMaxDb = 6.0;
constexpr double kButterworthQ = 0.7071067811865476;

inline double biquadMagLinear(const juce::dsp::IIR::Coefficients<double>& coeffs, double hz, double sampleRate) {
    return coeffs.getMagnitudeForFrequency(hz, sampleRate);
}

inline double ladderMagLinear(double hz, double cutoffHz, double R, int order) {
    // Moog ladder small-signal linear transfer function:
    //   G(jω) = 1 / (1 + jω/ωc)
    //   |H_N(jω)| = |G^N / (1 + R·G^N)|
    // (1 + R) bass-makeup is applied by the caller in dB.
    const double omega = 2.0 * kPi * hz;
    const double wc = 2.0 * kPi * cutoffHz;
    const std::complex<double> denom(1.0, omega / wc);
    const std::complex<double> G = 1.0 / denom;
    std::complex<double> GN = G;
    for (int i = 1; i < order; ++i) {
        GN *= G;
    }
    const std::complex<double> H = GN / (1.0 + R * GN);
    return std::abs(H);
}

inline double ladderHighpassMagLinear(double hz, double cutoffHz, double R, int order) {
    // Moog HP cascade small-signal linear transfer function:
    //   G_HP(jω) = (jω/ωc) / (1 + jω/ωc)
    //   |H_N(jω)| = |G_HP^N / (1 + R·G_HP^N)|
    // (1 + R) passband-makeup is applied by the caller in dB.
    const double omega = 2.0 * kPi * hz;
    const double wc = 2.0 * kPi * cutoffHz;
    const std::complex<double> num(0.0, omega / wc);
    const std::complex<double> denom(1.0, omega / wc);
    const std::complex<double> G = num / denom;
    std::complex<double> GN = G;
    for (int i = 1; i < order; ++i) {
        GN *= G;
    }
    const std::complex<double> H = GN / (1.0 + R * GN);
    return std::abs(H);
}

inline double linearToDb(double mag) {
    constexpr double kMinLinear = 1e-9; // -180 dB floor — well past visualizer dbRange
    return 20.0 * std::log10(std::max(mag, kMinLinear));
}

} // namespace tone_response_detail

/**
 * Compute the magnitude response of an IR (`monoSamples`, `numSamples` at
 * `irSampleRate`) on the same log-spaced frequency grid the rest of the tone
 * visualizer uses. Result is peak-normalized so max(magsDb) == 0.
 *
 * FFT size is the next power of two ≥ numSamples, clamped to [4096, 65536].
 * 4096 keeps low-frequency resolution sensible for short cab IRs; 65536 caps
 * the cost for long reverb tails. No window — we want the IR's true spectrum,
 * not a windowed slice. Bandpower averaging over ±1/12 octave (= 1/6 octave
 * total) smooths the comb ripples a finite-length IR's phase response would
 * otherwise put on screen.
 *
 * UI thread only — allocates an internal FFT scratch and the juce::dsp::FFT
 * object. Cost is one-shot per IR file change; the editor caches the result.
 */
inline void computeIRMagnitudeResponseDb(const float* monoSamples, int numSamples,
                                         double irSampleRate, const double* freqsHz,
                                         double* magsDb, int count) {
    using namespace tone_response_detail;

    if (monoSamples == nullptr || numSamples <= 0 || irSampleRate <= 0.0 ||
        freqsHz == nullptr || magsDb == nullptr || count <= 0) {
        if (magsDb != nullptr && count > 0) {
            std::fill_n(magsDb, count, 0.0);
        }
        return;
    }

    // FFT size: next power of two >= numSamples, clamped to [4096, 65536].
    constexpr int kMinFftSize = 4096;
    constexpr int kMaxFftSize = 65536;
    int fftSize = kMinFftSize;
    while (fftSize < numSamples && fftSize < kMaxFftSize) {
        fftSize <<= 1;
    }
    fftSize = juce::jlimit(kMinFftSize, kMaxFftSize, fftSize);

    int order = 0;
    for (int v = fftSize; v > 1; v >>= 1) {
        ++order;
    }

    juce::dsp::FFT fft(order);

    // performRealOnlyForwardTransform expects a buffer of size 2*fftSize.
    std::vector<float> scratch(static_cast<size_t>(fftSize) * 2, 0.0f);
    const int copyCount = std::min(numSamples, fftSize);
    std::memcpy(scratch.data(), monoSamples, static_cast<size_t>(copyCount) * sizeof(float));

    fft.performRealOnlyForwardTransform(scratch.data());

    // Real-only forward output is interleaved [re0, im0, re1, im1, ...] for
    // bins 0..N/2. Convert to power per bin.
    const int numBins = fftSize / 2 + 1;
    std::vector<double> power(static_cast<size_t>(numBins), 0.0);
    for (int k = 0; k < numBins; ++k) {
        const double re = scratch[static_cast<size_t>(k) * 2];
        const double im = scratch[static_cast<size_t>(k) * 2 + 1];
        power[static_cast<size_t>(k)] = re * re + im * im;
    }

    // Hz per FFT bin.
    const double binHz = irSampleRate / static_cast<double>(fftSize);
    const double nyquist = irSampleRate * 0.5;

    // Two-pass smoothing:
    //  1) per-target log-linear bin lerp — gives a continuous power curve
    //     even at low frequencies where consecutive LUT entries would
    //     otherwise read the same FFT bin (the previous stair-step bug).
    //  2) triangular smoothing across the LUT itself, ±1/12 octave wide
    //     in log-frequency space. Suppresses the comb ripples a finite-
    //     length IR's phase response produces at HF without adding the
    //     LF plateaus that hard-band-edge averaging causes.
    constexpr double kMinPower = 1e-12; // -120 dB floor pre-normalization.

    std::vector<double> rawDb(static_cast<size_t>(count), -60.0);
    for (int i = 0; i < count; ++i) {
        const double f = freqsHz[i];
        if (f <= 0.0 || f >= nyquist) {
            continue;
        }
        // Log-linear interpolate power between the two FFT bins straddling f.
        // Skip the DC bin (k=0): for very-low targets just read bin 1 — DC
        // is not part of the response and would pull the floor down sharply.
        const double kReal = f / binHz;
        const int k0 = juce::jlimit(1, numBins - 1, static_cast<int>(std::floor(kReal)));
        const int k1 = juce::jlimit(1, numBins - 1, k0 + 1);
        const double frac = juce::jlimit(0.0, 1.0, kReal - std::floor(kReal));
        const double p = (1.0 - frac) * power[static_cast<size_t>(k0)] +
                         frac * power[static_cast<size_t>(k1)];
        rawDb[static_cast<size_t>(i)] = 10.0 * std::log10(std::max(p, kMinPower));
    }

    // Triangular smoothing kernel half-width = number of LUT bins covering
    // 1/12 octave. The LUT is log-spaced, so a fixed bin count gives a
    // fixed octave bandwidth — no need to recompute per target.
    int kernelHalfWidth = 1;
    if (count >= 2) {
        const double octavesPerBin = std::log2(freqsHz[count - 1] / freqsHz[0]) /
                                     static_cast<double>(count - 1);
        if (octavesPerBin > 0.0) {
            kernelHalfWidth = juce::jmax(1, static_cast<int>(std::round((1.0 / 12.0) / octavesPerBin)));
        }
    }

    for (int i = 0; i < count; ++i) {
        const int jMin = juce::jmax(0, i - kernelHalfWidth);
        const int jMax = juce::jmin(count - 1, i + kernelHalfWidth);
        double sum = 0.0;
        double wsum = 0.0;
        for (int j = jMin; j <= jMax; ++j) {
            // Triangular: 1 at center, 0 at ±(kernelHalfWidth+1). Smoothing
            // in dB domain matches how the visualizer reads the LUT (linear
            // dB axis), so the resulting curve is visually-uniform.
            const double w = 1.0 - static_cast<double>(std::abs(i - j)) /
                                       static_cast<double>(kernelHalfWidth + 1);
            sum += w * rawDb[static_cast<size_t>(j)];
            wsum += w;
        }
        magsDb[i] = (wsum > 0.0) ? sum / wsum : rawDb[static_cast<size_t>(i)];
    }

    // Peak-normalize so max == 0 dB; floor at -60 dB so the trace stays well
    // inside the visualizer's ±24 dB display range and log math stays sane.
    double peak = magsDb[0];
    for (int i = 1; i < count; ++i) {
        peak = std::max(peak, magsDb[i]);
    }
    for (int i = 0; i < count; ++i) {
        magsDb[i] = std::max(magsDb[i] - peak, -60.0);
    }
}

/**
 * Fill `magnitudesDb` (length `count`) with the strategy's frequency response
 * in dB at each frequency in `freqsHz`. Caller pre-allocates both arrays.
 * Off type writes a flat 0 dB curve.
 *
 * Thread-safe with respect to ToneStage's atomic params because the caller
 * passes a snapshot; no shared mutable state.
 */
inline void computeFrequencyResponseDb(const ToneFrequencyResponseParams& params,
                                       const double* freqsHz, double* magnitudesDb, int count) {
    using namespace tone_response_detail;

    if (params.type == ToneStage::Type::Off || params.type == ToneStage::Type::Hysteresis) {
        // Hysteresis has no linear tone shaping — the saturator is nonlinear and
        // runs at the pipeline-level HysteresisStage. Draw a flat (0 dB) curve.
        for (int i = 0; i < count; ++i) {
            magnitudesDb[i] = 0.0;
        }
        return;
    }

    if (params.type == ToneStage::Type::ImpulseResponse) {
        // Editor computes the IR's FFT-based magnitude once per IR file change
        // and passes a borrowed pointer in via params.irMagnitudesDb. Null
        // means no IR is loaded — fall back to a flat trace so the overlay
        // still draws cleanly while the user picks a file.
        if (params.irMagnitudesDb == nullptr) {
            std::fill_n(magnitudesDb, count, 0.0);
            return;
        }
        // Blend the IR curve against a 0 dB dry reference. The dry path is unity
        // (0 dB, flat), so in the linear domain the blended magnitude is
        //   |H| = (1 - mix)·1 + mix·10^(irDb/20)
        // a phase-aligned sum: at mix=1 it reproduces the pure-IR trace, at
        // mix=0 it is flat 0 dB, and partial mixes fill the IR's notches the
        // way a parallel dry path does. Magnitude-only (the cached IR trace is
        // peak-normalized, phase discarded) — the in-phase sum is the intuitive
        // best-case fill and the right visualization for a dry/wet blend.
        const double mix = juce::jlimit(0.0, 1.0, params.irMix);
        for (int i = 0; i < count; ++i) {
            const double irLinear = std::pow(10.0, params.irMagnitudesDb[i] / 20.0);
            const double blended = (1.0 - mix) + mix * irLinear;
            magnitudesDb[i] = linearToDb(blended);
        }
        return;
    }

    const double sr = params.sampleRate > 0.0 ? params.sampleRate : 48000.0;
    const double cutoffHz = juce::jlimit(20.0, 20000.0, params.cutoffHz);
    const double gainDb = juce::jlimit(-24.0, 24.0, params.shelfGainDb);
    const double gainLinear = juce::Decibels::decibelsToGain(gainDb);

    switch (params.type) {
        case ToneStage::Type::Lowpass12dB:
        case ToneStage::Type::Lowpass24dB: {
            const int order = (params.type == ToneStage::Type::Lowpass12dB) ? 2 : 4;
            const double R = juce::jlimit(0.0, 1.0, params.resonanceNorm) * kToneMaxResonance;
            const double bassMakeupDb = 20.0 * std::log10(1.0 + R);
            const double fatPct = juce::jlimit(0.0, 100.0, params.fatPercent);
            const double fatGainDb = (fatPct / 100.0) * kFatShelfMaxDb;
            // Fat shelf coefficients built once per recompute, evaluated at every frequency.
            // makeLowShelf allocates a ReferenceCountedPtr — one allocation per recompute, not per sample.
            juce::dsp::IIR::Coefficients<double>::Ptr fatCoeffs;
            if (fatGainDb > 0.0) {
                fatCoeffs = juce::dsp::IIR::Coefficients<double>::makeLowShelf(
                    sr, kFatShelfHz, kButterworthQ, juce::Decibels::decibelsToGain(fatGainDb));
            }
            for (int i = 0; i < count; ++i) {
                double mag = ladderMagLinear(freqsHz[i], cutoffHz, R, order);
                if (fatCoeffs != nullptr) {
                    mag *= biquadMagLinear(*fatCoeffs, freqsHz[i], sr);
                }
                magnitudesDb[i] = linearToDb(mag) + bassMakeupDb;
            }
            break;
        }
        case ToneStage::Type::LowShelf: {
            auto coeffs = juce::dsp::IIR::Coefficients<double>::makeLowShelf(sr, cutoffHz, kButterworthQ, gainLinear);
            for (int i = 0; i < count; ++i) {
                magnitudesDb[i] = linearToDb(biquadMagLinear(*coeffs, freqsHz[i], sr));
            }
            break;
        }
        case ToneStage::Type::HighShelf: {
            auto coeffs = juce::dsp::IIR::Coefficients<double>::makeHighShelf(sr, cutoffHz, kButterworthQ, gainLinear);
            for (int i = 0; i < count; ++i) {
                magnitudesDb[i] = linearToDb(biquadMagLinear(*coeffs, freqsHz[i], sr));
            }
            break;
        }
        case ToneStage::Type::Smile: {
            const double lsRatio = juce::jlimit(0.015, 0.5, params.lowShelfRatio);
            const double lsHz = juce::jlimit(20.0, 20000.0, cutoffHz * lsRatio);
            auto hsCoeffs = juce::dsp::IIR::Coefficients<double>::makeHighShelf(sr, cutoffHz, kButterworthQ, gainLinear);
            auto lsCoeffs = juce::dsp::IIR::Coefficients<double>::makeLowShelf(sr, lsHz, kButterworthQ, gainLinear);
            for (int i = 0; i < count; ++i) {
                const double mag = biquadMagLinear(*hsCoeffs, freqsHz[i], sr) * biquadMagLinear(*lsCoeffs, freqsHz[i], sr);
                magnitudesDb[i] = linearToDb(mag);
            }
            break;
        }
        case ToneStage::Type::Bell: {
            const double q = juce::jlimit(0.1, 10.0, params.bellQ);
            auto coeffs = juce::dsp::IIR::Coefficients<double>::makePeakFilter(sr, cutoffHz, q, gainLinear);
            for (int i = 0; i < count; ++i) {
                magnitudesDb[i] = linearToDb(biquadMagLinear(*coeffs, freqsHz[i], sr));
            }
            break;
        }
        case ToneStage::Type::Highpass12dB:
        case ToneStage::Type::Highpass24dB: {
            const int order = (params.type == ToneStage::Type::Highpass12dB) ? 2 : 4;
            const double R = juce::jlimit(0.0, 1.0, params.resonanceNorm) * kToneMaxResonance;
            const double passbandMakeupDb = 20.0 * std::log10(1.0 + R);
            for (int i = 0; i < count; ++i) {
                const double mag = ladderHighpassMagLinear(freqsHz[i], cutoffHz, R, order);
                magnitudesDb[i] = linearToDb(mag) + passbandMakeupDb;
            }
            break;
        }
        case ToneStage::Type::Off:
        case ToneStage::Type::Hysteresis:
        case ToneStage::Type::ImpulseResponse:
            // Handled above by the early return — included here so the switch
            // covers every enum value (silences -Wswitch).
            break;
    }
}

} // namespace dsp_core::audio_pipeline
