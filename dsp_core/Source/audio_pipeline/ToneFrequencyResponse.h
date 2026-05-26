#pragma once

#include "ToneStage.h"
#include <juce_dsp/juce_dsp.h>
#include <cmath>
#include <complex>

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
 *  - Hysteresis: net linear cascade of the NAB pre/de-emphasis pair (the
 *    saturator is fundamentally nonlinear and is not represented here).
 */
struct ToneFrequencyResponseParams {
    ToneStage::Type type = ToneStage::Type::Off;
    double cutoffHz = 3000.0;
    double resonanceNorm = 0.0; // [0, 1] — the pre-mapping UI value
    double shelfGainDb = 0.0;
    double fatPercent = 0.0;
    double lowShelfRatio = 0.0625;
    double bellQ = 1.0;
    double emphNorm = 0.5;
    double sampleRate = 48000.0;
};

namespace tone_response_detail {

constexpr double kPi = 3.14159265358979323846;

// Matches ToneStage's UI→ladder resonance mapping (see ToneStage.cpp).
constexpr double kToneMaxResonance = 3.5;

// Fat: 0%→0 dB, 100%→+6 dB low shelf at 200 Hz, Butterworth Q.
constexpr double kFatShelfHz = 200.0;
constexpr double kFatShelfMaxDb = 6.0;
constexpr double kButterworthQ = 0.7071067811865476;

// Hysteresis NAB constants (mirror HysteresisStrategy.h).
constexpr double kHystHfCornerHz = 3183.0;
constexpr double kHystLfCornerHz = 50.0;
constexpr double kHystLfBumpDb = 1.5;
constexpr double kHystPreEmphMaxDb = 12.0;
constexpr double kHystDeEmphMaxDb = 9.0;

// 1st-order shelf magnitude using the analog prototype (matches NAB shelf
// behaviour at frequencies well below Nyquist, which covers the entire
// 10 Hz–20 kHz display range).
inline double firstOrderShelfMagLinear(double hz, double cornerHz, double gainDb, bool highShelf) {
    if (gainDb == 0.0) {
        return 1.0;
    }
    const double A = std::pow(10.0, gainDb / 20.0);
    const double alpha = std::sqrt(A);
    const double omega = 2.0 * kPi * hz;
    const double omega0 = 2.0 * kPi * cornerHz;
    // HighShelf: H(s) = A·(s + ω0/α) / (s + ω0·α)
    // LowShelf:  H(s) =     (s + ω0·α) / (s + ω0/α)
    if (highShelf) {
        const std::complex<double> num(omega0 / alpha, omega);
        const std::complex<double> den(omega0 * alpha, omega);
        return A * std::abs(num) / std::abs(den);
    }
    const std::complex<double> num(omega0 * alpha, omega);
    const std::complex<double> den(omega0 / alpha, omega);
    return std::abs(num) / std::abs(den);
}

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

inline double linearToDb(double mag) {
    constexpr double kMinLinear = 1e-9; // -180 dB floor — well past visualizer dbRange
    return 20.0 * std::log10(std::max(mag, kMinLinear));
}

} // namespace tone_response_detail

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

    if (params.type == ToneStage::Type::Off) {
        for (int i = 0; i < count; ++i) {
            magnitudesDb[i] = 0.0;
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
        case ToneStage::Type::Hysteresis: {
            // Net linear cascade of the NAB pre/de-emphasis pair plus the de-side LF bump.
            // Pre: +emph·12 dB high shelf @ 3183 Hz. De: -emph·9 dB high shelf @ 3183 Hz, plus +1.5 dB LF shelf @ 50 Hz.
            const double emph = juce::jlimit(0.0, 1.0, params.emphNorm);
            const double preDb = +emph * kHystPreEmphMaxDb;
            const double deDb = -emph * kHystDeEmphMaxDb;
            for (int i = 0; i < count; ++i) {
                const double hz = freqsHz[i];
                double mag = firstOrderShelfMagLinear(hz, kHystHfCornerHz, preDb, /*highShelf=*/true)
                           * firstOrderShelfMagLinear(hz, kHystHfCornerHz, deDb, /*highShelf=*/true)
                           * firstOrderShelfMagLinear(hz, kHystLfCornerHz, kHystLfBumpDb, /*highShelf=*/false);
                magnitudesDb[i] = linearToDb(mag);
            }
            break;
        }
        case ToneStage::Type::Off:
            break;
    }
}

} // namespace dsp_core::audio_pipeline
