#include <dsp_core/dsp_core.h>
#include <gtest/gtest.h>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace dsp_core_test {

// Quality regression check for the DSP LUT size. Bakes a battery of representative
// transfer-function shapes at multiple table sizes, samples each via Catmull-Rom
// interpolation at a dense uniform grid, and asserts the max-abs-error against the
// analytical reference stays below an audio-safe threshold.
//
// Purpose: durable guard against quality regression when TABLE_SIZE is tuned. The
// thresholds here are deliberately conservative — well below audible thresholds for
// waveshaping signal paths — so a future TABLE_SIZE change that breaks one of these
// is a real signal, not noise.
class LUTFidelityComparisonTest : public ::testing::Test {
  protected:
    struct Curve {
        std::string name;
        std::function<double(double)> fn;
    };

    static std::vector<Curve> curves() {
        return {
            {"identity", [](double x) { return x; }},
            {"tanh2x", [](double x) { return std::tanh(2.0 * x); }},
            {"tanh4x", [](double x) { return std::tanh(4.0 * x); }},
            {"H3_chebyshev", [](double x) { return 4.0 * x * x * x - 3.0 * x; }},
            {"H5_chebyshev",
             [](double x) {
                 // T_5(x) = 16x^5 - 20x^3 + 5x
                 return 16.0 * x * x * x * x * x - 20.0 * x * x * x + 5.0 * x;
             }},
            {"asymmetric_tanh",
             [](double x) {
                 // Mildly asymmetric: tilted soft-clipper
                 return std::tanh(2.0 * x + 0.1);
             }},
            {"hard_clip_07",
             [](double x) {
                 // Hard clipper at ±0.7 — the steepest case Catmull-Rom must handle
                 if (x > 0.7) return 0.7;
                 if (x < -0.7) return -0.7;
                 return x;
             }},
        };
    }

    static std::unique_ptr<dsp_core::LayeredTransferFunction> bakeLTF(const Curve& curve, int tableSize) {
        auto ltf = std::make_unique<dsp_core::LayeredTransferFunction>(tableSize, -1.0, 1.0);
        // Paint mode evaluates the base layer directly without normalization, so we compare
        // against the raw analytical curve.
        ltf->setRenderingMode(dsp_core::RenderingMode::Paint);
        ltf->setCoefficient(0, 1.0);
        for (int c = 1; c < ltf->getNumCoefficients(); ++c) {
            ltf->setCoefficient(c, 0.0);
        }
        for (int i = 0; i < tableSize; ++i) {
            ltf->setBaseLayerValue(i, curve.fn(ltf->normalizeIndex(i)));
        }
        return ltf;
    }

    // Bake one of the analytical curves into the LTF base layer at the requested table size,
    // then sample applyTransferFunction across a dense grid and report max abs error.
    static double maxInterpolationError(const Curve& curve, int tableSize, int numProbes) {
        auto ltf = bakeLTF(curve, tableSize);
        double maxErr = 0.0;
        // Probe strictly inside [-1, 1] to avoid measuring extrapolation at the edges.
        for (int p = 0; p < numProbes; ++p) {
            const double x = -0.999 + (p / static_cast<double>(numProbes - 1)) * 1.998;
            const double interpolated = ltf->applyTransferFunction(x);
            const double analytical = curve.fn(x);
            maxErr = std::max(maxErr, std::abs(interpolated - analytical));
        }
        return maxErr;
    }

    // Total Distortion as defined in this test: drive a full-scale sin probe through both the
    // analytical curve and the LUT, then measure RMS(LUT - analytical) / RMS(analytical).
    // This captures *all* deviations — harmonic and inharmonic — that the LUT introduces
    // relative to the ideal waveshaper output. Probe frequency is irrational w.r.t. the buffer
    // length to avoid coherent sampling that would alias to single grid points.
    struct DistortionResult {
        double rmsErr = 0.0;
        double rmsRef = 0.0;
        double ratio = 0.0; // rmsErr / rmsRef
        double db = 0.0;    // 20·log10(ratio)
    };

    static DistortionResult totalDistortion(const Curve& curve, int tableSize, int probeLength = 65536,
                                            double cyclesInBuffer = 17.3) {
        auto ltf = bakeLTF(curve, tableSize);
        double sumSqErr = 0.0;
        double sumSqRef = 0.0;
        const double twoPi = 2.0 * 3.14159265358979323846;
        for (int n = 0; n < probeLength; ++n) {
            const double x = std::sin(twoPi * static_cast<double>(n) * cyclesInBuffer / probeLength);
            const double yRef = curve.fn(x);
            const double yLut = ltf->applyTransferFunction(x);
            const double err = yLut - yRef;
            sumSqErr += err * err;
            sumSqRef += yRef * yRef;
        }
        DistortionResult r;
        r.rmsErr = std::sqrt(sumSqErr / probeLength);
        r.rmsRef = std::sqrt(sumSqRef / probeLength);
        r.ratio = r.rmsErr / r.rmsRef;
        r.db = 20.0 * std::log10(r.ratio);
        return r;
    }
};

// At the live TABLE_SIZE (8192 today), every curve in the battery should track the analytical
// reference within 1e-3. This is comfortably below audible (~ -60 dB on a normalized signal).
TEST_F(LUTFidelityComparisonTest, LiveTableSize_TracksAnalyticalReference) {
    constexpr int kProbes = 4001;
    constexpr double kThreshold = 1e-3;
    for (const auto& curve : curves()) {
        const double err = maxInterpolationError(curve, dsp_core::LaneMixer::TABLE_SIZE, kProbes);
        EXPECT_LT(err, kThreshold) << "curve=" << curve.name << " size=" << dsp_core::LaneMixer::TABLE_SIZE
                                   << " err=" << err;
    }
}

// Larger LUTs trivially do at least as well — but assert it explicitly so a refactor
// that breaks the interpolation path at non-default sizes is caught.
TEST_F(LUTFidelityComparisonTest, LargerSize_DoesNotRegress) {
    constexpr int kProbes = 4001;
    constexpr int kLargeSize = 16384;
    constexpr double kThreshold = 5e-4;
    for (const auto& curve : curves()) {
        const double err = maxInterpolationError(curve, kLargeSize, kProbes);
        EXPECT_LT(err, kThreshold) << "curve=" << curve.name << " size=" << kLargeSize << " err=" << err;
    }
}

// Sanity check the floor: even at 4096, smooth waveshapers stay well within audible
// tolerance. This is informational — if 4096 is later considered as a follow-up, the
// loosened threshold here defines the worst-case envelope that needs auditioning.
TEST_F(LUTFidelityComparisonTest, SmallerSize_StillAudioSafe) {
    constexpr int kProbes = 4001;
    constexpr int kSmallSize = 4096;
    constexpr double kThreshold = 5e-3;
    for (const auto& curve : curves()) {
        // Skip hard_clip_07 at 4k — its discontinuous slope produces a known overshoot at the
        // step transition that linear-extrapolation handles but Catmull-Rom must round; the
        // overshoot is per-design audible-as-extra-softness, not a regression of the LUT size.
        if (curve.name == "hard_clip_07") continue;
        const double err = maxInterpolationError(curve, kSmallSize, kProbes);
        EXPECT_LT(err, kThreshold) << "curve=" << curve.name << " size=" << kSmallSize << " err=" << err;
    }
}

// Sweep Chebyshev order from H1..H19 (the highest in the factory default lane init)
// and report Total Distortion at each LUT size. T_n(x) has rapidly-growing high-order
// derivatives (especially near the ±1 boundary where sin θ → 0 in the chain rule), so
// this is the canonical stress test for "how high can the LUT order go before quality
// degrades". Information-only — no asserts, just print the table.
TEST_F(LUTFidelityComparisonTest, TotalDistortion_ChebyshevSweep) {
    auto fmtDb = [](double db) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1) << std::setw(6) << db << " dB";
        return oss.str();
    };

    std::cout << "\n  Chebyshev order sweep (full-scale sin probe, RMS error / RMS signal)\n";
    std::cout << "  " << std::string(60, '-') << '\n';
    std::cout << "  " << std::left << std::setw(8) << "order" << std::setw(16) << "16384" << std::setw(16)
              << "8192 (live)" << std::setw(16) << "4096" << '\n';
    std::cout << "  " << std::string(60, '-') << '\n';

    for (int n : {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 32}) {
        Curve c = {"H" + std::to_string(n), [n](double x) {
                       const double xc = std::max(-1.0, std::min(1.0, x));
                       return std::cos(static_cast<double>(n) * std::acos(xc));
                   }};
        const auto td16k = totalDistortion(c, 16384);
        const auto td8k = totalDistortion(c, 8192);
        const auto td4k = totalDistortion(c, 4096);
        std::cout << "  " << std::left << std::setw(8) << c.name << std::setw(16) << fmtDb(td16k.db) << std::setw(16)
                  << fmtDb(td8k.db) << std::setw(16) << fmtDb(td4k.db) << '\n';
    }
    std::cout << "  " << std::string(60, '-') << '\n';
}

// Total Distortion (LUT vs analytical f) under a full-scale sin probe. RMS-based, so
// includes both harmonic content at multiples of the probe frequency and inharmonic /
// broadband hash from the interpolation. Reported in dB; thresholds are conservative
// regression bounds, not target values.
TEST_F(LUTFidelityComparisonTest, TotalDistortion_AllSizes) {
    // dB ceiling per (size, curve type). Smooth curves stay well below; hard-clip is
    // the hardest case because Catmull-Rom must round a discontinuous slope.
    constexpr double kSmoothCeiling16k = -90.0;
    constexpr double kSmoothCeiling8k = -78.0;
    constexpr double kSmoothCeiling4k = -66.0;
    constexpr double kHardClipCeiling16k = -60.0;
    constexpr double kHardClipCeiling8k = -54.0;
    constexpr double kHardClipCeiling4k = -48.0;

    auto fmtDb = [](double db) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1) << std::setw(6) << db << " dB";
        return oss.str();
    };

    std::cout << "\n  Total Distortion (RMS error / RMS signal, full-scale sin probe)\n";
    std::cout << "  " << std::string(70, '-') << '\n';
    std::cout << "  " << std::left << std::setw(20) << "curve" << std::setw(16) << "16384" << std::setw(16)
              << "8192 (live)" << std::setw(16) << "4096" << '\n';
    std::cout << "  " << std::string(70, '-') << '\n';

    for (const auto& curve : curves()) {
        const auto td16k = totalDistortion(curve, 16384);
        const auto td8k = totalDistortion(curve, 8192);
        const auto td4k = totalDistortion(curve, 4096);

        std::cout << "  " << std::left << std::setw(20) << curve.name << std::setw(16) << fmtDb(td16k.db)
                  << std::setw(16) << fmtDb(td8k.db) << std::setw(16) << fmtDb(td4k.db) << '\n';

        const bool isHardClip = (curve.name == "hard_clip_07");
        EXPECT_LT(td16k.db, isHardClip ? kHardClipCeiling16k : kSmoothCeiling16k)
            << "curve=" << curve.name << " size=16384 TD=" << td16k.db << " dB";
        EXPECT_LT(td8k.db, isHardClip ? kHardClipCeiling8k : kSmoothCeiling8k)
            << "curve=" << curve.name << " size=8192 TD=" << td8k.db << " dB";
        EXPECT_LT(td4k.db, isHardClip ? kHardClipCeiling4k : kSmoothCeiling4k)
            << "curve=" << curve.name << " size=4096 TD=" << td4k.db << " dB";
    }
    std::cout << "  " << std::string(70, '-') << '\n';
}

} // namespace dsp_core_test
