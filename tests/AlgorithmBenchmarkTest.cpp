#include <dsp_core/dsp_core.h>
#include <gtest/gtest.h>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

namespace dsp_core_test {

/**
 * Performance benchmark comparing Akima vs Fritsch-Carlson tangent algorithms
 * Tests typical waveshaping curves to inform default algorithm selection
 */
class AlgorithmBenchmark : public ::testing::Test {
  protected:
    void SetUp() override {
        // Use production table size (16384) for realistic benchmarks
        ltf = std::make_unique<dsp_core::LayeredTransferFunction>(16384, -1.0, 1.0);
    }

    struct BenchmarkResult {
        std::string algorithmName;
        std::string curveName;
        double fittingTimeMs = 0.0;
        int numAnchors = 0;
        double maxError = 0.0;
        double avgError = 0.0;
        int spuriousExtrema = 0;

        void print() const {
            std::cout << std::left << std::setw(20) << curveName << std::setw(18) << algorithmName << std::setw(10)
                      << std::fixed << std::setprecision(3) << fittingTimeMs << " ms" << std::setw(12) << numAnchors
                      << std::setw(15) << std::scientific << std::setprecision(4) << maxError << std::setw(15)
                      << avgError << std::setw(10) << spuriousExtrema << '\n';
        }
    };

    BenchmarkResult benchmarkAlgorithm(dsp_core::TangentAlgorithm algo, const std::string& algoName,
                                       const std::string& curveName) {

        BenchmarkResult result;
        result.algorithmName = algoName;
        result.curveName = curveName;

        // Configure fitting (use smooth preset as baseline)
        dsp_core::SplineFitConfig config;
        config.positionTolerance = 0.01;
        config.maxAnchors = 64; // Generous limit for fair comparison
        config.tangentAlgorithm = algo;

        // Extract curve data for fitting
        auto curveData = extractCurveData();

        // Time the fitting operation
        auto start = std::chrono::high_resolution_clock::now();
        auto fitResult =
            dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                       ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> const duration = end - start;
        result.fittingTimeMs = duration.count();

        // Extract quality metrics
        result.numAnchors = fitResult.numAnchors;
        result.maxError = fitResult.maxError;
        result.avgError = fitResult.averageError;

        // Count spurious extrema
        result.spuriousExtrema = countSpuriousExtrema(fitResult.anchors);

        return result;
    }

    // Simplified extrema count - just count sign changes in fitted spline derivative
    // This is a proxy metric (not perfect, but good enough for comparison)
    int countSpuriousExtrema(const std::vector<dsp_core::SplineAnchor>& anchors) {
        if (anchors.size() < 3) {
            return 0;
        }

        const int NUM_SAMPLES = 1000;
        int extremaCount = 0;
        double prevY = ltf->getSplineLayer().evaluate(-1.0);
        double prevSlope = 0.0;
        bool firstSlope = true;

        for (int i = 1; i < NUM_SAMPLES; ++i) {
            double const x = -1.0 + (2.0 * i) / (NUM_SAMPLES - 1);
            double const y = ltf->getSplineLayer().evaluate(x);
            double const slope = y - prevY;

            if (!firstSlope && prevSlope * slope < 0.0) {
                extremaCount++;
            }

            prevY = y;
            prevSlope = slope;
            firstSlope = false;
        }

        return extremaCount;
    }

    void setupCurve(std::function<double(double)> func) {
        for (int i = 0; i < ltf->getTableSize(); ++i) {
            double const x = ltf->normalizeIndex(i);
            double const y = func(x);
            ltf->setBaseLayerValue(i, y);
        }
    }

    std::vector<double> extractCurveData() const {
        std::vector<double> data(ltf->getTableSize());
        for (int i = 0; i < ltf->getTableSize(); ++i) {
            data[i] = ltf->getBaseLayerValue(i);
        }
        return data;
    }

    std::unique_ptr<dsp_core::LayeredTransferFunction> ltf;
};

/**
 * Benchmark Suite: Test all common waveshaping curves
 */
TEST_F(AlgorithmBenchmark, ComprehensiveComparison) {
    std::cout << "\n" << std::string(110, '=') << "\n";
    std::cout << "  ALGORITHM PERFORMANCE BENCHMARK (Production Table Size: 16384)\n";
    std::cout << std::string(110, '=') << "\n\n";

    std::cout << std::left << std::setw(20) << "Curve Type" << std::setw(18) << "Algorithm" << std::setw(10) << "Time"
              << std::setw(12) << "Anchors" << std::setw(15) << "Max Error" << std::setw(15) << "Avg Error"
              << std::setw(10) << "Extrema" << '\n';
    std::cout << std::string(110, '-') << '\n';

    struct TestCurve {
        std::string name;
        std::function<double(double)> func;
    };

    std::vector<TestCurve> const curves = {
        // 1. Soft Saturation (most common in music production)
        {"Soft Saturation", [](double x) { return std::tanh(3.0 * x); }},

        // 2. Gentle Saturation
        {"Gentle Saturation", [](double x) { return std::tanh(1.5 * x); }},

        // 3. Hard Clipping (asymptotic)
        {"Hard Clipping", [](double x) { return std::tanh(10.0 * x); }},

        // 4. Asymmetric Distortion (even harmonics)
        {"Asymmetric Dist", [](double x) { return std::tanh(3.0 * x) + 0.2 * x * x; }},

        // 5. Cubic Soft Clip
        {"Cubic Soft Clip",
         [](double x) {
             if (x < -1.0) {
                 return -2.0 / 3.0;
             }
             if (x > 1.0) {
                 return 2.0 / 3.0;
             }
             return x - (x * x * x) / 3.0;
         }},

        // 6. S-Curve (tube-like)
        {"Tube S-Curve", [](double x) { return x * x * x * 0.7 + x * 0.3; }},

        // 7. Gentle Wavefolder
        {"Gentle Wavefolder", [](double x) { return std::sin(1.5 * M_PI * x); }},

        // 8. Complex Transfer Function (multiple features)
        {"Complex Multi", [](double x) { return std::tanh(2.0 * x) * (1.0 + 0.3 * std::sin(3.0 * M_PI * x)); }}};

    std::vector<BenchmarkResult> allResults;

    for (const auto& curve : curves) {
        setupCurve(curve.func);

        // Benchmark Akima
        auto akimaResult = benchmarkAlgorithm(dsp_core::TangentAlgorithm::Akima, "Akima", curve.name);
        akimaResult.print();
        allResults.push_back(akimaResult);

        // Benchmark Fritsch-Carlson
        auto fcResult = benchmarkAlgorithm(dsp_core::TangentAlgorithm::FritschCarlson, "Fritsch-Carlson", curve.name);
        fcResult.print();
        allResults.push_back(fcResult);

        std::cout << std::string(110, '-') << '\n';
    }

    // Summary statistics
    std::cout << "\n" << std::string(110, '=') << "\n";
    std::cout << "  SUMMARY STATISTICS\n";
    std::cout << std::string(110, '=') << "\n\n";

    auto computeStats = [&allResults](dsp_core::TangentAlgorithm algo, const std::string& name) {
        std::vector<double> times;
        std::vector<int> anchors;
        std::vector<double> maxErrors;
        std::vector<int> ripples;

        for (const auto& r : allResults) {
            if ((algo == dsp_core::TangentAlgorithm::Akima && r.algorithmName == "Akima") ||
                (algo == dsp_core::TangentAlgorithm::FritschCarlson && r.algorithmName == "Fritsch-Carlson")) {
                times.push_back(r.fittingTimeMs);
                anchors.push_back(r.numAnchors);
                maxErrors.push_back(r.maxError);
                ripples.push_back(r.spuriousExtrema);
            }
        }

        auto avg = [](const std::vector<double>& v) {
            return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
        };
        auto avgInt = [](const std::vector<int>& v) {
            return std::accumulate(v.begin(), v.end(), 0) / static_cast<double>(v.size());
        };

        std::cout << name << ":\n";
        std::cout << "  Avg Fitting Time:  " << std::fixed << std::setprecision(3) << avg(times) << " ms\n";
        std::cout << "  Avg Anchors:       " << std::fixed << std::setprecision(1) << avgInt(anchors) << "\n";
        std::cout << "  Avg Max Error:     " << std::scientific << std::setprecision(4) << avg(maxErrors) << "\n";
        std::cout << "  Avg Ripples:       " << std::fixed << std::setprecision(1) << avgInt(ripples) << "\n";
        std::cout << "  Max Ripples:       " << *std::max_element(ripples.begin(), ripples.end()) << "\n\n";
    };

    computeStats(dsp_core::TangentAlgorithm::Akima, "Akima");
    computeStats(dsp_core::TangentAlgorithm::FritschCarlson, "Fritsch-Carlson");

    std::cout << "\n" << std::string(110, '=') << "\n";
    std::cout << "  CONCLUSION\n";
    std::cout << std::string(110, '=') << "\n";
    std::cout
        << "\nBenchmark complete. Review timing, anchor count, and extrema metrics to inform default algorithm.\n\n";
}

/**
 * Stress Test: Very steep curves (hard clipping)
 */
TEST_F(AlgorithmBenchmark, SteepCurveStressTest) {
    std::cout << "\n" << std::string(100, '=') << "\n";
    std::cout << "  STRESS TEST: Extremely Steep Curves (tanh(Nx))\n";
    std::cout << std::string(100, '=') << "\n\n";

    std::cout << std::left << std::setw(20) << "Steepness (N)" << std::setw(18) << "Algorithm" << std::setw(10)
              << "Time" << std::setw(12) << "Anchors" << std::setw(15) << "Max Error" << std::setw(10) << "Extrema"
              << '\n';
    std::cout << std::string(100, '-') << '\n';

    std::vector<double> const steepness = {1.0, 3.0, 5.0, 10.0, 20.0, 50.0};

    for (double const N : steepness) {
        setupCurve([N](double x) { return std::tanh(N * x); });

        auto akima = benchmarkAlgorithm(dsp_core::TangentAlgorithm::Akima, "Akima",
                                        "tanh(" + std::to_string(static_cast<int>(N)) + "x)");

        auto fc = benchmarkAlgorithm(dsp_core::TangentAlgorithm::FritschCarlson, "Fritsch-Carlson",
                                     "tanh(" + std::to_string(static_cast<int>(N)) + "x)");

        std::cout << std::left << std::setw(20) << ("tanh(" + std::to_string(static_cast<int>(N)) + "x)")
                  << std::setw(18) << akima.algorithmName << std::setw(10) << std::fixed << std::setprecision(3)
                  << akima.fittingTimeMs << " ms" << std::setw(12) << akima.numAnchors << std::setw(15)
                  << std::scientific << std::setprecision(4) << akima.maxError << std::setw(10) << akima.spuriousExtrema
                  << '\n';

        std::cout << std::left << std::setw(20) << "" << std::setw(18) << fc.algorithmName << std::setw(10)
                  << std::fixed << std::setprecision(3) << fc.fittingTimeMs << " ms" << std::setw(12) << fc.numAnchors
                  << std::setw(15) << std::scientific << std::setprecision(4) << fc.maxError << std::setw(10)
                  << fc.spuriousExtrema << '\n';
        std::cout << std::string(100, '-') << '\n';
    }

    std::cout << "\nStress test complete.\n\n";
}

/**
 * Memory/Cache Test: Repeated fitting operations
 */
TEST_F(AlgorithmBenchmark, RepeatedFittingTest) {
    std::cout << "\n" << std::string(100, '=') << "\n";
    std::cout << "  REPEATED FITTING TEST: UI Responsiveness Simulation\n";
    std::cout << "  (100 fits on typical soft saturation curve)\n";
    std::cout << std::string(100, '=') << "\n\n";

    setupCurve([](double x) { return std::tanh(3.0 * x); });

    const int NUM_ITERATIONS = 100;

    auto curveData = extractCurveData();

    // Warm up
    for (int i = 0; i < 10; ++i) {
        dsp_core::SplineFitConfig config;
        config.tangentAlgorithm = dsp_core::TangentAlgorithm::Akima;
        dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                   ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);
    }

    // Benchmark Akima
    auto akimaStart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        dsp_core::SplineFitConfig config;
        config.tangentAlgorithm = dsp_core::TangentAlgorithm::Akima;
        dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                   ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);
    }
    auto akimaEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> const akimaDuration = akimaEnd - akimaStart;

    // Benchmark Fritsch-Carlson
    auto fcStart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        dsp_core::SplineFitConfig config;
        config.tangentAlgorithm = dsp_core::TangentAlgorithm::FritschCarlson;
        dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                   ltf->getMinSignalValue(), ltf->getMaxSignalValue(), config);
    }
    auto fcEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> const fcDuration = fcEnd - fcStart;

    std::cout << "Akima:            " << std::fixed << std::setprecision(2) << akimaDuration.count() << " ms total, "
              << akimaDuration.count() / NUM_ITERATIONS << " ms per fit\n";
    std::cout << "Fritsch-Carlson:  " << std::fixed << std::setprecision(2) << fcDuration.count() << " ms total, "
              << fcDuration.count() / NUM_ITERATIONS << " ms per fit\n";
    std::cout << "\nSpeedup factor:   " << std::fixed << std::setprecision(2)
              << fcDuration.count() / akimaDuration.count() << "x\n";
    std::cout << "\nUI Target: <16ms per frame (60fps) - ";

    if (akimaDuration.count() / NUM_ITERATIONS < 16.0) {
        std::cout << "✓ Akima meets target\n";
    } else {
        std::cout << "✗ Akima exceeds target\n";
    }

    if (fcDuration.count() / NUM_ITERATIONS < 16.0) {
        std::cout << "                                   ✓ Fritsch-Carlson meets target\n\n";
    } else {
        std::cout << "                                   ✗ Fritsch-Carlson exceeds target\n\n";
    }
}

} // namespace dsp_core_test
