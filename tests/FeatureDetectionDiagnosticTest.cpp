#include <dsp_core/dsp_core.h>
#include <gtest/gtest.h>
#include <iostream>
#include <iomanip>

namespace dsp_core_test {

/**
 * Diagnostic test for feature detection
 *
 * This test reproduces the ArbitraryPositions scenario and reports
 * exactly what features are detected and where anchors are placed.
 */
TEST(FeatureDetectionDiagnostic, ArbitraryPositions_FeatureAnalysis) {
    std::cout << "\n=== Feature Detection Diagnostic: ArbitraryPositions ===" << '\n';

    // Step 1: Create the exact same 4-anchor curve from ArbitraryPositions test
    std::vector<dsp_core::SplineAnchor> originalAnchors = {
        {-1.0, -0.87, false, 0.0}, {-0.412, -0.653, false, 0.0}, {0.234, 0.123, false, 0.0}, {1.0, 0.92, false, 0.0}};

    auto config = dsp_core::SplineFitConfig::tight();
    dsp_core::Services::SplineFitter::computeTangents(originalAnchors, config);

    std::cout << "\nOriginal anchors (4):" << '\n';
    for (size_t i = 0; i < originalAnchors.size(); ++i) {
        std::cout << "  [" << i << "] x=" << std::setw(8) << std::fixed << std::setprecision(4) << originalAnchors[i].x
                  << ", y=" << std::setw(8) << originalAnchors[i].y << ", m=" << std::setw(8)
                  << originalAnchors[i].tangent << '\n';
    }

    // Step 2: Render to LayeredTransferFunction (like backtranslation does)
    auto ltf = std::make_unique<dsp_core::LayeredTransferFunction>(16384, -1.0, 1.0);

    // Render spline to base layer
    for (int i = 0; i < ltf->getTableSize(); ++i) {
        double const x = ltf->normalizeIndex(i);
        double const y = dsp_core::Services::SplineEvaluator::evaluate(originalAnchors, x);
        ltf->setBaseLayerValue(i, y);
    }

    std::cout << "\nRendered to LTF with " << ltf->getTableSize() << " samples" << '\n';

    // Extract curve data for service calls
    std::vector<double> curveData(ltf->getTableSize());
    for (int i = 0; i < ltf->getTableSize(); ++i) {
        curveData[i] = ltf->getBaseLayerValue(i);
    }

    // Step 3: Run feature detection with various configurations
    std::cout << "\n--- Feature Detection Analysis ---" << '\n';

    // Test A: Zero threshold (detect everything)
    {
        dsp_core::FeatureDetectionConfig featureConfig;
        featureConfig.significanceThreshold = 0.0; // Detect ALL features
        featureConfig.derivativeThreshold = 1e-6;

        auto features = dsp_core::Services::CurveFeatureDetector::detectFeatures(
            curveData.data(), static_cast<int>(curveData.size()), ltf->getMinSignalValue(), ltf->getMaxSignalValue(),
            featureConfig);

        std::cout << "\nConfig: significanceThreshold=0.0 (detect all)" << '\n';
        std::cout << "  Local extrema: " << features.localExtrema.size() << '\n';
        for (size_t i = 0; i < features.localExtrema.size(); ++i) {
            int const idx = features.localExtrema[i];
            double const x = ltf->normalizeIndex(idx);
            double const y = ltf->evaluateBaseAndHarmonics(x);
            std::cout << "    [" << i << "] idx=" << idx << ", x=" << std::setprecision(6) << x << ", y=" << y << '\n';
        }

        std::cout << "  Total mandatory anchors: " << features.mandatoryAnchors.size() << '\n';
    }

    // Test B: Production threshold (0.001)
    {
        dsp_core::FeatureDetectionConfig featureConfig;
        featureConfig.significanceThreshold = 0.001;
        featureConfig.derivativeThreshold = 1e-6;

        auto features = dsp_core::Services::CurveFeatureDetector::detectFeatures(
            curveData.data(), static_cast<int>(curveData.size()), ltf->getMinSignalValue(), ltf->getMaxSignalValue(),
            featureConfig);

        std::cout << "\nConfig: significanceThreshold=0.001 (production)" << '\n';
        std::cout << "  Local extrema: " << features.localExtrema.size() << '\n';
        std::cout << "  Total mandatory anchors: " << features.mandatoryAnchors.size() << '\n';
    }

    // Test C: Higher derivative threshold
    {
        dsp_core::FeatureDetectionConfig featureConfig;
        featureConfig.significanceThreshold = 0.001;
        featureConfig.derivativeThreshold = 1e-5; // 10x higher

        auto features = dsp_core::Services::CurveFeatureDetector::detectFeatures(
            curveData.data(), static_cast<int>(curveData.size()), ltf->getMinSignalValue(), ltf->getMaxSignalValue(),
            featureConfig);

        std::cout << "\nConfig: Higher derivative threshold (1e-5)" << '\n';
        std::cout << "  Local extrema: " << features.localExtrema.size() << '\n';
        std::cout << "  Total mandatory anchors: " << features.mandatoryAnchors.size() << '\n';
    }

    // Step 4: Refit with feature detection enabled
    std::cout << "\n--- Spline Refitting ---" << '\n';

    {
        auto fitConfig = dsp_core::SplineFitConfig::tight();
        fitConfig.enableFeatureDetection = true;
        fitConfig.featureConfig.significanceThreshold = 0.001;
        fitConfig.featureConfig.derivativeThreshold = 1e-6;

        auto result =
            dsp_core::Services::SplineFitter::fitCurve(curveData.data(), static_cast<int>(curveData.size()),
                                                       ltf->getMinSignalValue(), ltf->getMaxSignalValue(), fitConfig);

        std::cout << "\nRefit with feature detection: " << result.anchors.size() << " anchors" << '\n';
        for (size_t i = 0; i < result.anchors.size(); ++i) {
            std::cout << "  [" << i << "] x=" << std::setw(8) << std::fixed << std::setprecision(4)
                      << result.anchors[i].x << ", y=" << std::setw(8) << result.anchors[i].y << '\n';
        }
    }

    std::cout << "\n=== Analysis Complete ===" << '\n';
}

} // namespace dsp_core_test
