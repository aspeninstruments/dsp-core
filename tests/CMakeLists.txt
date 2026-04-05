# DSP Core - Unit Tests

# Enable testing
enable_testing()

# Create test executable
add_executable(layered_transfer_function_tests
    LayeredTransferFunctionTest.cpp
)

# Sets the necessary include directories
target_include_directories(layered_transfer_function_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Link against dsp_core module and gtest_main
target_link_libraries(layered_transfer_function_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

# Adds googletest-specific CMake commands
include(GoogleTest)

# Add all tests defined with googletest to the CMake metadata
# Note: gtest_discover_tests may not work in all build configurations (e.g., CPM external projects)
# so we also add a simple test target that can be run directly
if(CMAKE_GENERATOR STREQUAL Xcode)
  # On macOS arm64, all binaries have to be signed before running
  gtest_discover_tests(layered_transfer_function_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(layered_transfer_function_tests)
endif()

# Also add a simple test that can always be run
add_test(NAME LayeredTransferFunctionTests COMMAND layered_transfer_function_tests)

# LaneMixer tests
add_executable(lane_mixer_tests
    LaneMixerTest.cpp
)

target_include_directories(lane_mixer_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(lane_mixer_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(lane_mixer_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(lane_mixer_tests)
endif()

add_test(NAME LaneMixerTests COMMAND lane_mixer_tests)

# DryWetMixStage tests
add_executable(dry_wet_mix_stage_tests
    DryWetMixStageTest.cpp
)

target_include_directories(dry_wet_mix_stage_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(dry_wet_mix_stage_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(dry_wet_mix_stage_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(dry_wet_mix_stage_tests)
endif()

add_test(NAME DryWetMixStageTests COMMAND dry_wet_mix_stage_tests)

# SplineFitter tests
add_executable(spline_fitter_tests
    SplineFitterTest.cpp
)

target_include_directories(spline_fitter_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(spline_fitter_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(spline_fitter_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(spline_fitter_tests)
endif()

add_test(NAME SplineFitterTests COMMAND spline_fitter_tests)

# SplineEvaluator tests
add_executable(spline_evaluator_tests
    SplineEvaluatorTest.cpp
)

target_include_directories(spline_evaluator_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(spline_evaluator_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(spline_evaluator_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(spline_evaluator_tests)
endif()

add_test(NAME SplineEvaluatorTests COMMAND spline_evaluator_tests)

# DCBlockingFilter tests
add_executable(dc_blocking_filter_tests
    DCBlockingFilterTest.cpp
)

target_include_directories(dc_blocking_filter_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(dc_blocking_filter_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(dc_blocking_filter_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(dc_blocking_filter_tests)
endif()

add_test(NAME DCBlockingFilterTests COMMAND dc_blocking_filter_tests)

# CurveFeatureDetector tests
add_executable(curve_feature_detector_tests
    CurveFeatureDetectorTest.cpp
)

target_include_directories(curve_feature_detector_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(curve_feature_detector_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(curve_feature_detector_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(curve_feature_detector_tests)
endif()

add_test(NAME CurveFeatureDetectorTests COMMAND curve_feature_detector_tests)

# SplineLayer tests
add_executable(spline_layer_tests
    SplineLayerTest.cpp
)

target_include_directories(spline_layer_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(spline_layer_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(spline_layer_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(spline_layer_tests)
endif()

add_test(NAME SplineLayerTests COMMAND spline_layer_tests)

# LayeredTransferFunctionSpline tests
add_executable(layered_transfer_function_spline_tests
    LayeredTransferFunctionSplineTest.cpp
)

target_include_directories(layered_transfer_function_spline_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(layered_transfer_function_spline_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(layered_transfer_function_spline_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(layered_transfer_function_spline_tests)
endif()

add_test(NAME LayeredTransferFunctionSplineTests COMMAND layered_transfer_function_spline_tests)

# Algorithm Benchmark tests
add_executable(algorithm_benchmark_tests
    AlgorithmBenchmarkTest.cpp
)

target_include_directories(algorithm_benchmark_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(algorithm_benchmark_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(algorithm_benchmark_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(algorithm_benchmark_tests)
endif()

add_test(NAME AlgorithmBenchmarkTests COMMAND algorithm_benchmark_tests)

# CoordinateSnapper tests
add_executable(coordinate_snapper_tests
    CoordinateSnapperTest.cpp
)

target_include_directories(coordinate_snapper_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(coordinate_snapper_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(coordinate_snapper_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(coordinate_snapper_tests)
endif()

add_test(NAME CoordinateSnapperTests COMMAND coordinate_snapper_tests)

# AdaptiveTolerance tests
add_executable(adaptive_tolerance_tests
    AdaptiveToleranceTest.cpp
)

target_include_directories(adaptive_tolerance_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(adaptive_tolerance_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(adaptive_tolerance_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(adaptive_tolerance_tests)
endif()

add_test(NAME AdaptiveToleranceTests COMMAND adaptive_tolerance_tests)

# SymmetryAnalyzer tests
add_executable(symmetry_analyzer_tests
    SymmetryAnalyzerTest.cpp
)

target_include_directories(symmetry_analyzer_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(symmetry_analyzer_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(symmetry_analyzer_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(symmetry_analyzer_tests)
endif()

add_test(NAME SymmetryAnalyzerTests COMMAND symmetry_analyzer_tests)

# SplineFitterIntegration tests
add_executable(spline_fitter_integration_tests
    SplineFitterIntegrationTest.cpp
)

target_include_directories(spline_fitter_integration_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(spline_fitter_integration_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(spline_fitter_integration_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(spline_fitter_integration_tests)
endif()

add_test(NAME SplineFitterIntegrationTests COMMAND spline_fitter_integration_tests)

# SplineFitterExactExtrema tests
add_executable(spline_fitter_exact_extrema_tests
    SplineFitterExactExtremaTest.cpp
)

target_include_directories(spline_fitter_exact_extrema_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(spline_fitter_exact_extrema_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(spline_fitter_exact_extrema_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(spline_fitter_exact_extrema_tests)
endif()

add_test(NAME SplineFitterExactExtremaTests COMMAND spline_fitter_exact_extrema_tests)

# ExtremaQualityMetrics tests (for parameter tuning)
add_executable(extrema_quality_metrics_tests
    ExtremaQualityMetricsTest.cpp
    AnchorEfficiencyMetricsTest.cpp
)

target_include_directories(extrema_quality_metrics_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(extrema_quality_metrics_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(extrema_quality_metrics_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(extrema_quality_metrics_tests)
endif()

add_test(NAME ExtremaQualityMetricsTests COMMAND extrema_quality_metrics_tests)

# FeatureDetectionDiagnostic tests (for root cause analysis)
add_executable(feature_detection_diagnostic_tests
    FeatureDetectionDiagnosticTest.cpp
)

target_include_directories(feature_detection_diagnostic_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(feature_detection_diagnostic_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(feature_detection_diagnostic_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(feature_detection_diagnostic_tests)
endif()

add_test(NAME FeatureDetectionDiagnosticTests COMMAND feature_detection_diagnostic_tests)

# HarmonicAnchorDistribution tests (for debugging anchor clustering)
add_executable(harmonic_anchor_distribution_tests
    HarmonicAnchorDistributionTest.cpp
)

target_include_directories(harmonic_anchor_distribution_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(harmonic_anchor_distribution_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(harmonic_anchor_distribution_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(harmonic_anchor_distribution_tests)
endif()

add_test(NAME HarmonicAnchorDistributionTests COMMAND harmonic_anchor_distribution_tests)

# FeatureDetectorDebug tests (for debugging extrema detection failure)
add_executable(feature_detector_debug_tests
    FeatureDetectorDebugTest.cpp
)

target_include_directories(feature_detector_debug_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(feature_detector_debug_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(feature_detector_debug_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(feature_detector_debug_tests)
endif()

add_test(NAME FeatureDetectorDebugTests COMMAND feature_detector_debug_tests)

# AnchorClusteringAnalysis tests (for validating symmetry hypothesis)
add_executable(anchor_clustering_analysis_tests
    AnchorClusteringAnalysisTest.cpp
)

target_include_directories(anchor_clustering_analysis_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(anchor_clustering_analysis_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(anchor_clustering_analysis_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(anchor_clustering_analysis_tests)
endif()

add_test(NAME AnchorClusteringAnalysisTests COMMAND anchor_clustering_analysis_tests)

# TransferFunctionOperations tests
add_executable(transfer_function_operations_tests
    TransferFunctionOperationsTest.cpp
)

target_include_directories(transfer_function_operations_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(transfer_function_operations_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(transfer_function_operations_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(transfer_function_operations_tests)
endif()

add_test(NAME TransferFunctionOperationsTests COMMAND transfer_function_operations_tests)

# SeamlessTransferFunction characterization tests
add_executable(seamless_characterization_tests
    SeamlessTransferFunctionCharacterizationTest.cpp
)

target_include_directories(seamless_characterization_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(seamless_characterization_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(seamless_characterization_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(seamless_characterization_tests)
endif()

add_test(NAME SeamlessCharacterizationTests COMMAND seamless_characterization_tests)

# AudioEngine tests
add_executable(audio_engine_tests
    AudioEngineTest.cpp
)

target_include_directories(audio_engine_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(audio_engine_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(audio_engine_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(audio_engine_tests)
endif()

add_test(NAME AudioEngineTests COMMAND audio_engine_tests)

# SeamlessTransferFunction performance tests (Task 8)
add_executable(seamless_transfer_function_performance_tests
    SeamlessTransferFunctionPerformanceTest.cpp
)

target_include_directories(seamless_transfer_function_performance_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(seamless_transfer_function_performance_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(seamless_transfer_function_performance_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(seamless_transfer_function_performance_tests)
endif()

add_test(NAME SeamlessTransferFunctionPerformanceTests COMMAND seamless_transfer_function_performance_tests)

# SymmetryDetectionComparison tests (experimental symmetry feature evaluation)
add_executable(symmetry_detection_comparison_tests
    SymmetryDetectionComparisonTest.cpp
)

target_include_directories(symmetry_detection_comparison_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(symmetry_detection_comparison_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(symmetry_detection_comparison_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(symmetry_detection_comparison_tests)
endif()

add_test(NAME SymmetryDetectionComparisonTests COMMAND symmetry_detection_comparison_tests)

# AudioPipelineBuilder tests
add_executable(audio_pipeline_builder_tests
    AudioPipelineBuilderTest.cpp
)

target_include_directories(audio_pipeline_builder_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(audio_pipeline_builder_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(audio_pipeline_builder_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(audio_pipeline_builder_tests)
endif()

add_test(NAME AudioPipelineBuilderTests COMMAND audio_pipeline_builder_tests)

# StereoHistoryBuffer tests
add_executable(stereo_history_buffer_tests
    StereoHistoryBufferTest.cpp
)

target_include_directories(stereo_history_buffer_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(stereo_history_buffer_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(stereo_history_buffer_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(stereo_history_buffer_tests)
endif()

add_test(NAME StereoHistoryBufferTests COMMAND stereo_history_buffer_tests)

# SoftClippingComparison tests (harmonic analysis of hard vs soft clipping)
add_executable(soft_clipping_comparison_tests
    SoftClippingComparisonTest.cpp
)

target_include_directories(soft_clipping_comparison_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(soft_clipping_comparison_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(soft_clipping_comparison_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(soft_clipping_comparison_tests)
endif()

add_test(NAME SoftClippingComparisonTests COMMAND soft_clipping_comparison_tests)

# HysteresisProcessor tests
add_executable(hysteresis_processor_tests
    HysteresisProcessorTest.cpp
)

target_include_directories(hysteresis_processor_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(hysteresis_processor_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(hysteresis_processor_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(hysteresis_processor_tests)
endif()

add_test(NAME HysteresisProcessorTests COMMAND hysteresis_processor_tests)

# EventDrivenRenderer tests
add_executable(event_driven_renderer_tests
    EventDrivenRendererTest.cpp
)

target_include_directories(event_driven_renderer_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(event_driven_renderer_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(event_driven_renderer_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(event_driven_renderer_tests)
endif()

add_test(NAME EventDrivenRendererTests COMMAND event_driven_renderer_tests)

# HysteresisStage tests
add_executable(hysteresis_stage_tests
    HysteresisStageTest.cpp
)

target_include_directories(hysteresis_stage_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(hysteresis_stage_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(hysteresis_stage_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(hysteresis_stage_tests)
endif()

add_test(NAME HysteresisStageTests COMMAND hysteresis_stage_tests)

