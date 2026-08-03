# DSP Core - Unit Tests

# Enable testing
enable_testing()

# MSVC's <cmath> only exposes M_PI / M_E / etc. when _USE_MATH_DEFINES is
# defined before <cmath>/<math.h> is included. libc++ (macOS) and libstdc++
# (Linux) expose them unconditionally, which masks the gap until Windows CI
# runs. Apply at directory scope so every test executable below picks it up.
add_compile_definitions(_USE_MATH_DEFINES)

# JUCE's juce_gui_extra.cpp includes <gtk/gtk.h> on Linux only when
# JUCE_WEB_BROWSER is enabled (default). dsp-core tests don't need WebKit,
# but they transitively compile juce_gui_extra via juce_audio_processors.
# Mirror what ui-core/tfe tests already set per-target so the GTK include
# is compiled out and Linux CI doesn't need libgtk-3-dev.
add_compile_definitions(JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0)

include(GoogleTest)

# Single consolidated test runner. Previously this file built one executable
# per test .cpp (56 of them), and because dsp_core is a juce_add_module
# INTERFACE library, EACH executable recompiled the full JUCE surface
# (juce_core/dsp/audio_processors/...) from scratch — 56x duplicated JUCE
# compiles and 56 link steps. Bundling every test source into one binary
# compiles that JUCE surface ONCE (a cache-independent win on all platforms),
# matching the AudioPluginTest / transfer_function_editor test pattern.
add_executable(dsp_core_tests
    LayeredTransferFunctionTest.cpp
    ExpressionEvaluatorTest.cpp
    ExpressionParserTest.cpp
    AnchorConstraintServiceTest.cpp
    LaneMixerTest.cpp
    DryWetMixStageTest.cpp
    SplineFitterTest.cpp
    SplineEvaluatorTest.cpp
    DCBlockingFilterTest.cpp
    LowpassStageTest.cpp
    LowShelfStageTest.cpp
    FirstOrderShelfStageTest.cpp
    HighShelfStageTest.cpp
    BellStageTest.cpp
    ImpulseResponseStrategyTest.cpp
    CurveFeatureDetectorTest.cpp
    SplineLayerTest.cpp
    LayeredTransferFunctionSplineTest.cpp
    AlgorithmBenchmarkTest.cpp
    CoordinateSnapperTest.cpp
    AdaptiveToleranceTest.cpp
    SymmetryAnalyzerTest.cpp
    SplineFitterIntegrationTest.cpp
    SplineFitterExactExtremaTest.cpp
    ExtremaQualityMetricsTest.cpp
    AnchorEfficiencyMetricsTest.cpp
    FeatureDetectionDiagnosticTest.cpp
    HarmonicAnchorDistributionTest.cpp
    FeatureDetectorDebugTest.cpp
    AnchorClusteringAnalysisTest.cpp
    TransferFunctionOperationsTest.cpp
    SeamlessTransferFunctionCharacterizationTest.cpp
    AudioEngineTest.cpp
    LUTFidelityComparisonTest.cpp
    SymmetryDetectionComparisonTest.cpp
    AudioPipelineBuilderTest.cpp
    StereoHistoryBufferTest.cpp
    SoftClippingComparisonTest.cpp
    HysteresisProcessorTest.cpp
    EventDrivenRendererTest.cpp
    MorphGestureTest.cpp
    LfoStageTest.cpp
    HysteresisStageTest.cpp
    SurgeStageTest.cpp
    BiasStageTest.cpp
    SignalRangeTapTest.cpp
    SlotVisualizerPublisherTest.cpp
    EnvelopeFollowerStageTest.cpp
    AutoGainStageTest.cpp
    ToneStageTest.cpp
    FatStageTest.cpp
    LadderTPTStageTest.cpp
    LadderHighpassTPTStageTest.cpp
    VirtualAnalogFilterStageTest.cpp
    FilterTopologyBenchmarkTest.cpp
    OversamplingWrapperTest.cpp
    NonlinearLadderStageTest.cpp
)

target_include_directories(dsp_core_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(dsp_core_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

# Register every gtest case in the bundle with ctest.
if(CMAKE_GENERATOR STREQUAL Xcode)
  # On macOS arm64, all binaries have to be signed before running.
  gtest_discover_tests(dsp_core_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(dsp_core_tests)
endif()

# --- Dev-only benchmarks: built but intentionally NOT registered with ctest ---
# These stay as their own executables so their timing-gated cases don't run in
# CI (they flake on shared runners). They are a small, deliberate exception to
# the consolidation above.

# SeamlessTransferFunction performance tests (Task 8)
# Not registered with ctest. Latency_InteractiveOperations has a 60ms gate
# that's brittle on shared CI runners (poller worst-case 50ms + render +
# system jitter routinely exceeds it). Useful for local profiling only.
# Run locally with: ./build/modules/dsp-core/tests/seamless_transfer_function_performance_tests
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

# HysteresisStage throughput benchmark (code-cleanse DSP-001). Not registered
# with ctest — prints timing for local before/after comparison of the RK4
# nonlinearity dispatch. Run locally, Release build:
#   ./build-release/modules/dsp-core/tests/hysteresis_stage_benchmark
add_executable(hysteresis_stage_benchmark
    HysteresisStageBenchmark.cpp
)

target_include_directories(hysteresis_stage_benchmark
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(hysteresis_stage_benchmark
    PRIVATE
    dsp_core
    GTest::gtest_main
)
