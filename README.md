---
verified: 2026-07-13
verified_modules: { dsp-core: 45f7854 }
status: current
---

# dsp-core Module

`dsp_core` is the JUCE module holding all non-UI signal processing for Black Diamond Distortion: the multi-lane transfer-function model (`LaneMixer`), the click-free LUT delivery system (`SeamlessTransferFunction`), the Jiles-Atherton hysteresis processor, the `audio_pipeline` stage library, and the curve-fitting/synthesis Services. It has no plugin-specific dependencies and is consumed by `transfer_function_editor` and `plugin`.

## Key files

| File | Role |
|---|---|
| `dsp_core/dsp_core.h` | JUCE module header (module declaration + unity include list) |
| `dsp_core/Source/LaneMixer.h/.cpp` | Production transfer-function model (up to 100 lanes, Blend/Scan/Series) |
| `dsp_core/Source/Lane.h` | Single lane: 16384-pt curve, atomic amplitude/depths, content metadata |
| `dsp_core/Source/SeamlessTransferFunction.h/.cpp` | Public facade: LUT delivery, lifecycle, restore paths |
| `dsp_core/Source/SeamlessTransferFunctionImpl.h/.cpp` | `AudioEngine`, `EventDrivenRenderer`, `VisualizerUpdateDispatcher` |
| `dsp_core/Source/HysteresisProcessor.h/.cpp` | Jiles-Atherton magnetization model (RK4, custom nonlinearity injection) |
| `dsp_core/Source/audio_pipeline/` | Processing stages + `AudioPipeline`/`AudioPipelineBuilder` |
| `dsp_core/Source/Services/` | Pure static services (fitting, synthesis, geometry) |
| `tests/` | 55 standalone gtest executables |

## Load-bearing facts

- The production editing model is `LaneMixer`: `MAX_LANES=100`, `TABLE_SIZE=16384`, range ±1.0, `MAX_HARMONIC_NUMBER=200` (`dsp_core/Source/LaneMixer.h:50-54`).
- The audio thread never reads `LaneMixer` directly; it reads the triple-buffered LUT rendered by `EventDrivenRenderer` (`dsp_core/Source/LaneMixer.h:37-41`, `dsp_core/Source/Lane.h:41-43`).
- `LayeredTransferFunction` is legacy/test-only — zero references in `plugin/source` or `transfer_function_editor` production sources; only its serialized ValueTree format survives, via `LaneMixer::fromLegacyLTFValueTree` (see [layered-transfer-function.md](docs/layered-transfer-function.md)).
- LUT generations crossfade over 5 ms with a smoothstep curve (`SeamlessConfig::CROSSFADE_DURATION_MS`, `dsp_core/Source/SeamlessTransferFunctionImpl.h:29`).
- Module dependencies: `juce_core`, `juce_data_structures`, `juce_audio_processors`, `juce_dsp`, `juce_audio_formats` (`dsp_core/dsp_core.h` module declaration). The repo builds against JUCE 8.0.12 with C++20 (root `CMakeLists.txt`).
- There is no umbrella test target: `tests/CMakeLists.txt` defines 55 separate executables (`lane_mixer_tests`, `layered_transfer_function_tests`, `spline_fitter_tests`, ...), each registered with ctest via `gtest_discover_tests`.

## Component map

### Transfer-function model (production)

| Class | File | Role |
|---|---|---|
| `LaneMixer` | `Source/LaneMixer.h/.cpp` | Dynamic lane container; `computeSum`/`computeScan`/`computeSeries`; version counters; serialization + legacy migration |
| `Lane` / `LaneContentType` | `Source/Lane.h` | One waveshape (Harmonic/Paint/Spline/Equation/Preset) + mix atomics + stable `laneId` |
| `HarmonicLayer` | `Source/HarmonicLayer.h/.cpp` | Chebyshev basis evaluator with precomputed tables (`sin(n·asin x)` / `cos(n·acos x)`) |
| `AnchorMorphGesture` | `Source/MorphGesture.h/.cpp` | Recorded 2D delta path per spline anchor; drives morphable lanes |
| `SplineAnchor`, `SplineFitConfig` | `Source/SplineTypes.h` | Anchor data (incl. `morphGesture`, `homeX/homeY`) and fit tuning presets |

### Seamless LUT delivery

| Class | File | Role |
|---|---|---|
| `SeamlessTransferFunction` | `Source/SeamlessTransferFunction.h/.cpp` | Facade + wiring; `getLaneMixer()`, `renderLUTImmediate()`, `getRenderTrigger()` |
| `AudioEngine` | `Source/SeamlessTransferFunctionImpl.h/.cpp` | Triple-buffered LUT playback, Catmull-Rom evaluation, crossfade, analytical derivative |
| `EventDrivenRenderer` | same | High-priority worker thread ("BDD-LUTRenderer"), 120 Hz render gate |
| `VisualizerUpdateDispatcher` | same | 60 Hz-capped 16k pull-source publication for the editor's visualizer |

Full architecture: [seamless-transfer-function.md](docs/seamless-transfer-function.md).

### audio_pipeline stage library

`AudioPipeline` runs an ordered list of `AudioProcessingStage`s built via the `AudioPipelineBuilder` fluent API (`.withDryWetMix()`, `.addStage<T>(StageTag)`, `.beginOversampledGroup()/.endOversampledGroup()`), returning the pipeline plus typed `StageHandles`. Principal stages:

| Stage | Role |
|---|---|
| `OversamplingWrapper` | Wraps a stage group in `juce::dsp::Oversampling<double>`, order 0-4 (1x-16x) |
| `HysteresisStage` | The plugin's waveshaper: `HysteresisProcessor` per channel when enabled, memoryless `SeamlessTransferFunction::processBuffer` fallback when disabled |
| `ToneStage` + `*Strategy` | Filter stage with strategy-pattern types (LP/HP/shelf/smile/bell/IR convolution) |
| `DryWetMixStage` | Dry capture + mix; dry path runs a matched half-band oversampler for IIR phase coherence |
| `GainStage` | 10 ms-smoothed gain (`juce::dsp::Gain<double>`) |
| `BiasStage`, `SurgeStage` | DC offset into the shaper; time-varying overflow response |
| `AutoSquashStage` / `AutoRestoreStage` / `AutoGainState` | Auto-normalize pair with per-sample gain history |
| `DCBlockingFilter` | 1st-order Butterworth HPF @ 5 Hz (`juce::dsp::IIR`) |
| `ModulatorSlotStage` (`EnvelopeFollowerStage` / `LfoStage`) | Modulator slot sources + `SlotVisualizerPublisher` |
| `AudioInputWriter` / `AudioInputBuffer` | Publishes input audio for UI visualization |
| `SoftClippingStage` / `SoftClippingSolver` | Soft-clip solver (also used for LUT input bounding) |

Which stages the plugin instantiates, and in what order, is owned by [../../docs/architecture/signal-chain.md](../../docs/architecture/signal-chain.md) — do not infer chain order from this table.

### Services (`Source/Services/`)

| Service | Role |
|---|---|
| `SplineFitter` | Curve → anchors: feature-based init + greedy error-driven refinement |
| `SplineEvaluator` | PCHIP cubic Hermite evaluation |
| `CurveFeatureDetector` | Extrema/inflection detection on raw curve data |
| `AdaptiveToleranceCalculator` | Dynamic fit-error tolerances |
| `SymmetryAnalyzer` | Odd/even symmetry detection |
| `CoordinateSnapper` | World-space grid snapping |
| `AnchorConstraintService` | Ideal → constrained anchor projection (Y-clamp, min-gap) |
| `GestureSmoother` | Raw drag samples → arc-length-uniform morph gesture (Catmull-Rom) |
| `PerlinNoiseService` | Noise for brushes + LFO Random shape |
| `TransferFunctionOperations` | Stateless transforms on raw curve vectors |
| `ExpressionParser` / `EquationLaneSynthesizer` | `f(x) ; f(n)` syntax parsing; compiled-equation lane synthesis |
| `SplineLaneSynthesizer` | Side-effect-free spline-lane LUT synthesis at a morph value |

`CoordinateMapper` is **not** a dsp-core service — it lives in `transfer_function_editor`. `BaseLayerSolver` no longer exists anywhere.

### Legacy / test-only

| Class | Status |
|---|---|
| `LayeredTransferFunction` (+ `SplineLayer`) | Superseded by `LaneMixer`; kept for tests and the legacy serialization format — [docs/layered-transfer-function.md](docs/layered-transfer-function.md) |
| `TransferFunction` | Pre-layered original; compiled and exported but no references outside its own files |
| `WaveshapingStage` | Not instantiated by the plugin (`HysteresisStage` subsumed the memoryless fallback); keeps an LTF constructor for tests |

### Utility

| Class | Role |
|---|---|
| `AudioHistoryBuffer` / `StereoHistoryBuffer` | Ring buffers for UI-facing audio history (stereo variant is a seqlock guaranteeing X/Y correspondence) |
| `ExpressionEvaluator` | exprtk wrapper: variables `x`, `n` (harmonic index), `m` (morph macro) |

## Documentation

Current:
- [seamless-transfer-function.md](docs/seamless-transfer-function.md) — LaneMixer model, event-driven rendering, triple buffering, lifecycle
- [layered-transfer-function.md](docs/layered-transfer-function.md) — legacy LTF status, surviving API, migration path
- [spline-curve-fitting.md](docs/spline-curve-fitting.md) — fitting pipeline (feature detection → greedy fitting → PCHIP tangents)
- [spline-algorithm-decision.md](docs/spline-algorithm-decision.md) — Fritsch-Carlson vs Akima, no-overshoot rationale

Historical:
- [spline-layer-refactoring-notes.md](docs/spline-layer-refactoring-notes.md)
- [spline-optimization-history.md](docs/spline-optimization-history.md)
- [curve-fitting-enhancements-summary.md](docs/curve-fitting-enhancements-summary.md)

## Testing

```bash
cmake --build build --target lane_mixer_tests -j 8   # one of 55 targets; no umbrella target
ctest --test-dir build -R LaneMixer --output-on-failure
```

## Related docs

- [../../docs/architecture/signal-chain.md](../../docs/architecture/signal-chain.md) — canonical processBlock chain (owns stage order)
- [../../docs/architecture/dsp-processing.md](../../docs/architecture/dsp-processing.md) — audio-thread safety rules, memory ordering
- [../../docs/architecture/event-based-rendering.md](../../docs/architecture/event-based-rendering.md) — plugin-side integration of the render pipeline
- [../../docs/architecture/hysteresis.md](../../docs/architecture/hysteresis.md) — HysteresisProcessor/Stage parameters and panels
- [../../docs/architecture/services.md](../../docs/architecture/services.md) — service extraction criteria
- [../../docs/guides/curve-fitting-tuning.md](../../docs/guides/curve-fitting-tuning.md) — fit parameter tuning workflow
- [../../docs/guides/testing.md](../../docs/guides/testing.md) — running module tests
