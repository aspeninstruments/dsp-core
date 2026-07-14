---
verified: 2026-07-13
verified_modules: { dsp-core: 45f7854 }
status: current
---

# Spline Curve Fitting Pipeline

How `SplineFitter` converts a raw curve array (a 16384-sample `LaneMixer` lane) into a compact set of cubic Hermite spline anchors, and how `SplineEvaluator` turns anchors back into a LUT. Covers the fit pipeline stages, `SplineFitConfig` (including which fields are live vs. inert), feature detection, adaptive tolerance, symmetric fitting, tangent algorithms, and the batch evaluation API. Tuning advice lives in the [curve-fitting tuning guide](../../../docs/guides/curve-fitting-tuning.md); the Fritsch-Carlson-only decision is in [spline-algorithm-decision.md](spline-algorithm-decision.md).

## Key files

| File | Role |
|---|---|
| `../dsp_core/Source/Services/SplineFitter.h/.cpp` | `fitCurve()` entry point, sampling/sanitize, greedy refinement, all 4 tangent algorithms |
| `../dsp_core/Source/SplineTypes.h` | `SplineAnchor`, `SplineFitResult`, `SplineFitConfig` + `tight()` preset, `TangentAlgorithm`, `SymmetryDetection` |
| `../dsp_core/Source/Services/CurveFeatureDetector.h/.cpp` | Local-extrema detection, `FeatureDetectionConfig` |
| `../dsp_core/Source/Services/AdaptiveToleranceCalculator.h/.cpp` | Anchor-density-scaled error tolerance (anti anchor-creep) |
| `../dsp_core/Source/Services/SymmetryAnalyzer.h/.cpp` | Odd-symmetry score via Pearson correlation |
| `../dsp_core/Source/Services/SplineEvaluator.h/.cpp` | Hermite evaluation: `evaluate` / `evaluateBatch` / `evaluateDerivative` |
| `../dsp_core/Source/Services/SplineLaneSynthesizer.h/.cpp` | Morph-value spline-lane LUT synthesis (fit-result consumer) |
| `../tests/SplineFitterTest.cpp` | Fit, backtranslation, harmonic, and performance suites (`spline_fitter_tests`) |

## Load-bearing facts

- Entry point takes raw arrays, not a model object: `SplineFitter::fitCurve(const double* curveData, int tableSize, double minValue, double maxValue, const SplineFitConfig& = SplineFitConfig::tight())` (`SplineFitter.h:37-38`).
- Production input is per-lane: `CurveEditorController::applySplineFit` passes `Lane::curveData` with `LaneMixer::MIN_VALUE`/`MAX_VALUE` (`CurveEditorController.cpp:1002-1003`); `LaneMixer::TABLE_SIZE = 16384`, domain [-1, 1] (`LaneMixer.h:51-53`).
- `SplineFitConfig::tight()` is the **only** preset (`positionTolerance = 0.005`, `maxAnchors = 128`, FritschCarlson) and the default argument (`SplineTypes.h:236-242`). `smooth()`/`monotonePreserving()` no longer exist.
- `positionTolerance`, `enableRefinement`, and `pinEndpoints` are **inert** config fields — declared with doc comments but never read by any fitting code (`SplineTypes.h:92,119,157`; repo-wide grep shows no functional reads).
- The effective error target is hard-coded: `adaptiveConfig.relativeErrorTarget = 0.01` (1% of vertical range) on every fit (`SplineFitter.cpp:588`). The `anchorDensityMultiplier` stays at its `Config` default of **8.0** (`AdaptiveToleranceCalculator.h:57`) — the inline comment at `SplineFitter.cpp:589` claiming 10.0 is wrong.
- Feature detection is on by default (`enableFeatureDetection = true`, `SplineTypes.h:216` — the header comment saying "disabled by default" is stale) and its `maxFeatures` is capped at 70% of `maxAnchors` (`SplineFitter.cpp:53-57`).
- `CurveFeatureDetector` detects **local extrema only**; inflection detection was removed and `FeatureResult::inflectionPoints` is never populated (`CurveFeatureDetector.cpp:47-98`, `CurveFeatureDetector.h:84-88`).
- Symmetric paired-anchor insertion is gated on `symmetryDetection == Auto` and `SymmetryAnalyzer::analyzeOddSymmetry(curve).score >= symmetryThreshold` (default 0.90) (`SplineFitter.cpp:592-597`, `SplineTypes.h:187,197`).
- Default tangent algorithm is `TangentAlgorithm::FritschCarlson` (`SplineTypes.h:171`); the `(default)` comment on the PCHIP enum entry (`SplineTypes.h:11`) is stale.
- `SplineEvaluator::evaluateBatch` assumes sorted x values and uses incremental segment search instead of per-sample binary search (`SplineEvaluator.cpp:30-78`).
- Anchor pruning, the zero-crossing drift check, and inflection detection were all removed (2025-12-29; see [tuning guide changelog](../../../docs/guides/curve-fitting-tuning.md)).

## Pipeline

```
Input: raw curve array (double*, tableSize, [minValue, maxValue])
   ↓
Stage 0: Feature detection (optional, on by default) → mandatory anchor indices
   ↓
Stage 1: Sampling + densification (Catmull-Rom midpoints, ~2× samples)
   ↓
Stage 2: Sanitization (sort, dedupe, optional monotonicity, clamp)
   ↓
Stage 3: Greedy error-driven refinement (adaptive tolerance, symmetric pairs)
   ↓
Stage 4: Final tangent computation (configured algorithm)
   ↓
Stage 5: Error analysis → SplineFitResult
```

`fitCurve` orchestrates all stages (`SplineFitter.cpp:41-108`).

### Stage 0: Feature detection

`CurveFeatureDetector::detectFeatures(curveData, tableSize, minValue, maxValue, config)` returns table indices of mandatory anchors (`CurveFeatureDetector.h:100-101`).

- An index `i` is a local extremum when the central-difference derivative changes sign (`deriv_prev * deriv < 0`) **and at least one** of the two derivative magnitudes exceeds `derivativeThreshold` (OR, not AND) (`CurveFeatureDetector.cpp:57-61`, derivative at `:130-151`).
- Significance = `|y - verticalCenter|`; with `enableSignificanceFiltering = true` it becomes local prominence over a `max(5, tableSize/1600)` window (`CurveFeatureDetector.cpp:67-95`).
- Endpoints `[0, tableSize-1]` are always mandatory. If `features + 2 > maxFeatures`, the top `maxFeatures - 2` extrema by significance are kept (`CurveFeatureDetector.cpp:100-128`).
- `fitCurve` caps `featureConfig.maxFeatures` at `maxAnchors * 0.7`, reserving 30% of the budget for error-driven refinement (`SplineFitter.cpp:53-57`).

`FeatureDetectionConfig` defaults (`CurveFeatureDetector.h:51-53`):

| Field | Default | Effect |
|---|---|---|
| `significanceThreshold` | 0.001 | Prominence floor (fraction of vertical range), only used when filtering enabled |
| `maxFeatures` | 100 | Hard cap before prioritization (further capped to 70% of `maxAnchors`) |
| `derivativeThreshold` | 1e-6 | First-derivative noise floor |
| `enableSignificanceFiltering` | false | Prominence filtering off — all sign-change extrema accepted |

### Stage 1: Sampling + densification

`sampleAndSanitize` (`SplineFitter.cpp:111-162`) emits one sample per table index at normalized x, then inserts a midpoint sample between each adjacent pair — sampled via **Catmull-Rom interpolation** (`SplineFitter::interpolateCurve`, `SplineFitter.cpp:17-39`), not linear interpolation or nearest-index lookup, for sub-sample accuracy on high-frequency curves. Result: `2N - 1` samples (~32,767 for a 16384 table).

### Stage 2: Sanitization

Applied in order (`SplineFitter.cpp:147-159`):

1. `sortByX` (`:164-166`)
2. `deduplicateNearVerticals` — groups samples with `|dx| < 1e-6`, averages them (`:168-193`)
3. `enforceMonotonicity` (only when `config.enforceMonotonicity`) — single forward pass of pairwise averaging; **not** a full PAVA isotonic regression (a lowered `y[i-1]` is never re-checked against `y[i-2]`; acknowledged in the TODO at `:213`) (`:195-215`)
4. `clampToRange` — clamps x and y to [-1, 1] (`:217-222`)

### Stage 3: Greedy error-driven refinement

`greedySplineFit` (`SplineFitter.cpp:552-633`):

1. **Seed anchors**: mandatory feature indices snapped to nearest samples, sorted, deduped at 1e-9 (`initializeAnchorsFromIndices`, `:637-669`); endpoints only if feature detection is off/empty (`:563-568`).
2. **Budget**: `remainingAnchors = max(0, maxAnchors - seeded)` (`:572`).
3. **Symmetry gate**: with `symmetryDetection == Auto`, runs `SymmetryAnalyzer::analyzeOddSymmetry` on the raw curve; symmetric mode when `score >= config.symmetryThreshold` (`:592-597`).
4. **Loop** (up to `remainingAnchors` iterations, `:601-627`): recompute tangents → `findWorstFitSample` (skips samples already anchored within 1e-9, `:515-550`) → stop when `maxError <= adaptiveTolerance` (`:609`) → insert anchor(s) at the worst-fit sample.
5. **Final tangents** with the configured algorithm (`:630`).

Insertion modes:

- **Asymmetric** (`insertAnchorAsymmetric`, `:735-749`): single anchor at the worst sample; sorted insert via `lower_bound`; refuses duplicates.
- **Symmetric** (`insertAnchorSymmetric`, `:671-733`): finds the complementary sample nearest `-x`. If neither position already has an anchor **and** the complementary error exceeds `adaptiveTolerance * 0.5`, inserts a pair at `(x, ySym)` and `(-x, -ySym)` with `ySymmetric = (yWorst - yComplementary) / 2` (`:707-721`); a pair consumes two loop iterations (`:619-621`). Otherwise falls back to a single anchor (so final counts can be one higher than an asymmetric fit).

### Adaptive tolerance

`AdaptiveToleranceCalculator::computeTolerance` (`AdaptiveToleranceCalculator.cpp:7-27`):

```
tolerance = verticalRange × relativeErrorTarget × (1 + min(1, currentAnchors/maxAnchors) × anchorDensityMultiplier)
```

- `SplineFitter` overrides `relativeErrorTarget` to **0.01** for every fit; there is no `positionTolerance` floor (`SplineFitter.cpp:583-589`).
- `anchorDensityMultiplier` stays at the `Config` default **8.0** (`AdaptiveToleranceCalculator.h:57`); the struct's `relativeErrorTarget` default of 0.004 is never used in production.
- Effect with multiplier 8.0: 10% capacity → 1.8× baseline, 50% → 5.0×, 100% → 9.0×.
- Purpose: prevents "anchor creeping" where refitting a fitted curve grows the anchor count each cycle. Verified by the `BacktranslationTest` suites (`../tests/SplineFitterTest.cpp:1525`). Tuning history: [spline-optimization-history.md](spline-optimization-history.md).

### Symmetry detection

`SymmetryAnalyzer::analyzeOddSymmetry(const std::vector<double>& curveData, Config)` (`SymmetryAnalyzer.cpp:51-99`):

- Hard gate: `|curveData[center]| > 0.1` → score 0, Asymmetric (`:66-71`).
- Samples 128 complementary point pairs and computes the Pearson correlation between `f(x)` and `-f(-x)` (`:73-88`, `:8-49`; sample count in `SymmetryAnalyzer.h:47`).
- Classification: Perfect ≥ 0.99, Approximate ≥ 0.90, else Asymmetric (`SymmetryAnalyzer.h:35,41`; `SymmetryAnalyzer.cpp:90-96`).
- Only odd symmetry is implemented — there is no even-symmetry analysis (`SymmetryAnalyzer.h:93-101`).

### Stage 4: Tangent computation

`SplineFitter::computeTangents` dispatches on `config.tangentAlgorithm` (`SplineFitter.cpp:226-246`). It is also public API, used to recompute tangents after manual anchor edits (`SplineFitter.h:42-43`).

| Algorithm | Implementation | Notes |
|---|---|---|
| `FritschCarlson` (default) | `SplineFitter.cpp:366-423` | Weighted harmonic-mean tangents; zero tangent at extrema; α²+β²≤9 no-overshoot constraint with τ = 3/√(α²+β²) (`:408-412`); one-sided boundary tangents |
| `PCHIP` | `SplineFitter.cpp:338-362` | Fritsch-Carlson base + iterative overshoot damping (4 interior samples at t = 0.2…0.8, tolerance 0.001, damping 0.7, max 3 iterations, `:286-319`) + long-segment tangent scaling (threshold 0.3, `:322-335`) |
| `Akima` | `SplineFitter.cpp:427-475` | Local weighted average with extrapolated boundary slopes |
| `FiniteDifference` | `SplineFitter.cpp:479-511` | Central/one-sided differences (baseline for comparison) |

All algorithms clamp tangents to `[minSlope, maxSlope]` (default ±8.0). Only FritschCarlson is selected by production code; the other three are exercised by tests only. Rationale: [spline-algorithm-decision.md](spline-algorithm-decision.md).

### Stage 5: Result

`SplineFitResult` carries `success`, `anchors`, `numAnchors`, `maxError`, `averageError`, and a user-facing `message` of the form `"Fitted N anchors, max error: X"` (`SplineTypes.h:47-57`, `SplineFitter.cpp:90-107`).

## SplineFitConfig reference

Struct defaults vs. the `tight()` preset (`SplineTypes.h:73-243`). The header's field doc comments predate several removals — treat this table (and the code) as authoritative.

| Field | Struct default | `tight()` | Live? | Read at |
|---|---|---|---|---|
| `positionTolerance` | 0.01 | 0.005 | **Inert** — never read | — |
| `maxAnchors` | 128 | 128 | Live | `SplineFitter.cpp:54,572,607` |
| `enableRefinement` | true | — | **Inert** — never read | — |
| `enforceMonotonicity` | true | — | Live | `SplineFitter.cpp:154` |
| `minSlope` / `maxSlope` | -8.0 / 8.0 | — | Live | tangent clamps, e.g. `SplineFitter.cpp:421` |
| `pinEndpoints` | true | — | **Inert** — never read | — |
| `tangentAlgorithm` | FritschCarlson | FritschCarlson | Live | `SplineFitter.cpp:228` |
| `symmetryDetection` | Auto | — | Live | `SplineFitter.cpp:593` |
| `symmetryThreshold` | 0.90 | — | Live | `SplineFitter.cpp:596` |
| `enableFeatureDetection` | true | — | Live | `SplineFitter.cpp:50` |
| `featureConfig` | see Stage 0 | — | Live | `SplineFitter.cpp:53` |

`SymmetryDetection` has exactly two values: `Auto` and `Never` (`SplineTypes.h:18-21`) — there is no `Always`.

`SplineAnchor` also carries morph-gesture state: `morphGesture` plus `homeX`/`homeY` home position, kept inline so gestures survive reordering, snapshot/restore, and serialization (`SplineTypes.h:24-44`). Morph synthesis consumes it via `SplineLaneSynthesizer::synthesizeSplineLaneLUT`, which projects gesture targets through `AnchorConstraintService`, recomputes tangents with `tight()`, and batch-evaluates into a LUT (`SplineLaneSynthesizer.cpp:13-60`).

## Evaluation: SplineEvaluator

- `evaluate(anchors, x)` — binary search for the segment (`lower_bound`, `SplineEvaluator.cpp:102-118`), clamps to endpoint y outside the anchor range (`:8-28`).
- `evaluateBatch(anchors, xValues, yValues, count)` — incremental segment search; **requires sorted ascending `xValues`**; fills 0.0 / single-anchor y for degenerate inputs (`:30-78`).
- `evaluateDerivative(anchors, x)` — Hermite derivative; returns endpoint tangents outside the range (`:80-100`, segment derivative at `:155-188`).
- `evaluateSegment(p0, p1, x)` — cubic Hermite basis `H(t) = h00·y0 + h10·Δx·m0 + h01·y1 + h11·Δx·m1` (`:120-153`); public for the fitter's overshoot detection.

Production consumers of `evaluateBatch`: `CurveEditorController::updateSplineAnchorsDirect` (evaluates anchors into the 16384-sample lane `curveData` on every anchor edit, `CurveEditorController.cpp:876-903`) and `SplineLaneSynthesizer` (`SplineLaneSynthesizer.cpp:56`).

## Production integration (transfer_function_editor)

Cross-module references below were verified against transfer_function_editor `5f15313`.

- `SplineTool` holds `currentConfig = SplineFitConfig::tight()` (`SplineTool.cpp:16`); on activation it calls `fitCurveToSpline` → `CurveEditorController::applySplineFit` (`SplineTool.cpp:108,156-158`).
- **Full-curve path**: `fitCurve(lane.curveData.data(), tableSize, LaneMixer::MIN_VALUE, LaneMixer::MAX_VALUE, config)` (`CurveEditorController.cpp:1000-1004`).
- **Odd-symmetry half-fit path** (when the lane's odd-symmetry toggle is on): resamples the left half of the lane onto [-1, 0] via Catmull-Rom, fits only that half, snaps any near-origin anchor (|x| < 1e-6) to exactly (0, 0), mirrors non-origin anchors to `(-x, -y)`, sorts, and recomputes tangents (`CurveEditorController.cpp:919-999`).
- The per-lane odd-symmetry toggle (`LaneMixer::isLaneOddSymmetryEnabled`, read at `CurveEditorController.cpp:1296-1301`) is a **different mechanism** from `SplineFitConfig::symmetryDetection`: the former forces mirrored editing/fitting at the editor level; the latter is the fitter-internal paired-insertion heuristic used on full-curve fits.

The legacy `LayeredTransferFunction` model still exists in dsp-core but is not part of this pipeline — see [layered-transfer-function.md](layered-transfer-function.md).

## Testing

`spline_fitter_tests` builds from `SplineFitterTest.cpp` (`../tests/CMakeLists.txt:180-202`). Key suites: `BacktranslationTest.*` (anchor-count stability across refits), `SplineFitterTest.AllHarmonics_*`, `SplineFitterTest.Performance_*`. Adjacent binaries: `spline_evaluator_tests`, `curve_feature_detector_tests`, `adaptive_tolerance_tests`, `symmetry_analyzer_tests`, `algorithm_benchmark_tests`, `spline_fitter_integration_tests`.

```bash
./build/modules/dsp-core/tests/spline_fitter_tests --gtest_filter='BacktranslationTest.*'
```

## Related docs

- [spline-algorithm-decision.md](spline-algorithm-decision.md) — why Fritsch-Carlson is the sole production tangent algorithm
- [curve-fitting tuning guide](../../../docs/guides/curve-fitting-tuning.md) — which knobs to turn for fit quality/anchor count
- [layered-transfer-function.md](layered-transfer-function.md) — legacy LTF model (migration/tests only)
- [spline-optimization-history.md](spline-optimization-history.md) — historical parameter-tuning record (Phase 4, archived)
- [curve-fitting-enhancements-summary.md](curve-fitting-enhancements-summary.md) — historical record of the symmetry-fitting enhancement phases (archived)
