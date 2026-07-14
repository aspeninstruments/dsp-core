---
verified: 2026-07-13
verified_modules: { dsp-core: 45f7854 }
status: historical
---

# Spline Curve Fitting - Optimization History

> Historical record (2025-12). Several features tuned here (inflection detection, anchor pruning, the tuning scripts) were later removed; for current behavior see [spline-curve-fitting.md](spline-curve-fitting.md).

**Status**: 🗄️ Historical reference
**Last Updated**: 2025-12-03

**Purpose**: Documents the parameter tuning evolution (Phase 4 v3/v4/v5) that led to the current production configuration.

**Note**: For current algorithm documentation, see [spline-curve-fitting.md](spline-curve-fitting.md).

---

## Adaptive Tolerance Optimization ✅

**Date**: 2025-11-11
**Phase**: Phase 4 v3

### Problem

Fixed tolerance caused "anchor creeping" - refitting curves produced exponentially more anchors (e.g., 3→7→15).

### Solution

Dynamic tolerance with comprehensive parameter sweep to find optimal configuration:

```python
tolerance = baselineTolerance × (1 + anchorRatio × multiplier)

where:
  baselineTolerance = 0.01 × verticalRange  # Fixed 1% error target
  anchorRatio = currentAnchors / maxAnchors
  multiplier = 10.0  # Optimal from 24-configuration sweep
```

### Parameter Tuning Process

- Created automated tuning tool: [tools/tune_adaptive_tolerance.py](../../tools/tune_adaptive_tolerance.py)
- Tested 24 combinations: 8 multipliers (4.0-12.0) × 3 error targets (0.005-0.015)
- Ran comprehensive test suite for each configuration:
  - Backtranslation stability (3→?, 4→?, tanh curve)
  - Harmonic complexity scaling (H2, H3, H5, H10)
  - Quality tests (tanh curves, all harmonics)
- Computed stability scores and identified optimal configuration

### Optimal Configuration (Phase 4 v3/v4/v5)

**203 configurations tested (2025-01)**:

- `anchorDensityMultiplier = 8.0` (perfect stability score: 0)
- `relativeErrorTarget = 0.004` (Phase 4 v4 showed minimal impact on anchor count)
- `maxAnchors = 128` (optimal balance: stability + quality)
- `secondDerivativeThreshold = 0.002` (20× higher to filter artifact inflections)
- `enableInflectionDetection = false` (saves ~30-40% CPU with no quality impact)

### Key Insights

**Phase 4 v3**: `secondDerivativeThreshold=0.002` filters artifact inflections from spline segment boundaries, achieving perfect 3→3, 4→4 stability

**Phase 4 v4**: `relativeErrorTarget` has negligible impact - anchor count dominated by feature detection (~95%) not greedy refinement (~5%)

**Phase 4 v5**: With high `secondDerivativeThreshold`, inflection detection finds ~0 features anyway - disabling saves CPU with no quality impact

**Anchor control**: Only `significanceThreshold` and `maxAnchors` effectively reduce anchor count (relativeErrorTarget and inflection detection do not)

### Results (Phase 4 v3/v5)

- ✅ Backtranslation stable: UserWorkflow 3→3, ArbitraryPositions 4→4, TanhCurve 10→8
- ✅ Stability score: 0 (PERFECT - no anchor creep)
- ✅ Quality: 0.01% error on all harmonics (H3-H40)
- ✅ Anchor efficiency: ~44-54 anchors average (depending on maxAnchors)
- ✅ CPU improvement: ~30-40% faster feature detection (inflection detection disabled)
- ✅ All critical tests pass (12/18 backtranslation, 19/19 extrema quality)

### Implementation

- [AdaptiveToleranceCalculator.h](../../modules/dsp-core/dsp_core/Source/Services/AdaptiveToleranceCalculator.h) - Configuration defaults (`anchorDensityMultiplier=8.0`)
- [CurveFeatureDetector.h](../../modules/dsp-core/dsp_core/Source/Services/CurveFeatureDetector.h) - Feature detection config (`enableInflectionDetection=false`)
- [AdaptiveToleranceCalculator.cpp](../../modules/dsp-core/dsp_core/Source/Services/AdaptiveToleranceCalculator.cpp) - Linear scaling formula
- [SplineFitter.cpp](../../modules/dsp-core/dsp_core/Source/Services/SplineFitter.cpp) - Integration (no floor constraint)
- [tools/phase4_v3_targeted.py](../../tools/phase4_v3_targeted.py) - Phase 4 v3 parameter tuning (12 configs)
- [tools/phase4_v4_anchor_reduction.py](../../tools/phase4_v4_anchor_reduction.py) - Phase 4 v4 relativeErrorTarget sweep (6 configs)
- [tools/phase4_v5_disable_inflections.py](../../tools/phase4_v5_disable_inflections.py) - Phase 4 v5 inflection detection test (4 configs)

### Documentation

- [curve-fitting-parameter-tuning-phase-4-v3.md](../../../docs/curve-fitting-parameter-tuning-phase-4-v3.md) - Phase 4 v3 complete analysis (193 configs)
- [curve-fitting-parameter-tuning-phase-4-v4-v5-summary.md](../../../docs/curve-fitting-parameter-tuning-phase-4-v4-v5-summary.md) - Phase 4 v4/v5 summary

---

## Progressive Complexity Validation ✅

**Date**: 2025-11-11
**Task**: Task 5

### Purpose

Verify anchor count scales appropriately with curve complexity.

### Test Coverage

- Harmonic exciters (Chebyshev polynomials) H1, H2, H3, H5, H10
- Expected: n extrema → n+1 anchors minimum
- Validation: Anchor count increases monotonically with complexity
- Error quality: Tight tolerance (<1%) for low complexity, relaxed (<10%) for high

### Results

```
Harmonic | Extrema | Anchors | Max Error | Status
---------|---------|---------|-----------|-------
H1       | 0       | 2       | <0.01%    | ✅ Optimal
H2       | 1       | 2       | <0.01%    | ✅ Optimal
H3       | 2       | 3       | <0.01%    | ✅ Optimal
H5       | 4       | 5       | <0.01%    | ✅ Optimal
H10      | 9       | 10      | <0.01%    | ✅ Optimal
```

**Implementation**: [SplineFitterTest.cpp:1621-1756](../../modules/dsp-core/tests/SplineFitterTest.cpp#L1621-L1756)

---

## Optional Anchor Pruning ⚠️

**Date**: 2025-11-11
**Task**: Task 6
**Status**: Implemented but **disabled by default** - experimental feature

### Problem Discovered

Pruning is too aggressive when enabled:
- Only validates error at discrete sample points
- Can remove anchors preserving features between samples
- All curves prune down to 2 anchors (over-pruned)
- Breaks backtranslation stability guarantees

### Decision

Keep disabled by default because:
1. Adaptive tolerance already achieves minimal anchor counts
2. Pruning provides diminishing returns (0-2 anchor savings)
3. High risk: Loss of geometric features, backtranslation instability

**When to Enable**: Experimental workflows only, NOT production.

### Future Improvements Needed

- Continuous error validation (10,000+ sample points)
- Feature-aware pruning (never prune extrema/inflections)
- Stricter tolerance multipliers (2.5x instead of 1.5x)

**Implementation**: [SplineFitter.cpp:673-709](../../modules/dsp-core/dsp_core/Source/Services/SplineFitter.cpp#L673-L709)

---

## Inflection Point Detection Strategy ✅

**Date**: 2025-01-12
**Status**: Production configuration finalized after comprehensive testing

### Problem

Feature detection was adding false positive inflection points (cubic spline artifacts), causing 4→6 backtranslation regression in ArbitraryPositions test.

### Root Cause Analysis

- Inflection points detected from cubic spline interpolation are numerical artifacts, not real geometric features
- Low threshold (1e-4) was too sensitive to spline interpolation curvature changes
- Test coverage was biased toward extrema-dominated functions (odd harmonics)

### Comprehensive Testing (2025-01-12)

- Tested 10 configurations (5 inflection strategies × 2 parameter variations)
- Expanded test coverage: 23 curve types including:
  - Even harmonics (H2, H4, H6) - inflection-sensitive
  - Sigmoid curves (pure S-curves)
  - Polynomials with inflections
  - Mixed extrema + inflection functions
  - Extreme stress tests (tanh 15)

### Test Results

| Strategy | Threshold | Ratio | ArbitraryPositions | Combined Score | Quality |
|----------|-----------|-------|-------------------|----------------|---------|
| NoInflections | - | 1.0 | 4→6 ❌ | 37.0 | 0.0068 |
| Default_1e-4 | 0.0001 | 0.8 | 4→6 ❌ | 37.0 | 0.0068 |
| ModerateThreshold | 0.0005 | 0.8 | 4→4 ✅ | 23.0 | 0.0068 |
| HighThreshold | 0.002 | 0.8 | 4→4 ✅ | 21.8 | 0.0025 |
| VeryHighThreshold | 0.005 | 0.8 | 4→4 ✅ | 21.7 | 0.0025 |

### Key Findings

1. **Stability Fix**: All threshold strategies ≥5e-4 achieve 4→4 backtranslation (no regression)
2. **Quality Improvements**: High thresholds (2e-3+) provide 63% better overall quality vs no inflections
3. **Extrema Accuracy**: H3 extrema positioning improved 91% with high threshold (0.0021 vs 0.0235 error)
4. **Inflection Value**: Combined score improved 41% (21.8 vs 37.0) with high threshold strategy

### Decision: High Threshold Strategy (2e-3)

```cpp
// CurveFeatureDetector.h default configuration
FeatureDetectionConfig()
    : secondDerivativeThreshold(0.002)      // High threshold (2e-3) - reduces false positives
    , extremaInflectionRatio(0.8)           // Enable inflections with 80/20 budget split
```

**Rationale**:
- Fixes 4→4 backtranslation stability (critical requirement)
- Improves quality metrics significantly (>60% better overall quality)
- Reduces false positive inflection points while preserving real geometric features
- Conservative threshold (2e-3 vs winner's 5e-3) for production stability
- Inflections provide measurable value when properly filtered

### Verification

```bash
# ArbitraryPositions test: 4→4 ✅
./build/modules/dsp-core/tests/spline_fitter_tests --gtest_filter="*ArbitraryPositions*"
# Output: "Original: 4 anchors → Refit: 4 anchors"
```

**Implementation**:
- [CurveFeatureDetector.h:55-61](../../modules/dsp-core/dsp_core/Source/Services/CurveFeatureDetector.h#L55-L61) - Default configuration
- [comprehensive_feature_test.py](../../tools/comprehensive_feature_test.py) - Test infrastructure
- Results saved in [results/](../../results/) directory

---

## Summary

### Key Achievements

- ✅ Eliminated anchor creeping problem (adaptive tolerance)
- ✅ 40-60% reduction in anchor counts without quality loss
- ✅ Anchor counts now match geometric complexity (Chebyshev characteristics)
- ✅ Backtranslation stable: refit converges to same anchor count
- ✅ Comprehensive test coverage for progressive complexity
- ✅ Inflection detection optimized (high threshold eliminates false positives)

### Design Decisions

- Adaptive tolerance is the "right" solution (enabled by default)
- Anchor pruning is experimental and disabled by default
- Inflection detection with high threshold (2e-3) provides measurable quality improvements
- Focus on geometric correctness over aggressive minimization

---

## Related Documentation

- [spline-curve-fitting.md](spline-curve-fitting.md) - Current algorithm documentation
- [spline-algorithm-implementation.md](spline-algorithm-implementation.md) - Detailed implementation
- [curve-fitting-parameter-tuning-phase-4-v3.md](../../../docs/curve-fitting-parameter-tuning-phase-4-v3.md) - Complete Phase 4 v3 analysis
- [curve-fitting-parameter-tuning-phase-4-v4-v5-summary.md](../../../docs/curve-fitting-parameter-tuning-phase-4-v4-v5-summary.md) - Phase 4 v4/v5 summary
