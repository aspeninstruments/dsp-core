# Layered Transfer Function Architecture

**Last Updated**: 2025-11-10

## Overview

The core innovation of TotalHarmonicControl is a **four-layer model** that separates user drawing from harmonic synthesis and spline interpolation, enabling independent manipulation of wavetable, harmonic, and spline content.

---

## Four-Layer Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                      LayeredTransferFunction                                  │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          │                         │                         │
          ▼                         ▼                         ▼
   ┌─────────────┐          ┌──────────────┐         ┌──────────────┐
   │ Base Layer  │          │ Harmonic     │         │ Spline       │
   │ (Drawable)  │          │ Layer        │         │ Layer        │
   │             │          │ (Coeffs[41]) │         │ (Anchors)    │
   │ User draws  │          │ WT Mix + 40  │  XOR    │ Direct PCHIP │
   │ here with   │          │ harmonics    │         │ evaluation   │
   │ PaintTool    │          │              │         │              │
   └─────────────┘          └──────────────┘         └──────────────┘
         │                          │                        │
         │                          │ Mutually Exclusive     │
         │                          │ (controlled by         │
         │                          │ splineLayerEnabled)    │
         │                          │                        │
         └──────────────────────────┴────────────────────────┘
                                    │
                          updateComposite()
                                    │
                     ┌──────────────┴──────────────┐
                     │                             │
              Harmonic Mode:              Spline Mode:
          Composite[i] = normalize(    Composite[i] =
            coeffs[0] * base[i] +        splineLayer->evaluate(x)
            harmonicLayer->evaluate()    (no normalization,
          )                               normScalar = 1.0)
                     │                             │
                     └──────────────┬──────────────┘
                                    ▼
                             ┌──────────────┐
                             │ Composite    │
                             │ (Read-only)  │
                             │ For audio    │
                             │ processing   │
                             └──────────────┘
```

---

## Layer Descriptions

### 1. Base Layer

```cpp
std::vector<std::atomic<double>> baseTable;  // 16384 points
```

**Purpose**: User-drawn wavetable
**Modified By**: PaintTool painting, EquationTool expressions, SplineTool curves
**Initially**: Linear (y = x)

**Thread Safety**:
- Uses atomics for lock-free access
- UI thread writes with `std::memory_order_release`
- Audio thread reads with `std::memory_order_acquire`

**Table Size**: 16384 points (production), 256 points (tests)
**Index to X mapping**: `normalizeIndex(i) → [-1, 1]`

### 2. Harmonic Layer

```cpp
HarmonicLayer harmonicLayer;
std::vector<double> coefficients;  // 41 coefficients
```

**Purpose**: Additive harmonic synthesis using Chebyshev polynomial basis
**Modified By**: LaneMixerPanel sliders

**Coefficients**:
- `coefficients[0]` = Wavetable (WT) mix (0.0 to 1.0)
  - 0.0 = pure harmonics only
  - 1.0 = pure wavetable only
  - 0.5 = equal mix
- `coefficients[1..40]` = Harmonic amplitudes (40 harmonics)
  - Odd harmonics (1, 3, 5, ..., 39): Use `sin(n * asin(x))` Chebyshev basis
  - Even harmonics (2, 4, 6, ..., 40): Use `cos(n * acos(x))` Chebyshev basis
  - Range: typically [-1.0, 1.0]

**Implementation**:
- Precomputed Chebyshev basis functions for performance
- Pure additive synthesis (no base layer modification)
- Formula: `harmonic = Σ(coeffs[n] * H_n(x))` where `H_n` is Chebyshev basis

### 3. Spline Layer

```cpp
std::unique_ptr<SplineLayer> splineLayer;
std::atomic<bool> splineLayerEnabled;
```

**Purpose**: PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) spline interpolation for smooth curve editing
**Modified By**: SplineTool anchor dragging, curve fitting
**Mutually Exclusive With**: Harmonic Layer (controlled by `splineLayerEnabled` flag)

**Architecture**:
- Direct evaluation: No wavetable precomputation, evaluates PCHIP on-demand
- Lock-free anchor storage: Uses `std::shared_ptr<const std::vector<SplineAnchor>>` with atomic load/store
- Dual-path optimization:
  - Cache invalid (during drag): Direct evaluation ~40-50ns per sample
  - Cache valid (after drag): Interpolation from composite ~5-10ns per sample

**Thread Safety**:
```cpp
// UI thread: Update anchors (lock-free)
void SplineLayer::setAnchors(const std::vector<SplineAnchor>& anchors) {
    auto newPtr = std::make_shared<const std::vector<SplineAnchor>>(anchors);
    std::atomic_store(&anchorsPtr, newPtr);  // Atomic swap
}

// Audio thread: Read anchors (lock-free)
double SplineLayer::evaluate(double x) const {
    auto ptr = std::atomic_load(&anchorsPtr);  // Atomic load
    if (!ptr || ptr->empty()) return 0.0;
    return Services::SplineEvaluator::evaluate(*ptr, x);
}
```

**Why No Wavetable Precomputation?**
- PCHIP evaluation is fast enough (~30-40ns) for direct audio thread usage
- Avoids 16,384 atomic stores on UI thread during drag (eliminates ~1-2ms bottleneck)
- Memory efficient: Few anchors (~8-12) vs. 16,384-point table

**Normalization Behavior**:
- **No normalization**: `normScalar = 1.0` (identity) in spline mode
- User controls amplitude directly via anchor Y-position
- Design rationale: Single-layer edit with visual feedback → user has direct control

**See Also**: [spline-curve-fitting.md](spline-curve-fitting.md) for curve fitting algorithm details

### 4. Composite Layer

```cpp
std::vector<std::atomic<double>> compositeTable;  // 16384 points
```

**Purpose**: Final output for audio processing
**Modified By**: Automatic recomputation only
**Read By**: Audio thread during `processBlock()`

**Update Formula**:
```cpp
for (int i = 0; i < tableSize; ++i) {
    double x = normalizeIndex(i);  // [-1, 1]
    double baseValue = baseTable[i].load(std::memory_order_acquire);
    double harmonicValue = harmonicLayer.evaluate(x, coefficients);

    double composite = coeffs[0] * baseValue + harmonicValue;
    composite = normalizeValue(composite);  // Clamp to [-1, 1]

    compositeTable[i].store(composite, std::memory_order_release);
}
```

**Thread Safety**:
- Only updated from UI thread in `updateComposite()`
- Audio thread reads lock-free using atomics
- No mutex required

---

## Paint Stroke Active (Frozen Normalization)

During interactive editing (paint strokes, spline dragging), normalization scalar can be frozen to prevent visual jumping.

```cpp
// Before paint stroke
ltf.updateNormalizationScalar();   // Cache current normalization scalar
ltf.setPaintStrokeActive(true);    // Freeze it during the stroke

// During drag
// ... multiple setBaseLayerValue() calls ...
// ... computeCompositeAt() uses frozen cached scalar ...

// After paint stroke
ltf.setPaintStrokeActive(false);   // Unfreeze (renderer will recompute at 120Hz rate gate)
```

**Why This Matters**:
- Without freezing: Each paint point could change the normalization scalar (if recomputed)
- Result: Curve would jump visually as user drags
- With freezing: Scalar frozen during stroke, recalculated by renderer after
- Result: Smooth, predictable painting experience

**Implementation** (Refactored 2025-12-03):
```cpp
class LayeredTransferFunction {
    bool paintStrokeActive = false;
    mutable std::atomic<double> normalizationScalar{1.0};

    double computeCompositeAt(int index) const {
        // ... compute unnormalized value ...
        const double unnormalized = wtCoeff * baseValue + harmonicValue;

        // Apply cached scalar (O(1), not O(n))
        if (normalizationEnabled) {
            const double normScalar = normalizationScalar.load(std::memory_order_acquire);
            return normScalar * unnormalized;
        }
        return unnormalized;
    }

    void updateNormalizationScalar() {
        // Scan entire table to find max (called explicitly, not on-the-fly)
        double maxAbsValue = 0.0;
        for (int i = 0; i < tableSize; ++i) {
            // ... compute unnormalized composite ...
            maxAbsValue = std::max(maxAbsValue, std::abs(unnormalized));
        }
        const double normScalar = (maxAbsValue > 1e-12) ? (1.0 / maxAbsValue) : 1.0;
        normalizationScalar.store(normScalar, std::memory_order_release);
    }
};
```

---

## Thread Safety Guarantees

### UI Thread (Writer)

```cpp
// Write to base layer
void setBaseLayerValue(int index, double value) {
    baseTable[index].store(value, std::memory_order_release);
}

// Recompute composite
void updateComposite() {
    for (int i = 0; i < tableSize; ++i) {
        double composite = computeValue(i);
        compositeTable[i].store(composite, std::memory_order_release);
    }
}
```

**Rules**:
- ALWAYS use `memory_order_release` for writes
- ALWAYS call `updateComposite()` after base layer changes
- NEVER modify from multiple threads

### Audio Thread (Reader)

```cpp
// Read composite for processing
void processBlock(AudioBuffer<float>& buffer) {
    for (int i = 0; i < buffer.getNumSamples(); ++i) {
        float input = buffer.getSample(0, i);

        // Map input to table index
        int index = inputToTableIndex(input);

        // Read composite (lock-free)
        double output = compositeTable[index].load(std::memory_order_acquire);

        buffer.setSample(0, i, static_cast<float>(output));
    }
}
```

**Rules**:
- ALWAYS use `memory_order_acquire` for reads
- NEVER write to base or composite
- NEVER allocate memory
- NEVER use locks or mutexes

---

## Normalization Strategy

### Purpose

Ensure composite output stays within `[-1, 1]` range to prevent audio clipping.

---

### Harmonic Mode Normalization (3 States)

In harmonic mode, normalization is **always applied** to prevent clipping when mixing base + harmonics:

| State | When | Behavior | Use Case |
|-------|------|----------|----------|
| **Normal** | `normalizationEnabled=true`<br>`paintStrokeActive=false` | Renderer recomputes scalar at 25Hz | Standard operation |
| **Frozen** | `paintStrokeActive=true` | Renderer uses cached scalar | Paint strokes, slider drags (prevents visual shift) |
| **Disabled** | `normalizationEnabled=false` | Scalar locked at `1.0` | Creative effects (allows clipping) |

**Algorithm**:
```cpp
double findMaxAbsValue() {
    double maxAbs = 0.0;
    for (int i = 0; i < tableSize; ++i) {
        double value = coeffs[0] * base[i] + harmonic[i];
        maxAbs = std::max(maxAbs, std::abs(value));
    }
    return maxAbs;
}

double normalizationScalar = (maxAbs > 1.0) ? (1.0 / maxAbs) : 1.0;
```

**When Normalization Occurs** (Harmonic Mode):
- ✅ After base layer changes (via `updateComposite()`)
- ✅ After harmonic coefficient changes
- ✅ After equation application
- ⚠️ DEFERRED during paint strokes and spline drags
- ⚠️ DEFERRED during harmonic slider drags (via transaction)

---

### Spline Mode Normalization

In spline mode, normalization is **always disabled**:

```cpp
// Spline mode: normScalar = 1.0 (identity, no scaling)
if (splineLayerEnabled) {
    normalizationScalar.store(1.0, std::memory_order_release);
}
```

**Why Different from Harmonic Mode?**

| Aspect | Harmonic Mode | Spline Mode |
|--------|---------------|-------------|
| **Layer Mixing** | Base + 40 harmonics (unpredictable amplitude) | Single spline layer only |
| **Amplitude Control** | Automatic (normalization required) | Manual (user places anchors) |
| **Visual Feedback** | Indirect (normalized after edit) | Direct (WYSIWYG) |
| **Normalization** | Auto-normalize to prevent clipping | No normalization (user controls amplitude) |

**Design Rationale**:
- Harmonic mode mixes multiple layers → unpredictable total amplitude → auto-normalize
- Spline mode is single-layer with direct visual feedback → user has full amplitude control → no normalization

**User Control**:
- Spline mode users control amplitude by placing anchor Y-positions
- If curve exceeds [-1, 1], it will clip (intentional creative effect)
- Visualizer shows actual amplitude, so user can see and adjust clipping

---

## Harmonic Layer Details

### Precomputed Basis Functions

For performance, harmonic basis functions are precomputed using Chebyshev polynomials:

```cpp
// Precompute at construction or when table size changes
void precomputeHarmonics() {
    precomputedHarmonics.resize(40);  // Harmonics 1-40

    for (int h = 0; h < 40; ++h) {
        precomputedHarmonics[h].resize(tableSize);

        for (int i = 0; i < tableSize; ++i) {
            double x = normalizeIndex(i);  // [-1, 1]
            int harmonicNumber = h + 1;    // Harmonic number (1-40)

            // Odd harmonics: sin(n * asin(x)) - Chebyshev U polynomials
            // Even harmonics: cos(n * acos(x)) - Chebyshev T polynomials
            if (harmonicNumber % 2 == 1) {
                precomputedHarmonics[h][i] = std::sin(harmonicNumber * std::asin(x));
            } else {
                precomputedHarmonics[h][i] = std::cos(harmonicNumber * std::acos(x));
            }
        }
    }
}
```

### Evaluation

```cpp
double HarmonicLayer::evaluate(double x, const std::vector<double>& coeffs) {
    double result = 0.0;

    // coeffs[0] is wavetable mix (not used here)
    // coeffs[1..40] are harmonic amplitudes

    for (int h = 0; h < 40; ++h) {
        int tableIndex = xToTableIndex(x);
        result += coeffs[h + 1] * precomputedHarmonics[h][tableIndex];
    }

    return result;
}
```

**Why Chebyshev Polynomials?**
- More efficient harmonic content generation compared to pure sine waves
- Better numerical stability for high-order harmonics
- Matches common waveshaping transfer function shapes

---

## Coordinate Systems

### Index Space → X Space

```cpp
double normalizeIndex(int index) {
    // Maps [0, tableSize-1] → [-1, 1]
    return (2.0 * index / (tableSize - 1)) - 1.0;
}
```

### X Space → Index Space

```cpp
int xToTableIndex(double x) {
    // Maps [-1, 1] → [0, tableSize-1]
    return static_cast<int>(((x + 1.0) / 2.0) * (tableSize - 1));
}
```

### Audio Input → Index

```cpp
int inputToTableIndex(float audioInput) {
    // Clamp input to [-1, 1]
    float clamped = std::clamp(audioInput, -1.0f, 1.0f);

    // Map to table index
    return xToTableIndex(static_cast<double>(clamped));
}
```

---

## Interpolation Modes

The composite table uses interpolation for sub-sample accuracy:

### Linear Interpolation (Default)

```cpp
double evaluateLinear(double x) {
    double scaledIndex = ((x + 1.0) / 2.0) * (tableSize - 1);
    int index0 = static_cast<int>(std::floor(scaledIndex));
    int index1 = std::min(index0 + 1, tableSize - 1);

    double frac = scaledIndex - index0;
    double y0 = compositeTable[index0].load(std::memory_order_acquire);
    double y1 = compositeTable[index1].load(std::memory_order_acquire);

    return y0 + frac * (y1 - y0);
}
```

### Cubic Interpolation

```cpp
double evaluateCubic(double x) {
    double scaledIndex = ((x + 1.0) / 2.0) * (tableSize - 1);
    int index1 = static_cast<int>(std::floor(scaledIndex));
    int index0 = std::max(index1 - 1, 0);
    int index2 = std::min(index1 + 1, tableSize - 1);
    int index3 = std::min(index1 + 2, tableSize - 1);

    double frac = scaledIndex - index1;

    double y0 = compositeTable[index0].load(std::memory_order_acquire);
    double y1 = compositeTable[index1].load(std::memory_order_acquire);
    double y2 = compositeTable[index2].load(std::memory_order_acquire);
    double y3 = compositeTable[index3].load(std::memory_order_acquire);

    // Catmull-Rom cubic interpolation
    return cubicInterpolate(y0, y1, y2, y3, frac);
}
```

---

## API Reference

### LayeredTransferFunction

**File**: [modules/dsp-core/dsp_core/Source/LayeredTransferFunction.h](../../modules/dsp-core/dsp_core/Source/LayeredTransferFunction.h)

#### Read Methods (Thread-Safe)

```cpp
double getBaseLayerValue(int index) const;
double getCompositeValue(int index) const;
double evaluate(double x) const;  // Interpolated composite
int getTableSize() const;
double normalizeIndex(int index) const;
```

#### Write Methods (UI Thread Only)

```cpp
void setBaseLayerValue(int index, double value);
void setCoefficient(int index, double value);  // index 0 = WT mix, 1..40 = harmonics
void updateNormalizationScalar();  // Explicit scalar computation
void setPaintStrokeActive(bool active);  // Freeze/unfreeze normalization
```

#### Utility Methods

```cpp
void copyFrom(const LayeredTransferFunction& other);
void reset();  // Reset to linear
std::vector<std::byte> serialize() const;
void deserialize(const std::vector<std::byte>& data);
```

---

## Usage Patterns

### Drawing to Base Layer

```cpp
// Begin paint stroke
ltf.updateNormalizationScalar();  // Cache current scalar
ltf.setPaintStrokeActive(true);   // Freeze normalization

// Paint multiple points
for (auto point : paintPath) {
    int index = xToTableIndex(point.x);
    double value = solveForBaseValue(point.y, harmonics);
    ltf.setBaseLayerValue(index, value);
}

// End paint stroke
ltf.setPaintStrokeActive(false);  // Unfreeze (renderer recomputes at 25Hz)
```

### Modifying Harmonics

```cpp
// Begin transaction
controller.beginTransaction("Adjust Harmonic 5");

// Modify coefficient
std::vector<double> newCoeffs = currentCoeffs;
newCoeffs[5] = sliderValue;

// Apply via command
controller.applyHarmonics(currentCoeffs, newCoeffs);

// End transaction
controller.endTransaction();
```

### Applying Mathematical Function

```cpp
// Create function
auto func = [](double x) { return std::sin(x * M_PI); };

// Apply to entire base layer
controller.applyExpression("sin(x * pi)");  // Internally uses ApplyFunctionCommand
```

---

## Mode Transitions and Layer Baking (CRITICAL)

### The Mode Transition Contract

When switching between editing modes (Harmonic ↔ Spline ↔ Paint), the mode coordinator enforces a strict **exit-then-entry** contract to maintain data integrity across the layered architecture:

```cpp
// ToolCoordinator::setEditingMode() enforces this contract
void setEditingMode(EditingMode newMode) {
    // 1. MODE EXIT: Bake current mode state to base layer
    if (oldMode == EditingMode::Harmonic && newMode != EditingMode::Harmonic) {
        controller.bakeHarmonicsToBase();  // Normalize + write to base layer
    }

    // 2. MODE TRANSITION: Deactivate old mode, activate new mode
    transitionToMode(newMode);

    // 3. MODE ENTRY: New mode reads baked base layer
    // (e.g., SplineTool::activate() fits anchors to base layer)
}
```

### Why This Contract Exists

The layered architecture has **mutually exclusive rendering paths**:

| Mode | Active Layer | Rendering Path | Base Layer Role |
|------|--------------|----------------|-----------------|
| **Paint** | Base only | Direct base read | User draws here |
| **Harmonic** | Base + Harmonic | `normalize(wt*base + harmonics)` | Mixed with harmonics |
| **Spline** | Spline only | Direct spline evaluation | Source for fitting |

**Problem**: Harmonic mode modifies the visual output via the harmonic layer WITHOUT modifying the base layer. When switching to Spline mode:
- ❌ **Without baking**: Spline fits to the base layer (which still contains the original pre-harmonic curve)
- ✅ **With baking**: Spline fits to the normalized composite (base + harmonics), capturing the user's intended curve

### Baking Methods

The `LayeredTransferFunction` provides two baking methods:

#### 1. `bakeHarmonicsToBase()`

**Purpose**: Normalize harmonic composite, then bake to base layer
**Used By**: Harmonic mode exit
**Thread**: UI thread only

```cpp
void LayeredTransferFunction::bakeHarmonicsToBase() {
    // Step 1: Update normalization scalar (scan full table)
    updateNormalizationScalar();

    // Step 2: Bake normalized composite to base
    for (int i = 0; i < tableSize; ++i) {
        double x = normalizeIndex(i);
        double baseValue = baseTable[i].load(std::memory_order_acquire);
        double harmonicValue = harmonicLayer.evaluate(x, coefficients);

        // Apply normalization
        double composite = normalizationScalar * (coeffs[0] * baseValue + harmonicValue);

        // Write back to base layer
        baseTable[i].store(composite, std::memory_order_release);
    }

    // Step 3: Reset harmonic coefficients to identity
    coefficients[0] = 1.0;  // WT mix = 100%
    for (int h = 1; h <= 40; ++h) {
        coefficients[h] = 0.0;  // Zero all harmonics
    }
}
```

**Key Steps**:
1. **Normalize first**: Compute normalization scalar via `updateNormalizationScalar()`
2. **Bake normalized composite**: Write `normScalar * (wt*base + harmonics)` to base layer
3. **Reset harmonics**: Set WT=1.0, H1-H40=0.0 (identity state)

#### 2. `bakeCompositeToBase()`

**Purpose**: Generic baking (used by other modes)
**Used By**: Equation mode, Draw mode (when needed)
**Thread**: UI thread only

```cpp
void LayeredTransferFunction::bakeCompositeToBase() {
    for (int i = 0; i < tableSize; ++i) {
        double composite = compositeTable[i].load(std::memory_order_acquire);
        baseTable[i].store(composite, std::memory_order_release);
    }
}
```

### Critical Fix #22: Double-Fit Bug (2025-12-03)

**Symptom**: Entering Spline mode from Harmonic mode produced incorrect spline fit on first entry, correct fit on second entry.

**Root Cause**: Duplicate fitting logic violated the mode transition contract:

1. ✅ **Correct Path**: `ToolCoordinator::setEditingMode()` → bake harmonics → `SplineTool::activate()` → fit curve
2. ❌ **Wrong Path**: `CurveEditorController::enterSplineToolInternal()` → manual bake + manual fit

**Problem**: The second baking happened AFTER the correct baking, overwriting the base layer with un-normalized harmonics.

**Fix**: Gutted `enterSplineToolInternal()` to only set model flags:

```cpp
// BEFORE (lines 770-887): Manual baking + fitting
void CurveEditorController::enterSplineToolInternal(...) {
    // ... 100+ lines of duplicate baking and fitting logic ...
}

// AFTER (lines 770-809): Flag-setting only
void CurveEditorController::enterSplineToolInternal(...) {
    // Enable spline layer (sets RenderingMode::Spline)
    layeredTransferFunction.setSplineLayerEnabled(true);

    // Notify UI listeners
    if (onSplineLayerStateChanged) {
        onSplineLayerStateChanged(true);
    }
}
```

**Lesson**: Enforce separation of concerns:
- **ToolCoordinator**: Enforces mode-exit baking contract
- **Mode::activate()**: Performs mode-entry setup (fitting, UI initialization)
- **Controller internal methods**: Only set model flags, no side effects

**See Also**: [seamless-transfer-function.md](seamless-transfer-function.md#mode-transition-contract-critical) for detailed mode transition architecture and Critical Fix #22 documentation.

---

## Performance Characteristics

### Table Size Trade-offs

| Size | Memory | Interpolation Quality | Update Cost |
|------|--------|----------------------|-------------|
| 256 | 4KB | Low (visible steps) | Fast (tests) |
| 4096 | 64KB | Medium | Medium |
| 16384 | 256KB | High (smooth) | Slower (production) |

**Production Uses**: 16384 points (defined in `PluginProcessor::tableSize`)
**Tests Use**: 256 points (faster test execution)

### Atomic Read Performance

- **16384 atomic reads** = ~0.5-1μs on modern CPUs (cache-resident)
- **Visualizer optimization**: Sample 2048 points instead of 16384 (87.5% reduction)
- **Audio thread**: Full 16384 resolution maintained for quality

---

## Related Documentation

- [mvc-patterns.md](../../../docs/architecture/mvc-patterns.md) - Controller and command patterns
- [odd-symmetry-editor-option.md](odd-symmetry-editor-option.md) - Harmonic filtering (zeroing even harmonics for odd symmetry)
- [services.md](../../../docs/architecture/services.md) - BaseLayerSolver for inverse solving
- [dsp-processing.md](../../../docs/architecture/dsp-processing.md) - Audio thread usage patterns
- [state-management.md](../../../docs/architecture/state-management.md) - Plugin state serialization (ValueTree format)
- [preset-management.md](preset-management.md) - Preset save/restore workflows
- [spline-curve-fitting.md](spline-curve-fitting.md) - Spline layer curve fitting algorithm
- [CLAUDE.md](../../CLAUDE.md) - Thread safety guidelines

---

## File Locations

| Component | File Path |
|-----------|-----------|
| LayeredTransferFunction | [modules/dsp-core/dsp_core/Source/LayeredTransferFunction.h](../../modules/dsp-core/dsp_core/Source/LayeredTransferFunction.h) |
| LayeredTransferFunction | [modules/dsp-core/dsp_core/Source/LayeredTransferFunction.cpp](../../modules/dsp-core/dsp_core/Source/LayeredTransferFunction.cpp) |
| SplineLayer | [modules/dsp-core/dsp_core/Source/SplineLayer.h](../../modules/dsp-core/dsp_core/Source/SplineLayer.h) |
| SplineLayer | [modules/dsp-core/dsp_core/Source/SplineLayer.cpp](../../modules/dsp-core/dsp_core/Source/SplineLayer.cpp) |
| SplineAnchor | [modules/dsp-core/dsp_core/Source/SplineTypes.h](../../modules/dsp-core/dsp_core/Source/SplineTypes.h) |
| BaseLayerSolver (service) | [modules/transfer_function_editor/transfer_function_editor/Source/Services/BaseLayerSolver.h](../../modules/transfer_function_editor/transfer_function_editor/Source/Services/BaseLayerSolver.h) |
| PluginProcessor (usage) | [plugin/source/PluginProcessor.cpp](../../plugin/source/PluginProcessor.cpp) |
