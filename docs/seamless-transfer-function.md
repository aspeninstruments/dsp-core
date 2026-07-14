# Seamless Transfer Function Architecture

## Overview

Enables glitch-free audio processing during transfer function edits by decoupling UI mutations from audio rendering through triple-buffered lookup tables (LUTs) and automatic crossfading.

**Key Achievement**: Users can paint/modify transfer functions in real-time without audio clicks, pops, or dropouts.

## The Problem

Transfer function updates require milliseconds (spline evaluation, 40-harmonic synthesis, normalization). The audio thread has only microseconds per buffer. Direct updates cause dropouts.

## The Solution

**Three-thread architecture** with lock-free communication and **event-driven rendering**:

```
                     SeamlessTransferFunction (self-contained)
                    ┌─────────────────────────────────────────┐
                    │                                         │
    EventDrivenRenderer (Worker)          VisualizerUpdateDispatcher (Message)
    (High-priority thread)                 (Rate-limited 60Hz)
    │                                      │
    ├─→ Version changed?                   ├─→ Triggered by LaneMixer version
    │   (via requestRender or timer)       │   or triggerAsyncUpdate
    │                                      ├─→ Rate-gate: 60Hz (16.7ms)
    ├─→ Rate-gate: 120Hz (8.33ms)         │
    │                                      └─→ Downsample 16K→1K
    ├─→ Render DSP LUT (16K samples)      └─→ visualizerCallback (paint)
    │   (~5-15ms)
    │
    └─→ Write to AudioEngine triple-buffer
        └─→ newLUTReady.store(true)
                       │
                Audio Thread
                   └─→ Check flag, rotate buffers
                       └─→ 5ms crossfade
```

**Key Design: Event-Driven with Rate Gates**
- **EventDrivenRenderer (Worker, 120Hz rate gate)**: Direct version-tracking renders, worker wakes via requestRender() or fallback timers
- **VisualizerUpdateDispatcher (Message, 60Hz rate gate)**: Async updates with downsampling from DSP LUT or direct LaneMixer reads

**Benefits**:
- Audio thread never blocks (wait-free reads)
- UI stays responsive (60Hz visualizer, decoupled from DSP)
- DSP renders at 120Hz rate gate (varies by version changes, save CPU via rate limiting)
- Guaranteed final delivery via event-driven wake pattern
- Crossfades mask discontinuities
- Lock-free communication via atomics

### Configuration Constants (SeamlessConfig)

**File**: [`modules/dsp-core/dsp_core/Source/SeamlessTransferFunctionImpl.h`](../../modules/dsp-core/dsp_core/Source/SeamlessTransferFunctionImpl.h#L24-L31)

All timing configuration is centralized in `SeamlessConfig` struct:

```cpp
struct SeamlessConfig {
    static constexpr int DSP_LUT_SIZE = LaneMixer::TABLE_SIZE;  // 16384 samples (sourced from TABLE_SIZE)
    static constexpr int VISUALIZER_LUT_SIZE = 1024;            // 1K samples for UI rendering
    static constexpr double MIN_VALUE = -1.0;
    static constexpr double MAX_VALUE = 1.0;
    static constexpr double CROSSFADE_DURATION_MS = 5.0;        // 5ms S-curve crossfade
    static constexpr int DSP_TIMER_HZ = 20;                     // Rate gate: 20 Hz (DSP only)
};
```

**Why Event-Driven with Rate Gates?**

| Component | Rate Gate | Workload | Rationale |
|-----------|-----------|----------|-----------|
| **EventDrivenRenderer (Worker)** | 120 Hz (8.33ms) | Render 16K-sample LUT via worker thread | Expensive (~5-15ms), high rate gate ensures responsive modulation, coalesces rapid edits, worker wakes via event not timer |
| **VisualizerUpdateDispatcher (Message)** | 60 Hz (16.7ms) | Downsample 16K→1K via message thread or read from renderer cache | Cheap (~0.5ms), responsive to UI drags, independent from DSP rate, decouples visualizer from DSP wake frequency |

**CPU Efficiency**: Event-driven workers wake only on changes (no idle polling), rate gates prevent overload (120Hz DSP, 60Hz visualizer). Worker applies the actual rate limit, not message-thread timers.

---

## Architecture Components

### 1. AudioEngine (Audio Thread)

**File**: [`modules/dsp-core/dsp_core/Source/SeamlessTransferFunctionImpl.h`](../../modules/dsp-core/dsp_core/Source/SeamlessTransferFunctionImpl.h#L60-L184)

**Triple Buffering**:
```cpp
LUTBuffer lutBuffers[3];

// lutBuffers[primaryIndex]   - Active LUT (currently playing)
// lutBuffers[secondaryIndex] - Previous LUT (during crossfade)
// lutBuffers[workerTargetIndex] - Worker writes here (safe from audio thread)
```

**Why Triple Buffering?**
- Audio reads from `[0, 1]` during crossfade
- Worker writes to `[2]` simultaneously
- No data races - worker never touches crossfade buffers

**Key Methods**:
- `prepareToPlay()` - Calculate sample-rate-adaptive crossfade duration (5ms)
- `processBuffer()` - Unified multi-channel processing with shared crossfade state
- `checkForNewLUT()` - Atomic flag check + buffer rotation (called once per buffer)

**Crossfade**: 5ms S-curve fade using smoothstep (cubic Hermite) interpolation (220 samples @ 44.1kHz). Uses ease-in/ease-out curve with zero derivative at endpoints for perceptually smooth transitions. Duration matches DSP poll rate for seamless updates.

### 2. EventDrivenRenderer Details (see #3 for full description)

### 3. EventDrivenRenderer (Worker Thread, 120Hz rate gate)

**File**: [`modules/dsp-core/dsp_core/Source/SeamlessTransferFunctionImpl.h`](../../modules/dsp-core/dsp_core/Source/SeamlessTransferFunctionImpl.h#L287-L426)

**Event-Driven Worker with Priority Isolation**:
- High-priority worker thread wakes via `requestRender()` (direct WaitableEvent signal)
- No message-thread hop: audio thread, UI thread, or 5Hz fallback timer can wake the worker directly
- Worker owns the 120Hz rate gate via `RENDER_MIN_INTERVAL_MS = 8.33ms` — limits renders to 120Hz max
- Applies version tracking (full + mix versions) for skip-optimization when unchanged

**Worker Algorithm**:
```cpp
void EventDrivenRenderer::run() override {
    while (!threadShouldExit()) {
        // Apply 120 Hz rate gate
        double remainingMs = nextAllowedRenderTimeMs - currentTimeMs;
        if (remainingMs > 0) {
            wait(remainingMs);  // Block until minimum interval elapsed
        }

        renderIfNeeded();       // Check version, then doRender() if changed
        nextAllowedRenderTimeMs = currentTimeMs + RENDER_MIN_INTERVAL_MS;  // 8.33ms
    }
}
```

**Why 120Hz Rate Gate?** 
- Provides 6 samples per cycle for 20Hz LFO targets (above "perceptually smooth" 4-sample floor)
- 5ms crossfade smooths remaining stair-stepping
- Prevents worker from hammering audio buffers during rapid modulation

**Wakeup Paths**:
- `requestRender()`: Audio thread modulation or UI thread drag → direct worker signal
- `handleAsyncUpdate()`: Message thread mutation (fallback forward)
- 5Hz safety timer: Catches edges missed by event path

### 4. VisualizerUpdateDispatcher (Message Thread, 60Hz rate gate)

**File**: [`modules/dsp-core/dsp_core/Source/SeamlessTransferFunctionImpl.h`](../../modules/dsp-core/dsp_core/Source/SeamlessTransferFunctionImpl.h#L428-L528)

**Event-Driven Visualizer with Downsampling** - AsyncUpdater dispatches curve/lane compute on message thread in response to LaneMixer version changes, rate-limited to 60Hz.

**Dispatcher Algorithm (60Hz)**:
```cpp
void VisualizerUpdateDispatcher::runUpdate(bool forceMixerRecompute) {
    uint64_t currentVersion = laneMixer.getVersion();

    if (currentVersion != lastSeenVersion) {
        lastSeenVersion = currentVersion;

        // Two paths:
        // 1. Fast: Read renderer's cached 16K LUT (updated by EventDrivenRenderer)
        // 2. Slow: Recompute 16K directly from LaneMixer (for forceUpdate at init)
        
        if (!forceMixerRecompute && sourceRenderer_) {
            sourceRenderer_->copyLastRenderedSum(sourceBuffer_);
        } else {
            laneMixer.computeFullSum(sourceBuffer_);  // Full 16K compute
        }

        // Downsample 16K → 1K for UI rendering
        downsampleForVisualization();
        
        if (onVisualizerUpdate) {
            onVisualizerUpdate();  // Triggers UI repaint
        }
    }
}
```

**Why 60Hz Rate Gate?**
- Decoupled from DSP render rate (120Hz)
- 16.7ms is one frame at 60Hz — below human discrimination for continuous knob drag
- Reduces message-thread queue depth and prevents starving DSP renders on slow hardware
- Reads renderer's cached buffer instead of recomputing (if available)

### 5. Version Tracking System

**File**: [`modules/dsp-core/dsp_core/Source/LayeredTransferFunction.h`](../../modules/dsp-core/dsp_core/Source/LayeredTransferFunction.h)

LayeredTransferFunction uses atomic version counter that increments on every mutation:

```cpp
private:
    std::atomic<uint64_t> versionCounter{0};

public:
    uint64_t getVersion() const {
        return versionCounter.load(std::memory_order_acquire);
    }
```

**Critical**: Always use wrapper methods (`setSplineAnchors()`, etc.) instead of direct layer access - bypasses version tracking.

**Version Counter Properties**:
- NOT serialized (runtime-only dirty tracking)
- Atomic (safe to read from timer callbacks while UI mutates editing model)
- Acquire/Release ordering (ensures mutations visible to timer callbacks)

### 5. Normalization Architecture

**File**: [`modules/dsp-core/dsp_core/Source/LayeredTransferFunction.h`](../../modules/dsp-core/dsp_core/Source/LayeredTransferFunction.h)

**Core Principle**: Normalization is the **renderer's responsibility**, computed at max 20Hz (DSP) or 60Hz (visualizer). UI code never recomputes normalization on-the-fly.

**Cached Scalar Pattern**:
```cpp
// Cached normalization scalar (computed explicitly, not on-the-fly)
mutable std::atomic<double> normalizationScalar{1.0};

// Explicit computation (call before baking or at paint stroke start)
void updateNormalizationScalar();

// Composite evaluation uses cached scalar (O(1), not O(n))
double computeCompositeAt(int index) const {
    // ... compute unnormalized value ...
    const double normScalar = normalizationScalar.load(std::memory_order_acquire);
    return normScalar * unnormalized;
}
```

**Why Explicit Caching?**
- **Performance**: Eliminates O(n²) bug where equation mode rendered 16K points, each calling `computeCompositeAt()` which scanned 16K table = 268M iterations
- **Simplicity**: Controller no longer needs complex defer normalization management
- **Correctness**: Renderer recomputes normalization at 20Hz (DSP) or 60Hz (visualizer), UI uses frozen scalar during interactive edits

**Paint Stroke Freezing**:
```cpp
// Controller pattern for paint strokes
void beginPaintStrokeDirect() {
    layeredTransferFunction.updateNormalizationScalar();  // Cache current scalar
    layeredTransferFunction.setPaintStrokeActive(true);   // Freeze it
    // ... paint operations use frozen scalar ...
}

void endPaintStrokeDirect() {
    layeredTransferFunction.setPaintStrokeActive(false);  // Unfreeze
    // Renderer will recompute normalization at next 20Hz DSP poll
}
```

**Baking Operations**:
```cpp
bool LayeredTransferFunction::bakeHarmonicsToBase() {
    // Step 1: Compute normalization BEFORE baking (captures visual state)
    updateNormalizationScalar();

    // Step 2: Bake composite (uses cached scalar)
    for (int i = 0; i < tableSize; ++i) {
        const double compositeValue = computeCompositeAt(i);  // Uses cached scalar
        setBaseLayerValue(i, compositeValue);
    }

    // Step 3-4: Zero harmonics, set WT mix to 1.0
    // ...

    // Step 5: Recalculate normalization (base now contains normalized values)
    updateNormalizationScalar();  // Scalar adapts to new state (~1.0)

    return true;
}
```

**Key Insight**: Baking preserves visual continuity by computing normalization BEFORE capturing composite values, then recalculating AFTER to adapt to the new base layer state.

**API Changes (Refactor 2025-12-03)**:
- ❌ **Removed**: `setDeferNormalization()`, `isNormalizationDeferred()` - complex defer pattern
- ✅ **Added**: `updateNormalizationScalar()` - explicit computation
- ✅ **Added**: `setPaintStrokeActive()`, `isPaintStrokeActive()` - cleaner freezing

### 6. Rendering Mode Evaluation Paths

**File**: [`modules/dsp-core/dsp_core/Source/LayeredTransferFunction.cpp`](../../modules/dsp-core/dsp_core/Source/LayeredTransferFunction.cpp)

**Critical Invariant**: Each RenderingMode has specific assumptions about data state and uses an optimized evaluation path.

#### Paint Mode → Direct Base Read (NO normalization)

```cpp
case RenderingMode::Paint:
    // Direct base layer output - NO normalization, NO harmonics
    // Invariant: Harmonics should be baked into base (wtCoeff = 1.0, all harmonics = 0)
    return getBaseValueAt(x);  // Single table lookup
```

**Assumptions**:
- Harmonics are baked into base layer (wtCoeff = 1.0, all harmonics = 0)
- Base layer already contains normalized values from previous mode exit

**Why no normalization**: Base already contains normalized values from the previous mode's exit baking operation. Scanning 16K values would only find max ≈ 1.0 (wasted work).

**Performance**: ~10-15% faster than full composite path due to skipping wtCoeff multiplication and normalization.

**Contract**: Previous mode MUST bake on exit (enforced by ToolCoordinator).

#### Harmonic Mode → Base + Harmonics (WITH normalization)

```cpp
case RenderingMode::Harmonic:
{
    const double wtCoeff = coefficients[0];
    const double baseValue = getBaseValueAt(x);

    // OPTIMIZATION: Early-exit if all harmonics are zero
    if (!hasNonZeroHarmonics()) {
        const double unnormalized = wtCoeff * baseValue;
        return normScalar * unnormalized;  // Base only
    }

    const double harmonicValue = harmonicLayer->evaluate(x, coefficients, tableSize);
    const double unnormalized = wtCoeff * baseValue + harmonicValue;
    return normScalar * unnormalized;
}
```

**Assumptions**:
- Base and harmonics are independent layers that need mixing
- Harmonic synthesis can produce values exceeding [-1, 1] bounds

**Why normalization**: Harmonic synthesis (sum of 40 sine waves) can easily exceed [-1, 1], requiring normalization to prevent clipping.

**Optimization**: Early-exit when `hasNonZeroHarmonics()` returns false. This skips 40 coefficient loads + sin/cos evaluation, making the common case (wtCoeff-only transitions) as fast as Paint mode.

**Contract**: Mode exit MUST bake harmonics to base for the next mode.

#### Spline Mode → Direct Spline (NO normalization)

```cpp
case RenderingMode::Spline:
    // Direct spline evaluation - NO normalization
    // Splines are UI-clamped to [-1, 1] during editing
    return splineLayer->evaluate(x);  // Catmull-Rom interpolation
```

**Assumptions**:
- Spline anchors are UI-clamped to [-1, 1] during editing
- Catmull-Rom interpolation preserves bounds (no overshoot beyond anchor values)

**Why no normalization**: Splines are already constrained to [-1, 1] by the UI. Scanning 16K spline evaluations would only find max ≈ 1.0, wasting CPU cycles for no benefit.

**Performance**: Eliminates unnecessary 16K scan on every spline change (20Hz DSP polling would trigger expensive normalization constantly).

**Contract**: Mode exit MUST bake spline to base WITHOUT normalization for the next mode.

#### Why This Matters

**Breaking these contracts causes**:
- **Paint mode reading un-baked harmonics**: Incorrect audio output (harmonics not applied)
- **Unnecessary normalization scans**: Performance regression (16K iterations at 20Hz = 320K ops/sec wasted)
- **Visual discontinuities**: Baking with wrong normalization scalar causes jumps in visualizer
- **Double-fit bug**: Fitting the base layer twice causes wrong curve to be fitted (see Critical Fix #22 below)

**When adding a new mode**: Understand which evaluation path to use and what state to leave for the next mode. The mode-exit contract is the glue that makes this architecture work.

---

### 8. Mode Transition Contract (CRITICAL)

**File**: [`modules/transfer_function_editor/transfer_function_editor/Source/ToolCoordinator.cpp`](../../modules/transfer_function_editor/transfer_function_editor/Source/ToolCoordinator.cpp)

The seamless architecture depends on a **strict mode transition contract** enforced by ToolCoordinator:

```cpp
void ToolCoordinator::setEditingTool(EditingTool newMode) {
    // 1. MODE EXIT: Bake current mode to base layer
    if (oldMode == EditingTool::Harmonic && newMode != EditingTool::Harmonic) {
        controller.bakeHarmonicsToBase();  // Normalize + write to base
    }

    // 2. MODE TRANSITION: Deactivate old, activate new
    transitionToMode(newMode);
        → deactivateEditingTool(oldMode)   // Cleanup callbacks
        → activateEditingTool(newMode)      // Setup callbacks, trigger fit
}
```

**The Contract**:

1. **Mode EXIT** (ToolCoordinator responsibility):
   - Previous mode MUST bake its state to base layer
   - Harmonic mode: `bakeHarmonicsToBase()` writes normalized composite → base
   - Spline mode: `exitSplineToolInternal()` writes evaluated spline → base
   - Paint mode: No-op (already in base layer)

2. **Mode ENTRY** (Mode's activate() responsibility):
   - New mode performs initial setup (e.g., SplineTool::activate() fits curve)
   - Assumes base layer contains correct baked curve from previous mode
   - Sets up callbacks, UI state, performs mode-specific initialization

3. **Controller's enterSplineToolInternal()** (Model state only):
   - Sets `splineLayerEnabled = true` (changes RenderingMode)
   - Fires `onSplineLayerStateChanged` callback for UI sync
   - Does NOT bake (already done by mode exit)
   - Does NOT fit (already done by activate())

**Example: Harmonic → Spline Transition**:

```cpp
// State: H3=1.0, WT=0.0, base=y=x

// 1. ToolCoordinator::setEditingTool(Spline) line 58-62
controller.bakeHarmonicsToBase();
   → updateNormalizationScalar()  // Find max of (0.0*y=x + H3)
   → for i: base[i] = normalize(0.0*y=x + H3)  // Write normalized H3 to base
   → Set WT=1.0, zero all harmonics
   → Base layer now contains normalized H3 curve ✅

// 2. ToolCoordinator::transitionToMode(Spline)
deactivateEditingTool(Harmonic);  // Cleanup callbacks
activateEditingTool(Spline);
   → SplineTool::activate()
      → fitCurveToSpline()  // Fit anchors to base layer (normalized H3)
      → Creates beautiful anchor fit ✅

// 3. enterSplineToolInternal() [called by perform()]
setSplineLayerEnabled(true);  // Just sets flag, no baking/fitting
onSplineLayerStateChanged(true);  // UI sync
```

**Why Two Entry Points?**

- `ToolCoordinator::setEditingTool()`: User-initiated mode switch (enforces exit contract)
- `controller.enterSplineTool()`: Undoable wrapper (calls setEditingTool + creates undo entry)
- `enterSplineToolInternal()`: Model state only (called by perform(), no side effects)


## Lifecycle Management

### Construction (Plugin Load)

```cpp
TotalHarmonicControlAudioProcessor::TotalHarmonicControlAudioProcessor() {
    transferFunction = std::make_unique<dsp_core::SeamlessTransferFunction>();
    transferFunction->startSeamlessUpdates();  // Creates timers + worker
}
```

### prepareToPlay() (Audio Start)

```cpp
void prepareToPlay(double sampleRate, int samplesPerBlock) override {
    transferFunction->prepareToPlay(sampleRate, samplesPerBlock);
    // Calculates sample-rate-adaptive crossfade duration (5ms)
}
```

### releaseResources() (Audio Stop)

**⚠️ CRITICAL DAW QUIRK**:

```cpp
void SeamlessTransferFunction::releaseResources() {
    // CRITICAL: Do NOT stop seamless updates here!
    //
    // WHY: DAWs call releaseResources() unpredictably (e.g., when stopping playback,
    // or even during initialization). The seamless update system should stay alive
    // for the plugin's entire lifetime because:
    //
    // 1. Timers run on message thread (UI-related, not audio resources)
    // 2. Worker thread is idle when not rendering (no CPU overhead)
    // 3. Audio engine is always needed (not just during playback)
    //
    // The only time we should stop seamless updates is in the destructor.
}
```

**Key Insight**: Seamless update system is a **"plugin lifetime" resource**, NOT a **"playback session" resource**.

**DAW Behavior**:
- Ableton Live: Calls `releaseResources()` during plugin initialization
- Logic Pro: Calls when stopping playback
- Reaper: Calls when deactivating plugin

### Destructor (Plugin Unload)

```cpp
SeamlessTransferFunction::~SeamlessTransferFunction() {
    stopSeamlessUpdates();  // Only cleanup point!
}
```

---

## Threading Model

### Thread Responsibilities

| Thread | Reads | Writes | Synchronization |
|--------|-------|--------|-----------------|
| **Message** | Editing model (non-atomic) | Editing model, enqueue jobs | Mutex-free (single-threaded) |
| **Worker** | Job queue (lock-free) | `lutBuffers[workerTargetIndex]` | Atomic flags |
| **Audio** | `lutBuffers[0,1]`, atomic indices | Crossfade state (local) | Atomic flags |

### Memory Ordering

```cpp
// Worker thread writes LUT, then signals
newLUTReady.store(true, std::memory_order_release);
   // ↑ Ensures LUT writes visible before flag becomes true

// Audio thread checks flag, then reads LUT
if (newLUTReady.load(std::memory_order_acquire)) {
   // ↓ Ensures LUT reads see worker's writes
   evaluateLUT(&lutBuffers[...], x);
}
```

**Acquire-Release Semantics**: Establish "happens-before" relationship between worker's LUT write and audio's LUT read.

### Critical Invariants

1. **Worker Target Isolation**: Audio thread NEVER reads `lutBuffers[workerTargetIndex]`
2. **Single Writer**: Only worker thread writes to `lutBuffers[workerTargetIndex]`
3. **Message Thread Only**: Timers ONLY run on message thread (JUCE Timer contract)
4. **No Allocations in Audio Thread**: All buffers pre-allocated in `prepareToPlay()`
5. **Atomic Crossfade State**: `crossfading`, `crossfadePosition` are audio-thread-local (mutable)

---

## Integration Guide

### Step 1: Create SeamlessTransferFunction

```cpp
// PluginProcessor.h
class MyAudioProcessor : public juce::AudioProcessor {
private:
    std::unique_ptr<dsp_core::SeamlessTransferFunction> transferFunction;
};

// PluginProcessor.cpp
MyAudioProcessor::MyAudioProcessor() {
    transferFunction = std::make_unique<dsp_core::SeamlessTransferFunction>();

    // CRITICAL: Create controller BEFORE starting seamless updates
    controller = std::make_unique<CurveEditorController>(
        transferFunction->getEditingTooll()
    );

    // Start seamless updates after controller is created
    transferFunction->startSeamlessUpdates();
}
```

### Step 2: Hook into Audio Pipeline

```cpp
void MyAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock) {
    transferFunction->prepareToPlay(sampleRate, samplesPerBlock);
}

void MyAudioProcessor::releaseResources() {
    transferFunction->releaseResources();  // Does nothing (by design!)
}

void MyAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                    juce::MidiBuffer&) {
    // Convert to double, process, convert back
    juce::AudioBuffer<double> doubleBuffer(buffer.getNumChannels(), buffer.getNumSamples());
    // ... conversion ...

    transferFunction->processBuffer(doubleBuffer);  // Unified multi-channel processing

    // ... convert back to float ...
}
```

### Step 3: Set Up Visualizer Callback

```cpp
// PluginEditor.cpp
MyAudioProcessorEditor::MyAudioProcessorEditor(MyAudioProcessor& p)
    : AudioProcessorEditor(&p), processor(p) {

    // Set up visualizer callback (called at 60Hz when model changes)
    processor.getTransferFunction().setVisualizerCallback([this]() {
        if (visualizerComponent != nullptr) {
            visualizerComponent->repaint();
        }
    });
}
```

**Note**: The internal `VisualizerUpdateTimer` (60Hz) and `LUTRenderTimer` (20Hz) handle all version tracking and update scheduling automatically. No Editor polling is required.

### Step 4: Mutate Editing Model (Message Thread Only)

```cpp
void onUserPaint(double x, double y) {
    auto& editingModel = transferFunction->getEditingTooll();
    editingModel.setSplineAnchor(index, {x, y});
    // Version change detected by internal timers:
    // - Visualizer: within ~17ms (60Hz)
    // - DSP: within ~55ms (20Hz: 50ms worst-case + 5ms render)
}
```

---

## Common Pitfalls

### 1. Calling stopSeamlessUpdates() in releaseResources()

```cpp
// ❌ WRONG
void releaseResources() {
    transferFunction->stopSeamlessUpdates();  // Breaks DAW support!
}

// ✅ CORRECT
void releaseResources() {
    transferFunction->releaseResources();  // Does nothing by design
}
```

### 2. Not Setting Visualizer Callback

```cpp
// ❌ WRONG (visualizer never updates)
MyAudioProcessorEditor::MyAudioProcessorEditor(MyAudioProcessor& p) {
    // Forgot to set callback!
}

// ✅ CORRECT (visualizer updates at 60Hz when model changes)
MyAudioProcessorEditor::MyAudioProcessorEditor(MyAudioProcessor& p) {
    processor.getTransferFunction().setVisualizerCallback([this]() {
        visualizerComponent->repaint();
    });
}
```

### 3. Mutating Editing Model from Non-Message Thread

```cpp
// ❌ WRONG (audio thread mutation)
void processBlock(...) {
    transferFunction->getEditingTooll().setCoefficient(0, 0.5);  // CRASH!
}

// ✅ CORRECT (message thread mutation)
void buttonClicked(juce::Button*) override {
    transferFunction->getEditingTooll().setCoefficient(0, 0.5);  // Safe
}
```

### 4. Visualizer Updates - Single Source of Truth

**DO NOT** add separate visualizer update callbacks - causes flicker during rapid edits.

```cpp
// ❌ WRONG (causes flicker)
controller.onSomeCallback = [this]() {
    visualizer->setData(sampleEditingTooll());  // Competes with worker thread!
};

// ✅ CORRECT (single update path)
// Let worker thread LUT updates be the ONLY visualizer update path
// Job coalescing ensures latest state shown with ~30ms latency (imperceptible)
```

**Why critical**: Multiple visualizer update paths compete for control, causing flicker as worker processes stale queued jobs. Trust the two-timer architecture - the 60Hz VisualizerUpdateTimer samples the model directly for the UI, while the 20Hz LUTRenderTimer handles DSP updates with guaranteed final delivery.

---

## Performance Characteristics

**Memory Overhead**: ~967KB
- 3 LUT buffers: 393KB
- Job queue: ~524KB
- Worker LayeredTransferFunction: ~50KB

**CPU Overhead**:
- Polling (Message Thread): <0.1% CPU
- Worker Thread: 0% when idle
- Audio Thread: <1% (dominated by LUT cache misses)

**Latency**:
- UI-to-Visualizer: <17ms (60Hz timer, worst-case ~17ms to detect change)
- UI-to-Audio: <55ms (20Hz timer worst-case 50ms + worker render 5-15ms, crossfade 5ms)
- Audio crossfade: 5ms (seamless update duration)

---

## Historical Notes

For detailed history of critical fixes (stereo crossfade coherence, visualizer flicker, normalization cleanup, mode transition double-fit bug):
- [seamless-transfer-function-critical-fixes.md](../../../docs/archive/seamless-transfer-function-critical-fixes.md)

**Key Lessons**:
- Single source of truth (avoid dual-update systems)
- Explicit caching (normalization belongs in renderer)
- Strict lifecycle contracts (mode entry/exit without duplication)

---

## Related Documentation

- [layered-transfer-function.md](layered-transfer-function.md) - Editing model architecture
- [dsp-processing.md](../../../docs/architecture/dsp-processing.md) - Audio thread safety patterns
- [testing-strategy.md](../../../docs/architecture/testing-strategy.md) - How to test seamless behavior

---

**Key Takeaway**: The seamless transfer function system is a **plugin lifetime resource**, not a **playback session resource**. It must stay alive from plugin load to plugin unload, regardless of DAW playback state.
