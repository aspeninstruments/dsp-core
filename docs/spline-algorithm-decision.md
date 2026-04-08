# Spline Algorithm Decision: Fritsch-Carlson Only

**Status**: 📚 Reference - Stable decision document
**Date:** 2025-11-01
**Decision:** Use Fritsch-Carlson as the sole tangent algorithm for spline curve fitting

---

## Context

The spline curve fitting feature requires a tangent computation algorithm to generate smooth cubic Hermite splines from user-drawn transfer functions. We evaluated two primary candidates:

1. **Akima** - Local weighted average (prioritizes visual smoothness)
2. **Fritsch-Carlson** - Monotone-preserving with α²+β²≤9 constraint (guarantees no overshoot)

---

## Decision

**We chose Fritsch-Carlson as the only algorithm** and removed the UI selector.

### Key Reasons

1. **No-Overshoot Guarantee Simplifies UI**
   - Fritsch-Carlson's α²+β²≤9 constraint mathematically guarantees that fitted curves won't overshoot beyond anchor points
   - This makes anchor placement predictable and intuitive for users
   - Users can directly place anchors knowing the curve will stay within bounds

2. **Performance Is Identical**
   - Benchmark results: Akima 270.4ms, Fritsch-Carlson 268.9ms (0.5% difference = noise)
   - Both achieve zero spurious extrema with feature-based anchor placement
   - Both fit within musical tolerance (< 1% error)

3. **Simpler UX**
   - Users shouldn't need to understand "tangent algorithm theory"
   - Removing the combo box reduces cognitive load
   - One less decision to make = faster, more intuitive workflow

4. **Feature Detection Solved Akima's Main Weakness**
   - Our `CurveFeatureDetector` anchors at local extrema and inflection points
   - This eliminated Akima's ripple problem (both algorithms now achieve 0 spurious extrema)
   - The 5% smoothness advantage of Akima is negligible in this context

### Why Akima Was Considered

Akima was attractive for its **perceptual smoothness**:
- Rounded peaks sound "warmer" and "more analog-like"
- Fewer high-frequency harmonics due to gentler curvature transitions
- Better for creative/artistic waveshaping where smoothness > accuracy

However, the **no-overshoot guarantee** of Fritsch-Carlson proved more valuable for UI predictability.

---

## Implementation Details

### Default Configuration

All presets now use Fritsch-Carlson ([SplineTypes.h:61](../../modules/dsp-core/dsp_core/Source/SplineTypes.h#L61)):

```cpp
TangentAlgorithm tangentAlgorithm = TangentAlgorithm::FritschCarlson;
```

### Removed UI Components

- Algorithm combo box (`InlineComboBoxFactory::makeAlgorithmCombo()` - preserved but commented for reference)
- Spline mode settings panel (`TransferFunctionPanel::splineModeSettingsPanel`)
- Processor state persistence (`PluginAudioProcessor::currentTangentAlgorithm`)
- Wiring methods (`TransferFunctionPanel::wireSplineToolToProcessor()`)

---

## Pattern: Panel Within Editor Mode (Preserved for Future Reference)

**Use Case:** When an editor mode (like SplineTool) needs its own settings panel that appears/disappears with mode activation.

### Architecture

```
TransferFunctionPanel (owner)
└── SplineTool (child component, overlay on visualizer)
    └── Settings Panel (HorizontalStrip with controls)
        └── Combo Box / Sliders / Buttons
```

### Implementation Pattern

#### 1. Panel Setup (in owning panel)

```cpp
// TransferFunctionPanel.h
private:
    std::unique_ptr<ui_panels::HorizontalStrip> splineModeSettingsPanel;
    ui_core::InlineComboBox* algorithmComboPtr = nullptr;  // Raw pointer for callbacks

// TransferFunctionPanel.cpp constructor
void TransferFunctionPanel::setupSplineToolSettingsPanel()
{
    // Create panel
    splineModeSettingsPanel = std::make_unique<ui_panels::HorizontalStrip>();
    splineModeSettingsPanel->setStyle(HorizontalStripRoles::Dark());

    // Create controls with factories
    auto algorithmCombo = InlineComboBoxFactory::makeAlgorithmCombo();
    algorithmComboPtr = algorithmCombo.get();  // Store raw pointer before moving

    // Wire up callback
    algorithmComboPtr->onChange = [this](int selectedId) {
        // Update mode state via controller
        // Note: Direct mode access (getSplineTool) was removed.
        // Use controller methods for mode-specific operations.
        controller->setSplineAlgorithm(selectedId);
    };

    // Add to panel
    splineModeSettingsPanel->addItem(std::move(algorithmCombo));

    // Panel starts hidden
    addChildComponent(*splineModeSettingsPanel);
}
```

#### 2. Panel Visibility (in owning panel's updateModeComponentsVisibility())

```cpp
void TransferFunctionPanel::updateModeComponentsVisibility()
{
    if (editingMode == ToolCoordinator::EditingTool::Spline)
    {
        // Show panel and sync state
        if (splineModeSettingsPanel)
        {
            splineModeSettingsPanel->setVisible(true);

            // Sync controls with mode state via controller
            // Note: Direct mode access (getSplineTool) was removed.
            // Query state through controller or model instead.
            auto value = controller->getSplineAlgorithm();

            // Temporarily disable callback to avoid recursive updates
            auto originalCallback = algorithmComboPtr->onChange;
            algorithmComboPtr->onChange = nullptr;
            algorithmComboPtr->setSelectedId(value);
            algorithmComboPtr->onChange = originalCallback;
        }
    }
    else
    {
        // Hide panel
        if (splineModeSettingsPanel)
        {
            splineModeSettingsPanel->setVisible(false);
        }
    }
}
```

#### 3. Panel Layout (in owning panel's resized())

```cpp
void TransferFunctionPanel::resized()
{
    auto area = getLocalBounds();

    // ... mode buttons, equation mode, harmonic mode ...

    // Spline settings panel (if visible)
    if (splineModeSettingsPanel && splineModeSettingsPanel->isVisible())
    {
        const int panelHeight = 40;  // Standard HorizontalStrip height
        splineModeSettingsPanel->setBounds(area.removeFromTop(panelHeight));
        area.removeFromTop(margin);  // Spacing
    }

    // ... visualizer fills remaining space ...
}
```

### Critical Patterns

1. **Ownership:** Panel lives in the owning component (TransferFunctionPanel), NOT in the mode
2. **Visibility:** Panel visibility managed by owning component based on active mode
3. **State Sync:** When mode activates, sync control values from mode state
4. **Callback Guards:** Temporarily disable callbacks during programmatic updates to prevent infinite loops
5. **Raw Pointers:** Store raw pointers to controls (after std::move) for callback access

### When to Use

Use this pattern when:
- ✅ Mode needs persistent settings controls (combos, sliders, buttons)
- ✅ Controls should only be visible when mode is active
- ✅ Multiple controls need coordinated layout (HorizontalStrip)
- ✅ Settings are mode-specific, not global

Don't use when:
- ❌ Mode is purely interactive (like PaintTool - no settings needed)
- ❌ Settings are global (use bottom panel or top panel instead)
- ❌ Only one or two buttons (use mode button bar instead)

---

## Benchmarks

See [algorithm-comparison-technical-analysis.md](../algorithm-comparison-technical-analysis.md) for full performance data.

**Summary:**
- **Avg Fitting Time:** Akima 270.4ms, Fritsch-Carlson 268.9ms (0.5% difference)
- **Avg Anchors:** Akima 14.6, Fritsch-Carlson 13.4 (8% fewer)
- **Avg Error:** Akima 8.85e-03, Fritsch-Carlson 6.79e-03 (23% lower)
- **Spurious Extrema:** Both 0 (feature detection success)

---

## References

1. Fritsch, F. N., & Carlson, R. E. (1980). "Monotone Piecewise Cubic Interpolation"
2. [SplineFitter.cpp](../../modules/dsp-core/dsp_core/Source/Services/SplineFitter.cpp) - Implementation
3. [AlgorithmBenchmarkTest.cpp](../../modules/dsp-core/tests/AlgorithmBenchmarkTest.cpp) - Benchmarks
4. [algorithm-comparison-technical-analysis.md](../algorithm-comparison-technical-analysis.md) - Full analysis

---

## Related Architecture Docs

- [mvc-patterns.md](../../../docs/architecture/mvc-patterns.md) - Mode system architecture
- [ui-design-system.md](ui-design-system.md) - Panel and control patterns
- [spline-layer-current-state.md](spline-layer-current-state.md) - Spline layer technical details
