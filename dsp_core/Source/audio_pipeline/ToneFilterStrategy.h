#pragma once

#include "AudioProcessingStage.h"

namespace dsp_core::audio_pipeline {

/**
 * Strategy interface for tone-stage filter families.
 *
 * ToneStage owns one concrete instance of each strategy (LP12, LP24, LowShelf,
 * ...) and dispatches to the active one based on the user-selected Type. Every
 * strategy implements the same universal setter surface; each strategy ignores
 * the parameters it does not use. This keeps PluginProcessor::processBlock free
 * of per-type branching, and lets idle strategies stay primed with the current
 * param values so type-switches that share params (e.g. LP12 <-> LP24 with the
 * same cutoff/resonance) are click-free.
 *
 * Adding a new filter type (e.g. HighShelf, Peak):
 *   1. Create a new <Name>Strategy.h that inherits ToneFilterStrategy.
 *   2. Forward the relevant setters to the underlying DSP and make the rest no-ops.
 *   3. Add a value to ToneStage::Type (append-only; never reorder).
 *   4. Add an instance member of the new strategy to ToneStage.
 *   5. Add a case to ToneStage::activeStrategy() returning that member.
 *   6. Add the prepare call in ToneStage::prepareToPlay.
 *   7. (Plugin) Append the type name to the Tone_Type choice array.
 *   8. (UI) Add a visibility-toggled sub-row to ToneOptionsPanel and, if the
 *      new type changes the hover-dial role, extend rebindTonePrimary().
 *
 * Exception — wrap-around-the-waveshaper effects: HysteresisStrategy is a
 * partial example. The Jiles-Atherton magnetisation loop itself runs at the
 * pipeline-level HysteresisStage (driven by Tone_Type), but the pre/post tone
 * stages around it host NAB-style pre-emphasis (before saturation) and
 * de-emphasis (after saturation) shelves. Each ToneStage instance is told its
 * role (PreEmphasis / DeEmphasis) at construction; the two share the same
 * Tone_Type but their hysteresis_ strategies process differently. See
 * HysteresisStrategy::Role and ToneStage::setHysteresisRole.
 *
 * Thread safety:
 *   - process(), reset() run on the audio thread.
 *   - Setters run on the UI thread; they may allocate (RBJ shelf coefficient
 *     updates do). Strategies must not allocate in process().
 */
class ToneFilterStrategy : public AudioProcessingStage {
  public:
    // AudioProcessingStage interface — concrete strategies must implement.
    void prepareToPlay(double sampleRate, int samplesPerBlock, int numChannels) override = 0;
    void process(juce::AudioBuffer<double>& buffer) override = 0;
    void reset() override = 0;

    /** Filter corner frequency in Hz (LP cutoff or LS frequency). Strategies
     *  that don't have a corner frequency should make this a no-op. */
    virtual void setFrequency(double frequencyHz) = 0;

    /** Resonance / Q as a normalized [0, 1] dial value. LP-only; other
     *  strategies should make this a no-op. */
    virtual void setResonance(double zeroToOne) = 0;

    /** Shelf gain in dB. LS-only; LP strategies should make this a no-op
     *  (Fat has its own knob — do NOT route shelf gain to it). */
    virtual void setShelfGainDb(double gainDb) = 0;

    /** "Fat" percent (LP-internal bass restoration). LP-only; other
     *  strategies should make this a no-op. */
    virtual void setFat(double percent) = 0;

    /** Low-shelf-to-high-shelf frequency ratio used by the Smile strategy
     *  (LS Hz = HS Hz × ratio). Smile-only; other strategies should make
     *  this a no-op. */
    virtual void setLowShelfRatio(double ratio) = 0;

    /** Bell Q (bandwidth). Bell-only; other strategies should make this
     *  a no-op. Lower Q = broader bell; higher Q = narrower bell. */
    virtual void setQ(double q) = 0;

    /** Hysteresis-emphasis macro [0, 1]. Hysteresis-only; other strategies
     *  should make this a no-op. Drives the pre-/de-emphasis high-shelf gain
     *  around the saturator (see HysteresisStrategy::Role). */
    virtual void setEmph(double zeroToOne) = 0;
};

} // namespace dsp_core::audio_pipeline
