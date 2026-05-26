#pragma once

#include "FirstOrderShelfStage.h"
#include "ToneFilterStrategy.h"

namespace dsp_core::audio_pipeline {

/**
 * Hysteresis tone-filter strategy — NAB-style pre/de-emphasis around the
 * Jiles-Atherton saturator.
 *
 * Unlike the buffer-filter strategies (Lowpass, LowShelf, ...), this strategy
 * coexists with the pipeline-level HysteresisStage that wraps the waveshaper:
 * the JA RK4 loop runs there (driven by Tone_Type), and the pre/post ToneStage
 * instances around it host this strategy in two complementary roles:
 *
 *   Role::PreEmphasis   (lives on tonePreStage_, before saturation)
 *     - 1st-order high shelf at 3183 Hz (NAB 50 µs / 15 ips)
 *     - gain = +emph · 12 dB   → pushes the highs harder into the saturator
 *
 *   Role::DeEmphasis    (lives on tonePostStage_, after saturation)
 *     - 1st-order high shelf at 3183 Hz (same corner as pre)
 *     - gain = -emph · 9 dB    → partial restoration (asymmetric by design;
 *                                leaves +3 dB net HF lift at emph=1, which
 *                                reads as "tape that's been pushed but still
 *                                has presence" instead of a perfect null)
 *     - PLUS: fixed +1.5 dB / 50 Hz low shelf (NAB low-frequency bump)
 *             — never scaled by emph, only on the de side so the waveshaper
 *               doesn't see it.
 *
 * Role is set once at plugin construction via setRole(); it does not change
 * at runtime (pre is always pre, post is always post). All other universal
 * setters from ToneFilterStrategy are no-ops; only setEmph() matters.
 */
class HysteresisStrategy : public ToneFilterStrategy {
  public:
    enum class Role { PreEmphasis, DeEmphasis };

    HysteresisStrategy() = default;

    /** Set the pre-vs-de role. Call once at plugin construction BEFORE
     *  prepareToPlay (so the prepareToPlay path can size only the shelves
     *  that this role actually uses). */
    void setRole(Role r) {
        role_ = r;
    }
    Role getRole() const {
        return role_;
    }

    void prepareToPlay(double sampleRate, int samplesPerBlock, int numChannels) override {
        hfShelf_.prepareToPlay(sampleRate, samplesPerBlock, numChannels);
        hfShelf_.setCutoffFrequency(kHfCornerHz);
        applyEmphToHfShelf();
        if (role_ == Role::DeEmphasis) {
            lfShelf_.prepareToPlay(sampleRate, samplesPerBlock, numChannels);
            lfShelf_.setCutoffFrequency(kLfCornerHz);
            lfShelf_.setGainDb(kLfBumpDb);
        }
    }

    void process(juce::AudioBuffer<double>& buffer) override {
        // Strict 0% bypass for the HF shelf: at emph=0 the shelf is at 0 dB,
        // which is mathematically identity, but skipping the IIR entirely
        // saves the per-sample work AND avoids accumulating denormal state
        // when the surrounding signal is silent.
        if (emphIsActive_) {
            hfShelf_.process(buffer);
        }
        // LF bump is fixed at +1.5 dB and only on the de side. It always
        // runs whenever this strategy is the de role — its gain does not
        // depend on emph.
        if (role_ == Role::DeEmphasis) {
            lfShelf_.process(buffer);
        }
    }

    void reset() override {
        hfShelf_.reset();
        if (role_ == Role::DeEmphasis) {
            lfShelf_.reset();
        }
    }

    juce::String getName() const override {
        return role_ == Role::PreEmphasis ? "HysteresisPreEmph" : "HysteresisDeEmph";
    }

    // ToneFilterStrategy universal setters
    void setFrequency(double /*frequencyHz*/) override {}
    void setResonance(double /*zeroToOne*/) override {}
    void setShelfGainDb(double /*gainDb*/) override {}
    void setFat(double /*percent*/) override {}
    void setLowShelfRatio(double /*ratio*/) override {}
    void setQ(double /*q*/) override {}

    void setEmph(double zeroToOne) override {
        emph_ = juce::jlimit(0.0, 1.0, zeroToOne);
        applyEmphToHfShelf();
    }

    /** Read current emph value — used by tests + getter parity with other
     *  strategies (e.g. LowpassStrategy::getFatPercent). */
    double getEmph() const {
        return emph_;
    }

  private:
    void applyEmphToHfShelf() {
        // Asymmetric law (decoupled by design): pre = +emph·12 dB, de = -emph·9 dB.
        // The pre stage drives the highs into the saturator; the de stage only
        // partially restores them, leaving a small net HF lift at high emph that
        // sounds like "tape that's been pushed but still has presence."
        const double db = (role_ == Role::PreEmphasis) ? (+emph_ * kPreEmphMaxDb) : (-emph_ * kDeEmphMaxDb);
        hfShelf_.setGainDb(db);
        emphIsActive_ = (emph_ > 0.0);
    }

    Role role_ = Role::PreEmphasis;
    double emph_ = 0.5;
    bool emphIsActive_ = true;
    FirstOrderShelfStage hfShelf_{FirstOrderShelfStage::Mode::HighShelf};
    FirstOrderShelfStage lfShelf_{FirstOrderShelfStage::Mode::LowShelf};

    // NAB 15 ips / 50 µs HF emphasis breakpoint. NAB also specifies the
    // low-frequency 3180 µs / 50 Hz bump on the output (de) side only.
    static constexpr double kHfCornerHz = 3183.0;
    static constexpr double kLfCornerHz = 50.0;
    static constexpr double kLfBumpDb = 1.5;

    // Emphasis law: pre boosts up to +12 dB, de cuts up to -9 dB. Net at full
    // emph: +12 + (-9) = +3 dB residual HF lift after pre and de cascade.
    static constexpr double kPreEmphMaxDb = 12.0;
    static constexpr double kDeEmphMaxDb = 9.0;
};

} // namespace dsp_core::audio_pipeline
