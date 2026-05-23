#include "VirtualAnalogFilterStage.h"
#include <algorithm>
#include <juce_core/juce_core.h>

namespace dsp_core::audio_pipeline {

namespace {
constexpr double kVaMinCutoffHz = 20.0;
constexpr double kVaNyquistMargin = 0.45;
constexpr double kVaMinResonance = 0.0;
constexpr double kVaMaxResonance = 4.0;
constexpr double kVaSmoothingTimeSec = 0.0002;
constexpr double kVaSixth = 1.0 / 6.0;

double vaMaxCutoffForSampleRate(double sampleRate) {
    return std::max(kVaMinCutoffHz, sampleRate * kVaNyquistMargin);
}
} // namespace

VirtualAnalogFilterStage::VirtualAnalogFilterStage() {
    setUniform(StageType::Ladder);
}

void VirtualAnalogFilterStage::setStage(int idx, std::unique_ptr<FilterStage> stage) {
    if (idx < 0 || idx >= kMaxStages) {
        return;
    }
    stages_[static_cast<std::size_t>(idx)] = std::move(stage);
}

void VirtualAnalogFilterStage::setUniform(StageType type) {
    for (int i = 0; i < kMaxStages; ++i) {
        if (type == StageType::Ladder) {
            stages_[static_cast<std::size_t>(i)] = std::make_unique<LadderFilterStage>();
        } else {
            stages_[static_cast<std::size_t>(i)] = std::make_unique<OTAFilterStage>();
        }
    }
}

void VirtualAnalogFilterStage::setNumStages(int n) {
    numStages_ = juce::jlimit(1, kMaxStages, n);
}

void VirtualAnalogFilterStage::prepareToPlay(double sampleRate, int /*samplesPerBlock*/,
                                             int numChannels) {
    sampleRate_ = sampleRate;
    T_ = 1.0 / sampleRate;

    const int ch = std::max(1, numChannels);
    channels_.assign(static_cast<std::size_t>(ch), ChannelState{});

    const double maxCutoff = vaMaxCutoffForSampleRate(sampleRate_);
    const double clampedCutoff =
        juce::jlimit(kVaMinCutoffHz, maxCutoff, cutoffHz_.load(std::memory_order_acquire));
    cutoffHz_.store(clampedCutoff, std::memory_order_release);

    smoothCutoff_.reset(sampleRate, kVaSmoothingTimeSec);
    smoothResonance_.reset(sampleRate, kVaSmoothingTimeSec);
    smoothCutoff_.setCurrentAndTargetValue(clampedCutoff);
    smoothResonance_.setCurrentAndTargetValue(resonance_.load(std::memory_order_acquire));
    lastCutoffTarget_ = clampedCutoff;
    lastResonanceTarget_ = resonance_.load(std::memory_order_acquire);
}

inline std::array<double, VirtualAnalogFilterStage::kMaxStages>
VirtualAnalogFilterStage::cascade(double V, const std::array<double, kMaxStages>& s, double gVal,
                                  double R) const noexcept {
    // Feedback always taps the last stored state slot (numStages_-1's value
    // is mirrored into the unused upper slots via the integration step below,
    // so s[kMaxStages-1] is correct regardless of numStages_).
    const double fb = g_tanhLUT.lookup(R * s[static_cast<std::size_t>(numStages_ - 1)]);

    std::array<double, kMaxStages> d{};
    d[0] = T_ * stages_[0]->computeDelta(V - fb, s[0], gVal);
    for (int i = 1; i < kMaxStages; ++i) {
        if (i < numStages_) {
            const auto u = static_cast<std::size_t>(i);
            d[u] = T_ * stages_[u]->computeDelta(s[u - 1] + d[u - 1], s[u], gVal);
        } else {
            // Mirrors the prototype's `(numStages > i) ? ... : prev` short-circuit
            // so unused upper stages don't waste a vcall + LUT lookup.
            d[static_cast<std::size_t>(i)] = d[static_cast<std::size_t>(i - 1)];
        }
    }
    return d;
}

void VirtualAnalogFilterStage::process(juce::AudioBuffer<double>& buffer) {
    if (!enabled_.load(std::memory_order_acquire)) {
        return;
    }

    const double cutTarget = cutoffHz_.load(std::memory_order_acquire);
    const double resTarget = resonance_.load(std::memory_order_acquire);
    if (cutTarget != lastCutoffTarget_) {
        smoothCutoff_.setTargetValue(cutTarget);
        lastCutoffTarget_ = cutTarget;
    }
    if (resTarget != lastResonanceTarget_) {
        smoothResonance_.setTargetValue(resTarget);
        lastResonanceTarget_ = resTarget;
    }

    const int numChannels = std::min(buffer.getNumChannels(), static_cast<int>(channels_.size()));
    const int numSamples = buffer.getNumSamples();
    constexpr double kTwoPi = 2.0 * juce::MathConstants<double>::pi;

    for (int i = 0; i < numSamples; ++i) {
        const double cutoff = smoothCutoff_.getNextValue();
        const double R = smoothResonance_.getNextValue();
        const double gNow = kTwoPi * cutoff;

        for (int ch = 0; ch < numChannels; ++ch) {
            auto& state = channels_[static_cast<std::size_t>(ch)];
            const double Vn = buffer.getWritePointer(ch)[i];

            const double g = gNow;
            const double g_mid = 0.5 * (g + state.g_n1);
            const double Vn_mid = 0.5 * (Vn + state.Vn_1);

            const auto k1 = cascade(state.Vn_1, state.y, state.g_n1, R);

            std::array<double, kMaxStages> s2{};
            for (int j = 0; j < kMaxStages; ++j) {
                s2[static_cast<std::size_t>(j)] =
                    state.y[static_cast<std::size_t>(j)] + 0.5 * k1[static_cast<std::size_t>(j)];
            }
            const auto k2 = cascade(Vn_mid, s2, g_mid, R);

            std::array<double, kMaxStages> s3{};
            for (int j = 0; j < kMaxStages; ++j) {
                s3[static_cast<std::size_t>(j)] =
                    state.y[static_cast<std::size_t>(j)] + 0.5 * k2[static_cast<std::size_t>(j)];
            }
            const auto k3 = cascade(Vn_mid, s3, g_mid, R);

            std::array<double, kMaxStages> s4{};
            for (int j = 0; j < kMaxStages; ++j) {
                s4[static_cast<std::size_t>(j)] =
                    state.y[static_cast<std::size_t>(j)] + k3[static_cast<std::size_t>(j)];
            }
            const auto k4 = cascade(Vn, s4, g, R);

            for (int j = 0; j < kMaxStages; ++j) {
                const auto u = static_cast<std::size_t>(j);
                if (j < numStages_) {
                    state.y[u] += (k1[u] + 2.0 * k2[u] + 2.0 * k3[u] + k4[u]) * kVaSixth;
                } else {
                    state.y[u] = state.y[static_cast<std::size_t>(numStages_ - 1)];
                }
            }

            state.Vn_1 = Vn;
            state.g_n1 = g;
            buffer.getWritePointer(ch)[i] = state.y[static_cast<std::size_t>(kMaxStages - 1)];
        }
    }
}

void VirtualAnalogFilterStage::reset() {
    for (auto& c : channels_) {
        c = ChannelState{};
    }
}

void VirtualAnalogFilterStage::setCutoffFrequency(double frequencyHz) {
    const double maxCutoff = vaMaxCutoffForSampleRate(sampleRate_);
    frequencyHz = juce::jlimit(kVaMinCutoffHz, maxCutoff, frequencyHz);
    cutoffHz_.store(frequencyHz, std::memory_order_release);
}

void VirtualAnalogFilterStage::setResonance(double r) {
    r = juce::jlimit(kVaMinResonance, kVaMaxResonance, r);
    resonance_.store(r, std::memory_order_release);
}

} // namespace dsp_core::audio_pipeline
