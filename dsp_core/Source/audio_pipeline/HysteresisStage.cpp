#include "HysteresisStage.h"

namespace dsp_core::audio_pipeline {

HysteresisStage::HysteresisStage(dsp_core::SeamlessTransferFunction& tf) : transferFunction_(&tf) {}

void HysteresisStage::prepareToPlay(double sampleRate, int samplesPerBlock, int /*numChannels*/) {
    // Forward to the shared transfer function at this (possibly oversampled) rate
    // so the LUT crossfade and surge-weight step both track wall time correctly.
    transferFunction_->prepareToPlay(sampleRate, samplesPerBlock);

    for (int ch = 0; ch < 2; ++ch) {
        processors_[ch].prepareToPlay(sampleRate);
        processors_[ch].setOperatingPoint(1.0); // Audio-range: LUT sees raw signal
        // LUT is now a pure memoryless function (Surge lives upstream as its own
        // pipeline stage). RK4's 4× intra-sample NL evaluations are safe to route
        // through applyTransferFunction directly.
        processors_[ch].setNonlinearity(
            [this, ch](double x) { return transferFunction_->applyTransferFunction(x, ch); });
        processors_[ch].setNonlinearityDerivative(
            [this, ch](double x) { return transferFunction_->applyTransferFunctionDerivative(x, ch); });
    }

    smoothedMakeupGain_.reset(sampleRate, 0.01); // 10ms ramp

    phaseSamples_ = static_cast<int>(sampleRate * 0.025); // 25ms per phase
    crossfadeState_ = CrossfadeState::Inactive;
    crossfadePosition_ = 0;
    previousEnabled_ = hysteresisEnabled_.load(std::memory_order_acquire);
}

void HysteresisStage::detectTransition(bool enabled) {
    // Only detect when no crossfade is active — mid-crossfade toggles are deferred.
    if (crossfadeState_ != CrossfadeState::Inactive) {
        return;
    }
    if (enabled && !previousEnabled_) {
        crossfadeState_ = CrossfadeState::WarmingUp;
        crossfadePosition_ = 0;
        for (auto& proc : processors_) {
            proc.reset();
        }
    } else if (!enabled && previousEnabled_) {
        crossfadeState_ = CrossfadeState::CrossfadingOut;
        crossfadePosition_ = 0;
    }
    previousEnabled_ = enabled;
}

int HysteresisStage::advanceCrossfadePhase(juce::AudioBuffer<double>& buffer, int pos, int remaining, bool enabled) {
    const int samplesLeft = phaseSamples_ - crossfadePosition_;
    const int toProcess = std::min(remaining, samplesLeft);

    switch (crossfadeState_) {
    case CrossfadeState::WarmingUp:
        processWarmup(buffer, pos, toProcess);
        crossfadePosition_ += toProcess;
        if (crossfadePosition_ >= phaseSamples_) {
            crossfadeState_ = CrossfadeState::CrossfadingIn;
            crossfadePosition_ = 0;
        }
        return toProcess;

    case CrossfadeState::CrossfadingIn:
        processCrossfadeIn(buffer, pos, toProcess);
        crossfadePosition_ += toProcess;
        if (crossfadePosition_ >= phaseSamples_) {
            previousEnabled_ = true;
            // Check for deferred disable toggle
            if (!enabled) {
                crossfadeState_ = CrossfadeState::CrossfadingOut;
                crossfadePosition_ = 0;
            } else {
                crossfadeState_ = CrossfadeState::Inactive;
            }
        }
        return toProcess;

    case CrossfadeState::CrossfadingOut:
        processCrossfadeOut(buffer, pos, toProcess);
        crossfadePosition_ += toProcess;
        if (crossfadePosition_ >= phaseSamples_) {
            previousEnabled_ = false;
            // Check for deferred enable toggle
            if (enabled) {
                crossfadeState_ = CrossfadeState::WarmingUp;
                crossfadePosition_ = 0;
                for (auto& proc : processors_) {
                    proc.reset();
                }
            } else {
                crossfadeState_ = CrossfadeState::Inactive;
            }
        }
        return toProcess;

    case CrossfadeState::Inactive:
        // Phase just completed mid-buffer — process remainder in steady state
        if (enabled) {
            processSteadyHysteresis(buffer, pos, remaining);
        } else {
            processSteadyWaveshaping(buffer, pos, remaining);
        }
        return remaining;
    }
    return 0;
}

void HysteresisStage::process(juce::AudioBuffer<double>& buffer) {
    const int numSamples = buffer.getNumSamples();
    if (numSamples == 0) {
        return;
    }

    const bool enabled = hysteresisEnabled_.load(std::memory_order_acquire);
    detectTransition(enabled);

    // Fast path: no transition active
    if (crossfadeState_ == CrossfadeState::Inactive) {
        if (enabled) {
            transferFunction_->beginBlock();
            processSteadyHysteresis(buffer, 0, numSamples);
        } else {
            transferFunction_->processBuffer(buffer);
        }
        return;
    }

    // Transition active — process in phases, handling mid-buffer completion
    transferFunction_->beginBlock();
    int pos = 0;
    while (pos < numSamples) {
        pos += advanceCrossfadePhase(buffer, pos, numSamples - pos, enabled);
    }
}

void HysteresisStage::processWarmup(juce::AudioBuffer<double>& buffer, int startSample, int numSamples) {
    const int numChannels = buffer.getNumChannels();
    const int channelsToProcess = std::min(numChannels, 2);

    for (int i = startSample; i < startSample + numSamples; ++i) {
        const double t =
            static_cast<double>(crossfadePosition_ + (i - startSample)) / static_cast<double>(phaseSamples_);
        const double s = smoothstep(t);
        smoothedMakeupGain_.getNextValue(); // keep makeup gain smoother advancing

        for (int ch = 0; ch < channelsToProcess; ++ch) {
            auto* data = buffer.getWritePointer(ch);
            const double input = data[i];

            // Warm up the ODE silently with scaled input
            processors_[ch].process(input * s);

            // Output is pure waveshaping
            data[i] = transferFunction_->applyTransferFunction(input, ch);
        }
        transferFunction_->advanceCrossfadeSample();
    }
}

void HysteresisStage::processCrossfadeIn(juce::AudioBuffer<double>& buffer, int startSample, int numSamples) {
    const int numChannels = buffer.getNumChannels();
    const int channelsToProcess = std::min(numChannels, 2);

    for (int i = startSample; i < startSample + numSamples; ++i) {
        const double t =
            static_cast<double>(crossfadePosition_ + (i - startSample)) / static_cast<double>(phaseSamples_);
        const double s = smoothstep(t);
        const double gain = smoothedMakeupGain_.getNextValue();

        for (int ch = 0; ch < channelsToProcess; ++ch) {
            auto* data = buffer.getWritePointer(ch);
            const double input = data[i];
            const double waveOut = transferFunction_->applyTransferFunction(input, ch);
            const double hystOut = processors_[ch].process(input) * gain;

            data[i] = waveOut * (1.0 - s) + hystOut * s;
        }
        transferFunction_->advanceCrossfadeSample();
    }
}

void HysteresisStage::processCrossfadeOut(juce::AudioBuffer<double>& buffer, int startSample, int numSamples) {
    const int numChannels = buffer.getNumChannels();
    const int channelsToProcess = std::min(numChannels, 2);

    for (int i = startSample; i < startSample + numSamples; ++i) {
        const double t =
            static_cast<double>(crossfadePosition_ + (i - startSample)) / static_cast<double>(phaseSamples_);
        const double s = smoothstep(t);
        const double gain = smoothedMakeupGain_.getNextValue();

        for (int ch = 0; ch < channelsToProcess; ++ch) {
            auto* data = buffer.getWritePointer(ch);
            const double input = data[i];
            const double waveOut = transferFunction_->applyTransferFunction(input, ch);
            const double hystOut = processors_[ch].process(input) * gain;

            data[i] = waveOut * s + hystOut * (1.0 - s);
        }
        transferFunction_->advanceCrossfadeSample();
    }
}

void HysteresisStage::processSteadyHysteresis(juce::AudioBuffer<double>& buffer, int startSample, int numSamples) {
    const int numChannels = buffer.getNumChannels();
    const int channelsToProcess = std::min(numChannels, 2);

    for (int i = startSample; i < startSample + numSamples; ++i) {
        const double gain = smoothedMakeupGain_.getNextValue();

        for (int ch = 0; ch < channelsToProcess; ++ch) {
            auto* data = buffer.getWritePointer(ch);
            data[i] = processors_[ch].process(data[i]) * gain;
        }
        transferFunction_->advanceCrossfadeSample();
    }
}

void HysteresisStage::processSteadyWaveshaping(juce::AudioBuffer<double>& buffer, int startSample, int numSamples) {
    const int numChannels = buffer.getNumChannels();
    const int channelsToProcess = std::min(numChannels, 2);

    for (int i = startSample; i < startSample + numSamples; ++i) {
        for (int ch = 0; ch < channelsToProcess; ++ch) {
            auto* data = buffer.getWritePointer(ch);
            data[i] = transferFunction_->applyTransferFunction(data[i], ch);
        }
        transferFunction_->advanceCrossfadeSample();
    }
}

void HysteresisStage::reset() {
    for (auto& proc : processors_) {
        proc.reset();
    }
    crossfadeState_ = CrossfadeState::Inactive;
    crossfadePosition_ = 0;
}

void HysteresisStage::setHysteresisEnabled(bool enabled) {
    hysteresisEnabled_.store(enabled, std::memory_order_release);
}

void HysteresisStage::requestWarmup() {
    if (hysteresisEnabled_.load(std::memory_order_acquire)) {
        crossfadeState_ = CrossfadeState::WarmingUp;
        crossfadePosition_ = 0;
        previousEnabled_ = false; // so the transition detection sees enabled && !previousEnabled_
        for (auto& proc : processors_) {
            proc.reset();
        }
    }
}

void HysteresisStage::setDrive(double drive) {
    for (auto& proc : processors_) {
        proc.setDrive(drive);
    }
}

void HysteresisStage::setSaturation(double sat) {
    for (auto& proc : processors_) {
        proc.setSaturation(sat);
    }
}

void HysteresisStage::setWidth(double width) {
    for (auto& proc : processors_) {
        proc.setWidth(width);
    }
}

void HysteresisStage::setMakeupGain(double gain) {
    smoothedMakeupGain_.setTargetValue(gain);
}

void HysteresisStage::setOperatingPoint(double Ms) {
    for (auto& proc : processors_) {
        proc.setOperatingPoint(Ms);
    }
}

double HysteresisStage::computeMakeupForWidth(double width) {
    // Piecewise-linear LUT, tuned at amp=1.0 with k = 0.4·M_s (see setOperatingPoint).
    // Low k/M_s ratio keeps peak loss roughly amplitude-independent in [0.5, 2.0]
    // and bounded by natural J-A saturation at higher amplitudes — so tuning at
    // full-scale means signals louder than unity undershoot (output bounded by
    // M_s ≈ 1), which is the preferred failure mode. Measurements from
    // DIAGNOSTIC_WidthSweep_Amp1_NewK. Re-run if J-A operating point changes.
    static constexpr std::array<double, 11> kTable = {
        1.000, // w=0.0
        1.039, // w=0.1
        1.081, // w=0.2
        1.126, // w=0.3
        1.175, // w=0.4
        1.229, // w=0.5
        1.288, // w=0.6
        1.352, // w=0.7
        1.424, // w=0.8
        1.504, // w=0.9
        1.593, // w=1.0
    };
    static constexpr int kN = static_cast<int>(kTable.size());
    static constexpr double kStep = 1.0 / (kN - 1);

    const double w = juce::jlimit(0.0, 1.0, width);
    const double f = w / kStep;
    const int i = std::min(static_cast<int>(f), kN - 2);
    const double t = f - static_cast<double>(i);
    return kTable[i] + t * (kTable[i + 1] - kTable[i]);
}

} // namespace dsp_core::audio_pipeline
