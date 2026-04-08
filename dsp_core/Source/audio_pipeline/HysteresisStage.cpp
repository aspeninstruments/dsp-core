#include "HysteresisStage.h"

namespace dsp_core::audio_pipeline {

HysteresisStage::HysteresisStage(const dsp_core::SeamlessTransferFunction& tf)
    : transferFunction_(&tf) {}

void HysteresisStage::prepareToPlay(double sampleRate, int /*samplesPerBlock*/) {
    for (int ch = 0; ch < 2; ++ch) {
        processors_[ch].prepareToPlay(sampleRate);
        processors_[ch].setOperatingPoint(1.0); // Audio-range: LUT sees raw signal
        processors_[ch].setNonlinearity([this](double x) {
            return transferFunction_->applyTransferFunction(x);
        });
    }

    smoothedMakeupGain_.reset(sampleRate, 0.01); // 10ms ramp

    phaseSamples_ = static_cast<int>(sampleRate * 0.025); // 25ms per phase
    crossfadeState_ = CrossfadeState::Inactive;
    crossfadePosition_ = 0;
    previousEnabled_ = hysteresisEnabled_.load(std::memory_order_acquire);
}

void HysteresisStage::process(juce::AudioBuffer<double>& buffer) {
    const int numSamples = buffer.getNumSamples();

    if (numSamples == 0) {
        return;
    }

    const bool enabled = hysteresisEnabled_.load(std::memory_order_acquire);

    // Detect transitions (only when no crossfade is active)
    if (crossfadeState_ == CrossfadeState::Inactive) {
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
        const int remaining = numSamples - pos;

        switch (crossfadeState_) {
            case CrossfadeState::WarmingUp: {
                const int samplesLeft = phaseSamples_ - crossfadePosition_;
                const int toProcess = std::min(remaining, samplesLeft);
                processWarmup(buffer, pos, toProcess);
                pos += toProcess;
                crossfadePosition_ += toProcess;
                if (crossfadePosition_ >= phaseSamples_) {
                    crossfadeState_ = CrossfadeState::CrossfadingIn;
                    crossfadePosition_ = 0;
                }
                break;
            }
            case CrossfadeState::CrossfadingIn: {
                const int samplesLeft = phaseSamples_ - crossfadePosition_;
                const int toProcess = std::min(remaining, samplesLeft);
                processCrossfadeIn(buffer, pos, toProcess);
                pos += toProcess;
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
                break;
            }
            case CrossfadeState::CrossfadingOut: {
                const int samplesLeft = phaseSamples_ - crossfadePosition_;
                const int toProcess = std::min(remaining, samplesLeft);
                processCrossfadeOut(buffer, pos, toProcess);
                pos += toProcess;
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
                break;
            }
            case CrossfadeState::Inactive: {
                // Phase just completed mid-buffer — process remainder in steady state
                if (enabled) {
                    processSteadyHysteresis(buffer, pos, remaining);
                } else {
                    processSteadyWaveshaping(buffer, pos, remaining);
                }
                pos += remaining;
                break;
            }
        }
    }
}

void HysteresisStage::processWarmup(juce::AudioBuffer<double>& buffer, int startSample, int numSamples) {
    const int numChannels = buffer.getNumChannels();
    const int channelsToProcess = std::min(numChannels, 2);

    for (int i = startSample; i < startSample + numSamples; ++i) {
        const double t = static_cast<double>(crossfadePosition_ + (i - startSample))
                       / static_cast<double>(phaseSamples_);
        const double s = smoothstep(t);
        smoothedMakeupGain_.getNextValue(); // keep makeup gain smoother advancing

        for (int ch = 0; ch < channelsToProcess; ++ch) {
            auto* data = buffer.getWritePointer(ch);
            const double input = data[i];

            // Warm up the ODE silently with scaled input
            processors_[ch].process(input * s);

            // Output is pure waveshaping
            data[i] = transferFunction_->applyTransferFunction(input);
        }
        transferFunction_->advanceCrossfadeSample();
    }
}

void HysteresisStage::processCrossfadeIn(juce::AudioBuffer<double>& buffer, int startSample, int numSamples) {
    const int numChannels = buffer.getNumChannels();
    const int channelsToProcess = std::min(numChannels, 2);

    for (int i = startSample; i < startSample + numSamples; ++i) {
        const double t = static_cast<double>(crossfadePosition_ + (i - startSample))
                       / static_cast<double>(phaseSamples_);
        const double s = smoothstep(t);
        const double gain = smoothedMakeupGain_.getNextValue();

        for (int ch = 0; ch < channelsToProcess; ++ch) {
            auto* data = buffer.getWritePointer(ch);
            const double input = data[i];
            const double waveOut = transferFunction_->applyTransferFunction(input);
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
        const double t = static_cast<double>(crossfadePosition_ + (i - startSample))
                       / static_cast<double>(phaseSamples_);
        const double s = smoothstep(t);
        const double gain = smoothedMakeupGain_.getNextValue();

        for (int ch = 0; ch < channelsToProcess; ++ch) {
            auto* data = buffer.getWritePointer(ch);
            const double input = data[i];
            const double waveOut = transferFunction_->applyTransferFunction(input);
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
            data[i] = transferFunction_->applyTransferFunction(data[i]);
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

} // namespace dsp_core::audio_pipeline
