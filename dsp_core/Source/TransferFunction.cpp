#include "TransferFunction.h"

namespace dsp_core {

TransferFunction::TransferFunction(int tableSize, double minSignalValue, double maxSignalValue)
    : tableSize(tableSize), minSignalValue(minSignalValue), maxSignalValue(maxSignalValue), table(tableSize) {
    reset();
}

void TransferFunction::reset() {
    for (int i = 0; i < tableSize; ++i) {
        setTableValue(i, normalizeIndex(i));
    }
}

double TransferFunction::normalizeIndex(int index) const {
    if (index < 0 || index >= tableSize) {
        return 0.0;
    }
    return juce::jmap(static_cast<double>(index), 0.0, static_cast<double>(tableSize - 1), minSignalValue,
                      maxSignalValue);
}

void TransferFunction::setExtrapolationMode(ExtrapolationMode mode) {
    extrapolationMode = mode;
}

void TransferFunction::setTableValue(int index, double value) {
    if (index >= 0 && index < tableSize) {
        table[index].store(value);
    }
}

void TransferFunction::setTableValues(const double* samples, const int numSamples) {
    for (int i = 0; i < numSamples && i < tableSize; ++i) {
        setTableValue(i, samples[i]);
    }
}

double TransferFunction::getTableValue(int index) const {
    if (index >= 0 && index < tableSize) {
        return table[index].load();
    }
    return 0.0;
}

double TransferFunction::applyTransferFunction(double x) const {
    const double x_proj = (x + 1.0) * 0.5 * (tableSize - 1);
    const int index = static_cast<int>(x_proj);
    const double t = x_proj - index;
    const double y0 = getSample(index - 1);
    const double y1 = getSample(index);
    const double y2 = getSample(index + 1);
    const double y3 = getSample(index + 2);
    return interpolateCatmullRom(y0, y1, y2, y3, t);
}

double TransferFunction::interpolateCatmullRom(double y0, double y1, double y2, double y3, double t) {
    // Catmull-Rom coefficients: 2.0, 5.0, 4.0, 3.0 are from the spline basis
    return 0.5 * ((2.0 * y1) + (-y0 + y2) * t + (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) * t * t +
                  (-y0 + 3.0 * y1 - 3.0 * y2 + y3) * t * t * t);
}

double TransferFunction::getSample(int i) const {
    const bool isBelow = (i < 0);
    const bool isAbove = (i > tableSize - 1);
    if (isBelow && (extrapolationMode == ExtrapolationMode::Linear)) {
        const double slope = table[1].load() - table[0].load();
        return table[0].load() + slope * i;
    }
    if (isAbove && extrapolationMode == ExtrapolationMode::Linear) {
        const double slope = table[tableSize - 1].load() - table[tableSize - 2].load();
        return table[tableSize - 1].load() + slope * (i - tableSize + 1);
    }
    if ((isBelow || isAbove) && extrapolationMode == ExtrapolationMode::Mirror) {
        const int period = 2 * (tableSize - 1);
        const int wrapped = ((i % period) + period) % period;
        const int idx = wrapped <= tableSize - 1 ? wrapped : period - wrapped;
        return table[idx].load();
    }
    const int clampedIdx = juce::jlimit(0, tableSize - 1, i);
    return table[clampedIdx].load();
}

void TransferFunction::processBlock(double* samples, int numSamples) const {
    for (int n = 0; n < numSamples; ++n) {
        const double inputSample = samples[n];
        samples[n] = applyTransferFunction(inputSample);
    }
}

} // namespace dsp_core
