#pragma once

#include <vector>
#include <atomic>
#include <juce_core/juce_core.h>

namespace dsp_core {

class TransferFunction {
  public:
    enum class ExtrapolationMode { Clamp, Linear, Mirror };

    TransferFunction(int tableSize, double minSignalValue = -1.0, double maxSignalValue = 1.0);

    void reset();
    double normalizeIndex(int index) const;

    double applyTransferFunction(double x) const;

    void setExtrapolationMode(ExtrapolationMode mode);

    void setTableValue(int index, double value);
    void setTableValues(const double* samples, const int numSamples);
    double getTableValue(int index) const;

    int getTableSize() const {
        return tableSize;
    }
    double getMinSignalValue() const {
        return minSignalValue;
    }
    double getMaxSignalValue() const {
        return maxSignalValue;
    }

    // For GUI/editor access
    const std::vector<std::atomic<double>>& getTable() const {
        return table;
    }
    std::vector<std::atomic<double>>& getTable() {
        return table;
    }

    void processBlock(double* samples, int numSamples) const;

  private:
    double getSample(int i) const;
    static double interpolateCatmullRom(double y0, double y1, double y2, double y3, double t);

    int tableSize;
    double minSignalValue, maxSignalValue;
    std::vector<std::atomic<double>> table;
    ExtrapolationMode extrapolationMode = ExtrapolationMode::Clamp;
};

} // namespace dsp_core
