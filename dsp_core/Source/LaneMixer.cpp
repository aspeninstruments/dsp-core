#include "LaneMixer.h"
#include <algorithm>
#include <cmath>

namespace dsp_core {

namespace {
    constexpr int kLaneMixerNumHarmonics = 40;
    constexpr double kLaneMixerNormEpsilon = 1e-12;
} // namespace

LaneMixer::LaneMixer() : harmonicLayer_(std::make_unique<HarmonicLayer>(kLaneMixerNumHarmonics)) {
    harmonicLayer_->precomputeBasisFunctions(TABLE_SIZE, MIN_VALUE, MAX_VALUE);
    initializeDefaults();
}

// ============================================================================
// Lane Access
// ============================================================================

const Lane& LaneMixer::getLane(int index) const {
    jassert(isValidIndex(index));
    return lanes_[static_cast<size_t>(index)];
}

// ============================================================================
// Lane Mutations
// ============================================================================

void LaneMixer::setLaneAmplitude(int index, double amplitude) {
    if (!isValidIndex(index))
        return;
    lanes_[static_cast<size_t>(index)].amplitude = amplitude;
    incrementVersionIfNotBatching();
}

double LaneMixer::getLaneAmplitude(int index) const {
    if (!isValidIndex(index))
        return 0.0;
    return lanes_[static_cast<size_t>(index)].amplitude;
}

void LaneMixer::setLaneContentType(int index, LaneContentType type) {
    if (!isValidIndex(index))
        return;
    lanes_[static_cast<size_t>(index)].contentType = type;
    incrementVersionIfNotBatching();
}

LaneContentType LaneMixer::getLaneContentType(int index) const {
    if (!isValidIndex(index))
        return LaneContentType::Harmonic;
    return lanes_[static_cast<size_t>(index)].contentType;
}

void LaneMixer::setLaneHarmonicNumber(int index, int harmonicNumber) {
    if (!isValidIndex(index))
        return;
    lanes_[static_cast<size_t>(index)].harmonicNumber = harmonicNumber;
    incrementVersionIfNotBatching();
}

int LaneMixer::getLaneHarmonicNumber(int index) const {
    if (!isValidIndex(index))
        return 0;
    return lanes_[static_cast<size_t>(index)].harmonicNumber;
}

void LaneMixer::setLaneCurveData(int index, const std::vector<double>& data) {
    if (!isValidIndex(index))
        return;
    auto& lane = lanes_[static_cast<size_t>(index)];
    lane.curveData = data;
    lane.curveData.resize(TABLE_SIZE, 0.0); // Pad or truncate to TABLE_SIZE
    incrementVersionIfNotBatching();
}

void LaneMixer::setLaneCurveData(int index, const double* data, int size) {
    if (!isValidIndex(index) || data == nullptr)
        return;
    auto& lane = lanes_[static_cast<size_t>(index)];
    lane.curveData.assign(data, data + std::min(size, TABLE_SIZE));
    lane.curveData.resize(TABLE_SIZE, 0.0);
    incrementVersionIfNotBatching();
}

void LaneMixer::fillLaneWithHarmonic(int index, int harmonicNumber) {
    if (!isValidIndex(index))
        return;

    auto& lane = lanes_[static_cast<size_t>(index)];
    lane.curveData.resize(TABLE_SIZE);
    lane.contentType = LaneContentType::Harmonic;
    lane.harmonicNumber = harmonicNumber;

    if (harmonicNumber == 0) {
        // Harmonic 0 = WT base, fill with tanh(2x)
        fillLaneWithTanh2x(index);
        lane.harmonicNumber = 0;
        return;
    }

    // Use precomputed basis functions from HarmonicLayer
    // Create a temporary coefficient array with only this harmonic active
    std::vector<double> coefficients(static_cast<size_t>(kLaneMixerNumHarmonics + 1), 0.0);
    if (harmonicNumber >= 1 && harmonicNumber <= kLaneMixerNumHarmonics) {
        coefficients[static_cast<size_t>(harmonicNumber)] = 1.0;
    }

    for (int i = 0; i < TABLE_SIZE; ++i) {
        const double x = normalizeIndex(i);
        lane.curveData[static_cast<size_t>(i)] = harmonicLayer_->evaluate(x, coefficients, TABLE_SIZE);
    }

    incrementVersionIfNotBatching();
}

void LaneMixer::fillLaneWithTanh2x(int index) {
    if (!isValidIndex(index))
        return;

    auto& lane = lanes_[static_cast<size_t>(index)];
    lane.curveData.resize(TABLE_SIZE);

    for (int i = 0; i < TABLE_SIZE; ++i) {
        const double x = normalizeIndex(i);
        lane.curveData[static_cast<size_t>(i)] = std::tanh(2.0 * x);
    }

    incrementVersionIfNotBatching();
}

// ============================================================================
// Mixing
// ============================================================================

void LaneMixer::computeSum(double* outputBuffer, int size) const {
    const int numSamples = std::min(size, TABLE_SIZE);

    // Zero the output buffer
    std::fill(outputBuffer, outputBuffer + numSamples, 0.0);

    // Accumulate weighted contributions from all active lanes
    for (int laneIdx = 0; laneIdx < NUM_LANES; ++laneIdx) {
        const auto& lane = lanes_[static_cast<size_t>(laneIdx)];
        if (!lane.isActive())
            continue;

        const double amp = lane.amplitude;
        const auto& curve = lane.curveData;
        const int curveSize = std::min(static_cast<int>(curve.size()), numSamples);

        for (int i = 0; i < curveSize; ++i) {
            outputBuffer[i] += amp * curve[static_cast<size_t>(i)];
        }
    }

    // Apply normalization if enabled
    if (normalizationEnabled_) {
        double maxAbs = 0.0;
        for (int i = 0; i < numSamples; ++i) {
            maxAbs = std::max(maxAbs, std::abs(outputBuffer[i]));
        }

        if (maxAbs > kLaneMixerNormEpsilon) {
            const double scalar = 1.0 / maxAbs;
            for (int i = 0; i < numSamples; ++i) {
                outputBuffer[i] *= scalar;
            }
        }
    }
}

double LaneMixer::evaluateSumAt(double x) const {
    // Clamp x to valid range
    x = std::max(MIN_VALUE, std::min(MAX_VALUE, x));

    // Map x to fractional table index
    const double fractionalIndex =
        (x - MIN_VALUE) / (MAX_VALUE - MIN_VALUE) * static_cast<double>(TABLE_SIZE - 1);

    // Compute sum at this position using linear interpolation between table points
    const int idx0 = static_cast<int>(std::floor(fractionalIndex));
    const int idx1 = std::min(idx0 + 1, TABLE_SIZE - 1);
    const double frac = fractionalIndex - static_cast<double>(idx0);

    double sum0 = 0.0;
    double sum1 = 0.0;

    for (int laneIdx = 0; laneIdx < NUM_LANES; ++laneIdx) {
        const auto& lane = lanes_[static_cast<size_t>(laneIdx)];
        if (!lane.isActive())
            continue;

        const double amp = lane.amplitude;
        const auto& curve = lane.curveData;
        if (static_cast<int>(curve.size()) > idx1) {
            sum0 += amp * curve[static_cast<size_t>(idx0)];
            sum1 += amp * curve[static_cast<size_t>(idx1)];
        }
    }

    return sum0 + frac * (sum1 - sum0);
}

// ============================================================================
// Normalization
// ============================================================================

void LaneMixer::setNormalizationEnabled(bool enabled) {
    normalizationEnabled_ = enabled;
    incrementVersionIfNotBatching();
}

// ============================================================================
// Version Tracking
// ============================================================================

void LaneMixer::beginBatchUpdate() {
    batchUpdateActive_ = true;
}

void LaneMixer::endBatchUpdate() {
    batchUpdateActive_ = false;
    versionCounter_.fetch_add(1, std::memory_order_release);
}

// ============================================================================
// Initialization
// ============================================================================

void LaneMixer::resetToDefaults() {
    initializeDefaults();
    incrementVersionIfNotBatching();
}

void LaneMixer::initializeDefaults() {
    // Lane 0: WT base layer — tanh(2x), amplitude=0.0
    {
        auto& lane = lanes_[0];
        lane.amplitude = 0.0;
        lane.contentType = LaneContentType::Harmonic;
        lane.harmonicNumber = 0;
        lane.curveData.resize(TABLE_SIZE);
        for (int i = 0; i < TABLE_SIZE; ++i) {
            const double x = normalizeIndex(i);
            lane.curveData[static_cast<size_t>(i)] = std::tanh(2.0 * x);
        }
        lane.splineAnchors.clear();
        lane.equationText.clear();
        lane.presetSourcePath.clear();
    }

    // Lanes 1-40: Chebyshev harmonics T_n(x)
    std::vector<double> coefficients(static_cast<size_t>(kLaneMixerNumHarmonics + 1), 0.0);

    for (int n = 1; n <= kLaneMixerNumHarmonics; ++n) {
        auto& lane = lanes_[static_cast<size_t>(n)];
        lane.amplitude = (n == 1) ? 1.0 : 0.0; // H1 = 1.0 (y=x), others = 0.0
        lane.contentType = LaneContentType::Harmonic;
        lane.harmonicNumber = n;
        lane.curveData.resize(TABLE_SIZE);
        lane.splineAnchors.clear();
        lane.equationText.clear();
        lane.presetSourcePath.clear();

        // Build coefficient array with only this harmonic active
        std::fill(coefficients.begin(), coefficients.end(), 0.0);
        coefficients[static_cast<size_t>(n)] = 1.0;

        for (int i = 0; i < TABLE_SIZE; ++i) {
            const double x = normalizeIndex(i);
            lane.curveData[static_cast<size_t>(i)] = harmonicLayer_->evaluate(x, coefficients, TABLE_SIZE);
        }
    }
}

// ============================================================================
// Serialization
// ============================================================================

juce::ValueTree LaneMixer::toValueTree() const {
    juce::ValueTree vt("LaneMixer");
    vt.setProperty("formatVersion", 1, nullptr);
    vt.setProperty("numLanes", NUM_LANES, nullptr);
    vt.setProperty("tableSize", TABLE_SIZE, nullptr);
    vt.setProperty("normalizationEnabled", normalizationEnabled_, nullptr);

    for (int i = 0; i < NUM_LANES; ++i) {
        const auto& lane = lanes_[static_cast<size_t>(i)];

        juce::ValueTree laneVT("Lane");
        laneVT.setProperty("index", i, nullptr);
        laneVT.setProperty("amplitude", lane.amplitude, nullptr);
        laneVT.setProperty("contentType", static_cast<int>(lane.contentType), nullptr);
        laneVT.setProperty("harmonicNumber", lane.harmonicNumber, nullptr);

        if (!lane.equationText.isEmpty()) {
            laneVT.setProperty("equationText", lane.equationText, nullptr);
        }

        if (!lane.presetSourcePath.isEmpty()) {
            laneVT.setProperty("presetSourcePath", lane.presetSourcePath, nullptr);
        }

        // Serialize spline anchors if present
        if (!lane.splineAnchors.empty()) {
            juce::ValueTree anchorsVT("SplineAnchors");
            for (const auto& anchor : lane.splineAnchors) {
                juce::ValueTree anchorVT("Anchor");
                anchorVT.setProperty("x", anchor.x, nullptr);
                anchorVT.setProperty("y", anchor.y, nullptr);
                if (anchor.hasCustomTangent) {
                    anchorVT.setProperty("tangent", anchor.tangent, nullptr);
                }
                anchorsVT.appendChild(anchorVT, nullptr);
            }
            laneVT.appendChild(anchorsVT, nullptr);
        }

        // Compress curve data with zlib
        if (!lane.curveData.empty()) {
            juce::MemoryBlock rawData(lane.curveData.data(),
                                      lane.curveData.size() * sizeof(double));
            juce::MemoryOutputStream compressedStream;
            {
                juce::GZIPCompressorOutputStream compressor(compressedStream);
                compressor.write(rawData.getData(), rawData.getSize());
            }
            laneVT.setProperty("curveData",
                                juce::var(compressedStream.getMemoryBlock()),
                                nullptr);
        }

        vt.appendChild(laneVT, nullptr);
    }

    return vt;
}

void LaneMixer::fromValueTree(const juce::ValueTree& vt) {
    if (!vt.isValid() || vt.getType().toString() != "LaneMixer")
        return;

    const int loadedTableSize = vt.getProperty("tableSize", TABLE_SIZE);
    normalizationEnabled_ = vt.getProperty("normalizationEnabled", true);

    for (int i = 0; i < vt.getNumChildren(); ++i) {
        const auto laneVT = vt.getChild(i);
        if (laneVT.getType().toString() != "Lane")
            continue;

        const int laneIndex = laneVT.getProperty("index", -1);
        if (!isValidIndex(laneIndex))
            continue;

        auto& lane = lanes_[static_cast<size_t>(laneIndex)];
        lane.amplitude = static_cast<double>(laneVT.getProperty("amplitude", 0.0));
        lane.contentType = static_cast<LaneContentType>(
            static_cast<int>(laneVT.getProperty("contentType", 0)));
        lane.harmonicNumber = laneVT.getProperty("harmonicNumber", 0);
        lane.equationText = laneVT.getProperty("equationText", juce::String()).toString();
        lane.presetSourcePath = laneVT.getProperty("presetSourcePath", juce::String()).toString();

        // Deserialize spline anchors
        lane.splineAnchors.clear();
        const auto anchorsVT = laneVT.getChildWithName("SplineAnchors");
        if (anchorsVT.isValid()) {
            for (int a = 0; a < anchorsVT.getNumChildren(); ++a) {
                const auto anchorVT = anchorsVT.getChild(a);
                SplineAnchor anchor;
                anchor.x = static_cast<double>(anchorVT.getProperty("x", 0.0));
                anchor.y = static_cast<double>(anchorVT.getProperty("y", 0.0));
                if (anchorVT.hasProperty("tangent")) {
                    anchor.hasCustomTangent = true;
                    anchor.tangent = static_cast<double>(anchorVT.getProperty("tangent", 0.0));
                }
                lane.splineAnchors.push_back(anchor);
            }
        }

        // Decompress curve data
        if (laneVT.hasProperty("curveData")) {
            const auto* compressedData = laneVT.getProperty("curveData").getBinaryData();
            if (compressedData != nullptr) {
                juce::MemoryInputStream compressedStream(*compressedData, false);
                juce::GZIPDecompressorInputStream decompressor(compressedStream);

                const size_t expectedSize = static_cast<size_t>(loadedTableSize) * sizeof(double);
                juce::MemoryBlock decompressed;
                decompressed.setSize(expectedSize);

                const auto bytesRead = decompressor.read(decompressed.getData(),
                                                          static_cast<int>(expectedSize));

                if (bytesRead > 0) {
                    const auto numDoubles = static_cast<size_t>(bytesRead) / sizeof(double);
                    lane.curveData.resize(TABLE_SIZE, 0.0);
                    const auto* src = static_cast<const double*>(decompressed.getData());
                    const auto copyCount = std::min(numDoubles, static_cast<size_t>(TABLE_SIZE));
                    std::copy(src, src + copyCount, lane.curveData.begin());
                }
            }
        } else {
            // No curve data in serialized form — regenerate from content type
            if (lane.contentType == LaneContentType::Harmonic) {
                if (lane.harmonicNumber == 0) {
                    fillLaneWithTanh2x(laneIndex);
                } else {
                    fillLaneWithHarmonic(laneIndex, lane.harmonicNumber);
                }
            } else {
                lane.initCurveData(TABLE_SIZE);
            }
        }
    }

    incrementVersionIfNotBatching();
}

// ============================================================================
// Utilities
// ============================================================================

double LaneMixer::normalizeIndex(int index) const {
    return juce::jmap(static_cast<double>(index),
                      0.0, static_cast<double>(TABLE_SIZE - 1),
                      MIN_VALUE, MAX_VALUE);
}

} // namespace dsp_core
