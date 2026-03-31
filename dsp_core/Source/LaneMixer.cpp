#include "LaneMixer.h"
#include <algorithm>
#include <cmath>

namespace dsp_core {

namespace {
    constexpr int kDefaultLaneCount = 13;       // Lane 0 (WT) + H1-H12
    constexpr int kDefaultHarmonicCount = 12;   // H1 through H12
    constexpr double kLaneMixerNormEpsilon = 1e-12;
} // namespace

LaneMixer::LaneMixer() : harmonicLayer_(std::make_unique<HarmonicLayer>(MAX_HARMONIC_NUMBER)) {
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

Lane& LaneMixer::getMutableLane(int index) {
    jassert(isValidIndex(index));
    return lanes_[static_cast<size_t>(index)];
}

// ============================================================================
// Dynamic Lane Management
// ============================================================================

int LaneMixer::addLane(int insertAfterIndex) {
    if (activeLaneCount_ >= MAX_LANES)
        return -1;

    // Compute insertion index
    const int insertionIndex = (insertAfterIndex < 0)
        ? activeLaneCount_
        : std::min(insertAfterIndex + 1, activeLaneCount_);

    // Shift existing lanes right to make room
    shiftLanesRight(insertionIndex);

    // Increment count BEFORE fillLaneWithHarmonic — it calls isValidIndex()
    // which checks index < activeLaneCount_. Must come after shiftLanesRight
    // which uses the old activeLaneCount_ as its loop bound.
    activeLaneCount_++;

    // Initialize the new lane
    auto& newLane = lanes_[static_cast<size_t>(insertionIndex)];
    newLane = Lane{}; // Reset to defaults
    newLane.contentType = LaneContentType::Harmonic;

    int harmonicNumber = findNextUnusedHarmonicNumber();
    if (harmonicNumber > MAX_HARMONIC_NUMBER) {
        harmonicNumber = 1; // Fallback
    }
    newLane.harmonicNumber = harmonicNumber;
    newLane.amplitude.store(0.0, std::memory_order_relaxed);
    newLane.laneId = nextLaneId_++;

    // Fill curve data
    fillLaneWithHarmonic(insertionIndex, harmonicNumber);
    incrementVersionIfNotBatching();
    return insertionIndex;
}

bool LaneMixer::removeLane(int index) {
    if (!isValidIndex(index) || activeLaneCount_ <= 1)
        return false;

    // Clear removed lane's curveData (release memory)
    lanes_[static_cast<size_t>(index)].curveData.clear();
    lanes_[static_cast<size_t>(index)].curveData.shrink_to_fit();

    // Shift lanes left to fill the gap
    shiftLanesLeft(index + 1);

    // Clear the now-inactive slot at the end
    auto& inactiveSlot = lanes_[static_cast<size_t>(activeLaneCount_ - 1)];
    inactiveSlot = Lane{};

    activeLaneCount_--;
    incrementVersionIfNotBatching();
    return true;
}

// ============================================================================
// Lane Identity
// ============================================================================

uint32_t LaneMixer::getLaneId(int index) const {
    jassert(isValidIndex(index));
    return lanes_[static_cast<size_t>(index)].laneId;
}

int LaneMixer::findLaneById(uint32_t laneId) const {
    for (int i = 0; i < activeLaneCount_; ++i) {
        if (lanes_[static_cast<size_t>(i)].laneId == laneId)
            return i;
    }
    return -1;
}

// ============================================================================
// Lane Mutations
// ============================================================================

void LaneMixer::setLaneAmplitude(int index, double amplitude) {
    if (!isValidIndex(index))
        return;
    lanes_[static_cast<size_t>(index)].amplitude.store(amplitude, std::memory_order_release);
    incrementVersionIfNotBatching();
}

double LaneMixer::getLaneAmplitude(int index) const {
    if (!isValidIndex(index))
        return 0.0;
    return lanes_[static_cast<size_t>(index)].amplitude.load(std::memory_order_acquire);
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

void LaneMixer::setLaneOddSymmetryEnabled(int index, bool enabled) {
    if (!isValidIndex(index))
        return;
    lanes_[static_cast<size_t>(index)].oddSymmetryEnabled = enabled;
    // No version increment — symmetry is a UI/editing constraint, not an audio change
}

bool LaneMixer::isLaneOddSymmetryEnabled(int index) const {
    if (!isValidIndex(index))
        return false;
    return lanes_[static_cast<size_t>(index)].oddSymmetryEnabled;
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

void LaneMixer::setLaneCurveValue(int laneIndex, int sampleIndex, double value) {
    if (!isValidIndex(laneIndex) || sampleIndex < 0 || sampleIndex >= TABLE_SIZE)
        return;
    lanes_[static_cast<size_t>(laneIndex)].curveData[static_cast<size_t>(sampleIndex)] = value;
    // NOTE: Does not increment version — caller is responsible for version management
    // (e.g., via BatchUpdateGuard or explicit incrementVersion() call)
}

void LaneMixer::incrementVersion() {
    versionCounter_.fetch_add(1, std::memory_order_release);
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

    // Direct Chebyshev trig computation — works for any harmonic number.
    // Odd harmonics: sin(n * asin(x)), Even harmonics: cos(n * acos(x))
    for (int i = 0; i < TABLE_SIZE; ++i) {
        const double x = std::clamp(normalizeIndex(i), -1.0, 1.0);
        if (harmonicNumber == 1) {
            lane.curveData[static_cast<size_t>(i)] = x;
        } else if (harmonicNumber % 2 == 0) {
            lane.curveData[static_cast<size_t>(i)] = std::cos(harmonicNumber * std::acos(x));
        } else {
            lane.curveData[static_cast<size_t>(i)] = std::sin(harmonicNumber * std::asin(x));
        }
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
    for (int laneIdx = 0; laneIdx < activeLaneCount_; ++laneIdx) {
        const auto& lane = lanes_[static_cast<size_t>(laneIdx)];
        if (!lane.isActive())
            continue;

        const double amp = lane.amplitude.load(std::memory_order_acquire);
        const auto& curve = lane.curveData;
        const int curveSize = std::min(static_cast<int>(curve.size()), numSamples);

        for (int i = 0; i < curveSize; ++i) {
            outputBuffer[i] += amp * curve[static_cast<size_t>(i)];
        }
    }

    // Normalize output to [-1, 1] range
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

    for (int laneIdx = 0; laneIdx < activeLaneCount_; ++laneIdx) {
        const auto& lane = lanes_[static_cast<size_t>(laneIdx)];
        if (!lane.isActive())
            continue;

        const double amp = lane.amplitude.load(std::memory_order_acquire);
        const auto& curve = lane.curveData;
        if (static_cast<int>(curve.size()) > idx1) {
            sum0 += amp * curve[static_cast<size_t>(idx0)];
            sum1 += amp * curve[static_cast<size_t>(idx1)];
        }
    }

    return sum0 + frac * (sum1 - sum0);
}

// ============================================================================
// Extrapolation Mode
// ============================================================================

void LaneMixer::setExtrapolationMode(ExtrapolationMode mode) {
    extrapolationMode_ = mode;
    incrementVersionIfNotBatching();
}

// ============================================================================
// Convenience Accessors
// ============================================================================

std::vector<double> LaneMixer::getAmplitudes() const {
    std::vector<double> result(static_cast<size_t>(activeLaneCount_));
    for (int i = 0; i < activeLaneCount_; ++i) {
        result[static_cast<size_t>(i)] = lanes_[static_cast<size_t>(i)].amplitude.load(std::memory_order_acquire);
    }
    return result;
}

void LaneMixer::clearLaneCurveData(int index) {
    if (!isValidIndex(index)) {
        return;
    }
    auto& lane = lanes_[static_cast<size_t>(index)];
    std::fill(lane.curveData.begin(), lane.curveData.end(), 0.0);
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
    // Lane 0: Equation mode — tanh(2x), amplitude=0.0
    {
        auto& lane = lanes_[0];
        lane = Lane{};
        lane.amplitude.store(0.0, std::memory_order_relaxed);
        lane.contentType = LaneContentType::Equation;
        lane.harmonicNumber = 0;
        lane.equationText = "tanh(2x)";
        lane.laneId = 0;
        lane.curveData.resize(TABLE_SIZE);
        for (int i = 0; i < TABLE_SIZE; ++i) {
            const double x = normalizeIndex(i);
            lane.curveData[static_cast<size_t>(i)] = std::tanh(2.0 * x);
        }
    }

    // Lanes 1-12: Chebyshev harmonics H1-H12
    for (int n = 1; n <= kDefaultHarmonicCount; ++n) {
        auto& lane = lanes_[static_cast<size_t>(n)];
        lane = Lane{};
        lane.amplitude.store((n == 1) ? 1.0 : 0.0, std::memory_order_relaxed);
        lane.contentType = LaneContentType::Harmonic;
        lane.harmonicNumber = n;
        lane.laneId = static_cast<uint32_t>(n);
        lane.curveData.resize(TABLE_SIZE);

        for (int i = 0; i < TABLE_SIZE; ++i) {
            const double x = std::clamp(normalizeIndex(i), -1.0, 1.0);
            if (n == 1) {
                lane.curveData[static_cast<size_t>(i)] = x;
            } else if (n % 2 == 0) {
                lane.curveData[static_cast<size_t>(i)] = std::cos(n * std::acos(x));
            } else {
                lane.curveData[static_cast<size_t>(i)] = std::sin(n * std::asin(x));
            }
        }
    }

    // Clear any remaining slots beyond the default count
    for (int i = kDefaultLaneCount; i < MAX_LANES; ++i) {
        lanes_[static_cast<size_t>(i)] = Lane{};
    }

    activeLaneCount_ = kDefaultLaneCount;
    nextLaneId_ = static_cast<uint32_t>(kDefaultLaneCount);
}

// ============================================================================
// Serialization
// ============================================================================

juce::ValueTree LaneMixer::toValueTree() const {
    juce::ValueTree vt("LaneMixer");
    vt.setProperty("formatVersion", 3, nullptr);
    vt.setProperty("numLanes", activeLaneCount_, nullptr);
    vt.setProperty("nextLaneId", static_cast<int>(nextLaneId_), nullptr);
    vt.setProperty("tableSize", TABLE_SIZE, nullptr);

    for (int i = 0; i < activeLaneCount_; ++i) {
        const auto& lane = lanes_[static_cast<size_t>(i)];

        juce::ValueTree laneVT("Lane");
        laneVT.setProperty("index", i, nullptr);
        laneVT.setProperty("laneId", static_cast<int>(lane.laneId), nullptr);
        laneVT.setProperty("amplitude", lane.amplitude.load(std::memory_order_acquire), nullptr);
        laneVT.setProperty("contentType", static_cast<int>(lane.contentType), nullptr);
        laneVT.setProperty("harmonicNumber", lane.harmonicNumber, nullptr);

        if (!lane.equationText.isEmpty()) {
            laneVT.setProperty("equationText", lane.equationText, nullptr);
        }

        if (!lane.presetSourcePath.isEmpty()) {
            laneVT.setProperty("presetSourcePath", lane.presetSourcePath, nullptr);
        }

        if (lane.oddSymmetryEnabled) {
            laneVT.setProperty("oddSymmetryEnabled", true, nullptr);
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

        // Skip curveData for Harmonic lanes — regenerated from harmonicNumber on load
        const bool canRegenerate = (lane.contentType == LaneContentType::Harmonic);
        if (!canRegenerate && !lane.curveData.empty()) {
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

    const int formatVersion = vt.getProperty("formatVersion", 2);
    const int loadedTableSize = vt.getProperty("tableSize", TABLE_SIZE);

    if (formatVersion >= 3) {
        // Dynamic format: read lane count, IDs, nextLaneId
        activeLaneCount_ = juce::jlimit(1, MAX_LANES,
            static_cast<int>(vt.getProperty("numLanes", kDefaultLaneCount)));
        nextLaneId_ = static_cast<uint32_t>(
            static_cast<int>(vt.getProperty("nextLaneId", activeLaneCount_)));
    } else {
        // Format v2: fixed 41 lanes -> assign sequential IDs
        activeLaneCount_ = 41;
        nextLaneId_ = 41;
        for (int i = 0; i < 41; ++i) {
            lanes_[static_cast<size_t>(i)].laneId = static_cast<uint32_t>(i);
        }
    }

    for (int i = 0; i < vt.getNumChildren(); ++i) {
        const auto laneVT = vt.getChild(i);
        if (laneVT.getType().toString() != "Lane")
            continue;

        const int laneIndex = laneVT.getProperty("index", -1);
        if (laneIndex < 0 || laneIndex >= activeLaneCount_)
            continue;

        auto& lane = lanes_[static_cast<size_t>(laneIndex)];

        // Read lane ID for v3+ format
        if (formatVersion >= 3) {
            lane.laneId = static_cast<uint32_t>(
                static_cast<int>(laneVT.getProperty("laneId", laneIndex)));
        }

        lane.amplitude.store(
            static_cast<double>(laneVT.getProperty("amplitude", 0.0)),
            std::memory_order_release);
        lane.contentType = static_cast<LaneContentType>(
            static_cast<int>(laneVT.getProperty("contentType", 0)));
        lane.harmonicNumber = laneVT.getProperty("harmonicNumber", 0);
        lane.equationText = laneVT.getProperty("equationText", juce::String()).toString();
        lane.presetSourcePath = laneVT.getProperty("presetSourcePath", juce::String()).toString();
        lane.oddSymmetryEnabled = static_cast<bool>(laneVT.getProperty("oddSymmetryEnabled", false));

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
// Legacy Migration
// ============================================================================

void LaneMixer::fromLegacyLTFValueTree(const juce::ValueTree& ltfVT) {
    // Default overload: treat as harmonic mode without odd symmetry (41 lanes)
    fromLegacyLTFValueTree(ltfVT, true, false);
}

void LaneMixer::fromLegacyLTFValueTree(const juce::ValueTree& ltfVT,
                                         bool wasHarmonicMode,
                                         bool oddSymmetryWasEnabled) {
    if (!ltfVT.isValid() || ltfVT.getType().toString() != "LayeredTransferFunction") {
        return;
    }

    // Parse coefficients from legacy format
    std::vector<double> legacyCoefficients;
    if (ltfVT.hasProperty("coefficients")) {
        const juce::Array<juce::var>* coeffArray = ltfVT.getProperty("coefficients").getArray();
        if (coeffArray != nullptr) {
            legacyCoefficients.reserve(static_cast<size_t>(coeffArray->size()));
            for (int i = 0; i < coeffArray->size(); ++i) {
                legacyCoefficients.push_back(static_cast<double>((*coeffArray)[i]));
            }
        }
    }

    // Parse BaseLayer curve data
    std::vector<double> baseLayerData;
    const auto baseVT = ltfVT.getChildWithName("BaseLayer");
    if (baseVT.isValid() && baseVT.hasProperty("tableData")) {
        const juce::MemoryBlock baseBlob = *baseVT.getProperty("tableData").getBinaryData();
        const auto* data = static_cast<const double*>(baseBlob.getData());
        const int numValues = static_cast<int>(baseBlob.getSize() / sizeof(double));
        const int copyCount = std::min(numValues, TABLE_SIZE);
        baseLayerData.assign(data, data + copyCount);
        baseLayerData.resize(TABLE_SIZE, 0.0);
    }

    // Parse spline anchors from legacy format
    std::vector<SplineAnchor> legacySplineAnchors;
    LaneContentType lane0ContentType = LaneContentType::Harmonic;
    const auto splineVT = ltfVT.getChildWithName("SplineLayer");
    if (splineVT.isValid()) {
        lane0ContentType = LaneContentType::Spline;
        const auto anchorsVT = splineVT.getChildWithName("Anchors");
        if (anchorsVT.isValid()) {
            for (int a = 0; a < anchorsVT.getNumChildren(); ++a) {
                const auto anchorVT = anchorsVT.getChild(a);
                SplineAnchor anchor;
                anchor.x = static_cast<double>(anchorVT.getProperty("x", 0.0));
                anchor.y = static_cast<double>(anchorVT.getProperty("y", 0.0));
                anchor.tangent = static_cast<double>(anchorVT.getProperty("tangent", 0.0));
                anchor.hasCustomTangent = static_cast<bool>(anchorVT.getProperty("hasCustomTangent", false));
                legacySplineAnchors.push_back(anchor);
            }
        }
    }

    // Clear all lanes
    for (int i = 0; i < MAX_LANES; ++i) {
        lanes_[static_cast<size_t>(i)] = Lane{};
    }

    if (wasHarmonicMode && oddSymmetryWasEnabled) {
        // Case B — harmonic mode with odd symmetry: Lane 0 + odd harmonics only
        // Lane 0: base layer
        {
            auto& lane0 = lanes_[0];
            lane0.laneId = 0;
            lane0.contentType = lane0ContentType;
            lane0.harmonicNumber = 0;
            lane0.splineAnchors = legacySplineAnchors;
            lane0.amplitude.store(
                legacyCoefficients.empty() ? 0.0 : legacyCoefficients[0],
                std::memory_order_relaxed);
            if (!baseLayerData.empty()) {
                lane0.curveData = baseLayerData;
            } else {
                lane0.curveData.resize(TABLE_SIZE);
                for (int i = 0; i < TABLE_SIZE; ++i) {
                    lane0.curveData[static_cast<size_t>(i)] = std::tanh(2.0 * normalizeIndex(i));
                }
            }
        }

        // Odd harmonics: H1, H3, H5, ..., H39 (20 lanes at indices 1-20)
        int laneIdx = 1;
        for (int h = 1; h <= 39; h += 2) {
            auto& lane = lanes_[static_cast<size_t>(laneIdx)];
            lane.laneId = static_cast<uint32_t>(laneIdx);
            lane.contentType = LaneContentType::Harmonic;
            lane.harmonicNumber = h;
            lane.amplitude.store(
                (static_cast<size_t>(h) < legacyCoefficients.size()) ? legacyCoefficients[static_cast<size_t>(h)] : 0.0,
                std::memory_order_relaxed);
            lane.curveData.resize(TABLE_SIZE);
            for (int i = 0; i < TABLE_SIZE; ++i) {
                const double x = std::clamp(normalizeIndex(i), -1.0, 1.0);
                if (h == 1) {
                    lane.curveData[static_cast<size_t>(i)] = x;
                } else {
                    lane.curveData[static_cast<size_t>(i)] = std::sin(h * std::asin(x));
                }
            }
            laneIdx++;
        }

        activeLaneCount_ = 21; // Lane 0 + 20 odd harmonics
        nextLaneId_ = 21;

    } else if (wasHarmonicMode) {
        // Case B — harmonic mode without odd symmetry: Lane 0 + H1-H39 (41 lanes)
        // Lane 0: base layer
        {
            auto& lane0 = lanes_[0];
            lane0.laneId = 0;
            lane0.contentType = lane0ContentType;
            lane0.harmonicNumber = 0;
            lane0.splineAnchors = legacySplineAnchors;
            lane0.amplitude.store(
                legacyCoefficients.empty() ? 0.0 : legacyCoefficients[0],
                std::memory_order_relaxed);
            if (!baseLayerData.empty()) {
                lane0.curveData = baseLayerData;
            } else {
                lane0.curveData.resize(TABLE_SIZE);
                for (int i = 0; i < TABLE_SIZE; ++i) {
                    lane0.curveData[static_cast<size_t>(i)] = std::tanh(2.0 * normalizeIndex(i));
                }
            }
        }

        // H1-H39 (40 harmonic lanes at indices 1-40)
        for (int n = 1; n <= 40; ++n) {
            auto& lane = lanes_[static_cast<size_t>(n)];
            lane.laneId = static_cast<uint32_t>(n);
            lane.contentType = LaneContentType::Harmonic;
            lane.harmonicNumber = n;
            lane.amplitude.store(
                (static_cast<size_t>(n) < legacyCoefficients.size()) ? legacyCoefficients[static_cast<size_t>(n)] : 0.0,
                std::memory_order_relaxed);
            lane.curveData.resize(TABLE_SIZE);
            for (int i = 0; i < TABLE_SIZE; ++i) {
                const double x = std::clamp(normalizeIndex(i), -1.0, 1.0);
                if (n == 1) {
                    lane.curveData[static_cast<size_t>(i)] = x;
                } else if (n % 2 == 0) {
                    lane.curveData[static_cast<size_t>(i)] = std::cos(n * std::acos(x));
                } else {
                    lane.curveData[static_cast<size_t>(i)] = std::sin(n * std::asin(x));
                }
            }
        }

        activeLaneCount_ = 41;
        nextLaneId_ = 41;

    } else {
        // Case C — non-harmonic mode (Paint, Equation, Spline): Lane 0 + H1-H12
        // Lane 0: saved state
        {
            auto& lane0 = lanes_[0];
            lane0.laneId = 0;
            lane0.contentType = lane0ContentType;
            lane0.harmonicNumber = 0;
            lane0.splineAnchors = legacySplineAnchors;
            lane0.amplitude.store(
                legacyCoefficients.empty() ? 0.0 : legacyCoefficients[0],
                std::memory_order_relaxed);
            if (!baseLayerData.empty()) {
                lane0.curveData = baseLayerData;
            } else {
                lane0.curveData.resize(TABLE_SIZE);
                for (int i = 0; i < TABLE_SIZE; ++i) {
                    lane0.curveData[static_cast<size_t>(i)] = std::tanh(2.0 * normalizeIndex(i));
                }
            }
        }

        // Lanes 1-12: default H1-H12, all at amplitude 0.0
        for (int n = 1; n <= kDefaultHarmonicCount; ++n) {
            auto& lane = lanes_[static_cast<size_t>(n)];
            lane.laneId = static_cast<uint32_t>(n);
            lane.contentType = LaneContentType::Harmonic;
            lane.harmonicNumber = n;
            lane.amplitude.store(0.0, std::memory_order_relaxed);
            lane.curveData.resize(TABLE_SIZE);
            for (int i = 0; i < TABLE_SIZE; ++i) {
                const double x = std::clamp(normalizeIndex(i), -1.0, 1.0);
                if (n == 1) {
                    lane.curveData[static_cast<size_t>(i)] = x;
                } else if (n % 2 == 0) {
                    lane.curveData[static_cast<size_t>(i)] = std::cos(n * std::acos(x));
                } else {
                    lane.curveData[static_cast<size_t>(i)] = std::sin(n * std::asin(x));
                }
            }
        }

        activeLaneCount_ = kDefaultLaneCount;
        nextLaneId_ = static_cast<uint32_t>(kDefaultLaneCount);
    }

    incrementVersionIfNotBatching();
}

// ============================================================================
// Private Helpers
// ============================================================================

int LaneMixer::findNextUnusedHarmonicNumber() const {
    // Scan active lanes for used harmonic numbers
    for (int candidate = 1; candidate <= MAX_HARMONIC_NUMBER; ++candidate) {
        bool used = false;
        for (int i = 0; i < activeLaneCount_; ++i) {
            const auto& lane = lanes_[static_cast<size_t>(i)];
            if (lane.contentType == LaneContentType::Harmonic
                && lane.harmonicNumber == candidate) {
                used = true;
                break;
            }
        }
        if (!used) {
            return candidate;
        }
    }
    // All 1..MAX_HARMONIC_NUMBER used — fallback to 1
    return 1;
}

void LaneMixer::shiftLanesRight(int fromIndex) {
    for (int i = activeLaneCount_; i > fromIndex; --i) {
        lanes_[static_cast<size_t>(i)] = std::move(lanes_[static_cast<size_t>(i - 1)]);
    }
}

void LaneMixer::shiftLanesLeft(int fromIndex) {
    for (int i = fromIndex; i < activeLaneCount_; ++i) {
        lanes_[static_cast<size_t>(i - 1)] = std::move(lanes_[static_cast<size_t>(i)]);
    }
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
