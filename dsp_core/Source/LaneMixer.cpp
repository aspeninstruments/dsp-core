#include "LaneMixer.h"
#include "Services/EquationLaneSynthesizer.h"
#include "Services/SplineLaneSynthesizer.h"
#include "Services/TransferFunctionOperations.h"
#include <algorithm>
#include <cmath>

namespace dsp_core {

namespace {
constexpr int kDefaultLaneCount = 10; // H1, H3, H5, ..., H19
constexpr std::array<int, 10> kDefaultHarmonics = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
constexpr std::array<double, 10> kDefaultAmplitudes = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
constexpr std::array<double, 10> kDefaultBlendDepths = {0.0,    1.0,     0.5,      0.25,      0.125,
                                                        0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625};
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
    const auto mutationGuard = acquireMutationGuard();

    if (activeLaneCount_ >= MAX_LANES) {
        return -1;
    }

    // Compute insertion index
    const int insertionIndex =
        (insertAfterIndex < 0) ? activeLaneCount_ : std::min(insertAfterIndex + 1, activeLaneCount_);

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

int LaneMixer::duplicateLane(int sourceLaneIndex) {
    if (activeLaneCount_ >= MAX_LANES) {
        return -1;
    }
    if (!isValidIndex(sourceLaneIndex)) {
        return -1;
    }

    // Copy the source lane BEFORE shifting (shift uses move semantics)
    Lane sourceCopy(lanes_[static_cast<size_t>(sourceLaneIndex)]);

    const int insertionIndex = sourceLaneIndex + 1;

    // Adjust scan position to maintain continuity across the insertion
    if (activeLaneCount_ > 1) {
        const double scanPos = scanPosition_.load(std::memory_order_acquire);
        const double f = scanPos * (activeLaneCount_ - 1);
        const double fNew = (static_cast<double>(insertionIndex) <= std::ceil(f)) ? f + 1.0 : f;
        const double newScanPos = std::clamp(fNew / activeLaneCount_, 0.0, 1.0);
        scanPosition_.store(newScanPos, std::memory_order_release);
    }

    shiftLanesRight(insertionIndex);
    activeLaneCount_++;

    auto& newLane = lanes_[static_cast<size_t>(insertionIndex)];
    newLane = std::move(sourceCopy);
    newLane.laneId = nextLaneId_++;
    newLane.amplitude.store(0.0, std::memory_order_relaxed);

    incrementVersionIfNotBatching();
    return insertionIndex;
}

bool LaneMixer::removeLane(int index) {
    const auto mutationGuard = acquireMutationGuard();

    if (!isValidIndex(index) || activeLaneCount_ <= 1) {
        return false;
    }

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
        if (lanes_[static_cast<size_t>(i)].laneId == laneId) {
            return i;
        }
    }
    return -1;
}

// ============================================================================
// Lane Mutations
// ============================================================================

void LaneMixer::setLaneAmplitude(int index, double amplitude) {
    if (!isValidIndex(index)) {
        return;
    }
    auto& slot = lanes_[static_cast<size_t>(index)];
    const double current = slot.amplitude.load(std::memory_order_acquire);
    if (current == amplitude) {
        return; // Early-return guard: avoid spamming the LUT remix path under static automation.
    }
    slot.amplitude.store(amplitude, std::memory_order_release);
    incrementMixVersionIfNotBatching();
    incrementVersionIfNotBatching();
}

double LaneMixer::getLaneAmplitude(int index) const {
    if (!isValidIndex(index)) {
        return 0.0;
    }
    return lanes_[static_cast<size_t>(index)].amplitude.load(std::memory_order_acquire);
}

void LaneMixer::setLaneBlendDepth(int index, double depth) {
    if (!isValidIndex(index)) {
        return;
    }
    const double clamped = std::clamp(depth, -1.0, 1.0);
    auto& slot = lanes_[static_cast<size_t>(index)];
    const double current = slot.blendDepth.load(std::memory_order_acquire);
    if (current == clamped) {
        return; // Early-return guard: same rationale as setLaneAmplitude.
    }
    slot.blendDepth.store(clamped, std::memory_order_release);
    incrementMixVersionIfNotBatching();
    incrementVersionIfNotBatching();
}

double LaneMixer::getLaneBlendDepth(int index) const {
    if (!isValidIndex(index)) {
        return 0.0;
    }
    return lanes_[static_cast<size_t>(index)].blendDepth.load(std::memory_order_acquire);
}

void LaneMixer::setLaneModulationDepth(int index, int slotIdx, double depth) {
    if (!isValidIndex(index) || slotIdx < 0 || slotIdx >= 2) {
        return;
    }
    const double clamped = std::clamp(depth, -1.0, 1.0);
    auto& slot = lanes_[static_cast<size_t>(index)];
    auto& depthSlot = slot.modulationDepth[static_cast<size_t>(slotIdx)];
    const double current = depthSlot.load(std::memory_order_acquire);
    if (current == clamped) {
        return; // Early-return guard: same rationale as setLaneAmplitude.
    }
    depthSlot.store(clamped, std::memory_order_release);
    incrementMixVersionIfNotBatching();
    incrementVersionIfNotBatching();
}

double LaneMixer::getLaneModulationDepth(int index, int slotIdx) const {
    if (!isValidIndex(index) || slotIdx < 0 || slotIdx >= 2) {
        return 0.0;
    }
    return lanes_[static_cast<size_t>(index)].modulationDepth[static_cast<size_t>(slotIdx)].load(
        std::memory_order_acquire);
}

void LaneMixer::setModulationEnvValue(int slotIdx, double env) {
    if (slotIdx < 0 || slotIdx >= 2) {
        return;
    }
    auto& slotEnv = modulationEnvValue_[static_cast<size_t>(slotIdx)];
    const double current = slotEnv.load(std::memory_order_acquire);
    if (current == env) {
        return; // Early-return guard: steady-state env → no LUT remix needed.
    }
    slotEnv.store(env, std::memory_order_release);

    // Skip the version bump when no lane is using *this slot's* modulation
    // depth — the env change has no audible effect, so a remix would be wasted
    // work. The other slot's depth is irrelevant here.
    bool anyLaneModulated = false;
    for (int i = 0; i < activeLaneCount_; ++i) {
        if (lanes_[static_cast<size_t>(i)].modulationDepth[static_cast<size_t>(slotIdx)].load(
                std::memory_order_acquire) != 0.0) {
            anyLaneModulated = true;
            break;
        }
    }
    if (!anyLaneModulated) {
        return;
    }

    // Per-block updates: bump mix version so the renderer recomputes the LUT
    // to reflect the new effective per-lane amplitudes (env * modulationDepth).
    // Same pattern as setBlendAmount under env-modulated Morph.
    incrementMixVersionIfNotBatching();
    incrementVersionIfNotBatching();
}

void LaneMixer::setLaneContentType(int index, LaneContentType type) {
    if (!isValidIndex(index)) {
        return;
    }
    lanes_[static_cast<size_t>(index)].contentType = type;
    incrementVersionIfNotBatching();
}

LaneContentType LaneMixer::getLaneContentType(int index) const {
    if (!isValidIndex(index)) {
        return LaneContentType::Harmonic;
    }
    return lanes_[static_cast<size_t>(index)].contentType;
}

void LaneMixer::setLaneHarmonicNumber(int index, int harmonicNumber) {
    if (!isValidIndex(index)) {
        return;
    }
    lanes_[static_cast<size_t>(index)].harmonicNumber = harmonicNumber;
    incrementVersionIfNotBatching();
}

int LaneMixer::getLaneHarmonicNumber(int index) const {
    if (!isValidIndex(index)) {
        return 0;
    }
    return lanes_[static_cast<size_t>(index)].harmonicNumber;
}

void LaneMixer::setLaneOddSymmetryEnabled(int index, bool enabled) {
    if (!isValidIndex(index)) {
        return;
    }
    lanes_[static_cast<size_t>(index)].oddSymmetryEnabled = enabled;
    // No version increment — symmetry is a UI/editing constraint, not an audio change
}

bool LaneMixer::isLaneOddSymmetryEnabled(int index) const {
    if (!isValidIndex(index)) {
        return false;
    }
    return lanes_[static_cast<size_t>(index)].oddSymmetryEnabled;
}

void LaneMixer::setLaneCurveData(int index, const std::vector<double>& data) {
    const auto mutationGuard = acquireMutationGuard();

    if (!isValidIndex(index)) {
        return;
    }
    auto& lane = lanes_[static_cast<size_t>(index)];
    lane.curveData = data;
    lane.curveData.resize(TABLE_SIZE, 0.0); // Pad or truncate to TABLE_SIZE
    incrementVersionIfNotBatching();
}

void LaneMixer::setLaneCurveData(int index, const double* data, int size) {
    const auto mutationGuard = acquireMutationGuard();

    if (!isValidIndex(index) || data == nullptr) {
        return;
    }
    auto& lane = lanes_[static_cast<size_t>(index)];
    lane.curveData.assign(data, data + std::min(size, TABLE_SIZE));
    lane.curveData.resize(TABLE_SIZE, 0.0);
    incrementVersionIfNotBatching();
}

void LaneMixer::setLaneCurveValue(int laneIndex, int sampleIndex, double value) {
    if (!isValidIndex(laneIndex) || sampleIndex < 0 || sampleIndex >= TABLE_SIZE) {
        return;
    }
    lanes_[static_cast<size_t>(laneIndex)].curveData[static_cast<size_t>(sampleIndex)] = value;
    // NOTE: Does not increment version — caller is responsible for version management
    // (e.g., via BatchUpdateGuard or explicit incrementVersion() call)
}

void LaneMixer::incrementVersion() {
    versionCounter_.fetch_add(1, std::memory_order_release);
    if (onVersionChanged_) {
        onVersionChanged_();
    }
}

void LaneMixer::fillLaneWithHarmonic(int index, int harmonicNumber, double strength) {
    if (!isValidIndex(index)) {
        return;
    }

    auto& lane = lanes_[static_cast<size_t>(index)];
    lane.curveData.resize(TABLE_SIZE);
    lane.contentType = LaneContentType::Harmonic;
    lane.harmonicNumber = harmonicNumber;

    if (harmonicNumber == 0) {
        // Harmonic 0 = WT base, fill with tanh(2x). Strength doesn't apply here.
        fillLaneWithTanh2x(index);
        lane.harmonicNumber = 0;
        lane.harmonicStrength = 1.0;
        return;
    }

    const double s = std::clamp(strength, -1.0, 1.0);
    lane.harmonicStrength = s;

    // Direct Chebyshev trig computation — works for any harmonic number.
    // Odd harmonics: sin(n * asin(x)), Even harmonics: cos(n * acos(x))
    for (int i = 0; i < TABLE_SIZE; ++i) {
        const double x = std::clamp(normalizeIndex(i), -1.0, 1.0);
        double value;
        if (harmonicNumber == 1) {
            value = x;
        } else if (harmonicNumber % 2 == 0) {
            value = std::cos(harmonicNumber * std::acos(x));
        } else {
            value = std::sin(harmonicNumber * std::asin(x));
        }
        lane.curveData[static_cast<size_t>(i)] = value * s;
    }

    incrementVersionIfNotBatching();
}

void LaneMixer::fillLaneWithTanh2x(int index) {
    if (!isValidIndex(index)) {
        return;
    }

    auto& lane = lanes_[static_cast<size_t>(index)];
    lane.curveData.resize(TABLE_SIZE);

    for (int i = 0; i < TABLE_SIZE; ++i) {
        const double x = normalizeIndex(i);
        lane.curveData[static_cast<size_t>(i)] = std::tanh(2.0 * x);
    }

    Services::TransferFunctionOperations::normalize(lane.curveData);

    incrementVersionIfNotBatching();
}

// ============================================================================
// Mixing
// ============================================================================

void LaneMixer::computeSum(double* outputBuffer, int size) const {
    const int numSamples = std::min(size, TABLE_SIZE);

    // Zero the output buffer
    std::fill(outputBuffer, outputBuffer + numSamples, 0.0);

    // Hoist the global blend amount and per-slot env values once outside the per-lane loop.
    const double macro = blendAmount_.load(std::memory_order_acquire);
    const double env0 = modulationEnvValue_[0].load(std::memory_order_acquire);
    const double env1 = modulationEnvValue_[1].load(std::memory_order_acquire);

    // Accumulate weighted contributions from all lanes whose effective amplitude is non-zero.
    // Effective amplitude = max(0, base + macro * depth + sum_s env[s] * modDepth[s]). Negative
    // depths let the macro/env pull the lane *down* toward silence; the lower clamp at 0
    // prevents inverted-curve contributions, which would invert the transfer function.
    // NB: Do NOT use Lane::isActive() here — that only checks `amplitude`, missing the case
    // amplitude=0, depth>0, blendAmount>0 where the lane *is* audibly contributing.
    for (int laneIdx = 0; laneIdx < activeLaneCount_; ++laneIdx) {
        const auto& lane = lanes_[static_cast<size_t>(laneIdx)];
        const double base = lane.amplitude.load(std::memory_order_acquire);
        const double depth = lane.blendDepth.load(std::memory_order_acquire);
        const double modDepth0 = lane.modulationDepth[0].load(std::memory_order_acquire);
        const double modDepth1 = lane.modulationDepth[1].load(std::memory_order_acquire);
        const double effective = std::max(0.0, base + macro * depth + env0 * modDepth0 + env1 * modDepth1);

        if (effective <= kLaneMixerNormEpsilon || lane.curveData.empty()) {
            continue;
        }

        const auto& curve = lane.curveData;
        const int curveSize = std::min(static_cast<int>(curve.size()), numSamples);

        for (int i = 0; i < curveSize; ++i) {
            outputBuffer[i] += effective * curve[static_cast<size_t>(i)];
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

void LaneMixer::computeScan(double* outputBuffer, int size) const {
    const int numSamples = std::min(size, TABLE_SIZE);

    if (activeLaneCount_ <= 0) {
        std::fill(outputBuffer, outputBuffer + numSamples, 0.0);
        return;
    }

    if (activeLaneCount_ == 1) {
        const auto& lane = lanes_[0];
        const int curveSize = std::min(static_cast<int>(lane.curveData.size()), numSamples);
        std::copy(lane.curveData.begin(), lane.curveData.begin() + curveSize, outputBuffer);
        if (curveSize < numSamples) {
            std::fill(outputBuffer + curveSize, outputBuffer + numSamples, 0.0);
        }
    } else {
        const double pos = scanPosition_.load(std::memory_order_acquire);
        const double f = pos * (activeLaneCount_ - 1);
        const int lo = std::clamp(static_cast<int>(std::floor(f)), 0, activeLaneCount_ - 1);
        const int hi = std::min(lo + 1, activeLaneCount_ - 1);
        const double t = f - lo;

        const auto& laneA = lanes_[static_cast<size_t>(lo)];
        const auto& laneB = lanes_[static_cast<size_t>(hi)];
        const int sizeA = static_cast<int>(laneA.curveData.size());
        const int sizeB = static_cast<int>(laneB.curveData.size());

        for (int i = 0; i < numSamples; ++i) {
            const double a = (i < sizeA) ? laneA.curveData[static_cast<size_t>(i)] : 0.0;
            const double b = (i < sizeB) ? laneB.curveData[static_cast<size_t>(i)] : 0.0;
            outputBuffer[i] = (1.0 - t) * a + t * b;
        }
    }

    // Normalize to [-1, 1]
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

namespace {
// Linear-interp LUT lookup. x must be in [-1, 1].
inline double lookupLaneLUT(const std::vector<double>& curve, double x, int tableSize) {
    const double fractionalIndex =
        (x - LaneMixer::MIN_VALUE) / (LaneMixer::MAX_VALUE - LaneMixer::MIN_VALUE) * static_cast<double>(tableSize - 1);
    const int idx0 = std::clamp(static_cast<int>(std::floor(fractionalIndex)), 0, tableSize - 1);
    const int idx1 = std::min(idx0 + 1, tableSize - 1);
    const double frac = fractionalIndex - static_cast<double>(idx0);
    const double a = curve[static_cast<size_t>(idx0)];
    const double b = curve[static_cast<size_t>(idx1)];
    return a + frac * (b - a);
}

// Map sample index s in [0, numSamples) to x in [-1, 1] for thumbnail eval.
inline double thumbnailXAt(int s, int numSamples) {
    if (numSamples == 1) {
        return LaneMixer::MIN_VALUE;
    }
    const double t = static_cast<double>(s) / static_cast<double>(numSamples - 1);
    return LaneMixer::MIN_VALUE + t * (LaneMixer::MAX_VALUE - LaneMixer::MIN_VALUE);
}

// Post-loop normalize matching computeSum / computeScan / computeSeries: scan
// row for max|y|, divide every sample by it so the curve fits [-1, 1] without
// flat-topping when the unscaled sum exceeds unity. Mirrors the live DSP path
// the thumbnail is meant to preview.
inline void normalizeThumbnailRow(float* row, int numSamples) {
    float maxAbs = 0.0f;
    for (int s = 0; s < numSamples; ++s) {
        maxAbs = std::max(maxAbs, std::abs(row[s]));
    }
    if (maxAbs > static_cast<float>(kLaneMixerNormEpsilon)) {
        const float scalar = 1.0f / maxAbs;
        for (int s = 0; s < numSamples; ++s) {
            row[s] *= scalar;
        }
    }
}

// Per-row thumbnail evaluators. Each writes `numSamples` normalized float
// samples into `row` (max|row| ≈ 1 unless the curve is silent), given an
// explicit morph value (no live atomics from the mixer's blendAmount_ /
// scanPosition_ / modulationEnvValue_ are read).
//
// `laneLuts[i]` is the LUT to use for lane i — for non-Equation lanes this
// points at lane.curveData; for Equation lanes the caller (evaluateStaticThumbnail)
// substitutes a scratch LUT freshly synthesized for the current morph value via
// dsp_core::Services::synthesizeLaneLUT, mirroring the live audio path's
// per-morph re-evaluation. `laneLuts` must have at least activeLaneCount
// non-null entries.
void evalThumbnailRowBlend(const Lane* lanes, int activeLaneCount, const std::vector<double>* const* laneLuts,
                           double morph, float* row, int numSamples) {
    constexpr int kTableSize = LaneMixer::TABLE_SIZE;
    for (int s = 0; s < numSamples; ++s) {
        const double x = thumbnailXAt(s, numSamples);
        double sum = 0.0;
        for (int laneIdx = 0; laneIdx < activeLaneCount; ++laneIdx) {
            const auto& lane = lanes[laneIdx];
            const auto& lut = *laneLuts[laneIdx];
            const double base = lane.amplitude.load(std::memory_order_acquire);
            const double depth = lane.blendDepth.load(std::memory_order_acquire);
            const double effective = std::max(0.0, base + morph * depth);
            if (effective <= kLaneMixerNormEpsilon || lut.empty()) {
                continue;
            }
            sum += effective * lookupLaneLUT(lut, x, kTableSize);
        }
        row[s] = static_cast<float>(sum);
    }
    normalizeThumbnailRow(row, numSamples);
}

void evalThumbnailRowScan(const Lane* lanes, int activeLaneCount, const std::vector<double>* const* laneLuts,
                          double morph, float* row, int numSamples) {
    constexpr int kTableSize = LaneMixer::TABLE_SIZE;
    if (activeLaneCount <= 0) {
        std::fill(row, row + numSamples, 0.0f);
        return;
    }
    if (activeLaneCount == 1) {
        const auto& lut = *laneLuts[0];
        for (int s = 0; s < numSamples; ++s) {
            const double y = lut.empty() ? 0.0 : lookupLaneLUT(lut, thumbnailXAt(s, numSamples), kTableSize);
            row[s] = static_cast<float>(y);
        }
        normalizeThumbnailRow(row, numSamples);
        return;
    }
    (void)lanes; // Scan mode lerps LUTs directly; lane atomics aren't read in this path.
    const double pos = std::clamp(morph, 0.0, 1.0);
    const double f = pos * (activeLaneCount - 1);
    const int lo = std::clamp(static_cast<int>(std::floor(f)), 0, activeLaneCount - 1);
    const int hi = std::min(lo + 1, activeLaneCount - 1);
    const double t = f - static_cast<double>(lo);
    const auto& lutA = *laneLuts[lo];
    const auto& lutB = *laneLuts[hi];
    for (int s = 0; s < numSamples; ++s) {
        const double x = thumbnailXAt(s, numSamples);
        const double a = lutA.empty() ? 0.0 : lookupLaneLUT(lutA, x, kTableSize);
        const double b = lutB.empty() ? 0.0 : lookupLaneLUT(lutB, x, kTableSize);
        const double y = (1.0 - t) * a + t * b;
        row[s] = static_cast<float>(y);
    }
    normalizeThumbnailRow(row, numSamples);
}

void evalThumbnailRowSeries(const Lane* lanes, int activeLaneCount, const std::vector<double>* const* laneLuts,
                            double morph, float* row, int numSamples) {
    constexpr int kTableSize = LaneMixer::TABLE_SIZE;
    constexpr double kLn4 = 1.3862943611198906;
    for (int s = 0; s < numSamples; ++s) {
        double y = thumbnailXAt(s, numSamples);
        for (int laneIdx = 0; laneIdx < activeLaneCount; ++laneIdx) {
            const auto& lane = lanes[laneIdx];
            const auto& lut = *laneLuts[laneIdx];
            if (lut.empty()) {
                continue;
            }
            const double base = lane.amplitude.load(std::memory_order_acquire);
            const double depth = lane.blendDepth.load(std::memory_order_acquire);
            const double aEff = std::clamp(base + morph * depth, 0.0, 1.0);
            const double gain = std::exp(kLn4 * (2.0 * aEff - 1.0));
            // Inner clamp protects the LUT input range — kept; matches computeSeries line ~636.
            const double driven =
                std::clamp(y * gain, LaneMixer::MIN_VALUE, LaneMixer::MAX_VALUE);
            y = lookupLaneLUT(lut, driven, kTableSize);
        }
        row[s] = static_cast<float>(y);
    }
    normalizeThumbnailRow(row, numSamples);
}
} // namespace

void LaneMixer::computeSeries(double* outputBuffer, int size) const {
    const int numSamples = std::min(size, TABLE_SIZE);

    // Seed with the identity x in [-1, 1] so the first lane sees the raw input.
    for (int i = 0; i < numSamples; ++i) {
        outputBuffer[i] = normalizeIndex(i);
    }

    // Empty mixer -> identity passthrough.
    if (activeLaneCount_ <= 0) {
        return;
    }

    // Gain mapping: a_eff in [0, 1] -> gain in [0.25, 4.0], with 0.5 -> 1.0.
    // gain = 4^(2*a_eff - 1) = exp(ln(4) * (2*a_eff - 1)).
    constexpr double kLn4 = 1.3862943611198906; // std::log(4.0)

    const double macro = blendAmount_.load(std::memory_order_acquire);
    const double env0 = modulationEnvValue_[0].load(std::memory_order_acquire);
    const double env1 = modulationEnvValue_[1].load(std::memory_order_acquire);

    // Apply each lane in order: y_{n+1} = lane_n.LUT(clamp(y_n * gain_n, -1, 1)).
    for (int laneIdx = 0; laneIdx < activeLaneCount_; ++laneIdx) {
        const auto& lane = lanes_[static_cast<size_t>(laneIdx)];
        if (lane.curveData.empty()) {
            continue; // No curve to apply; preserves running signal.
        }

        const double base = lane.amplitude.load(std::memory_order_acquire);
        const double depth = lane.blendDepth.load(std::memory_order_acquire);
        const double modDepth0 = lane.modulationDepth[0].load(std::memory_order_acquire);
        const double modDepth1 = lane.modulationDepth[1].load(std::memory_order_acquire);
        const double aEff = std::clamp(base + macro * depth + env0 * modDepth0 + env1 * modDepth1, 0.0, 1.0);
        const double gain = std::exp(kLn4 * (2.0 * aEff - 1.0));

        for (int i = 0; i < numSamples; ++i) {
            const double driven = std::clamp(outputBuffer[i] * gain, MIN_VALUE, MAX_VALUE);
            outputBuffer[i] = lookupLaneLUT(lane.curveData, driven, TABLE_SIZE);
        }
    }

    // Normalize output to [-1, 1] range (match computeSum / computeScan).
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
    const double fractionalIndex = (x - MIN_VALUE) / (MAX_VALUE - MIN_VALUE) * static_cast<double>(TABLE_SIZE - 1);

    // Compute sum at this position using linear interpolation between table points
    const int idx0 = static_cast<int>(std::floor(fractionalIndex));
    const int idx1 = std::min(idx0 + 1, TABLE_SIZE - 1);
    const double frac = fractionalIndex - static_cast<double>(idx0);

    double sum0 = 0.0;
    double sum1 = 0.0;

    const double macro = blendAmount_.load(std::memory_order_acquire);
    const double env0 = modulationEnvValue_[0].load(std::memory_order_acquire);
    const double env1 = modulationEnvValue_[1].load(std::memory_order_acquire);

    for (int laneIdx = 0; laneIdx < activeLaneCount_; ++laneIdx) {
        const auto& lane = lanes_[static_cast<size_t>(laneIdx)];
        const double base = lane.amplitude.load(std::memory_order_acquire);
        const double depth = lane.blendDepth.load(std::memory_order_acquire);
        const double modDepth0 = lane.modulationDepth[0].load(std::memory_order_acquire);
        const double modDepth1 = lane.modulationDepth[1].load(std::memory_order_acquire);
        const double effective = std::max(0.0, base + macro * depth + env0 * modDepth0 + env1 * modDepth1);

        if (effective <= kLaneMixerNormEpsilon || lane.curveData.empty()) {
            continue;
        }

        const auto& curve = lane.curveData;
        if (static_cast<int>(curve.size()) > idx1) {
            sum0 += effective * curve[static_cast<size_t>(idx0)];
            sum1 += effective * curve[static_cast<size_t>(idx1)];
        }
    }

    return sum0 + frac * (sum1 - sum0);
}

bool LaneMixer::evaluateStaticThumbnail(float* out, int numSamples, const float* morphPositions,
                                        int numMorphPositions) const noexcept {
    if (out == nullptr || morphPositions == nullptr || numSamples <= 0 || numMorphPositions <= 0) {
        return false;
    }

    const MixerMode mode = mixerMode_;
    const Lane* lanes = lanes_.data();

    // Equation and Spline lanes: their `lane.curveData` is a snapshot baked at
    // the LIVE morph value, so reading it for all 5 thumbnail rows would freeze
    // them identically. The live audio path re-synthesizes the LUT on each
    // morph change:
    //   - Equation: ExpressionOperationHandler::applyMorphToEquationLanes
    //               → dsp_core::Services::synthesizeLaneLUT
    //   - Spline:   CurveEditorController::morphLaneAnchors
    //               → dsp_core::Services::synthesizeSplineLaneLUT
    // We mirror both here by compiling/preparing once per lane and
    // re-synthesizing into a scratch LUT per morph row. Wrap heap allocations
    // in try/catch to honor this function's noexcept contract.
    try {
        const int activeLaneCount = activeLaneCount_;
        std::vector<dsp_core::Services::CompiledLaneEquation> compiledEqns(static_cast<std::size_t>(activeLaneCount));
        std::vector<std::vector<double>> scratchLuts(static_cast<std::size_t>(activeLaneCount));
        std::vector<const std::vector<double>*> laneLutPointers(static_cast<std::size_t>(activeLaneCount));

        // Per-lane setup: compile equations, mark spline lanes for per-morph re-bake,
        // pass everything else through to the live curveData.
        enum class LaneSynth : std::uint8_t { Snapshot, Equation, Spline };
        std::vector<LaneSynth> synthKind(static_cast<std::size_t>(activeLaneCount), LaneSynth::Snapshot);

        for (int li = 0; li < activeLaneCount; ++li) {
            const auto& lane = lanes_[static_cast<std::size_t>(li)];
            const auto idx = static_cast<std::size_t>(li);

            if (lane.contentType == LaneContentType::Equation && !lane.equationText.isEmpty() &&
                dsp_core::Services::compileLaneEquation(lane.equationText, /*normalize=*/true, compiledEqns[idx])) {
                synthKind[idx] = LaneSynth::Equation;
                laneLutPointers[idx] = &scratchLuts[idx]; // filled per morph row
            } else if (lane.contentType == LaneContentType::Spline && !lane.splineAnchors.empty() &&
                       std::any_of(lane.splineAnchors.begin(), lane.splineAnchors.end(),
                                   [](const SplineAnchor& a) { return !a.morphGesture.empty(); })) {
                // Spline lane with at least one gesture-bearing anchor — re-bake per morph.
                // Spline lanes without gestures are static; fall through to Snapshot.
                synthKind[idx] = LaneSynth::Spline;
                laneLutPointers[idx] = &scratchLuts[idx];
            } else {
                // Non-equation, non-gesture-spline, or compile failure → use the live
                // curveData snapshot. (Compile failures fall through silently: the
                // editor would have rejected the user's input at entry time, so a
                // failure here means a corrupt preset; better to render the snapshot
                // than to abort the whole thumbnail.)
                laneLutPointers[idx] = &lane.curveData;
            }
        }

        std::vector<SplineAnchor> splineScratchAnchors; // reusable working buffer for spline morph

        for (int m = 0; m < numMorphPositions; ++m) {
            const auto morph = static_cast<double>(morphPositions[m]);

            // Re-synthesize equation/spline lanes at this morph value.
            for (int li = 0; li < activeLaneCount; ++li) {
                const auto idx = static_cast<std::size_t>(li);
                bool ok = true;
                switch (synthKind[idx]) {
                case LaneSynth::Equation:
                    ok = dsp_core::Services::synthesizeLaneLUT(compiledEqns[idx], morph, *harmonicLayer_, TABLE_SIZE,
                                                                scratchLuts[idx]);
                    break;
                case LaneSynth::Spline:
                    ok = dsp_core::Services::synthesizeSplineLaneLUT(lanes_[idx].splineAnchors, morph, TABLE_SIZE,
                                                                      splineScratchAnchors, scratchLuts[idx]);
                    break;
                case LaneSynth::Snapshot:
                    continue; // pointer already aimed at lane.curveData
                }
                if (!ok) {
                    // Synthesis failed (NaN, degenerate input, etc.) → fall back to live
                    // curveData for THIS row only. Other rows may still succeed.
                    laneLutPointers[idx] = &lanes_[idx].curveData;
                } else {
                    laneLutPointers[idx] = &scratchLuts[idx];
                }
            }

            float* row = out + (static_cast<std::ptrdiff_t>(m) * numSamples);
            switch (mode) {
            case MixerMode::Blend:
                evalThumbnailRowBlend(lanes, activeLaneCount, laneLutPointers.data(), morph, row, numSamples);
                break;
            case MixerMode::Scan:
                evalThumbnailRowScan(lanes, activeLaneCount, laneLutPointers.data(), morph, row, numSamples);
                break;
            case MixerMode::Series:
                evalThumbnailRowSeries(lanes, activeLaneCount, laneLutPointers.data(), morph, row, numSamples);
                break;
            }
        }
    } catch (...) {
        return false; // bad_alloc or unexpected throw — caller expects noexcept-equivalent failure mode
    }

    const auto totalSamples = static_cast<std::ptrdiff_t>(numSamples) * numMorphPositions;
    for (std::ptrdiff_t i = 0; i < totalSamples; ++i) {
        if (!std::isfinite(out[i])) {
            return false;
        }
    }
    return true;
}

// ============================================================================
// Mixer Mode
// ============================================================================

void LaneMixer::setMixerMode(MixerMode mode) {
    mixerMode_ = mode;
    incrementVersionIfNotBatching();
}

void LaneMixer::setScanPosition(double position) {
    const double clamped = std::clamp(position, 0.0, 1.0);
    const double current = scanPosition_.load(std::memory_order_acquire);
    if (current == clamped) {
        return; // Early-return guard: prevents per-block LUT remix under static automation.
    }
    scanPosition_.store(clamped, std::memory_order_release);
    incrementMixVersionIfNotBatching();
    incrementVersionIfNotBatching();
}

void LaneMixer::setBlendAmount(double amount) {
    const double clamped = std::clamp(amount, 0.0, 1.0);
    const double current = blendAmount_.load(std::memory_order_acquire);
    if (current == clamped) {
        return; // Early-return guard: same rationale as setScanPosition.
    }
    blendAmount_.store(clamped, std::memory_order_release);
    incrementMixVersionIfNotBatching();
    incrementVersionIfNotBatching();
}

// ============================================================================
// Extrapolation Mode
// ============================================================================

void LaneMixer::setExtrapolationMode(ExtrapolationMode mode) {
    extrapolationMode_ = mode;
    incrementVersionIfNotBatching();
}

void LaneMixer::setSoftClipEnabled(bool enabled) {
    softClipEnabled_ = enabled;
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
    batchHasMixChange_ = false;
}

void LaneMixer::endBatchUpdate() {
    batchUpdateActive_ = false;
    versionCounter_.fetch_add(1, std::memory_order_release);
    if (batchHasMixChange_) {
        mixVersionCounter_.fetch_add(1, std::memory_order_release);
        batchHasMixChange_ = false;
    }
    if (onVersionChanged_)
        onVersionChanged_();
}

// ============================================================================
// Initialization
// ============================================================================

void LaneMixer::resetToDefaults() {
    const auto mutationGuard = acquireMutationGuard();

    initializeDefaults();
    incrementVersionIfNotBatching();
}

void LaneMixer::initializeDefaults() {
    const auto mutationGuard = acquireMutationGuard();

    // Harmonic-foldback factory layout: H1 fully on, odd harmonics H3..H19
    // preloaded with halving blendDepth so amplitude sweeps morph toward foldback.
    for (int laneIdx = 0; laneIdx < kDefaultLaneCount; ++laneIdx) {
        const int n = kDefaultHarmonics[static_cast<size_t>(laneIdx)];
        auto& lane = lanes_[static_cast<size_t>(laneIdx)];
        lane = Lane{};
        lane.amplitude.store(kDefaultAmplitudes[static_cast<size_t>(laneIdx)], std::memory_order_relaxed);
        lane.blendDepth.store(kDefaultBlendDepths[static_cast<size_t>(laneIdx)], std::memory_order_relaxed);
        lane.contentType = LaneContentType::Harmonic;
        lane.harmonicNumber = n;
        lane.oddSymmetryEnabled = true;
        lane.laneId = static_cast<uint32_t>(laneIdx);
        lane.customName = "H" + juce::String(n);
        lane.curveData.resize(TABLE_SIZE);
        if (n == 1) {
            for (int i = 0; i < TABLE_SIZE; ++i) {
                lane.curveData[static_cast<size_t>(i)] = normalizeIndex(i);
            }
        } else {
            for (int i = 0; i < TABLE_SIZE; ++i) {
                const double x = std::clamp(normalizeIndex(i), -1.0, 1.0);
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
    mixerMode_ = MixerMode::Blend;
}

// ============================================================================
// Serialization
// ============================================================================

namespace {
juce::ValueTree serializeSplineAnchor(const SplineAnchor& anchor) {
    juce::ValueTree anchorVT("Anchor");
    anchorVT.setProperty("x", anchor.x, nullptr);
    anchorVT.setProperty("y", anchor.y, nullptr);
    if (anchor.hasCustomTangent) {
        anchorVT.setProperty("tangent", anchor.tangent, nullptr);
    }
    if (!anchor.morphGesture.empty()) {
        anchorVT.setProperty("homeX", anchor.homeX, nullptr);
        anchorVT.setProperty("homeY", anchor.homeY, nullptr);
        anchorVT.appendChild(anchor.morphGesture.toValueTree(), nullptr);
    }
    return anchorVT;
}

juce::ValueTree serializeSplineAnchors(const std::vector<SplineAnchor>& anchors) {
    juce::ValueTree anchorsVT("SplineAnchors");
    for (const auto& anchor : anchors) {
        anchorsVT.appendChild(serializeSplineAnchor(anchor), nullptr);
    }
    return anchorsVT;
}

juce::var compressCurveData(const std::vector<double>& curveData) {
    const juce::MemoryBlock rawData(curveData.data(), curveData.size() * sizeof(double));
    juce::MemoryOutputStream compressedStream;
    {
        juce::GZIPCompressorOutputStream compressor(compressedStream);
        compressor.write(rawData.getData(), rawData.getSize());
    }
    return {compressedStream.getMemoryBlock()};
}

juce::var compressCurveDataF32(const std::vector<double>& curveData) {
    std::vector<float> floats;
    floats.reserve(curveData.size());
    for (const double v : curveData) {
        floats.push_back(static_cast<float>(v));
    }
    const juce::MemoryBlock rawData(floats.data(), floats.size() * sizeof(float));
    juce::MemoryOutputStream compressedStream;
    {
        juce::GZIPCompressorOutputStream compressor(compressedStream);
        compressor.write(rawData.getData(), rawData.getSize());
    }
    return {compressedStream.getMemoryBlock()};
}

void writeOptionalLaneStrings(juce::ValueTree& laneVT, const Lane& lane) {
    if (!lane.equationText.isEmpty()) {
        laneVT.setProperty("equationText", lane.equationText, nullptr);
    }
    if (!lane.presetSourcePath.isEmpty()) {
        laneVT.setProperty("presetSourcePath", lane.presetSourcePath, nullptr);
    }
    if (!lane.customName.isEmpty()) {
        laneVT.setProperty("customName", lane.customName, nullptr);
    }
    if (lane.oddSymmetryEnabled) {
        laneVT.setProperty("oddSymmetryEnabled", true, nullptr);
    }
}

juce::ValueTree serializeLane(const Lane& lane, int index) {
    juce::ValueTree laneVT("Lane");
    laneVT.setProperty("index", index, nullptr);
    laneVT.setProperty("laneId", static_cast<int>(lane.laneId), nullptr);
    laneVT.setProperty("amplitude", lane.amplitude.load(std::memory_order_acquire), nullptr);
    laneVT.setProperty("blendDepth", lane.blendDepth.load(std::memory_order_acquire), nullptr);
    // formatVersion 7+ stores per-slot modulation depth (one property per slot).
    laneVT.setProperty("modulationDepth_slot1", lane.modulationDepth[0].load(std::memory_order_acquire), nullptr);
    laneVT.setProperty("modulationDepth_slot2", lane.modulationDepth[1].load(std::memory_order_acquire), nullptr);
    laneVT.setProperty("contentType", static_cast<int>(lane.contentType), nullptr);
    laneVT.setProperty("harmonicNumber", lane.harmonicNumber, nullptr);
    laneVT.setProperty("harmonicStrength", lane.harmonicStrength, nullptr);

    writeOptionalLaneStrings(laneVT, lane);

    if (!lane.splineAnchors.empty()) {
        laneVT.appendChild(serializeSplineAnchors(lane.splineAnchors), nullptr);
    }

    // Skip curveData when the lane's definition can fully recreate the curve on load:
    //   - Harmonic lanes (formatVersion >= 3) regenerate from harmonicNumber/harmonicStrength.
    //     Exception: harmonicNumber==0 (WT base layer) may hold legacy unnormalized data
    //     that can't be perfectly regenerated, so always serialize it.
    //   - Equation lanes (formatVersion >= 8) regenerate from equationText.
    //     Equation lanes with empty equationText fall back to saving curveData.
    //   - Spline lanes (formatVersion >= 9) regenerate from splineAnchors via
    //     synthesizeSplineLaneLUT(morph=0). Spline lanes with empty anchors fall back.
    const bool canRegenerateHarmonic = (lane.contentType == LaneContentType::Harmonic && lane.harmonicNumber != 0);
    const bool canRegenerateEquation = (lane.contentType == LaneContentType::Equation && !lane.equationText.isEmpty());
    const bool canRegenerateSpline = (lane.contentType == LaneContentType::Spline && !lane.splineAnchors.empty());
    const bool canRegenerate = canRegenerateHarmonic || canRegenerateEquation || canRegenerateSpline;
    if (!canRegenerate && !lane.curveData.empty()) {
        laneVT.setProperty("curveDataF32", compressCurveDataF32(lane.curveData), nullptr);
    }
    return laneVT;
}
} // namespace

juce::ValueTree LaneMixer::toValueTree() const {
    juce::ValueTree vt("LaneMixer");
    // formatVersion 8:
    //   - equation lanes omit curveData (regenerated from equationText on load)
    //   - non-regenerated curves are stored as gzip(float32) under the new `curveDataF32` property.
    //     v<=7 wrote gzip(double[]) under `curveData`; the loader still accepts that for legacy files.
    // formatVersion 9:
    //   - spline lanes omit curveData when splineAnchors is non-empty
    //     (regenerated from anchors via synthesizeSplineLaneLUT(morph=0) on load).
    vt.setProperty("formatVersion", 9, nullptr);
    vt.setProperty("numLanes", activeLaneCount_, nullptr);
    vt.setProperty("nextLaneId", static_cast<int>(nextLaneId_), nullptr);
    vt.setProperty("tableSize", TABLE_SIZE, nullptr);
    vt.setProperty("mixerMode", static_cast<int>(mixerMode_), nullptr);
    vt.setProperty("blendAmount", blendAmount_.load(std::memory_order_acquire), nullptr);

    for (int i = 0; i < activeLaneCount_; ++i) {
        vt.appendChild(serializeLane(lanes_[static_cast<size_t>(i)], i), nullptr);
    }

    return vt;
}

namespace {
SplineAnchor deserializeAnchor(const juce::ValueTree& anchorVT) {
    SplineAnchor anchor;
    anchor.x = static_cast<double>(anchorVT.getProperty("x", 0.0));
    anchor.y = static_cast<double>(anchorVT.getProperty("y", 0.0));
    if (anchorVT.hasProperty("tangent")) {
        anchor.hasCustomTangent = true;
        anchor.tangent = static_cast<double>(anchorVT.getProperty("tangent", 0.0));
    }
    const auto gestureVT = anchorVT.getChildWithName("Gesture");
    if (gestureVT.isValid()) {
        anchor.morphGesture = AnchorMorphGesture::fromValueTree(gestureVT);
        anchor.homeX = static_cast<double>(anchorVT.getProperty("homeX", anchor.x));
        anchor.homeY = static_cast<double>(anchorVT.getProperty("homeY", anchor.y));
    }
    return anchor;
}

std::vector<SplineAnchor> deserializeAnchors(const juce::ValueTree& anchorsVT) {
    std::vector<SplineAnchor> result;
    if (!anchorsVT.isValid()) {
        return result;
    }
    result.reserve(static_cast<size_t>(anchorsVT.getNumChildren()));
    for (int a = 0; a < anchorsVT.getNumChildren(); ++a) {
        result.push_back(deserializeAnchor(anchorsVT.getChild(a)));
    }
    return result;
}

// Linear-resample a source array of N samples (covering normalized [0, 1]) into `out`, which
// will be sized to `targetSize`. Used when a preset's stored tableSize differs from the live
// TABLE_SIZE (e.g. 16384-era preset loaded against a 8192 build, or vice versa). Linear is
// adequate for this one-shot load path — the audio thread re-interpolates via Catmull-Rom.
template <typename SrcT>
void resampleLinearInto(const SrcT* src, size_t numSrc, std::vector<double>& out, int targetSize) {
    out.assign(static_cast<size_t>(targetSize), 0.0);
    if (numSrc == 0 || targetSize <= 0) {
        return;
    }
    if (numSrc == 1) {
        std::fill(out.begin(), out.end(), static_cast<double>(src[0]));
        return;
    }
    const double scale = static_cast<double>(numSrc - 1) / static_cast<double>(targetSize - 1);
    for (int i = 0; i < targetSize; ++i) {
        const double srcPos = i * scale;
        const auto lo = static_cast<size_t>(srcPos);
        const auto hi = std::min(lo + 1, numSrc - 1);
        const double frac = srcPos - static_cast<double>(lo);
        out[i] = static_cast<double>(src[lo]) * (1.0 - frac) + static_cast<double>(src[hi]) * frac;
    }
}

bool decompressCurveDataInto(const juce::var& curveDataProp, std::vector<double>& out, int loadedTableSize,
                             int tableSize) {
    const auto* compressedData = curveDataProp.getBinaryData();
    if (compressedData == nullptr) {
        return false;
    }
    juce::MemoryInputStream compressedStream(*compressedData, false);
    juce::GZIPDecompressorInputStream decompressor(compressedStream);

    const size_t expectedSize = static_cast<size_t>(loadedTableSize) * sizeof(double);
    juce::MemoryBlock decompressed;
    decompressed.setSize(expectedSize);

    const auto bytesRead = decompressor.read(decompressed.getData(), static_cast<int>(expectedSize));
    if (bytesRead <= 0) {
        return false;
    }
    const auto numDoubles = static_cast<size_t>(bytesRead) / sizeof(double);
    const auto* src = static_cast<const double*>(decompressed.getData());
    if (static_cast<int>(numDoubles) == tableSize) {
        out.assign(src, src + numDoubles);
    } else {
        resampleLinearInto(src, numDoubles, out, tableSize);
    }
    return true;
}

bool decompressCurveDataF32Into(const juce::var& curveDataProp, std::vector<double>& out, int loadedTableSize,
                                int tableSize) {
    const auto* compressedData = curveDataProp.getBinaryData();
    if (compressedData == nullptr) {
        return false;
    }
    juce::MemoryInputStream compressedStream(*compressedData, false);
    juce::GZIPDecompressorInputStream decompressor(compressedStream);

    const size_t expectedSize = static_cast<size_t>(loadedTableSize) * sizeof(float);
    juce::MemoryBlock decompressed;
    decompressed.setSize(expectedSize);

    const auto bytesRead = decompressor.read(decompressed.getData(), static_cast<int>(expectedSize));
    if (bytesRead <= 0) {
        return false;
    }
    const auto numFloats = static_cast<size_t>(bytesRead) / sizeof(float);
    const auto* src = static_cast<const float*>(decompressed.getData());
    if (static_cast<int>(numFloats) == tableSize) {
        out.assign(static_cast<size_t>(tableSize), 0.0);
        for (size_t i = 0; i < numFloats; ++i) {
            out[i] = static_cast<double>(src[i]);
        }
    } else {
        resampleLinearInto(src, numFloats, out, tableSize);
    }
    return true;
}
} // namespace

void LaneMixer::applyLaneFromValueTree(Lane& lane, const juce::ValueTree& laneVT, int formatVersion, int laneIndex) {
    // Read lane ID for v3+ format
    if (formatVersion >= 3) {
        lane.laneId = static_cast<uint32_t>(static_cast<int>(laneVT.getProperty("laneId", laneIndex)));
    }

    lane.amplitude.store(static_cast<double>(laneVT.getProperty("amplitude", 0.0)), std::memory_order_release);
    // formatVersion 4+ adds per-lane blendDepth; default 0 for older versions.
    lane.blendDepth.store(static_cast<double>(laneVT.getProperty("blendDepth", 0.0)), std::memory_order_release);
    // formatVersion 7+ stores per-slot modulationDepth as two separate properties.
    // formatVersion 6 stored a single legacy `modulationDepth` — load that into
    // slot 1 only (slot 2 stays at 0) so old presets retain audible behavior.
    // Pre-v6 presets have neither and stay at 0/0.
    if (formatVersion >= 7) {
        lane.modulationDepth[0].store(static_cast<double>(laneVT.getProperty("modulationDepth_slot1", 0.0)),
                                      std::memory_order_release);
        lane.modulationDepth[1].store(static_cast<double>(laneVT.getProperty("modulationDepth_slot2", 0.0)),
                                      std::memory_order_release);
    } else {
        lane.modulationDepth[0].store(static_cast<double>(laneVT.getProperty("modulationDepth", 0.0)),
                                      std::memory_order_release);
        lane.modulationDepth[1].store(0.0, std::memory_order_release);
    }
    lane.contentType = static_cast<LaneContentType>(static_cast<int>(laneVT.getProperty("contentType", 0)));
    lane.harmonicNumber = laneVT.getProperty("harmonicNumber", 0);
    // formatVersion 5+ adds per-lane harmonicStrength; default 1.0 for older presets.
    lane.harmonicStrength = juce::jlimit(-1.0, 1.0, static_cast<double>(laneVT.getProperty("harmonicStrength", 1.0)));
    lane.equationText = laneVT.getProperty("equationText", juce::String()).toString();
    lane.presetSourcePath = laneVT.getProperty("presetSourcePath", juce::String()).toString();
    lane.customName = laneVT.getProperty("customName", juce::String()).toString();
    lane.oddSymmetryEnabled = static_cast<bool>(laneVT.getProperty("oddSymmetryEnabled", false));

    lane.splineAnchors = deserializeAnchors(laneVT.getChildWithName("SplineAnchors"));
}

void LaneMixer::loadOrRegenerateLaneCurve(Lane& lane, const juce::ValueTree& laneVT, int laneIndex,
                                          int loadedTableSize) {
    // formatVersion >= 8 emits curveDataF32 (gzip(float32[])); older versions emitted
    // curveData (gzip(double[])). Prefer F32, fall back to legacy doubles.
    if (laneVT.hasProperty("curveDataF32")) {
        decompressCurveDataF32Into(laneVT.getProperty("curveDataF32"), lane.curveData, loadedTableSize, TABLE_SIZE);
        return;
    }
    if (laneVT.hasProperty("curveData")) {
        decompressCurveDataInto(laneVT.getProperty("curveData"), lane.curveData, loadedTableSize, TABLE_SIZE);
        return;
    }
    // No curve data in serialized form — regenerate from content type.
    if (lane.contentType == LaneContentType::Harmonic) {
        if (lane.harmonicNumber == 0) {
            fillLaneWithTanh2x(laneIndex);
        } else {
            fillLaneWithHarmonic(laneIndex, lane.harmonicNumber, lane.harmonicStrength);
        }
        return;
    }
    if (lane.contentType == LaneContentType::Equation && !lane.equationText.isEmpty()) {
        // Mirrors evaluateStaticThumbnail's compile+synthesize path. normalize=true
        // matches the codebase's canonical rehydrate default; load-time morph=0 is
        // short-lived (the live audio path re-synthesizes on every morph change).
        dsp_core::Services::CompiledLaneEquation compiled;
        if (lane.curveData.size() != static_cast<size_t>(TABLE_SIZE)) {
            lane.initCurveData(TABLE_SIZE);
        }
        if (dsp_core::Services::compileLaneEquation(lane.equationText, /*normalize=*/true, compiled) &&
            dsp_core::Services::synthesizeLaneLUT(compiled, /*morph=*/0.0, *harmonicLayer_, TABLE_SIZE,
                                                  lane.curveData)) {
            return;
        }
        // compile or synthesis failed (corrupt equationText, NaN/Inf) — fall through to zero-fill.
    }
    if (lane.contentType == LaneContentType::Spline && !lane.splineAnchors.empty()) {
        // Mirrors evaluateStaticThumbnail's Spline branch (LaneMixer.cpp ~795). Load-time
        // morph=0 is short-lived; the live audio path re-synthesizes when morph state
        // arrives via APVTS (CurveEditorController::morphLaneAnchors).
        if (lane.curveData.size() != static_cast<size_t>(TABLE_SIZE)) {
            lane.initCurveData(TABLE_SIZE);
        }
        std::vector<SplineAnchor> scratchAnchors;
        if (dsp_core::Services::synthesizeSplineLaneLUT(lane.splineAnchors, /*morph=*/0.0, TABLE_SIZE,
                                                         scratchAnchors, lane.curveData)) {
            return;
        }
        // synthesis failed (degenerate anchors, NaN) — fall through to zero-fill.
    }
    lane.initCurveData(TABLE_SIZE);
}

void LaneMixer::resynthesizeMorphableLanes(double morph) {
    const auto mutationGuard = acquireMutationGuard();

    bool anyChanged = false;
    for (int i = 0; i < activeLaneCount_; ++i) {
        auto& lane = lanes_[static_cast<size_t>(i)];

        if (lane.contentType == LaneContentType::Equation && !lane.equationText.isEmpty()) {
            dsp_core::Services::CompiledLaneEquation compiled;
            if (lane.curveData.size() != static_cast<size_t>(TABLE_SIZE)) {
                lane.initCurveData(TABLE_SIZE);
            }
            if (dsp_core::Services::compileLaneEquation(lane.equationText, /*normalize=*/true, compiled) &&
                dsp_core::Services::synthesizeLaneLUT(compiled, morph, *harmonicLayer_, TABLE_SIZE, lane.curveData)) {
                anyChanged = true;
            }
            // compile/synthesis failure → leave existing curveData untouched.
            continue;
        }

        if (lane.contentType == LaneContentType::Spline && !lane.splineAnchors.empty()) {
            // Mirrors CurveEditorController::morphLaneAnchors — write-back morphed
            // anchor positions so subsequent edits operate on current positions
            // (homeX/homeY are preserved per anchor for future morph re-evaluation).
            std::vector<SplineAnchor> morphedAnchors;
            std::vector<double> outLut;
            if (dsp_core::Services::synthesizeSplineLaneLUT(lane.splineAnchors, morph, TABLE_SIZE, morphedAnchors,
                                                             outLut)) {
                lane.splineAnchors = std::move(morphedAnchors);
                lane.curveData = std::move(outLut);
                anyChanged = true;
            }
            continue;
        }
    }
    if (anyChanged) {
        incrementVersionIfNotBatching();
    }
}

void LaneMixer::fromValueTree(const juce::ValueTree& vt) {
    const auto mutationGuard = acquireMutationGuard();

    if (!vt.isValid() || vt.getType().toString() != "LaneMixer") {
        return;
    }

    const int formatVersion = vt.getProperty("formatVersion", 2);
    const int loadedTableSize = vt.getProperty("tableSize", TABLE_SIZE);

    mixerMode_ = static_cast<MixerMode>(static_cast<int>(vt.getProperty("mixerMode", 0)));

    // formatVersion 4+ adds blendAmount; default 0 for older versions.
    blendAmount_.store(static_cast<double>(vt.getProperty("blendAmount", 0.0)), std::memory_order_release);

    if (formatVersion >= 3) {
        // Dynamic format: read lane count, IDs, nextLaneId
        activeLaneCount_ = juce::jlimit(1, MAX_LANES, static_cast<int>(vt.getProperty("numLanes", kDefaultLaneCount)));
        nextLaneId_ = static_cast<uint32_t>(static_cast<int>(vt.getProperty("nextLaneId", activeLaneCount_)));
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
        if (laneVT.getType().toString() != "Lane") {
            continue;
        }

        const int laneIndex = laneVT.getProperty("index", -1);
        if (laneIndex < 0 || laneIndex >= activeLaneCount_) {
            continue;
        }

        auto& lane = lanes_[static_cast<size_t>(laneIndex)];
        applyLaneFromValueTree(lane, laneVT, formatVersion, laneIndex);
        loadOrRegenerateLaneCurve(lane, laneVT, laneIndex, loadedTableSize);
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

namespace {
std::vector<double> parseLegacyCoefficients(const juce::ValueTree& ltfVT) {
    std::vector<double> result;
    if (!ltfVT.hasProperty("coefficients")) {
        return result;
    }
    const juce::Array<juce::var>* coeffArray = ltfVT.getProperty("coefficients").getArray();
    if (coeffArray == nullptr) {
        return result;
    }
    result.reserve(static_cast<size_t>(coeffArray->size()));
    for (const auto& coeff : *coeffArray) {
        result.push_back(static_cast<double>(coeff));
    }
    return result;
}

std::vector<double> parseLegacyBaseLayer(const juce::ValueTree& ltfVT, int tableSize) {
    std::vector<double> result;
    const auto baseVT = ltfVT.getChildWithName("BaseLayer");
    if (!baseVT.isValid() || !baseVT.hasProperty("tableData")) {
        return result;
    }
    const juce::MemoryBlock baseBlob = *baseVT.getProperty("tableData").getBinaryData();
    const auto* data = static_cast<const double*>(baseBlob.getData());
    const int numValues = static_cast<int>(baseBlob.getSize() / sizeof(double));
    if (numValues == tableSize) {
        result.assign(data, data + numValues);
    } else {
        // Legacy presets baked the BaseLayer at the era's TABLE_SIZE (typically 16384).
        // Resample to the live size so the full x∈[-1,1] curve is preserved.
        resampleLinearInto(data, static_cast<size_t>(numValues), result, tableSize);
    }
    return result;
}

SplineAnchor parseLegacyAnchor(const juce::ValueTree& anchorVT) {
    SplineAnchor anchor;
    anchor.x = static_cast<double>(anchorVT.getProperty("x", 0.0));
    anchor.y = static_cast<double>(anchorVT.getProperty("y", 0.0));
    anchor.tangent = static_cast<double>(anchorVT.getProperty("tangent", 0.0));
    anchor.hasCustomTangent = static_cast<bool>(anchorVT.getProperty("hasCustomTangent", false));
    return anchor;
}

// Returns (anchors, lane0ContentType). lane0ContentType is Spline iff a SplineLayer was present.
std::pair<std::vector<SplineAnchor>, LaneContentType> parseLegacySplineAnchors(const juce::ValueTree& ltfVT) {
    std::vector<SplineAnchor> anchors;
    LaneContentType contentType = LaneContentType::Harmonic;
    const auto splineVT = ltfVT.getChildWithName("SplineLayer");
    if (!splineVT.isValid()) {
        return {anchors, contentType};
    }
    contentType = LaneContentType::Spline;
    const auto anchorsVT = splineVT.getChildWithName("Anchors");
    if (anchorsVT.isValid()) {
        for (int a = 0; a < anchorsVT.getNumChildren(); ++a) {
            anchors.push_back(parseLegacyAnchor(anchorsVT.getChild(a)));
        }
    }
    return {anchors, contentType};
}

// Pure Chebyshev harmonic curve: H1=identity, even=cos(n*acos(x)), odd=sin(n*asin(x)).
void fillChebyshevHarmonic(std::vector<double>& curveData, int tableSize, int harmonicNumber) {
    curveData.resize(static_cast<size_t>(tableSize));
    for (int i = 0; i < tableSize; ++i) {
        const double x = std::clamp(LaneMixer::normalizeIndex(i), -1.0, 1.0);
        double value;
        if (harmonicNumber == 1) {
            value = x;
        } else if (harmonicNumber % 2 == 0) {
            value = std::cos(harmonicNumber * std::acos(x));
        } else {
            value = std::sin(harmonicNumber * std::asin(x));
        }
        curveData[static_cast<size_t>(i)] = value;
    }
}

// Odd-only variant used by the Case-A symmetric branch (skips the cos branch).
void fillOddHarmonic(std::vector<double>& curveData, int tableSize, int harmonicNumber) {
    curveData.resize(static_cast<size_t>(tableSize));
    for (int i = 0; i < tableSize; ++i) {
        const double x = std::clamp(LaneMixer::normalizeIndex(i), -1.0, 1.0);
        curveData[static_cast<size_t>(i)] = (harmonicNumber == 1) ? x : std::sin(harmonicNumber * std::asin(x));
    }
}

double coefficientOrZero(const std::vector<double>& coefficients, size_t index) {
    return index < coefficients.size() ? coefficients[index] : 0.0;
}
} // namespace

void LaneMixer::initializeLegacyLane0(Lane& lane0, LaneContentType contentType,
                                      const std::vector<SplineAnchor>& anchors, double amplitude,
                                      const std::vector<double>& baseLayerData) {
    lane0.laneId = 0;
    lane0.contentType = contentType;
    lane0.harmonicNumber = 0;
    lane0.splineAnchors = anchors;
    lane0.amplitude.store(amplitude, std::memory_order_relaxed);
    if (!baseLayerData.empty()) {
        lane0.curveData = baseLayerData;
        return;
    }
    lane0.curveData.resize(TABLE_SIZE);
    for (int i = 0; i < TABLE_SIZE; ++i) {
        lane0.curveData[static_cast<size_t>(i)] = std::tanh(2.0 * normalizeIndex(i));
    }
    Services::TransferFunctionOperations::normalize(lane0.curveData);
}

void LaneMixer::restoreLegacyHarmonicSymmetricMode(const std::vector<double>& legacyCoefficients,
                                                   const std::vector<double>& baseLayerData,
                                                   const std::vector<SplineAnchor>& legacySplineAnchors,
                                                   LaneContentType lane0ContentType) {
    // Case A — harmonic mode with odd symmetry: Lane 0 + odd harmonics only (21 lanes)
    initializeLegacyLane0(lanes_[0], lane0ContentType, legacySplineAnchors, coefficientOrZero(legacyCoefficients, 0),
                          baseLayerData);

    // Odd harmonics: H1, H3, H5, ..., H39 (20 lanes at indices 1-20)
    int laneIdx = 1;
    for (int h = 1; h <= 39; h += 2) {
        auto& lane = lanes_[static_cast<size_t>(laneIdx)];
        lane.laneId = static_cast<uint32_t>(laneIdx);
        lane.contentType = LaneContentType::Harmonic;
        lane.harmonicNumber = h;
        lane.amplitude.store(coefficientOrZero(legacyCoefficients, static_cast<size_t>(h)), std::memory_order_relaxed);
        fillOddHarmonic(lane.curveData, TABLE_SIZE, h);
        laneIdx++;
    }

    activeLaneCount_ = 21; // Lane 0 + 20 odd harmonics
    nextLaneId_ = 21;
}

void LaneMixer::restoreLegacyHarmonicMode(const std::vector<double>& legacyCoefficients,
                                          const std::vector<double>& baseLayerData,
                                          const std::vector<SplineAnchor>& legacySplineAnchors,
                                          LaneContentType lane0ContentType) {
    // Case B — harmonic mode without odd symmetry: Lane 0 + H1-H40 (41 lanes)
    initializeLegacyLane0(lanes_[0], lane0ContentType, legacySplineAnchors, coefficientOrZero(legacyCoefficients, 0),
                          baseLayerData);

    for (int n = 1; n <= 40; ++n) {
        auto& lane = lanes_[static_cast<size_t>(n)];
        lane.laneId = static_cast<uint32_t>(n);
        lane.contentType = LaneContentType::Harmonic;
        lane.harmonicNumber = n;
        lane.amplitude.store(coefficientOrZero(legacyCoefficients, static_cast<size_t>(n)), std::memory_order_relaxed);
        fillChebyshevHarmonic(lane.curveData, TABLE_SIZE, n);
    }

    activeLaneCount_ = 41;
    nextLaneId_ = 41;
}

void LaneMixer::restoreLegacyNonHarmonicMode(const std::vector<double>& legacyCoefficients,
                                             const std::vector<double>& baseLayerData,
                                             const std::vector<SplineAnchor>& legacySplineAnchors,
                                             LaneContentType lane0ContentType) {
    // Case C — non-harmonic mode (Paint, Equation, Spline): Lane 0 + H1 + 9 odd harmonics (11 lanes)
    initializeLegacyLane0(lanes_[0], lane0ContentType, legacySplineAnchors, coefficientOrZero(legacyCoefficients, 0),
                          baseLayerData);

    // Lane 1: H1 (identity), amplitude=0.0
    {
        auto& lane1 = lanes_[1];
        lane1.laneId = 1;
        lane1.contentType = LaneContentType::Harmonic;
        lane1.harmonicNumber = 1;
        lane1.oddSymmetryEnabled = true;
        lane1.amplitude.store(0.0, std::memory_order_relaxed);
        lane1.curveData.resize(TABLE_SIZE);
        for (int i = 0; i < TABLE_SIZE; ++i) {
            lane1.curveData[static_cast<size_t>(i)] = normalizeIndex(i);
        }
    }

    // Lanes 2-10: Odd harmonics H3,H5,...,H19, amplitude=0.0
    constexpr std::array<int, 9> kLegacyOddHarmonics = {3, 5, 7, 9, 11, 13, 15, 17, 19};
    for (size_t laneIdx = 0; laneIdx < kLegacyOddHarmonics.size(); ++laneIdx) {
        const int n = kLegacyOddHarmonics[laneIdx];
        auto& lane = lanes_[laneIdx + 2];
        lane.laneId = static_cast<uint32_t>(laneIdx + 2);
        lane.contentType = LaneContentType::Harmonic;
        lane.harmonicNumber = n;
        lane.oddSymmetryEnabled = true;
        lane.amplitude.store(0.0, std::memory_order_relaxed);
        fillOddHarmonic(lane.curveData, TABLE_SIZE, n);
    }

    // Legacy LTF Case C lays out lane 0 (saved) + H1 + 9 odd harmonics = 11 lanes,
    // independent of the new factory default count.
    activeLaneCount_ = 11;
    nextLaneId_ = 11;
}

void LaneMixer::fromLegacyLTFValueTree(const juce::ValueTree& ltfVT, bool wasHarmonicMode, bool oddSymmetryWasEnabled) {
    if (!ltfVT.isValid() || ltfVT.getType().toString() != "LayeredTransferFunction") {
        return;
    }

    const auto legacyCoefficients = parseLegacyCoefficients(ltfVT);
    const auto baseLayerData = parseLegacyBaseLayer(ltfVT, TABLE_SIZE);
    const auto [legacySplineAnchors, lane0ContentType] = parseLegacySplineAnchors(ltfVT);

    // Clear all lanes
    for (int i = 0; i < MAX_LANES; ++i) {
        lanes_[static_cast<size_t>(i)] = Lane{};
    }

    if (wasHarmonicMode && oddSymmetryWasEnabled) {
        restoreLegacyHarmonicSymmetricMode(legacyCoefficients, baseLayerData, legacySplineAnchors, lane0ContentType);
    } else if (wasHarmonicMode) {
        restoreLegacyHarmonicMode(legacyCoefficients, baseLayerData, legacySplineAnchors, lane0ContentType);
    } else {
        restoreLegacyNonHarmonicMode(legacyCoefficients, baseLayerData, legacySplineAnchors, lane0ContentType);
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
            if (lane.contentType == LaneContentType::Harmonic && lane.harmonicNumber == candidate) {
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

double LaneMixer::normalizeIndex(int index) {
    return juce::jmap(static_cast<double>(index), 0.0, static_cast<double>(TABLE_SIZE - 1), MIN_VALUE, MAX_VALUE);
}

} // namespace dsp_core
