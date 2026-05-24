#include <gtest/gtest.h>

#include "../dsp_core/Source/audio_pipeline/FilterStage.h"
#include "../dsp_core/Source/audio_pipeline/LadderTPTStage.h"
#include "../dsp_core/Source/audio_pipeline/LowpassStage.h"
#include "../dsp_core/Source/audio_pipeline/Tanh2xLUT.h"
#include "../dsp_core/Source/audio_pipeline/VirtualAnalogFilterStage.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace dsp_core_test {

using namespace dsp_core::audio_pipeline;

namespace {

constexpr double kSampleRate = 48000.0;
constexpr int kBlockSize = 512;
constexpr int kNumChannels = 2;
constexpr int kTotalSamples = 1024 * 1024; // ~21.8 s @ 48 kHz per channel
constexpr int kWarmupPasses = 2;
constexpr int kMeasurementPasses = 5;
constexpr double kBenchCutoffHz = 1000.0;
constexpr double kBenchResonance = 2.0;

struct BenchmarkResult {
    std::string name;
    double nsPerSample = 0.0;        // median, per stereo-sample-pair
    double ratioVsBaseline = 1.0;
    double pctRealtime = 0.0;        // % of 48 kHz stereo CPU
    double outputPeak = 0.0;
    bool finite = true;
};

void printHeader(const std::string& buildType) {
    std::cout << "\n========================================================================\n";
    std::cout << "  FilterTopologyBenchmark   build=" << buildType
              << "   fs=" << static_cast<int>(kSampleRate) << " Hz   stereo   "
              << kTotalSamples << " samples\n";
    std::cout << "  cutoff=" << kBenchCutoffHz << " Hz   R=" << kBenchResonance << "\n";
    if (buildType != "Release" && buildType != "RelWithDebInfo") {
        std::cout << "  WARNING: non-Release build. Ratios are still informative; absolute\n"
                     "  ns/sample is not representative of production performance.\n";
    }
    std::cout << "------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(36) << "Filter"
              << std::right << std::setw(12) << "ns/sample"
              << std::setw(10) << "x SVF"
              << std::setw(10) << "% CPU"
              << std::setw(14) << "peak |out|" << "\n";
    std::cout << "------------------------------------------------------------------------\n";
}

void printRow(const BenchmarkResult& r) {
    std::cout << std::left << std::setw(36) << r.name
              << std::right << std::fixed << std::setprecision(2) << std::setw(12) << r.nsPerSample
              << std::setw(10) << std::setprecision(2) << r.ratioVsBaseline
              << std::setw(10) << std::setprecision(3) << r.pctRealtime
              << std::setw(14) << std::setprecision(4) << r.outputPeak << "\n";
}

std::string getBuildType() {
#ifdef NDEBUG
    return "Release";
#else
    return "Debug";
#endif
}

void fillWhiteNoise(std::vector<double>& buf, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    for (auto& s : buf) {
        s = dist(rng);
    }
}

// Copies `src` channels into `dst` (in-place buffer the filter mutates).
void copyToBuffer(juce::AudioBuffer<double>& dst, const std::vector<std::vector<double>>& src,
                  int offset, int length) {
    for (int ch = 0; ch < dst.getNumChannels(); ++ch) {
        auto* w = dst.getWritePointer(ch);
        const auto& s = src[static_cast<std::size_t>(ch)];
        for (int i = 0; i < length; ++i) {
            w[i] = s[static_cast<std::size_t>(offset + i)];
        }
    }
}

double peakAbs(const juce::AudioBuffer<double>& buf) {
    double p = 0.0;
    for (int ch = 0; ch < buf.getNumChannels(); ++ch) {
        const auto* r = buf.getReadPointer(ch);
        for (int i = 0; i < buf.getNumSamples(); ++i) {
            const double a = std::abs(r[i]);
            if (std::isfinite(a)) {
                p = std::max(p, a);
            } else {
                return std::numeric_limits<double>::infinity();
            }
        }
    }
    return p;
}

template <typename StageT, typename Configure>
BenchmarkResult runBench(const std::string& name, Configure configure,
                         const std::vector<std::vector<double>>& noise) {
    BenchmarkResult result;
    result.name = name;

    StageT stage;
    configure(stage);
    stage.prepareToPlay(kSampleRate, kBlockSize, kNumChannels);

    juce::AudioBuffer<double> block(kNumChannels, kBlockSize);

    auto runOnePass = [&]() -> std::chrono::nanoseconds {
        stage.reset();
        // (re-prepare not needed; reset clears state, params unchanged)
        std::chrono::nanoseconds elapsed{0};
        double localPeak = 0.0;
        int offset = 0;
        while (offset + kBlockSize <= kTotalSamples) {
            copyToBuffer(block, noise, offset, kBlockSize);
            auto t0 = std::chrono::high_resolution_clock::now();
            stage.process(block);
            auto t1 = std::chrono::high_resolution_clock::now();
            elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0);
            const double p = peakAbs(block);
            if (!std::isfinite(p)) {
                result.finite = false;
            }
            localPeak = std::max(localPeak, p);
            offset += kBlockSize;
        }
        result.outputPeak = std::max(result.outputPeak, localPeak);
        return elapsed;
    };

    // Warmup
    for (int i = 0; i < kWarmupPasses; ++i) {
        (void)runOnePass();
    }

    // Measurement passes — collect, take median
    std::vector<double> nsPerSampleSamples;
    nsPerSampleSamples.reserve(kMeasurementPasses);
    const int samplesPerPass = (kTotalSamples / kBlockSize) * kBlockSize;
    for (int i = 0; i < kMeasurementPasses; ++i) {
        const auto elapsed = runOnePass();
        const double nsPerSample = static_cast<double>(elapsed.count()) / samplesPerPass;
        nsPerSampleSamples.push_back(nsPerSample);
    }
    std::sort(nsPerSampleSamples.begin(), nsPerSampleSamples.end());
    result.nsPerSample = nsPerSampleSamples[nsPerSampleSamples.size() / 2];
    // % of realtime at 48 kHz stereo: per stereo-sample-pair we have 1/48000 s
    // budget per channel-sample; total budget for both channels = 2/48000 s =
    // 41.67 us. The measured ns/sample already covers both channels (filter
    // processes the whole buffer including both channels).
    const double budgetNs = (1.0e9 * 2.0) / kSampleRate;
    result.pctRealtime = 100.0 * result.nsPerSample / budgetNs;

    return result;
}

} // namespace

class FilterTopologyBenchmark : public ::testing::Test {
  protected:
    std::vector<std::vector<double>> noise_;

    void SetUp() override {
        noise_.assign(static_cast<std::size_t>(kNumChannels),
                      std::vector<double>(kTotalSamples, 0.0));
        fillWhiteNoise(noise_[0], 0x5eedu);
        fillWhiteNoise(noise_[1], 0xc0ffeeu);
    }
};

TEST_F(FilterTopologyBenchmark, CompareAllVariants) {
    printHeader(getBuildType());

    // Baseline: SVF lowpass. Q ≈ 1/sqrt(2) is Butterworth (no resonance peak).
    // We're measuring compute cost, not voicing, so the SVF runs with
    // Butterworth Q to avoid biasing the comparison with extreme Q values.
    auto svf = runBench<LowpassStage>(
        "SVF (LowpassStage, baseline)",
        [](LowpassStage& s) {
            s.setCutoffFrequency(kBenchCutoffHz);
            s.setResonance(0.7071067811865476);
        },
        noise_);

    auto ladderTPT12 = runBench<LadderTPT12dBStage>(
        "TPT-ZDF    Ladder 12dB (2-stage)",
        [](LadderTPT12dBStage& s) {
            s.setCutoffFrequency(kBenchCutoffHz);
            s.setResonance(kBenchResonance);
        },
        noise_);

    auto ladderTPT24 = runBench<LadderTPT24dBStage>(
        "TPT-ZDF    Ladder 24dB (4-stage)",
        [](LadderTPT24dBStage& s) {
            s.setCutoffFrequency(kBenchCutoffHz);
            s.setResonance(kBenchResonance);
        },
        noise_);

    auto modLadder = runBench<VirtualAnalogFilterStage>(
        "Modular    Ladder x4",
        [](VirtualAnalogFilterStage& s) {
            s.setUniform(VirtualAnalogFilterStage::StageType::Ladder);
            s.setNumStages(4);
            s.setCutoffFrequency(kBenchCutoffHz);
            s.setResonance(kBenchResonance);
        },
        noise_);

    auto modOta = runBench<VirtualAnalogFilterStage>(
        "Modular    OTA    x4",
        [](VirtualAnalogFilterStage& s) {
            s.setUniform(VirtualAnalogFilterStage::StageType::OTA);
            s.setNumStages(4);
            s.setCutoffFrequency(kBenchCutoffHz);
            s.setResonance(kBenchResonance);
        },
        noise_);

    auto modHetero = runBench<VirtualAnalogFilterStage>(
        "Modular    Ladder/OTA/Ladder/OTA",
        [](VirtualAnalogFilterStage& s) {
            s.setStage(0, std::make_unique<LadderFilterStage>());
            s.setStage(1, std::make_unique<OTAFilterStage>());
            s.setStage(2, std::make_unique<LadderFilterStage>());
            s.setStage(3, std::make_unique<OTAFilterStage>());
            s.setNumStages(4);
            s.setCutoffFrequency(kBenchCutoffHz);
            s.setResonance(kBenchResonance);
        },
        noise_);

    const double baselineNs = svf.nsPerSample > 0.0 ? svf.nsPerSample : 1.0;
    for (auto* r : {&svf, &ladderTPT12, &ladderTPT24, &modLadder, &modOta, &modHetero}) {
        r->ratioVsBaseline = r->nsPerSample / baselineNs;
    }

    printRow(svf);
    printRow(ladderTPT12);
    printRow(ladderTPT24);
    printRow(modLadder);
    printRow(modOta);
    printRow(modHetero);
    std::cout << "========================================================================\n\n";

    // Sanity gates — every variant must stay finite and below a sane peak.
    // Filter input is white noise [-0.5, 0.5]; tanh saturation in the feedback
    // path means output should not blow up even at R=2.
    for (const auto* r : {&svf, &ladderTPT12, &ladderTPT24, &modLadder, &modOta, &modHetero}) {
        EXPECT_TRUE(r->finite) << r->name << " produced NaN/Inf";
        EXPECT_LT(r->outputPeak, 10.0)
            << r->name << " peak output too large: " << r->outputPeak;
    }
}

// How much do we save by running fewer stages (lower-order filter response)?
// Useful for sizing the cost of a slope/pole-count knob on the modular path.
TEST_F(FilterTopologyBenchmark, StageCountSweep) {
    std::cout << "\n------------------------------------------------------------------------\n";
    std::cout << "  StageCountSweep   (modular Ladder, varying numStages)\n";
    std::cout << "------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(36) << "Configuration"
              << std::right << std::setw(12) << "ns/sample"
              << std::setw(10) << "x 4-stg"
              << std::setw(10) << "% CPU" << "\n";
    std::cout << "------------------------------------------------------------------------\n";

    std::vector<BenchmarkResult> results;
    for (int n : {1, 2, 3, 4}) {
        auto r = runBench<VirtualAnalogFilterStage>(
            "Modular Ladder x" + std::to_string(n),
            [n](VirtualAnalogFilterStage& s) {
                s.setUniform(VirtualAnalogFilterStage::StageType::Ladder);
                s.setNumStages(n);
                s.setCutoffFrequency(kBenchCutoffHz);
                s.setResonance(kBenchResonance);
            },
            noise_);
        results.push_back(r);
    }

    const double fourStageNs = results.back().nsPerSample;
    for (auto& r : results) {
        r.ratioVsBaseline = r.nsPerSample / fourStageNs;
        std::cout << std::left << std::setw(36) << r.name
                  << std::right << std::fixed << std::setprecision(2) << std::setw(12)
                  << r.nsPerSample
                  << std::setw(10) << std::setprecision(2) << r.ratioVsBaseline
                  << std::setw(10) << std::setprecision(3) << r.pctRealtime << "\n";
    }
    std::cout << "========================================================================\n\n";

    // Sanity: fewer stages should not be slower.
    for (std::size_t i = 1; i < results.size(); ++i) {
        EXPECT_LE(results[i - 1].nsPerSample, results[i].nsPerSample * 1.1)
            << results[i - 1].name << " unexpectedly slower than " << results[i].name;
    }
}

// Run the same configuration through a sweep of resonance values to confirm
// stability — non-finite output at any R between 0 and just-below-self-osc
// would mean the feedback nonlinearity isn't doing its job.
TEST_F(FilterTopologyBenchmark, StabilityAcrossResonance) {
    const std::vector<double> resonanceValues{0.0, 1.0, 2.0, 3.0, 3.5, 3.9};

    auto driveDC = [](juce::AudioBuffer<double>& buf, double value) {
        for (int ch = 0; ch < kNumChannels; ++ch) {
            for (int i = 0; i < kBlockSize; ++i) {
                buf.getWritePointer(ch)[i] = value;
            }
        }
    };

    auto checkLadderTPT = [&]<int N>(double r) {
        LadderTPTStage<N> s;
        s.setCutoffFrequency(kBenchCutoffHz);
        s.setResonance(r);
        s.prepareToPlay(kSampleRate, kBlockSize, kNumChannels);
        juce::AudioBuffer<double> buf(kNumChannels, kBlockSize);
        driveDC(buf, 0.3);
        for (int b = 0; b < 1000; ++b) {
            s.process(buf);
        }
        const double p = peakAbs(buf);
        EXPECT_TRUE(std::isfinite(p))
            << "LadderTPTStage<" << N << "> non-finite at R=" << r;
        EXPECT_LT(p, 5.0)
            << "LadderTPTStage<" << N << "> runaway at R=" << r << " peak=" << p;
    };

    auto checkModular = [&](VirtualAnalogFilterStage::StageType type, double r,
                            const char* label) {
        VirtualAnalogFilterStage s;
        s.setUniform(type);
        s.setNumStages(4);
        s.setCutoffFrequency(kBenchCutoffHz);
        s.setResonance(r);
        s.prepareToPlay(kSampleRate, kBlockSize, kNumChannels);
        juce::AudioBuffer<double> buf(kNumChannels, kBlockSize);
        driveDC(buf, 0.3);
        for (int b = 0; b < 1000; ++b) {
            s.process(buf);
        }
        const double p = peakAbs(buf);
        EXPECT_TRUE(std::isfinite(p)) << label << " non-finite at R=" << r;
        EXPECT_LT(p, 5.0) << label << " runaway at R=" << r << " peak=" << p;
    };

    for (double r : resonanceValues) {
        checkLadderTPT.template operator()<2>(r);
        checkLadderTPT.template operator()<4>(r);
        checkModular(VirtualAnalogFilterStage::StageType::Ladder, r, "Modular Ladder");
        checkModular(VirtualAnalogFilterStage::StageType::OTA, r, "Modular OTA");
    }
}

// Tiny LUT correctness check — guards against off-by-one indexing / wrong range.
// g_tanh2xLUT.lookup(x) returns tanh(2*x); compare against std::tanh(2*x) over
// the supported input range.
TEST(Tanh2xLUT, MatchesStdTanhWithinTolerance) {
    constexpr int kNumSamples = 200;
    double maxErr = 0.0;
    for (int i = 0; i < kNumSamples; ++i) {
        const double x = -4.0 + (8.0 * i) / (kNumSamples - 1);
        const double ref = std::tanh(2.0 * x);
        const double got = g_tanh2xLUT.lookup(x);
        maxErr = std::max(maxErr, std::abs(ref - got));
    }
    // 8192-entry lerp table over [-4,4] should be well under 1e-5.
    EXPECT_LT(maxErr, 1e-5) << "Tanh2xLUT max error vs std::tanh(2x): " << maxErr;
}

} // namespace dsp_core_test
