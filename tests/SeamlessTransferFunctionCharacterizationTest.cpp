#include <dsp_core/dsp_core.h>
#include <gtest/gtest.h>
#include <cmath>

namespace dsp_core_test {

/**
 * Characterization tests for SeamlessTransferFunction.
 *
 * These tests pin the current audio output behavior so we can verify
 * that the LaneMixer integration (Phase 2) is transparent — same input
 * produces identical output before and after the refactor.
 *
 * Approach: Use renderLUTImmediate() for synchronous rendering, then
 * process a ramp buffer through the audio engine and verify output.
 */
class SeamlessTransferFunctionCharacterizationTest : public ::testing::Test {
  protected:
    void SetUp() override {
        stf = std::make_unique<dsp_core::SeamlessTransferFunction>();
        // prepareToPlay with a standard sample rate
        stf->prepareToPlay(44100.0, 512);
    }

    std::unique_ptr<dsp_core::SeamlessTransferFunction> stf;

    /**
     * Render LUT synchronously and process a ramp buffer through the audio engine.
     * Returns the output samples for verification.
     *
     * The ramp goes from -1.0 to +1.0 across numSamples.
     * A warmup buffer is processed first to flush any active crossfade.
     */
    std::vector<double> renderAndProcess(int numSamples = 1024) {
        // Render LUT synchronously (bypasses worker thread)
        stf->renderLUTImmediate();

        // Process a warmup buffer to flush the crossfade (50ms @ 44100 = 2205 samples)
        juce::AudioBuffer<double> warmup(1, 4096);
        warmup.clear();
        stf->processBuffer(warmup);

        // Create a mono ramp buffer from -1 to +1
        juce::AudioBuffer<double> buffer(1, numSamples);
        auto* data = buffer.getWritePointer(0);
        for (int i = 0; i < numSamples; ++i) {
            data[i] = -1.0 + 2.0 * (i / static_cast<double>(numSamples - 1));
        }

        // Process through audio engine (crossfade completed, pure new LUT)
        stf->processBuffer(buffer);

        // Return output
        std::vector<double> result(data, data + numSamples);
        return result;
    }

    /**
     * Get max absolute value in a buffer.
     */
    static double maxAbs(const std::vector<double>& buf) {
        double m = 0.0;
        for (auto v : buf) m = std::max(m, std::abs(v));
        return m;
    }
};

TEST_F(SeamlessTransferFunctionCharacterizationTest, DefaultLUT_IsIdentityFunction) {
    // Default: WT=0.0, H1=1.0, base=tanh(2x), renderingMode=Harmonic
    // Composite = normScalar * (0.0 * tanh(2x) + 1.0 * x) = normScalar * x
    // max(|x|) on [-1,1] = 1.0, so normScalar = 1.0
    // Output: y = x (identity)

    auto output = renderAndProcess();

    for (int i = 0; i < static_cast<int>(output.size()); i += 50) {
        const double x = -1.0 + 2.0 * (i / static_cast<double>(output.size() - 1));
        EXPECT_NEAR(output[static_cast<size_t>(i)], x, 0.01)
            << "Default TF should be identity (y=x) at sample " << i;
    }
}

TEST_F(SeamlessTransferFunctionCharacterizationTest, AfterHarmonicChange_LUTReflectsNewCurve) {
    auto& ltf = stf->getEditingModel();

    // Set H1=0.5, H3=0.5 (mix of linear and cubic harmonics)
    auto coeffs = ltf.getHarmonicCoefficients();
    coeffs[1] = 0.5;  // H1
    coeffs[3] = 0.5;  // H3
    ltf.setHarmonicCoefficients(coeffs);

    auto output = renderAndProcess();

    // The output should NOT be identity anymore
    // At x=0.5: H1 contributes 0.5*0.5=0.25, H3 contributes 0.5*sin(3*asin(0.5))
    // Just verify it's different from identity and in [-1, 1]
    bool differsFromIdentity = false;
    for (int i = 0; i < static_cast<int>(output.size()); i += 50) {
        const double x = -1.0 + 2.0 * (i / static_cast<double>(output.size() - 1));
        if (std::abs(output[static_cast<size_t>(i)] - x) > 0.05) {
            differsFromIdentity = true;
        }
        EXPECT_GE(output[static_cast<size_t>(i)], -1.01);
        EXPECT_LE(output[static_cast<size_t>(i)], 1.01);
    }
    EXPECT_TRUE(differsFromIdentity) << "H1+H3 mix should differ from identity";
}

TEST_F(SeamlessTransferFunctionCharacterizationTest, PaintModeRender_MixerStillActive) {
    auto& ltf = stf->getEditingModel();

    // Set WT mix to 1.0, all harmonics to zero, switch to Paint mode
    // With the always-on mixer, Paint mode keeps all lanes active.
    // Only lane 0 (WT=1.0, tanh(2x)) contributes here since harmonics are zero.
    // Normalization is applied: output = tanh(2x) / max(|tanh(2x)|)
    auto coeffs = ltf.getHarmonicCoefficients();
    std::fill(coeffs.begin(), coeffs.end(), 0.0);
    coeffs[0] = 1.0;  // WT mix = 1.0
    ltf.setHarmonicCoefficients(coeffs);
    ltf.setRenderingMode(dsp_core::RenderingMode::Paint);

    auto output = renderAndProcess();

    // With normalization, the output is tanh(2x) scaled so max abs = 1.0
    const double maxVal = maxAbs(output);
    EXPECT_NEAR(maxVal, 1.0, 0.02) << "Normalized tanh(2x) should have max abs ≈ 1.0";

    // Verify shape is still tanh-like (monotonic, passes through zero)
    EXPECT_NEAR(output[output.size() / 2], 0.0, 0.02) << "Midpoint should be near zero";
    EXPECT_GT(output.back(), 0.9) << "Right endpoint should be positive and large";
    EXPECT_LT(output.front(), -0.9) << "Left endpoint should be negative and large";
}

TEST_F(SeamlessTransferFunctionCharacterizationTest, NormalizationEnabled_MaxAbsIsOne) {
    auto& ltf = stf->getEditingModel();

    // Set WT=1.0, H1=1.0 in Harmonic mode (composite will be tanh(2x) + x)
    // The max abs of tanh(2x)+x on [-1,1] is tanh(2)+1 ≈ 1.964
    // With normalization, output should be scaled to max abs = 1.0
    auto coeffs = ltf.getHarmonicCoefficients();
    coeffs[0] = 1.0;  // WT mix
    coeffs[1] = 1.0;  // H1
    ltf.setHarmonicCoefficients(coeffs);

    auto output = renderAndProcess();

    const double maxVal = maxAbs(output);
    EXPECT_NEAR(maxVal, 1.0, 0.02) << "Normalized output should have max abs ≈ 1.0";
}

} // namespace dsp_core_test
