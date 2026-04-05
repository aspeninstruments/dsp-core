#include <dsp_core/dsp_core.h>
#include <gtest/gtest.h>
#include <cmath>

namespace dsp_core_test {

/**
 * Test fixture for EventDrivenRenderer
 *
 * Tests the event-driven rendering pipeline that replaces the timer-based
 * LUTRenderTimer + LUTRendererThread system.
 *
 * Since handleAsyncUpdate() requires a JUCE message loop, we test via:
 * - forceRender(): synchronous render for initialization and testing
 * - renderLUTImmediate(): full SeamlessTransferFunction integration path
 * - Direct state verification on AudioEngine/LaneMixer
 */
class EventDrivenRendererTest : public ::testing::Test {
  protected:
    void SetUp() override {
        stf = std::make_unique<dsp_core::SeamlessTransferFunction>();
        stf->prepareToPlay(44100.0, 512);
    }

    std::unique_ptr<dsp_core::SeamlessTransferFunction> stf;

    /// Helper: render and flush crossfade so audio reflects the latest LUT
    void renderAndFlushCrossfade() {
        stf->renderLUTImmediate();
        const int crossfadeSamples = static_cast<int>(
            44100.0 * dsp_core::SeamlessConfig::CROSSFADE_DURATION_MS / 1000.0);
        juce::AudioBuffer<double> warmup(1, crossfadeSamples + 100);
        warmup.clear();
        stf->processBuffer(warmup);
    }

    /// Helper: evaluate the transfer function at a single point after flushing
    double evaluateAt(double x) {
        return stf->applyTransferFunction(x);
    }
};

// ============================================================================
// ForceRender Tests
// ============================================================================

TEST_F(EventDrivenRendererTest, ForceRender_WritesToTripleBuffer) {
    // After renderLUTImmediate, the audio engine should have a new LUT ready
    // Default state: H1=1.0 → identity y=x
    stf->renderLUTImmediate();

    // Process a buffer to consume the LUT
    juce::AudioBuffer<double> buffer(1, 1);
    buffer.setSample(0, 0, 0.5);
    stf->processBuffer(buffer);

    // Identity function: output should be ~0.5
    EXPECT_NEAR(buffer.getSample(0, 0), 0.5, 0.01);
}

TEST_F(EventDrivenRendererTest, ForceRender_SumMatchesLaneMixer) {
    auto& mixer = stf->getLaneMixer();

    // Set H3 amplitude to 1.0 (in addition to H1=1.0)
    mixer.setLaneAmplitude(2, 1.0); // Lane 2 = H3

    renderAndFlushCrossfade();

    // Compute expected: H1(x) + H3(x) = x + sin(3*asin(x)), normalized
    // At x=0.5: H1=0.5, H3=sin(3*asin(0.5))=sin(π/2)=1.0 → sum=1.5 → normalized=0.5/1.5*...
    // Just verify it's NOT the identity function anymore
    const double output = evaluateAt(0.5);
    EXPECT_NE(output, 0.5) << "After adding H3, output should differ from identity";

    // Cross-check with direct computeSum
    std::vector<double> expected(dsp_core::LaneMixer::TABLE_SIZE, 0.0);
    mixer.computeSum(expected.data(), dsp_core::LaneMixer::TABLE_SIZE);

    // Evaluate at x=0.5 via table lookup (index for x=0.5)
    const int idx = static_cast<int>((0.5 - dsp_core::LaneMixer::MIN_VALUE) /
        (dsp_core::LaneMixer::MAX_VALUE - dsp_core::LaneMixer::MIN_VALUE) *
        (dsp_core::LaneMixer::TABLE_SIZE - 1));
    EXPECT_NEAR(output, expected[static_cast<size_t>(idx)], 0.02)
        << "Audio output should match direct computeSum";
}

// ============================================================================
// Amplitude Change Tests (should render immediately, no rate limiting)
// ============================================================================

TEST_F(EventDrivenRendererTest, AmplitudeChange_ReflectedInAudio) {
    auto& mixer = stf->getLaneMixer();

    // Start with identity (H1=1.0 only)
    renderAndFlushCrossfade();
    EXPECT_NEAR(evaluateAt(0.0), 0.0, 0.01) << "Identity at x=0 should be 0";

    // Enable H3 (which has a non-zero value at x=0 for even harmonics... actually
    // H3 is odd, sin(3*asin(0))=0. Use WT lane (tanh(2x)) which is odd too.
    // Let's use a simpler test: set all amplitudes to 0, verify silence
    mixer.setLaneAmplitude(1, 0.0); // Turn off H1
    renderAndFlushCrossfade();

    // All lanes now have amplitude 0 → sum is zero → identity fallback
    const double output = evaluateAt(0.5);
    // With all amplitudes zero, computeSum produces all zeros
    // and normalization doesn't kick in (maxAbs < epsilon)
    EXPECT_NEAR(output, 0.0, 0.01) << "With all amplitudes zero, output should be ~0";

    // Re-enable H1
    mixer.setLaneAmplitude(1, 1.0);
    renderAndFlushCrossfade();
    EXPECT_NEAR(evaluateAt(0.5), 0.5, 0.01) << "Re-enabled H1 should restore identity";
}

TEST_F(EventDrivenRendererTest, ScanChange_MixVersionIncremented) {
    // Scan position changes should increment the mix version counter
    // (enabling fast-path rendering in EventDrivenRenderer)
    auto& mixer = stf->getLaneMixer();
    const auto mv0 = mixer.getMixVersion();
    mixer.setScanPosition(0.5);
    EXPECT_GT(mixer.getMixVersion(), mv0)
        << "Scan position change should increment mix version";
}

// ============================================================================
// Version Tracking Tests
// ============================================================================

TEST_F(EventDrivenRendererTest, VersionTracking_MultipleChangesCoalesce) {
    auto& mixer = stf->getLaneMixer();

    // Make many rapid amplitude changes
    for (int i = 0; i < 100; ++i) {
        mixer.setLaneAmplitude(1, static_cast<double>(i) / 100.0);
    }

    // Single render should capture the final state
    renderAndFlushCrossfade();

    // Final amplitude was 0.99 → should be close to identity * 0.99
    // But with normalization, it's still scaled to [-1,1]
    EXPECT_NEAR(evaluateAt(0.5), 0.5, 0.01)
        << "After many changes, render should reflect final state";
}

// ============================================================================
// Integration: Full SeamlessTransferFunction lifecycle
// ============================================================================

TEST_F(EventDrivenRendererTest, Integration_DefaultState_IsIdentity) {
    renderAndFlushCrossfade();

    const std::vector<double> testInputs = {-1.0, -0.5, 0.0, 0.5, 1.0};
    for (double x : testInputs) {
        EXPECT_NEAR(evaluateAt(x), x, 0.01) << "Default state should be identity at x=" << x;
    }
}

TEST_F(EventDrivenRendererTest, Integration_CurveDataChange_ReflectedInAudio) {
    auto& mixer = stf->getLaneMixer();

    // Replace H1 curve data with a constant 0.7
    // After normalization by computeSum, constant 0.7 → constant 1.0
    std::vector<double> constant(dsp_core::LaneMixer::TABLE_SIZE, 0.7);
    mixer.setLaneCurveData(1, constant);

    renderAndFlushCrossfade();

    // Normalization scales maxAbs to 1.0, so constant 0.7 becomes 1.0
    EXPECT_NEAR(evaluateAt(0.0), 1.0, 0.02) << "Normalized constant curve at x=0";
    EXPECT_NEAR(evaluateAt(0.5), 1.0, 0.02) << "Normalized constant curve at x=0.5";

    // Verify it's NOT the identity function anymore
    // (Identity would give 0.0 at x=0.0 and 0.5 at x=0.5)
    EXPECT_GT(std::abs(evaluateAt(0.0) - 0.0), 0.5)
        << "Should differ from identity at x=0";
}

} // namespace dsp_core_test
