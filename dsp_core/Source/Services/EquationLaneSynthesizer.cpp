#include "EquationLaneSynthesizer.h"
#include "../ExpressionEvaluator.h"
#include "../HarmonicLayer.h"
#include "../LaneMixer.h"
#include "ExpressionParser.h"

#include <cmath>

namespace dsp_core::Services {

namespace {
constexpr int kNumHarmonicsForFn = 40;
constexpr double kSynthesizeNormEpsilon = 1e-12;

void normalizeInPlace(std::vector<double>& values) {
    double maxAbs = 0.0;
    for (double v : values) {
        maxAbs = std::max(maxAbs, std::abs(v));
    }
    if (maxAbs > kSynthesizeNormEpsilon) {
        const double scalar = 1.0 / maxAbs;
        for (auto& v : values) {
            v *= scalar;
        }
    }
}
} // namespace

bool compileLaneEquation(const juce::String& equationText, bool normalize, CompiledLaneEquation& out) {
    out.evaluator.reset();
    out.sourceText = {};
    out.normalize = normalize;
    out.isHarmonic = false;

    auto parsed = ExpressionParser::parse(equationText);

    if (!parsed.hasBaseExpression && !parsed.hasHarmonicExpression) {
        return false;
    }

    // Combined `f(x); f(n)` is no longer supported (the editor's
    // applyUnifiedExpression rejects it explicitly). For the cached-state
    // rehydrate path used by morph drive and thumbnail bake, prefer base
    // when both somehow exist — that mirrors the prior behavior.
    const bool pickBase = parsed.hasBaseExpression;
    const bool pickHarmonic = !pickBase && parsed.hasHarmonicExpression;
    if (!pickBase && !pickHarmonic) {
        return false;
    }

    const juce::String& src = pickBase ? parsed.baseExpression : parsed.harmonicExpression;

    auto evaluator = std::make_unique<ExpressionEvaluator>();
    if (!evaluator->compile(src.toStdString())) {
        return false;
    }

    out.evaluator = std::move(evaluator);
    out.sourceText = equationText;
    out.isHarmonic = pickHarmonic;
    return true;
}

bool synthesizeLaneLUT(CompiledLaneEquation& compiled, double morph, const HarmonicLayer& harmonicLayer, int tableSize,
                       std::vector<double>& outLut) {
    if (compiled.evaluator == nullptr || tableSize <= 0) {
        return false;
    }

    compiled.evaluator->setMorph(morph);

    outLut.assign(static_cast<std::size_t>(tableSize), 0.0);

    if (compiled.isHarmonic) {
        std::vector<double> coeffs(kNumHarmonicsForFn + 1, 0.0); // [0] unused by HarmonicLayer
        for (int n = 1; n <= kNumHarmonicsForFn; ++n) {
            const double v = compiled.evaluator->evaluateHarmonic(static_cast<double>(n));
            if (!std::isfinite(v)) {
                return false;
            }
            coeffs[static_cast<std::size_t>(n)] = v;
        }
        for (int i = 0; i < tableSize; ++i) {
            const double x = LaneMixer::normalizeIndex(i);
            outLut[static_cast<std::size_t>(i)] = harmonicLayer.evaluate(x, coeffs, tableSize);
        }
        // HarmonicLayer.evaluate is a deterministic sum of finite coeffs × finite
        // basis values → guaranteed finite if coeffs are finite (validated above).
    } else {
        for (int i = 0; i < tableSize; ++i) {
            const double x = LaneMixer::normalizeIndex(i);
            const double y = compiled.evaluator->evaluate(x);
            if (!std::isfinite(y)) {
                return false;
            }
            outLut[static_cast<std::size_t>(i)] = y;
        }
    }

    if (compiled.normalize) {
        normalizeInPlace(outLut);
    }

    return true;
}

} // namespace dsp_core::Services
