#pragma once

#include <juce_core/juce_core.h>
#include <memory>
#include <vector>

namespace dsp_core {
class ExpressionEvaluator;
class HarmonicLayer;
} // namespace dsp_core

namespace dsp_core::Services {

/**
 * Cached state for an equation-driven lane: the compiled exprtk evaluator plus
 * the bookkeeping needed to (a) detect cache invalidation by source-text change
 * and (b) drive base vs. harmonic synthesis.
 *
 * Constructed via compileLaneEquation. Held by the caller (e.g. the in-plugin
 * editor's ExpressionOperationHandler caches one per lane; the static-thumbnail
 * baker compiles one fresh and uses it across all 5 morph rows).
 */
struct CompiledLaneEquation {
    std::unique_ptr<ExpressionEvaluator> evaluator;
    juce::String sourceText;       // for cache invalidation when lane.equationText changes
    bool normalize = true;         // post-fill scale to [-1, 1] (matches editor default)
    bool isHarmonic = false;       // true → f(n) form (synthesize via HarmonicLayer)
};

/**
 * Compile a lane's equation text into a reusable CompiledLaneEquation. Parses
 * unified `f(x); f(n)` syntax via ExpressionParser, picks the base form if
 * present (else harmonic), compiles the chosen branch, and stamps the flags.
 *
 * Returns false if `equationText` has no usable expression, contains both
 * forms (no longer supported), or fails to compile. On false, `out` is reset
 * to empty defaults; on true, `out.evaluator` is non-null and ready for
 * synthesizeLaneLUT() calls.
 *
 * Note: caller decides the `normalize` flag. The editor's first-entry path
 * passes whatever the user requested; the cache rehydrate path and the
 * thumbnail bake both default to true (matches production).
 */
bool compileLaneEquation(const juce::String& equationText, bool normalize, CompiledLaneEquation& out);

/**
 * Evaluate a previously-compiled lane equation at the given morph value and
 * fill `outLut` with `tableSize` samples for x in [-1, 1].
 *
 * For base expressions (isHarmonic=false): evaluates f(x) at `tableSize`
 * evenly-spaced x values.
 *
 * For harmonic expressions (isHarmonic=true): evaluates f(n) for n=1..40,
 * yielding 40 Chebyshev coefficients, then synthesizes the LUT via
 * `harmonicLayer.evaluate(x, coeffs, tableSize)`.
 *
 * After the fill, applies post-loop normalize if compiled.normalize is true.
 *
 * Returns false on NaN/Inf or invalid inputs; on false `outLut`'s contents
 * are undefined and the caller should not persist them. Mutates the evaluator
 * (setMorph), so `compiled` is taken by non-const reference.
 */
bool synthesizeLaneLUT(CompiledLaneEquation& compiled, double morph, const HarmonicLayer& harmonicLayer, int tableSize,
                       std::vector<double>& outLut);

} // namespace dsp_core::Services
