#pragma once
#include <juce_core/juce_core.h>

namespace dsp_core::Services {

/**
 * ExpressionParser - Parses unified expression syntax for transfer function editing.
 *
 * Syntax:
 *   - "f(x)"          → Base layer expression only
 *   - "f(n)"          → Harmonic coefficients expression only
 *   - "f(x) ; f(n)"   → Both base and harmonic expressions
 *   - "; f(n)"        → Harmonic only (clears base layer)
 *   - "f(x) ;"        → Base only (clears harmonics)
 *
 * Delimiter: semicolon (;)
 *
 * This service is stateless and pure - it only performs parsing. Lives in
 * dsp_core because it's used both by the in-plugin equation editor (via
 * transfer_function_editor's ExpressionOperationHandler) AND by the static
 * thumbnail evaluator (LaneMixer::evaluateStaticThumbnail) that bakes morph
 * snapshots for the community-website preview.
 */
class ExpressionParser {
  public:
    struct ParsedExpression {
        juce::String baseExpression;     // f(x) part (empty if missing)
        juce::String harmonicExpression; // f(n) part (empty if missing)
        bool hasBaseExpression{};        // true if f(x) present
        bool hasHarmonicExpression{};    // true if f(n) present
    };

    static ParsedExpression parse(const juce::String& unifiedExpression);

    static bool isValid(const juce::String& expression);

    static constexpr const char* getDelimiter() {
        return ";";
    }

  private:
    static juce::String trim(const juce::String& str);
};

} // namespace dsp_core::Services
