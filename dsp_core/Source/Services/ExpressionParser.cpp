#include "ExpressionParser.h"

namespace dsp_core::Services {

ExpressionParser::ParsedExpression ExpressionParser::parse(const juce::String& unifiedExpression) {
    ParsedExpression result;

    if (unifiedExpression.trim().isEmpty()) {
        return result;
    }

    const juce::StringArray parts = juce::StringArray::fromTokens(unifiedExpression, getDelimiter(),
                                                                  "" // No quote character
    );

    if (parts.size() == 1) {
        // No semicolon - convention: contains 'x' → base expression, else harmonic coefficients.
        const juce::String trimmed = trim(parts[0]);

        if (trimmed.contains("x") || trimmed.contains("X")) {
            result.baseExpression = trimmed;
            result.hasBaseExpression = true;
        } else {
            result.harmonicExpression = trimmed;
            result.hasHarmonicExpression = true;
        }
    } else if (parts.size() >= 2) {
        const juce::String basePart = trim(parts[0]);
        const juce::String harmonicPart = trim(parts[1]);

        if (basePart.isNotEmpty()) {
            result.baseExpression = basePart;
            result.hasBaseExpression = true;
        }

        if (harmonicPart.isNotEmpty()) {
            result.harmonicExpression = harmonicPart;
            result.hasHarmonicExpression = true;
        }
    }

    return result;
}

bool ExpressionParser::isValid(const juce::String& expression) {
    return expression.trim().isNotEmpty();
}

juce::String ExpressionParser::trim(const juce::String& str) {
    return str.trim();
}

} // namespace dsp_core::Services
