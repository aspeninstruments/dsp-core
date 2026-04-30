#include "../dsp_core/Source/Services/ExpressionParser.h"
#include <gtest/gtest.h>

namespace dsp_core_test {

using namespace dsp_core::Services;

// ============================================================================
// Base Expression Only Tests
// ============================================================================

TEST(ExpressionParserTest, ParseBaseExpressionOnly_Simple) {
    auto result = ExpressionParser::parse("sin(x)");

    EXPECT_TRUE(result.hasBaseExpression);
    EXPECT_FALSE(result.hasHarmonicExpression);
    EXPECT_EQ(result.baseExpression, "sin(x)");
    EXPECT_TRUE(result.harmonicExpression.isEmpty());
}

TEST(ExpressionParserTest, ParseBaseExpressionOnly_Complex) {
    auto result = ExpressionParser::parse("x^2 * cos(x * pi)");

    EXPECT_TRUE(result.hasBaseExpression);
    EXPECT_FALSE(result.hasHarmonicExpression);
    EXPECT_EQ(result.baseExpression, "x^2 * cos(x * pi)");
}

TEST(ExpressionParserTest, ParseBaseExpressionOnly_WithWhitespace) {
    auto result = ExpressionParser::parse("  sin(x)  ");

    EXPECT_TRUE(result.hasBaseExpression);
    EXPECT_EQ(result.baseExpression, "sin(x)"); // Trimmed
}

// ============================================================================
// Harmonic Expression Only Tests
// ============================================================================

TEST(ExpressionParserTest, ParseHarmonicExpressionOnly_WithSemicolon) {
    auto result = ExpressionParser::parse("; 0.8, 0.5, 0.3");

    EXPECT_FALSE(result.hasBaseExpression);
    EXPECT_TRUE(result.hasHarmonicExpression);
    EXPECT_TRUE(result.baseExpression.isEmpty());
    EXPECT_EQ(result.harmonicExpression, "0.8, 0.5, 0.3");
}

TEST(ExpressionParserTest, ParseHarmonicExpressionOnly_NoXVariable) {
    // If no semicolon and no 'x', treat as harmonic
    auto result = ExpressionParser::parse("1.0, 0.5, 0.25");

    EXPECT_FALSE(result.hasBaseExpression);
    EXPECT_TRUE(result.hasHarmonicExpression);
    EXPECT_EQ(result.harmonicExpression, "1.0, 0.5, 0.25");
}

// ============================================================================
// Unified Expression Tests (Both Parts)
// ============================================================================

TEST(ExpressionParserTest, ParseUnified_BothParts) {
    auto result = ExpressionParser::parse("x^2 ; 0.8, 0.5, 0.3");

    EXPECT_TRUE(result.hasBaseExpression);
    EXPECT_TRUE(result.hasHarmonicExpression);
    EXPECT_EQ(result.baseExpression, "x^2");
    EXPECT_EQ(result.harmonicExpression, "0.8, 0.5, 0.3");
}

TEST(ExpressionParserTest, ParseUnified_WithWhitespace) {
    auto result = ExpressionParser::parse("  sin(x)  ;  1.0, 0.5  ");

    EXPECT_TRUE(result.hasBaseExpression);
    EXPECT_TRUE(result.hasHarmonicExpression);
    EXPECT_EQ(result.baseExpression, "sin(x)");
    EXPECT_EQ(result.harmonicExpression, "1.0, 0.5");
}

TEST(ExpressionParserTest, ParseUnified_ComplexExpressions) {
    auto result = ExpressionParser::parse("cos(x * 2 * pi) * 0.5 + 0.5 ; 0.9, 0.1, 0.3, 0.0, 0.2");

    EXPECT_TRUE(result.hasBaseExpression);
    EXPECT_TRUE(result.hasHarmonicExpression);
    EXPECT_EQ(result.baseExpression, "cos(x * 2 * pi) * 0.5 + 0.5");
    EXPECT_EQ(result.harmonicExpression, "0.9, 0.1, 0.3, 0.0, 0.2");
}

// ============================================================================
// Clear Base Layer Tests
// ============================================================================

TEST(ExpressionParserTest, ParseClearBase_HarmonicOnly) {
    auto result = ExpressionParser::parse("; 1.0, 0.5");

    EXPECT_FALSE(result.hasBaseExpression);
    EXPECT_TRUE(result.hasHarmonicExpression);
    EXPECT_TRUE(result.baseExpression.isEmpty());
    EXPECT_EQ(result.harmonicExpression, "1.0, 0.5");
}

// ============================================================================
// Clear Harmonics Tests
// ============================================================================

TEST(ExpressionParserTest, ParseClearHarmonics_BaseOnly) {
    auto result = ExpressionParser::parse("sin(x) ;");

    EXPECT_TRUE(result.hasBaseExpression);
    EXPECT_FALSE(result.hasHarmonicExpression);
    EXPECT_EQ(result.baseExpression, "sin(x)");
    EXPECT_TRUE(result.harmonicExpression.isEmpty());
}

TEST(ExpressionParserTest, ParseClearHarmonics_WithWhitespace) {
    auto result = ExpressionParser::parse("cos(x)  ;  ");

    EXPECT_TRUE(result.hasBaseExpression);
    EXPECT_FALSE(result.hasHarmonicExpression);
    EXPECT_EQ(result.baseExpression, "cos(x)");
}

// ============================================================================
// Empty Input Tests
// ============================================================================

TEST(ExpressionParserTest, ParseEmpty_EmptyString) {
    auto result = ExpressionParser::parse("");

    EXPECT_FALSE(result.hasBaseExpression);
    EXPECT_FALSE(result.hasHarmonicExpression);
    EXPECT_TRUE(result.baseExpression.isEmpty());
    EXPECT_TRUE(result.harmonicExpression.isEmpty());
}

TEST(ExpressionParserTest, ParseEmpty_OnlyWhitespace) {
    auto result = ExpressionParser::parse("   ");

    EXPECT_FALSE(result.hasBaseExpression);
    EXPECT_FALSE(result.hasHarmonicExpression);
}

TEST(ExpressionParserTest, ParseEmpty_OnlySemicolon) {
    auto result = ExpressionParser::parse(";");

    EXPECT_FALSE(result.hasBaseExpression);
    EXPECT_FALSE(result.hasHarmonicExpression);
}

TEST(ExpressionParserTest, ParseEmpty_MultipleSemicolons) {
    auto result = ExpressionParser::parse(";;;");

    // Should handle gracefully (implementation dependent)
    // Just verify it doesn't crash
}

// ============================================================================
// Validation Tests
// ============================================================================

TEST(ExpressionParserTest, IsValid_ValidExpression) {
    EXPECT_TRUE(ExpressionParser::isValid("sin(x)"));
    EXPECT_TRUE(ExpressionParser::isValid("x^2 ; 1.0, 0.5"));
    EXPECT_TRUE(ExpressionParser::isValid("; 0.8"));
}

TEST(ExpressionParserTest, IsValid_InvalidExpression) {
    EXPECT_FALSE(ExpressionParser::isValid(""));
    EXPECT_FALSE(ExpressionParser::isValid("   "));
}

// ============================================================================
// Delimiter Tests
// ============================================================================

TEST(ExpressionParserTest, GetDelimiter) {
    EXPECT_STREQ(ExpressionParser::getDelimiter(), ";");
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(ExpressionParserTest, ParseEdgeCase_UppercaseX) {
    auto result = ExpressionParser::parse("SIN(X)");

    EXPECT_TRUE(result.hasBaseExpression);
    EXPECT_EQ(result.baseExpression, "SIN(X)");
}

TEST(ExpressionParserTest, ParseEdgeCase_MixedCase) {
    auto result = ExpressionParser::parse("CoS(x) ; 1.0");

    EXPECT_TRUE(result.hasBaseExpression);
    EXPECT_TRUE(result.hasHarmonicExpression);
}

TEST(ExpressionParserTest, ParseEdgeCase_NoSpacesAroundSemicolon) {
    auto result = ExpressionParser::parse("x^2;0.5");

    EXPECT_TRUE(result.hasBaseExpression);
    EXPECT_TRUE(result.hasHarmonicExpression);
    EXPECT_EQ(result.baseExpression, "x^2");
    EXPECT_EQ(result.harmonicExpression, "0.5");
}

TEST(ExpressionParserTest, ParseEdgeCase_LotsOfWhitespace) {
    auto result = ExpressionParser::parse("    sin(x)    ;    0.8 , 0.5    ");

    EXPECT_TRUE(result.hasBaseExpression);
    EXPECT_TRUE(result.hasHarmonicExpression);
    EXPECT_EQ(result.baseExpression, "sin(x)");
    EXPECT_EQ(result.harmonicExpression, "0.8 , 0.5");
}

} // namespace dsp_core_test
