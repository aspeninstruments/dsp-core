/**
 * ExpressionEvaluatorTest.cpp
 * Unit tests for dsp_core::ExpressionEvaluator, focused on the `m` (morph
 * macro) variable binding used by equation-mode morph automation.
 */

#include <gtest/gtest.h>
#include <dsp_core/dsp_core.h>
#include <cmath>

using dsp_core::ExpressionEvaluator;

TEST(ExpressionEvaluatorTest, PlainXStillEvaluates) {
    ExpressionEvaluator e;
    ASSERT_TRUE(e.compile("x"));
    EXPECT_DOUBLE_EQ(e.evaluate(0.0), 0.0);
    EXPECT_DOUBLE_EQ(e.evaluate(0.5), 0.5);
    EXPECT_DOUBLE_EQ(e.evaluate(-1.0), -1.0);
}

TEST(ExpressionEvaluatorTest, MorphDefaultsToZero) {
    ExpressionEvaluator e;
    ASSERT_TRUE(e.compile("x + m"));
    // Without setMorph, m defaults to 0 so evaluate == x.
    EXPECT_DOUBLE_EQ(e.evaluate(0.25), 0.25);
}

TEST(ExpressionEvaluatorTest, MorphIsReadEachEvaluate) {
    ExpressionEvaluator e;
    ASSERT_TRUE(e.compile("x + m"));

    e.setMorph(0.5);
    EXPECT_DOUBLE_EQ(e.evaluate(0.25), 0.75);

    e.setMorph(0.1);
    EXPECT_DOUBLE_EQ(e.evaluate(0.25), 0.35);

    // x may still vary independently.
    EXPECT_DOUBLE_EQ(e.evaluate(-0.5), -0.4);
}

TEST(ExpressionEvaluatorTest, MorphInNonTrivialExpression) {
    ExpressionEvaluator e;
    ASSERT_TRUE(e.compile("sin(pi * x * (1 + m))"));

    e.setMorph(0.0);
    // sin(pi * 0.5 * 1) == sin(pi/2) == 1
    EXPECT_NEAR(e.evaluate(0.5), 1.0, 1e-12);

    e.setMorph(1.0);
    // sin(pi * 0.5 * 2) == sin(pi) == 0
    EXPECT_NEAR(e.evaluate(0.5), 0.0, 1e-12);
}

TEST(ExpressionEvaluatorTest, EquationWithoutMorphUnaffected) {
    ExpressionEvaluator e;
    ASSERT_TRUE(e.compile("x * x"));
    e.setMorph(0.7); // should be ignored — no m symbol in expression
    EXPECT_DOUBLE_EQ(e.evaluate(0.5), 0.25);
}

TEST(ExpressionEvaluatorTest, InvalidExpressionFailsToCompile) {
    ExpressionEvaluator e;
    EXPECT_FALSE(e.compile("sin("));
}
