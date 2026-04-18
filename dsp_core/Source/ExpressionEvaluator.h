#pragma once

#include "exprtk.hpp"

namespace dsp_core {

class ExpressionEvaluator {
  public:
    ExpressionEvaluator();

    bool compile(const std::string& expression);
    double evaluate(double x) const;

    // Harmonic-coefficient evaluation. Sets the `n` variable and returns
    // expression.value(). Used by the f(n) Chebyshev synthesis path where
    // the expression produces a coefficient per harmonic index n=1..N.
    // `x` is not touched — harmonic expressions should not reference it.
    double evaluateHarmonic(double n) const;

    // Morph macro (variable `m` in expressions). 0..1. Set before a batch
    // of evaluate() calls when the macro moves; unused symbols cost nothing.
    void setMorph(double m);

  private:
    typedef exprtk::symbol_table<double> SymbolTable;
    typedef exprtk::expression<double> Expression;
    typedef exprtk::parser<double> Parser;

    double xVar = 0.0;
    double nVar = 0.0;
    double mVar = 0.0;
    SymbolTable symbolTable;
    Expression expression;
    Parser parser;
};

} // namespace dsp_core
