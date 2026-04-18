#pragma once

#include "exprtk.hpp"

namespace dsp_core {

class ExpressionEvaluator {
  public:
    ExpressionEvaluator();

    bool compile(const std::string& expression);
    double evaluate(double x) const;

    // Morph macro (variable `m` in expressions). 0..1. Set before a batch
    // of evaluate() calls when the macro moves; unused symbols cost nothing.
    void setMorph(double m);

  private:
    typedef exprtk::symbol_table<double> SymbolTable;
    typedef exprtk::expression<double> Expression;
    typedef exprtk::parser<double> Parser;

    double xVar = 0.0;
    double mVar = 0.0;
    SymbolTable symbolTable;
    Expression expression;
    Parser parser;
};

} // namespace dsp_core
