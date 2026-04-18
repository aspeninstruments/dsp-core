#include "ExpressionEvaluator.h"

namespace dsp_core {

ExpressionEvaluator::ExpressionEvaluator() {
    symbolTable.add_variable("x", xVar);
    symbolTable.add_variable("m", mVar);
    symbolTable.add_constants();
    expression.register_symbol_table(symbolTable);
}

bool ExpressionEvaluator::compile(const std::string& expressionStr) {
    return parser.compile(expressionStr, expression);
}

double ExpressionEvaluator::evaluate(double x) const {
    const_cast<double&>(xVar) = x;
    return expression.value();
}

void ExpressionEvaluator::setMorph(double m) {
    mVar = m;
}

} // namespace dsp_core
