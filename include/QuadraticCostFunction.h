#ifndef QUADRATIC_COST_FUNCTION
#define QUADRATIC_COST_FUNCTION
#include "CostFunction.h"
class QuadraticCostFunction : public CostFunction
{
public:
	const FloatingType fn(const FloatingType a, const FloatingType y);
	const FloatingType delta(const FloatingType z, const FloatingType a, const FloatingType y);

};
#endif