#ifndef COST_FUNCTION_H
#define COST_FUNCTION_H
#include <math.h>
#include "Types.h"
class CostFunction
{
public:
	CostFunction(){}
	virtual const FloatingType fn(const FloatingType a, const FloatingType y) = 0;
	virtual const FloatingType delta(const FloatingType z, const FloatingType a, const FloatingType y) = 0;
private:
};
#endif