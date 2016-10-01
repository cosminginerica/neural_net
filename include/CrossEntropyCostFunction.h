#ifndef CROSSENTROPYCOSTFUNCTION_H
#define CROSSENTROPYCOSTFUNCTION_H
#include "CostFunction.h"
class CrossEntropyCostFunction : public CostFunction
{
public:
	const FloatingType fn(const FloatingType a, const FloatingType y);
	const FloatingType delta(const FloatingType z, const FloatingType a, const FloatingType y);
};
#endif