#include "QuadraticCostFunction.h"

const FloatingType QuadraticCostFunction::fn(const FloatingType a, const FloatingType y)
{
	return 0.5 * pow(fabs(a - y), 2);
}

const FloatingType QuadraticCostFunction::delta(const FloatingType z, const FloatingType a, const FloatingType y)
{
	return (a - y) * HelperFunctions::sigmoid_prime(z);
}