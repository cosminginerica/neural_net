#include "CrossEntropyCostFunction.h"

const FloatingType CrossEntropyCostFunction::fn(const FloatingType a, const FloatingType y)
{
	return -y * log(a) - (1 - y) * log(1 - a);
}

const FloatingType CrossEntropyCostFunction::delta(const FloatingType z, const FloatingType a, const FloatingType y)
{
	return a - y;
}