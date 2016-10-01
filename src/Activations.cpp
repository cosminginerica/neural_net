#include "Activations.h"

const FloatingType Sigmoid::calcActivation(const FloatingType val)
{
	return HelperFunctions::sigmoid(val);
}

FloatingType* Sigmoid::activationVec(FloatingType* vec, const unsigned size)
{
	return HelperFunctions::sigmoid_vec(vec, size);
}

FloatingType* Sigmoid::activationPrimeVec(FloatingType* vec, const unsigned size)
{
	return HelperFunctions::sigmoid_prime_vec(vec, size);
}

FloatingType** Sigmoid::activationMat(FloatingType** m, const unsigned rows, const unsigned cols)
{
	return HelperFunctions::sigmoid_mat(m, rows, cols);
}

FloatingType** Sigmoid::activationPrimeMat(FloatingType** m, const unsigned rows, const unsigned cols)
{
	return HelperFunctions::sigmoid_prime_mat(m, rows, cols);
}

const FloatingType RELU::calcActivation(const FloatingType val)
{
	return HelperFunctions::relu(val);
}

FloatingType* RELU::activationVec(FloatingType* vec, const unsigned size)
{
	return HelperFunctions::relu_vec(vec, size);
}

FloatingType* RELU::activationPrimeVec(FloatingType* vec, const unsigned size)
{
	return HelperFunctions::relu_prime_vec(vec, size);
}

FloatingType** RELU::activationMat(FloatingType** m, const unsigned rows, const unsigned cols)
{
	return HelperFunctions::relu_mat(m, rows, cols);
}

FloatingType** RELU::activationPrimeMat(FloatingType** m, const unsigned rows, const unsigned cols)
{
	return HelperFunctions::relu_prime_mat(m, rows, cols);
}