#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H
#include "Types.h"
class Activation
{
public:
    Activation(){}
	Activation(FloatingType val) : inputVal(val) {}
	virtual const FloatingType calcActivation(const FloatingType val) = 0;
	virtual const FloatingType calcActivationPrime(const FloatingType val) = 0;

	virtual FloatingType* activationVec(FloatingType* vec, const unsigned size) = 0;
	virtual FloatingType* activationPrimeVec(FloatingType* vec, const unsigned size) = 0;
	virtual FloatingType** activationMat(FloatingType** mat, const unsigned rows, const unsigned cols) = 0;
	virtual FloatingType** activationPrimeMat(FloatingType** mat, const unsigned rows, const unsigned cols) = 0;
	virtual const std::string getClassId() = 0;
protected:
	FloatingType inputVal;
};

class Sigmoid : public Activation
{
public:
	const FloatingType calcActivation(const FloatingType val);
	const FloatingType calcActivationPrime(const FloatingType val){ return HelperFunctions::sigmoid_prime(val); }
	FloatingType* activationVec(FloatingType* vec, const unsigned size);
	FloatingType* activationPrimeVec(FloatingType* vec, const unsigned size);
	FloatingType** activationMat(FloatingType** mat, const unsigned rows, const unsigned cols);
	FloatingType** activationPrimeMat(FloatingType** mat, const unsigned rows, const unsigned cols);
	const std::string getClassId(){ return "Sigmoid"; }
};

class RELU : public Activation
{
public:
	const FloatingType calcActivation(const FloatingType val);
	const FloatingType calcActivationPrime(const FloatingType val){ return HelperFunctions::relu_prime(val); }
	FloatingType* activationVec(FloatingType* vec, const unsigned size);
	FloatingType* activationPrimeVec(FloatingType* vec, const unsigned size);
	FloatingType** activationMat(FloatingType** mat, const unsigned rows, const unsigned cols);
	FloatingType** activationPrimeMat(FloatingType** mat, const unsigned rows, const unsigned cols);
	const std::string getClassId(){ return "RELU"; }
};
#endif
