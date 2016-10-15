#ifndef MAXPOOLINGLAYER_H
#define MAXPOOLINGLAYER_H
#include "NeuronLayer.h"
#include "Types.h"
#include <assert.h>
class MaxPoolLayer
{
public:
	MaxPoolingLayer(const unsigned maxPoolSize, int activation) : maxPoolSize(maxPoolSize){}
	MaxPoolingLayer(){}
	void calculateOutputs();
	FloatingType** getOutputs();
	const int getOutputSize();
	void backPropagate(const int label);
	FloatingType ** getWeights();

	void init();
	std::string const getClassId(){ return "MaxPool"; }
	FloatingType **getDelta();
	ostream& serialize(ostream& f)const;

private:
	unsigned maxPoolSize;
	unsigned nrNeuronsPerPool;
	NeuronLayer *outputLayer, *inputLayer;
};
#endif
