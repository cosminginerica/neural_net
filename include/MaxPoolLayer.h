#ifndef MAXPOOLINGLAYER_H
#define MAXPOOLINGLAYER_H
#include "NeuronLayer.h"
#include "Types.h"
#include "Activations.h"
class MaxPoolLayer
{
public:
	MaxPoolLayer(const unsigned maxPoolSize, int activation) : maxPoolSize(maxPoolSize){}
	MaxPoolLayer(){}
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
	unsigned numberOfNeurons;
	unsigned rows;
	unsigned cols;
	NeuronLayer *outputLayer, *inputLayer;
	FloatingType **activations;
	FloatingType **zs;
	FloatingType **deltas;
	Activation *activation;
	
};
#endif
