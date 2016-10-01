#ifndef NEURONLAYER_H
#define NEURONLAYER_H
#include "Types.h"
#include "Activations.h"
#include "CostFunction.h"
class NeuronLayer
{
public:
    virtual ~NeuronLayer(){}
	virtual void calculateOutputs() = 0;
	virtual void initializeWeights() = 0;
	virtual void initializeBiases() = 0;
	virtual FloatingType** getOutputs() = 0;
	virtual const int getOutputSize() = 0;
	virtual void backPropagate(const int label) = 0;
	virtual FloatingType **getWeights() = 0;
	virtual void updateNablaB() = 0;
	virtual void updateNablaW() = 0;
	virtual void resetNablaB() = 0;
	virtual void resetNablaW() = 0;
	virtual void updateWeights(const FloatingType eta, const FloatingType lambda, const int numberOfSamples, const int miniBatchSize) = 0;
	virtual void updateBiases(const FloatingType eta, const int miniBatchSize) = 0;
	virtual void init() = 0;
    virtual std::string const getClassId() = 0;

    virtual const unsigned getNumberOfNeurons(){ return numberOfNeurons; }
    virtual const unsigned getNumberOfInputs(){ return inputLayer->getOutputSize(); }
    FloatingType** getDelta(){ return deltas; }
	const unsigned getRows()const{ return rows; }
	const unsigned getCols()const{ return cols; }
	virtual FloatingType** getZ(){return zs;}
	virtual void setInputLayer(NeuronLayer* layer){ this->inputLayer = layer; }
	virtual void setOutputLayer(NeuronLayer* layer){ this->outputLayer = layer; }
	virtual Activation* getActivation(){ return activation; }
    virtual void setActivation(Activation* activation){ this->activation = activation; }
	virtual void setCostFunction(CostFunction* cost){ this->costFunction = cost; }
    virtual void setInputs(FloatingType **inputs, const unsigned rowBs, const unsigned cols){}

protected:
	NeuronLayer* inputLayer;
	NeuronLayer* outputLayer;
	FloatingType **weights;
	FloatingType **biases;
	FloatingType **inputs;
	FloatingType **activations;
	FloatingType **zs;
	FloatingType **deltas;
	FloatingType **desiredOutputs;
	unsigned cols, rows;
	unsigned numberOfNeurons;
	bool isInput, isOutput;
	int numberOfSamples;
	std::string classId;
	Activation *activation;
	CostFunction *costFunction;
};
#endif
