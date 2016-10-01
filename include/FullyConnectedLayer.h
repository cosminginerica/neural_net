#ifndef FULLYCONNECTEDLAYER_H
#define FULLYCONNECTEDLAYER_H
#include "NeuronLayer.h"
#include "CostFunction.h"
#include "CrossEntropyCostFunction.h"
#include "QuadraticCostFunction.h"

class FullyConnectedLayer : public NeuronLayer
{
public:
	FullyConnectedLayer()
	{
		classId = "FullyConnectedLayer";
	}
    ~FullyConnectedLayer();
    FullyConnectedLayer(const unsigned _rows, const unsigned _cols, const bool _isInput,
                        const bool _isOutput, const int _costFunction, const int _activation);
	void initInputs();
    void setInputs(const FloatingType **inputs, const unsigned rows, const unsigned cols);
	virtual void calculateOutputs();
	void initializeWeights();
	void initializeBiases();
	FloatingType** getOutputs();
	void makeInputLayer();
	virtual void backPropagate(const int label);

	virtual const int getOutputSize();

	virtual FloatingType ** getWeights();
	void initDelta();

	virtual void updateNablaB();

	virtual void updateNablaW();

	virtual void resetNablaB();

	virtual void resetNablaW();

	void updateWeights(const FloatingType eta, const FloatingType lambda, const int numberOfSamples, const int miniBatchSize);

	virtual void updateBiases(const FloatingType eta, const int miniBatchSize);
    const FloatingType calcDelta(const unsigned crtRow, const unsigned crtCol)
    {
        return costFunction->delta(zs[crtRow][crtCol], activations[crtRow][crtCol], desiredOutputs[crtRow][crtCol]);
    }

	virtual void init();
	void initZs();
	void initActivations();
	std::string const getClassId(){ return classId; }

	FullyConnectedLayer& operator=(const FullyConnectedLayer& other);
private:
	FloatingType **nablaB;
	FloatingType **nablaW;
	int inputsSize;

};
#endif
