#ifndef CONVNEURALNET_H
#define CONVNEURALNET_H
#include <iostream>
#include <fstream>
#include <assert.h>
#include "Types.h"
#include "NeuronLayer.h"
#include "FullyConnectedLayer.h"
#include "DataLoader.h"
#include "MNISTLoader.h"


enum LayerTypes
{
	FULLY_CONNECTED,
	FEATURE_MAP,
	POOL
};
class NeuralNetwork
{

public:
    NeuralNetwork() : numLayers(0) {}
	bool initNeuronLayers();
    void encodeInput(const unsigned rows, const unsigned cols, const int activation);
    void calculateLayerOutput(const unsigned layer);
    void backPropagate(const unsigned layer, const int label);
    NeuronLayer* const getLayerById(const unsigned layer);
    NeuronLayer* getLayer(const LayerTypes type, const unsigned layer);
	void addFullyConnectedLayer(const unsigned rows, const unsigned cols, const bool isInput, const bool isOutput, const unsigned costFunction, const unsigned layer, int activation);
	void connectLayers(const unsigned l1, const unsigned l2);
	void setNumLayers(const unsigned nL){ numLayers = nL; }

	void SGD(const FloatingType eta, const FloatingType lambda, const unsigned epochs, const unsigned miniBatchSize);
	void updateMiniBatch(FloatingType **miniBatch, int * miniBatchLabels, const FloatingType eta, const FloatingType lambda, const unsigned miniBatchSize);
	void loadTrainingData(int dataSource, const char * dataFileName, const char * labelFileName);
	void loadTestData(int dataSource, const char * dataFileName, const char * labelFileName);
	const int accuracy(const int source);

private:
    std::vector<std::pair<NeuronLayer*, unsigned> > neuronLayers;
	unsigned numLayers;
	unsigned numberOfSamples, numberOfTestSamples;
	unsigned sizeOfInputs;
	unsigned trainingRows;
	unsigned trainingCols;
	DataLoader *trainingData, *testData, *validationData, *evaluationData;
	void setInputs(FloatingType** input);
	void feedforward(FloatingType** input);
	void backPropagate(const unsigned label);
	void updateParameters(const FloatingType eta, const FloatingType lambda, const unsigned miniBatchSize);
    void addFullyConnectedLayer(NeuronLayer* other, const unsigned layer){ neuronLayers.push_back(std::make_pair(other, layer)); }
};
#endif
