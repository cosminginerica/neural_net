#include "NeuralNetwork.h"

void NeuralNetwork::connectLayers(const unsigned l1, const unsigned l2)
{	
    getLayerById(l1)->setOutputLayer(getLayerById(l2));
    getLayerById(l2)->setInputLayer(getLayerById(l1));
}

bool NeuralNetwork::initNeuronLayers()
{
	for (unsigned i = 0; i < numLayers - 1; ++i)
	{
		connectLayers(i, i + 1);
	}

	for (unsigned i = 0; i < numLayers; ++i)
	{
		NeuronLayer* nL = getLayerById(i);
        nL->init();

	}
	return true;
}

void NeuralNetwork::setInputs(FloatingType** input)
{
    neuronLayers[0].first->setInputs(input, neuronLayers[0].first->getRows(), neuronLayers[0].first->getCols());
}

void NeuralNetwork::encodeInput(const unsigned rows, const unsigned cols, const int activation)
{
	FullyConnectedLayer *inputLayer = new FullyConnectedLayer(rows, cols, true, false, CROSS_ENTROPY, activation);
	neuronLayers.push_back(std::make_pair(inputLayer, 0));
}
void NeuralNetwork::addFullyConnectedLayer(const unsigned rows, const unsigned cols, const bool isInput, const bool isOutput, const unsigned costFunction, const unsigned layer, int activation)
{
	FullyConnectedLayer *neuronLayer= new FullyConnectedLayer(rows, cols, isInput, isOutput, costFunction, activation);
	neuronLayers.push_back(std::make_pair(neuronLayer, layer));
	numLayers++;
}

NeuronLayer* const NeuralNetwork::getLayerById(const unsigned layer)
{
	for (unsigned i = 0; i < neuronLayers.size(); ++i)
	{
		if (neuronLayers[i].second == layer)
		{
			return neuronLayers[i].first;
		}
	}
	return NULL;
}


void NeuralNetwork::calculateLayerOutput(const unsigned layer)
{
    NeuronLayer *workingLayer = getLayerById(layer);
	assert(workingLayer);
	if (workingLayer)
	{
		workingLayer->calculateOutputs();
	}
	else
	{
		std::cout << "The neuron layer for which you want to calculate the output does not exist" << std::endl;
	}
}

void NeuralNetwork::backPropagate(const unsigned layer, const int label)
{
    NeuronLayer *workingLayer = getLayerById(layer);
	assert(workingLayer);
	if (workingLayer)
	{
		workingLayer->backPropagate(label);
	}
	else
	{
		std::cout << "The neuron layer for which you want to back propagate does not exist" << std::endl;
	}
}

void NeuralNetwork::SGD(const FloatingType eta, const FloatingType lambda, const unsigned epochs, const unsigned miniBatchSize)
{
	for (unsigned i = 0; i < epochs; ++i)
	{
		clock_t startTime = clock();
		trainingData->shuffleData();
		for (unsigned j = 0; j < numberOfSamples / miniBatchSize; ++j)
		{

			FloatingType **currentMiniBatch = trainingData->getMiniBatch(miniBatchSize, j);
			int *labels = trainingData->getMiniBatchLabels(miniBatchSize, j);
			updateMiniBatch(currentMiniBatch, labels, eta, lambda, miniBatchSize);
			for (unsigned k = 0; k < miniBatchSize; ++k)
			{
				delete[]currentMiniBatch[k];
			}
			delete[]currentMiniBatch;
			delete[]labels;
		}
		clock_t finish = clock();
		if (!testData)
		{
			std::cout << "Epoch " << i << " finished. " << std::endl;
		}
		else
		{
			int acc = accuracy(TEST_DATA);
			int accT = accuracy(TRAINING_DATA);
			std::cout << "Epoch " << i << " : " << acc << " / " << numberOfTestSamples << " / " << 1000 * (double)(finish - startTime) / CLOCKS_PER_SEC << std::endl;
			std::cout << "Training data: " << accT << " / " << numberOfSamples << std::endl;
		}
	}
}

void NeuralNetwork::feedforward(FloatingType** inputMat)
{
	setInputs(inputMat);
	for (unsigned j = 0; j < numLayers; ++j)
	{
		NeuronLayer *nL = getLayerById(j);
		if (nL->getClassId() == "FullyConnectedLayer")
		{
			calculateLayerOutput(j);
		}
		else
		{
			std::cout << "You want an invalid layer" << std::endl;
			return;
		}
	}
}

void NeuralNetwork::backPropagate(const unsigned label)
{
	for (unsigned j = numLayers - 1; j > 0; --j)
	{
		NeuronLayer *nL = getLayerById(j);
		if (nL->getClassId() == "FullyConnectedLayer")
		{
			backPropagate(j, label);
		}
		else
		{
			std::cout << "The layer you want to propagate through is invalid";
			return;
		}
	}
}

void NeuralNetwork::updateParameters(const FloatingType eta, const FloatingType lambda, const unsigned miniBatchSize)
{
	for (unsigned j = 0; j < numLayers; ++j)
	{
		NeuronLayer *nL = getLayerById(j);
		if (nL->getClassId() == "FullyConnectedLayer")
		{
			nL->updateBiases(eta, miniBatchSize);
			nL->updateWeights(eta, lambda, numberOfSamples, miniBatchSize);
		}
		else
		{
			std::cout << "The layer you want to update weights for is invalid";
			return;
		}
	}
}

void NeuralNetwork::updateMiniBatch(FloatingType **miniBatch, int *miniBatchLabels, const FloatingType eta, const FloatingType lambda, const unsigned miniBatchSize)
{
	for (unsigned i = 0; i < neuronLayers.size(); ++i)
	{
		neuronLayers[i].first->resetNablaB();
		neuronLayers[i].first->resetNablaW();
	}
	for (unsigned i = 0; i < miniBatchSize; ++i)
	{
		unsigned rows = getLayerById(0)->getRows();
		unsigned cols = getLayerById(0)->getCols();
		FloatingType **inputMat = HelperFunctions::vec2mat(miniBatch[i], sizeOfInputs, rows, cols);
		
		//feedforward
		feedforward(inputMat);

		//backpropagate
		backPropagate(miniBatchLabels[i]);

		//update weights and biases
		updateParameters(eta, lambda, miniBatchSize);


		for (unsigned j = 0; j < getLayerById(0)->getRows(); ++j)
			delete[]inputMat[j];
		delete []inputMat;
	}
}

void NeuralNetwork::loadTrainingData(int dataSource, const char * dataFileName, const char * labelFileName)
{
	switch (dataSource)
	{
	case DATA_MNIST:
		trainingData = new MNISTLoader;
		break;
	default:
		trainingData = new MNISTLoader;
		break;
	}
	if (trainingData)
	{
		trainingData->loadData(dataFileName, labelFileName);
		sizeOfInputs = trainingData->getSizeOfInputs();
		numberOfSamples = trainingData->getNumberOfSamples();
		trainingCols = dynamic_cast<MNISTLoader*>(trainingData)->getNrCols();
		trainingRows = dynamic_cast<MNISTLoader*>(trainingData)->getNrRows();
	}
}

const int NeuralNetwork::accuracy(const int source)
{
	int goodOne = 0;
	switch (source)
	{
	case TEST_DATA:
		evaluationData = testData;
		break;
	case TRAINING_DATA:
		evaluationData = trainingData;
		break;
	default:
		evaluationData = testData;
		break;
	}
	const unsigned int numberOfSamples = evaluationData->getNumberOfSamples();
	unsigned rows = neuronLayers[0].first->getRows();
	unsigned cols = neuronLayers[0].first->getCols();
	for (unsigned int i = 0; i < numberOfSamples; ++i)
	{
		FloatingType** inputAsMat = HelperFunctions::vec2mat(evaluationData->getDataById(i), evaluationData->getSizeOfInputs(), rows, cols);
		feedforward(inputAsMat);
		int max = 0;
		FloatingType peakValue = 0.0;
		NeuronLayer *layer = getLayerById(numLayers - 1);
		unsigned outputRows = layer->getRows();
		unsigned outputCols = layer->getCols();
		FloatingType *outputs = HelperFunctions::mat2vec(layer->getOutputs(), outputRows, outputCols);
		for (unsigned j = 0; j < outputRows * outputCols; ++j)
		{
			if (outputs[j] > peakValue)
			{
				max = j;
				peakValue = outputs[j];
			}
		}

		int crtLabel = (int)evaluationData->getLabelById(i);
		if (max == crtLabel)
		{
			goodOne++;
		}
		delete[]outputs;
		for (unsigned j = 0; j < rows; ++j)
			delete[]inputAsMat[j];
		delete[]inputAsMat;
	}
	return goodOne;
}

void NeuralNetwork::loadTestData(int dataSource, const char * dataFileName, const char * labelFileName)
{
	switch (dataSource)
	{
	case DATA_MNIST:
		testData = new MNISTLoader;
		break;
	default:
		testData = new MNISTLoader;
		break;
	}
	if (testData)
	{
		testData->loadData(dataFileName, labelFileName);
		numberOfTestSamples = testData->getNumberOfSamples();
	}
}
