#include "FullyConnectedLayer.h"
#include <fstream>

FullyConnectedLayer::FullyConnectedLayer(const unsigned _rows, const unsigned _cols, const bool _isInput,
                                         const bool _isOutput, const int _costFunction, const int _activation)
{
    numberOfNeurons = _rows * _cols;
    isInput = _isInput;
    this->isOutput = _isOutput;
    this->rows = _rows;
    this->cols = _cols;
    if (_isInput)
	{
		init();
	}
    switch (_costFunction)
	{
	case QUADRATIC_COST:
		this->costFunction = new QuadraticCostFunction;
		break;
	case CROSS_ENTROPY:
	default:
		this->costFunction = new CrossEntropyCostFunction;
	}

    switch (_activation)
	{
	case SIGMOID_ACTIVATION:
	default:
		this->setActivation(new Sigmoid);
		break;
	}
	classId = "FullyConnectedLayer";

}
void FullyConnectedLayer::init()
{
	initializeWeights();
	initializeBiases();
	if (isInput)
	{
		initInputs();
	}
	initDelta();
	initZs();
	initActivations();
}
void FullyConnectedLayer::initInputs()
{
	inputsSize = numberOfNeurons;
	zs = new FloatingType*[rows];
	for (unsigned i = 0; i < rows; ++i)
	{
		zs[i] = new FloatingType[cols];
	}
	inputs = new FloatingType*[rows];
	for (unsigned i = 0; i < rows; ++i)
	{
		inputs[i] = new FloatingType[cols];
	}
}

void FullyConnectedLayer::setInputs(const FloatingType **inputs, const unsigned rows, const unsigned cols)
{
	for (unsigned i = 0; i < rows; ++i)
	{
		for (unsigned j = 0; j < cols; ++j)
		{
			this->inputs[i][j] = inputs[i][j];
		}		
	}
}

void FullyConnectedLayer::initializeWeights()
{
	if (!isInput)
	{
		weights = new FloatingType *[numberOfNeurons];
		nablaW = new FloatingType *[numberOfNeurons];
		const unsigned prevNrNeurons = inputLayer->getNumberOfNeurons();
		for (unsigned i = 0; i < numberOfNeurons; ++i)
		{
			weights[i] = new FloatingType[prevNrNeurons];
			nablaW[i] = new FloatingType[prevNrNeurons];
		}
		for (unsigned i = 0; i < numberOfNeurons; ++i)
		{
			for (unsigned j = 0; j < prevNrNeurons; ++j)
			{
				weights[i][j] = (HelperFunctions::randomNumber() - 0.5) / (sqrt((FloatingType)prevNrNeurons));
				nablaW[i][j] = 0;
			}
		}
	}
}

void FullyConnectedLayer::calculateOutputs()
{
	if (!isInput)
	{
		const int prevNrNeurons = inputLayer->getNumberOfNeurons();
		FloatingType **prevActivations = inputLayer->getOutputs();
		FloatingType *prevActivationsVec = HelperFunctions::mat2vec(prevActivations, inputLayer->getRows(), inputLayer->getCols());
		FloatingType *biasesVec = HelperFunctions::mat2vec(biases, rows, cols);
		FloatingType *w_a = HelperFunctions::matVecMul(weights, numberOfNeurons, prevNrNeurons, prevActivationsVec, prevNrNeurons);
		FloatingType* c = HelperFunctions::vectorAddition(w_a, numberOfNeurons, biasesVec, numberOfNeurons);
		unsigned crtIdx = 0;
		for (unsigned i = 0; i < rows; ++i)
		{
			for (unsigned j = 0; j < cols; ++j)
			{
				zs[i][j] = c[crtIdx++];
			}
			
		}
		FloatingType **a = activation->activationMat(zs, rows, cols); //HelperFunctions::sigmoid_mat(zs, rows, cols);
		for (unsigned i = 0; i < rows; ++i)
		{
			for (unsigned j = 0; j < cols; ++j)
			{
				activations[i][j] = a[i][j];
			}
			
		}
		for (unsigned i = 0; i < rows; ++i)
		{
			delete[]a[i];
		}
		delete[]a;
		delete[]w_a;
		delete[]c;
		delete[]biasesVec;
		delete[]prevActivationsVec;
	}
	else
	{
		activations = inputs;
	}
}

FloatingType **FullyConnectedLayer::getOutputs()
{
	if (!isInput)
		return activations;
	else
		return inputs;
}
void FullyConnectedLayer::initializeBiases()
{
	if (!isInput)
	{
		biases = new FloatingType*[rows];
		nablaB = new FloatingType*[rows];
		for (unsigned i = 0; i < rows; ++i)
		{
			biases[i] = new FloatingType[cols];
			nablaB[i] = new FloatingType[cols];
		}
		for (unsigned i = 0; i < rows; ++i)
		{
			for (unsigned j = 0; j < cols; ++j)
			{
				biases[i][j] = HelperFunctions::randomNumber() - 0.5;
				nablaB[i][j] = 0;
			}
			
		}
	}

}

void FullyConnectedLayer::makeInputLayer()
{
	isInput = true;
}

void FullyConnectedLayer::backPropagate(const int label)
{
	if (!isInput)
	{
		if (isOutput)
		{
			desiredOutputs = HelperFunctions::label2Mat(label, rows, cols);
			for (unsigned i = 0; i < rows; ++i)
			{
				for (unsigned j = 0; j < cols; ++j)
				{
					deltas[i][j] = costFunction->delta(zs[i][j], activations[i][j], desiredOutputs[i][j]);
				}
				
			}
			for (unsigned i = 0; i < rows; ++i)
				delete[]desiredOutputs[i];
			delete[]desiredOutputs;
		}
		else
		{
			FloatingType **nextWeights = outputLayer->getWeights();
			FloatingType **w_t = HelperFunctions::matrixTranspose(nextWeights, 
			                                                      outputLayer->getNumberOfNeurons(),
			                                                      numberOfNeurons);
			FloatingType **nextDelta = outputLayer->getDelta();
			
			FloatingType *nextDeltaVec = HelperFunctions::mat2vec(nextDelta, 
			                                                      outputLayer->getRows(), 
			                                                      outputLayer->getCols());
			                                                      
			FloatingType *w_d = HelperFunctions::matVecMul(w_t, 
			                                               numberOfNeurons, 
			                                               outputLayer->getNumberOfNeurons(), 
			                                               nextDeltaVec, 
			                                               outputLayer->getNumberOfNeurons());
			                                               
			FloatingType *zsVec = HelperFunctions::mat2vec(zs, rows, cols);
			FloatingType* sigmoids = activation->activationPrimeVec(zsVec, numberOfNeurons);
			FloatingType* d = HelperFunctions::hadamardProduct(w_d, sigmoids, numberOfNeurons);
			unsigned crtIdx = 0;
			for (unsigned i = 0; i < rows; ++i)
			{
				for (unsigned j = 0; j < cols; ++j)
				{
					deltas[i][j] = d[crtIdx];
					crtIdx++;
				}
			}
			delete[]zsVec;
			delete[]d;
			delete[]sigmoids;
			delete[]w_d;
			delete[]nextDeltaVec;
			for (unsigned i = 0; i < numberOfNeurons; ++i)
			{
				delete[]w_t[i];
			}
			delete[]w_t;

		}
		updateNablaB();
		updateNablaW();
	}
}

const int FullyConnectedLayer::getOutputSize()
{
	return numberOfNeurons;
}

FloatingType ** FullyConnectedLayer::getWeights()
{
	return weights;
}

void FullyConnectedLayer::initDelta()
{
	deltas = new FloatingType*[rows];
	for (unsigned i = 0; i < rows; ++i)
	{
		deltas[i] = new FloatingType[cols];
	}
}

void FullyConnectedLayer::updateNablaB()
{
	if (!isInput)
	{
		for (unsigned i = 0; i < rows; ++i)
		{
			for (unsigned j = 0; j < cols; ++j)
			{
				nablaB[i][j] += deltas[i][j];
			}
			
		}
	}
}

void FullyConnectedLayer::updateNablaW()
{
	if (!isInput)
	{
		unsigned prevLayerNumNeurons = inputLayer->getNumberOfNeurons();
		FloatingType **prevActivations = inputLayer->getOutputs();
		FloatingType *prevActivationsVec = HelperFunctions::mat2vec(prevActivations, inputLayer->getRows(), inputLayer->getCols());
		FloatingType *deltasVec = HelperFunctions::mat2vec(deltas, rows, cols);
		for (unsigned i = 0; i < numberOfNeurons; ++i)
		{
			for (unsigned j = 0; j < prevLayerNumNeurons; ++j)
			{
				nablaW[i][j] += prevActivationsVec[j] * deltasVec[i];
			}
		}
		delete[]prevActivationsVec;
		delete[]deltasVec;
	}
}

void FullyConnectedLayer::resetNablaB()
{
	if (!isInput)
	{
		for (unsigned i = 0; i < rows; ++i)
		{
			for (unsigned j = 0; j < cols; ++j)
			{
				nablaB[i][j] = 0;
			}
		}
	}
}

void FullyConnectedLayer::resetNablaW()
{
	if (!isInput)
	{
		unsigned prevNeuronNr = inputLayer->getNumberOfNeurons();
		for (unsigned i = 0; i < numberOfNeurons; ++i)
		{
			for (unsigned j = 0; j < prevNeuronNr; ++j)
			{
				nablaW[i][j] = 0;
			}
		}
	}
}

void FullyConnectedLayer::updateWeights(const FloatingType eta, const FloatingType lambda, const int numberOfSamples, const int miniBatchSize)
{
	if (!isInput)
	{
		unsigned prevNumNeurons = inputLayer->getNumberOfNeurons();
		for (unsigned i = 0; i < numberOfNeurons; ++i)
		{
			for (unsigned j = 0; j < prevNumNeurons; ++j)
			{
				weights[i][j] = (1 - eta * lambda / numberOfSamples) * weights[i][j] - (eta / miniBatchSize) * nablaW[i][j];
			}
		}
	}
}

void FullyConnectedLayer::updateBiases(const FloatingType eta, const int miniBatchSize)
{
	if (!isInput)
	{
		for (unsigned i = 0; i < rows; ++i)
		{
			for (unsigned j = 0; j < cols; ++j)
			{
				biases[i][j] -= (eta / miniBatchSize) * nablaB[i][j];
			}
			
		}
	}
}

void FullyConnectedLayer::initZs()
{
	zs = new FloatingType*[rows];
	for (unsigned i = 0; i < rows; ++i)
	{
		zs[i] = new FloatingType[cols];
	}
}

void FullyConnectedLayer::initActivations()
{
	activations = new FloatingType*[rows];
	for (unsigned i = 0; i < rows; ++i)
		activations[i] = new FloatingType[cols];
}

FullyConnectedLayer& FullyConnectedLayer::operator=(const FullyConnectedLayer& other)
{
	if (this != &other)
	{
		this->cols = other.cols;
		this->rows = other.rows;
		this->classId = other.classId;
		this->setCostFunction(other.costFunction);
		this->setActivation(other.activation);
		this->isInput = other.isInput;
		this->isOutput = other.isOutput;
		this->numberOfNeurons = other.numberOfNeurons;
		this->numberOfSamples = other.numberOfSamples;
		
		//create weights
		this->weights = new FloatingType*[numberOfNeurons];
		for (unsigned i = 0; i < numberOfNeurons; ++i)
		{
			this->weights[i] = new FloatingType[inputLayer->getNumberOfNeurons()];
		}

		for (unsigned i = 0; i < numberOfNeurons; ++i)
		{
			for (unsigned j = 0; j < inputLayer->getNumberOfNeurons(); ++j)
			{
				this->weights[i][j] = other.weights[i][j];
			}
		}
		
		//create biases
		this->biases = new FloatingType*[rows];
		for (unsigned i = 0; i < rows; ++i)
		{
			this->biases[i] = new FloatingType[cols];
		}
		for (unsigned i = 0; i < rows; ++i)
		{
			for (unsigned j = 0; j < cols; ++j)
			{
				this->biases[i][j] = other.biases[i][j];
			}
		}

		//create zs
		this->zs = new FloatingType*[rows];
		for (unsigned i = 0; i < rows; ++i)
		{
			this->zs[i] = new FloatingType[cols];
		}
		for (unsigned i = 0; i < rows; ++i)
		{
			for (unsigned j = 0; j < cols; ++j)
			{
				this->zs[i][j] = other.zs[i][j];
			}
		}

		//create deltas
		this->deltas = new FloatingType*[rows];
		for (unsigned i = 0; i < rows; ++i)
		{
			this->deltas[i] = new FloatingType[cols];
		}
		for (unsigned i = 0; i < rows; ++i)
		{
			for (unsigned j = 0; j < cols; ++j)
			{
				this->deltas[i][j] = other.deltas[i][j];
			}
		}

		//create activations
		this->activations = new FloatingType*[rows];
		for (unsigned i = 0; i < rows; ++i)
		{
			this->activations[i] = new FloatingType[cols];
		}
		for (unsigned i = 0; i < rows; ++i)
		{
			for (unsigned j = 0; j < cols; ++j)
			{
				this->activations[i][j] = other.activations[i][j];
			}
		}

	}
	return *this;
}

FullyConnectedLayer::~FullyConnectedLayer()
{
    if (nablaB)
    {
        for (unsigned i = 0; i < rows; ++i)
        {
            delete []nablaB[i];
        }
        delete []nablaB;
        nablaB = NULL;
    }
    if (biases)
    {
        for (unsigned i = 0; i < rows; ++i)
        {
            delete []biases[i];
        }
        delete []biases;
        biases = NULL;
    }

    if (nablaW)
    {
        for (unsigned i = 0; i < numberOfNeurons; ++i)
        {
            delete []nablaW[i];
        }
        delete []nablaW;
        nablaW = NULL;
    }
    if (weights)
    {
        for (unsigned i = 0; i < numberOfNeurons; ++i)
        {
            delete []weights[i];
        }
        delete []weights;
        weights = NULL;
    }
    if (zs)
    {
        for (unsigned i = 0; i < rows; ++i)
        {
            delete []zs[i];
        }
        delete []zs;
        zs = NULL;
    }
    if (deltas)
    {
        for (unsigned i = 0; i < rows; ++i)
        {
            delete []deltas[i];
        }
        delete []deltas;
        deltas = NULL;
    }
    if (activations)
    {
        for (unsigned i = 0; i < rows; ++i)
        {
            delete []activations[i];
        }
        delete []activations;
        activations = NULL;
    }
    if (activation)
    {
        delete activation;
        activation = NULL;
    }
}
