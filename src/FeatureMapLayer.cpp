#include "FeatureMapLayer.h"

FeatureMapLayer::FeatureMapLayer(const unsigned _localReceptiveFieldSize, 
                const unsigned _strideSize,
                const int _costFunction,
                const int _activation) : 
    localReceptiveFieldSize(_localReceptiveFieldSize), 
    strideSize(_strideSize)
{
    rows = (inputLayer->getRows() - localReceptiveFieldSize + 1) / strideSize;
    cols = (inputLayer->getCols() - localReceptiveFieldSize + 1) / strideSize;
    numberOfNeurons = rows * cols; 
                      ;
    switch (_costFunction)
    {
        case QUADRATIC_COST:
            this->costFunction = new QuadraticCostFunction;
            break;
        case CROSS_ENTROPY:
        default:
            this->costFunction = new CrossEntropyCostFunction;
            break;
    }
    
    switch (_activation)
    {
        case SIGMOID_ACTIVATION:
        default:
            this->setActivation(new Sigmoid);
            break;
    }
    
    classId = "FeatureMap";
}

void FeatureMapLayer::calculateOutputs()
{
	FloatingType **prevActivations = inputLayer->getOutputs();

	zs = HelperFunctions::conv(prevActivations, 
	                           inputLayer->getRows(), 
	                           inputLayer->getCols() ,
	                           weights, 
	                           localReceptiveFieldSize,
	                           localReceptiveFieldSize, 
	                           strideSize);
    for (unsigned i = 0; i < rows; ++i)
	{
		for (unsigned j = 0; j < cols; ++j)
		{
		    zs[i][j] += bias;
			activations[i][j] = activation->calcActivation(zs[i][j]);
		}	
	}
}

void FeatureMapLayer::initWeights()
{
	weights = new FloatingType *[localReceptiveFieldSize];
	nablaW = new FloatingType *[localReceptiveFieldSize];
	for (unsigned i = 0; i < localReceptiveFieldSize; ++i)
	{
		weights[i] = new FloatingType[localReceptiveFieldSize];
		nablaW[i] = new FloatingType[localReceptiveFieldSize];
	}
	for (unsigned i = 0; i < localReceptiveFieldSize; ++i)
	{
		for (unsigned j = 0; j < localReceptiveFieldSize; ++j)
		{
			weights[i][j] = (HelperFunctions::randomNumber() - 0.5);
			nablaW[i][j] = 0;
		}
	}
	unsigned prevNumberOfNeurons = inputLayer->getNumberOfNeurons();
	nablaWExpanded = new FloatingType *[numberOfNeurons];
	for (unsigned i = 0; i < numberOfNeurons; ++i)
	{
		nablaWExpanded[i] = new FloatingType[prevNumberOfNeurons];
	}
	for (unsigned i = 0; i < numberOfNeurons; ++i)
	{
		for (unsigned j = 0; j < prevNumberOfNeurons; ++j)
		{
			nablaWExpanded[i][j] = 0;
		}
	}
}

void FeatureMapLayer::initBias()
{
	bias = HelperFunctions::randomNumber() - 0.5;
	nablaB = 0;
}

void FeatureMapLayer::backPropagate()
{

    FloatingType **nextDeltasExpanded = mp2fm();
    for (unsigned i = 0; i < rows; ++i)
    {
        for (unsigned j = 0; j < cols; ++j)
        {
            deltas[i][j] = nextDeltasExpanded[i][j] * activation->calcActivationPrime(zs[i][j]);
        }
    }

    for (unsigned i = 0; i < rows; ++i)
    {
        delete[]nextDeltasExpanded[i];
    }
    delete[]nextDeltasExpanded;

	updateNablaB();
	updateNablaW();
}

void FeatureMapLayer::updateNablaB()
{
	nablaB += HelperFunctions::sumMat(deltas, rows, cols);
}

void FeatureMapLayer::updateNablaW()
{
	unsigned prevNumNeurons = inputLayer->getNumberOfNeurons();
	FloatingType **prevActivations = inputLayer->getOutputs();
	FloatingType* prevActivationsVec = HelperFunctions::mat2vec(prevActivations, inputLayer->getRows(), inputLayer->getCols());
	FloatingType* deltasVec = HelperFunctions::mat2vec(deltas, rows, cols);
	for (unsigned i = 0; i < numberOfNeurons; ++i)
	{
		for (unsigned j = 0; j < prevNumNeurons; ++j)
		{
			nablaWExpanded[i][j] += prevActivationsVec[j] * deltasVec[i];
		}
	}
	delete[]prevActivationsVec;
	delete[]deltasVec;
}

void FeatureMapLayer::resetNablaB()
{
	nablaB = 0;
}

void FeatureMapLayer::resetNablaW()
{
	for (unsigned i = 0; i < localReceptiveFieldSize; ++i)
	{
		for (unsigned j = 0; j < localReceptiveFieldSize; ++j)
		{
			nablaW[i][j] = 0;
		}
	}
	unsigned prevSize = inputLayer->getNumberOfNeurons();
	for (unsigned i = 0; i < numberOfNeurons; ++i)
	{
		for (unsigned j = 0; j < prevSize; ++j)
		{
			nablaWExpanded[i][j] = 0;
		}
	}
}

void FeatureMapLayer::updateWeights(const FloatingType eta, 
                                    const FloatingType lambda, 
                                    const int numberOfSamples, 
                                    const int miniBatchSize)
{
	FloatingType sum = HelperFunctions::sumMat(nablaWExpanded, 
	                                           numberOfNeurons, 
	                                           inputLayer->getNumberOfNeurons());
	for (unsigned i = 0; i < localReceptiveFieldSize; ++i)
	{
		for (unsigned j = 0; j < localReceptiveFieldSize; ++j)
		{
			nablaW[i][j] = sum;
		}
	}
	for (unsigned i = 0; i < localReceptiveFieldSize; ++i)
	{
		for (unsigned j = 0; j < localReceptiveFieldSize; ++j)
		{
			weights[i][j] = (1 - eta * lambda / numberOfSamples) * weights[i][j] - 
			                (eta / miniBatchSize) * nablaW[i][j];
		}
	}
}

void FeatureMapLayer::updateBias(const FloatingType eta, const int miniBatchSize)
{
	bias -= (eta / miniBatchSize) * nablaB;
}

void FeatureMapLayer::init()
{
	numberOfNeurons = cols * rows;
	initWeights();
	initBias();
	initDelta();
	initZs();
	initActivations();
}

void FeatureMapLayer::initDelta()
{
	deltas = new FloatingType*[rows];
	for (unsigned i = 0; i < rows; ++i)
	{
		deltas[i] = new FloatingType[cols];
	}
}

void FeatureMapLayer::initZs()
{
	zs = new FloatingType*[rows];
	for (unsigned i = 0; i < rows; ++i)
	{
		zs[i] = new FloatingType[cols];
	}
}

void FeatureMapLayer::initActivations()
{
	activations = new FloatingType*[rows];
	for (unsigned i = 0; i < rows; ++i)
	{
		activations[i] = new FloatingType[cols];
	}
}

FloatingType ** FeatureMapLayer::mp2fm()
{

	unsigned nRows = outputLayer->getRows();
	unsigned nCols = outputLayer->getCols();
	unsigned maxPoolReductionFactor = numberOfNeurons / outputLayer->getNumberOfNeurons() / 2;
	FloatingType **nextDeltas = outputLayer->getDelta();

	FloatingType **expandedRes;
	expandedRes = new FloatingType *[maxPoolReductionFactor * nRows];
	for (unsigned i = 0; i < maxPoolReductionFactor * nRows; ++i)
	{
		expandedRes[i] = new FloatingType[maxPoolReductionFactor * nCols];
	}
	for (unsigned i = 0; i < nRows; ++i)
	{
		for (unsigned j = 0; j < nCols; ++j)
		{
			for (unsigned k = 0; k < maxPoolReductionFactor; ++k)
			{
				for (unsigned l = 0; l < maxPoolReductionFactor; ++l)
				{
					expandedRes[i * maxPoolReductionFactor + k][j * maxPoolReductionFactor + l] = nextDeltas[i][j];
				}
			}
		}
	}

	return expandedRes;
}

FloatingType** FeatureMapLayer::getExpandedWeightMatrix()
{
	FloatingType **res = new FloatingType*[numberOfNeurons];
	unsigned prevNumNeurons = inputLayer->getNumberOfNeurons();
	for (unsigned i = 0; i < numberOfNeurons; ++i)
	{
		res[i] = new FloatingType[prevNumNeurons];
	}

	for (unsigned i = 0; i < numberOfNeurons; ++i)
	{
		for (unsigned j = 0; j < prevNumNeurons; ++j)
		{
			res[i][j] = 0;
		}
	}

	for (unsigned k = 0; k < numberOfNeurons; k++)
		for (unsigned wRows = 0; wRows < localReceptiveFieldSize; wRows++)
			for (unsigned wCols = 0; wCols < localReceptiveFieldSize; wCols++)
			{
				unsigned x = (k / cols) * inputLayer->getCols() + 
				             k % cols + 
				             wRows * inputLayer->getCols() + 
				             wCols;
				             
				res[k][x] = weights[wRows][wCols]; // weights values;
			}

	return res;
}
