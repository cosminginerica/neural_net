#include "MaxPoolLayer.h"


void MaxPoolLayer::calculateOutputs()
{
	FloatingType **prevOutputs = inputLayer->getOutputs();
	FloatingType **prevZs = inputLayer->getZ();
	int prevRows = inputLayer->getRows();
	unsigned rows = prevRows / maxPoolSize;
	int prevCols = inputLayer->getCols();
	unsigned cols = prevCols / maxPoolSize;
	for (unsigned i = 0; i < rows; i++)
	{
		for (unsigned j = 0; j < cols; j++)
		{
			FloatingType max = FLT_MIN;
			FloatingType maxZ = max;
			for (unsigned k = 0; k < maxPoolSize; ++k)
			{
				for (unsigned l = 0; l < maxPoolSize; ++l)
				{
					if (prevOutputs[i * maxPoolSize + k][j * maxPoolSize + l] > max)
					{
						max = prevOutputs[i * maxPoolSize + k][j * maxPoolSize + l];
						maxZ = prevZs[i * maxPoolSize + k][j * maxPoolSize + l];
					}
				}
			}
			activations[i][j] = max;
			zs[i][j] = maxZ;
		}
	}

}

FloatingType** MaxPoolLayer::getOutputs()
{
	return activations;
}

const int MaxPoolLayer::getOutputSize()
{
	return numberOfNeurons;
}

void MaxPoolLayer::backPropagate(const int label)
{
	FloatingType **nextWeights;
	nextWeights= outputLayer->getWeights();
	FloatingType **nextDeltas = outputLayer->getDelta();
	FloatingType *nextDeltasVec = HelperFunctions::mat2vec(nextDeltas, 
	                                                       outputLayer->getRows(), 
	                                                       outputLayer->getCols());
	                                                       
	FloatingType **w_t = HelperFunctions::matrixTranspose(nextWeights, 
	                                                      outputLayer->getNumberOfNeurons(),
	                                                      numberOfNeurons);
	                                                      
	FloatingType *w_d = HelperFunctions::matVecMul(w_t, 
	                                               numberOfNeurons, 
	                                               outputLayer->getNumberOfNeurons(), 
	                                               nextDeltasVec, 
	                                               outputLayer->getNumberOfNeurons());
	
	FloatingType **sigmoids = activation->activationPrimeMat(zs, rows, cols);
	FloatingType *sigmoidsVec = HelperFunctions::mat2vec(sigmoids, rows, cols);
	FloatingType *d = HelperFunctions::hadamardProduct(w_d, sigmoidsVec, numberOfNeurons);
	unsigned crtIdx = 0;
	for (unsigned i = 0; i < rows; ++i)
	{
		for (unsigned j = 0; j < cols; ++j)
		{
			deltas[i][j] = d[crtIdx++];
		}
	}
	delete[]nextDeltasVec;
	for (unsigned i = 0; i < rows; ++i)
		delete[]sigmoids[i];
	delete[]sigmoids;
	delete[]sigmoidsVec;
	delete[]d;
	delete[]w_d;
	for (int i = 0; i < numberOfNeurons; ++i)
	{
		delete[]w_t[i];
	}
	delete[]w_t;
}

void MaxPoolLayer::init()
{
	numberOfNeurons = inputLayer->getNumberOfNeurons() / (2 * maxPoolSize);
	this->rows = inputLayer->getRows() / maxPoolSize;
	this->cols = inputLayer->getCols() / maxPoolSize;
	activations = new FloatingType*[rows];
	zs = new FloatingType*[rows];
	deltas = new FloatingType*[rows];
	for (unsigned i = 0; i < rows; ++i)
	{
		activations[i] = new FloatingType[cols];
		zs[i] = new FloatingType[cols];
		deltas[i] = new FloatingType[cols];
	}
}

FloatingType** MaxPoolLayer::getDelta()
{
	return deltas;
}

ostream& MaxPoolLayer::serialize(ostream& f)const
{
	f << "Class ID:" << "MaxPool" << std::endl;
	f << "Number of neurons:" << this->numberOfNeurons << std::endl;
	f << "Rows:" << this->rows << std::endl;
	f << "Cols:" << this->cols << std::endl;
	f << "Max pool size:" << this->maxPoolSize << std::endl;
	f << "Deltas:" << std::endl;
	for (unsigned i = 0; i < rows; ++i)
	{
		for (unsigned j = 0; j < cols; ++j)
		{
			f << deltas[i][j] << " ";
		}
		f << std::endl;
	}
	return f;

}
