#include "MNISTLoader.h"
#include <stdio.h>
MNISTLoader::MNISTLoader(MNISTLoader& other, const unsigned sizeOfData)
{
	this->data = &other.data[other.nrTrainingSamples - sizeOfData];
	this->labels = &other.labels[other.nrTrainingSamples - sizeOfData];
	this->indices = &other.indices[other.nrTrainingSamples - sizeOfData];
	this->nrTrainingSamples = other.nrTrainingSamples - sizeOfData;
	this->sizeOfData = other.sizeOfData;
}

RESULT MNISTLoader::loadData(const char * dataFileName, const char * labelFileName)
{
	std::cout << "Loading data" << std::endl;
	clock_t startTime = clock();
	FILE *pFileData = fopen(dataFileName, "rb");
	int tempMagicNr, tempNrSamples, tempNrRows, tempNrCols;
	if (!pFileData)
	{
		return E_NOT_OK;
	}
	else
	{
		fread(&tempMagicNr, sizeof(int), 1, pFileData);
		int magicNr = HelperFunctions::reverseBytes(tempMagicNr);
		fread(&tempNrSamples, sizeof(int), 1, pFileData);
		int nrSamples = HelperFunctions::reverseBytes(tempNrSamples);
#ifdef DEBUG_NETWORK
		nrSamples = NUM_DBG_SAMPLES;
#endif
		fread(&tempNrRows, sizeof(int), 1, pFileData);
		nrRows = HelperFunctions::reverseBytes(tempNrRows);
		fread(&tempNrCols, sizeof(int), 1, pFileData);
		nrCols = HelperFunctions::reverseBytes(tempNrCols);
		nrTrainingSamples = nrSamples;
		sizeOfData = nrCols * nrRows;
		data = new FloatingType *[nrSamples];
		for (int i = 0; i < nrSamples; ++i)
		{
			data[i] = new FloatingType[nrRows * nrCols];
		}
		for (int i = 0; i < nrSamples; ++i)
		{
			unsigned char currentVal;
			for (unsigned j = 0; j < nrCols * nrRows; ++j)
			{
				fread(&currentVal, sizeof(unsigned char), 1, pFileData);
				data[i][j] = (FloatingType)currentVal / 255.;
			}
		}

		fclose(pFileData);
	}
	FILE *pLabelData = fopen(labelFileName, "rb");
	if (!pLabelData)
	{
		return E_NOT_OK;
	}
	else
	{
		fread(&tempMagicNr, sizeof(int), 1, pLabelData);
		int magicNr = HelperFunctions::reverseBytes(tempMagicNr);
		fread(&tempNrSamples, sizeof(int), 1, pLabelData);
		int nrSamples = HelperFunctions::reverseBytes(tempNrSamples);
#ifdef DEBUG_NETWORK
		nrSamples = NUM_DBG_SAMPLES;
#endif
		labels = new int[nrSamples];
		indices = new int[nrSamples];
		for (int i = 0; i < nrSamples; ++i)
		{
			indices[i] = i;
			unsigned char tmpVal;
			fread(&tmpVal, sizeof(unsigned char), 1, pLabelData);
			labels[i] = (int)tmpVal;
		}
		fclose(pLabelData);
	}
	std::cout << "Loading data took " << 1000 * (double)(clock() - startTime) / CLOCKS_PER_SEC << " ms." << std::endl;
	return E_OK;
}
FloatingType* MNISTLoader::getTrainingDataById(const unsigned id)
{
	return data[indices[id]];
}

const int MNISTLoader::getLabelById(const unsigned id)
{
	return labels[indices[id]];
}
void MNISTLoader::shuffleData()
{
	std::random_shuffle(indices, indices + nrTrainingSamples);
}

FloatingType* MNISTLoader::getDataById(const int id)
{
	return data[indices[id]];
}

FloatingType** MNISTLoader::getMiniBatch(const unsigned miniBatchSize, const unsigned batchNr)
{
	FloatingType** res;
	res = new FloatingType*[miniBatchSize];
	for (int i = 0; i < miniBatchSize; ++i)
	{
		res[i] = new FloatingType[sizeOfData];
	}
	for (int i = 0; i < miniBatchSize; ++i)
	{
		for (int j = 0; j < sizeOfData; ++j)
		{
			res[i][j] = data[indices[i + miniBatchSize * batchNr]][j];
		}
	}
	return res;
}

int* MNISTLoader::getMiniBatchLabels(const unsigned miniBatchSize, const unsigned batchNr)
{
	int* res;
	res = new int[miniBatchSize];
	for (int i = 0; i < miniBatchSize; ++i)
	{
		res[i] = labels[indices[i + miniBatchSize * batchNr]];		
	}
	return res;
}

MNISTLoader::~MNISTLoader()
{
	for (unsigned int i = 0; i < nrTrainingSamples; ++i)
	{
		delete []data[i];
		data[i] = NULL;
	}
	delete []data;
	delete []labels;
}

void MNISTLoader::setNrTrainingSamples(const int nr)
{
	nrTrainingSamples = nr;
}

void MNISTLoader::setTrainingData(FloatingType **data, int *labels, const unsigned numberOfSamples, const unsigned sizeOfData)
{
	this->data = new FloatingType *[numberOfSamples];
	this->labels = new int[numberOfSamples];
	this->indices = new int[numberOfSamples];
	for (int i = 0; i < numberOfSamples; ++i)
	{
		this->data[i] = new FloatingType[sizeOfData];
	}
	for (int i = 0; i < numberOfSamples; ++i)
	{
		for (int j = 0; j < sizeOfData; ++j)
		{
			this->data[i][j] = data[i][j];
		}
		this->labels[i] = labels[i];
		this->indices[i] = i;
	}
	this->nrTrainingSamples = numberOfSamples;
	this->sizeOfData = sizeOfData;
}

void MNISTLoader::trim(const int numberToTrim)
{
	FloatingType **newData;
	newData = new FloatingType *[nrTrainingSamples - numberToTrim];
	for (unsigned i = 0; i < nrTrainingSamples - numberToTrim; ++i)
	{
		newData[i] = new FloatingType[this->sizeOfData];
	}
	for (unsigned i = 0; i < nrTrainingSamples - numberToTrim; ++i)
	{
		for (int j = 0; j < this->sizeOfData; ++j)
		{
			newData[i][j] = data[i][j];
		}
	}
	for (unsigned i = 0; i < nrTrainingSamples; ++i)
	{
		delete []data[i];
	}
	delete []data;
	data = newData;
	int *newLabels = new int[nrTrainingSamples - numberToTrim];
	for (unsigned i = 0; i < nrTrainingSamples - numberToTrim; ++i)
	{
		newLabels[i] = labels[i];
	}
	delete []labels;
	labels = newLabels;
	int *newIndices = new int[nrTrainingSamples - numberToTrim];
	for (unsigned i = 0; i < nrTrainingSamples - numberToTrim; ++i)
	{
		newIndices[i] = indices[i];
	}
	delete []indices;
	indices = newIndices;
	nrTrainingSamples -= numberToTrim;
}
