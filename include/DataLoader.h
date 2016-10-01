#ifndef DATALOADER_H
#define DATALOADER_H
#include "Types.h"

class DataLoader
{
public:
	/************************************************************************/
	/* Load training data from a file                                       */
	/************************************************************************/
    DataLoader(){}
    DataLoader(DataLoader&, int){}
	virtual RESULT loadData(const char * dataFileName, const char * labelFileName) = 0;
	const unsigned int getNumberOfSamples(){return nrTrainingSamples;}
    virtual FloatingType* getTrainingDataById(const unsigned id) = 0;
    virtual const int getLabelById(const unsigned id) = 0;
	virtual void shuffleData() = 0;
	virtual FloatingType* getDataById(const int id) = 0;
    virtual FloatingType** getMiniBatch(const unsigned miniBatchSize, const unsigned batchNr) = 0;
    virtual int* getMiniBatchLabels(const unsigned miniBatchSize, const unsigned batchNr) = 0;
	virtual const int getSizeOfInputs(){return sizeOfData;}
	virtual void setNrTrainingSamples(const int nr) = 0;
	virtual ~DataLoader(){}
    virtual void setTrainingData(FloatingType **data, int *labels, const unsigned numberOfSamples, const unsigned sizeOfData)= 0;
    virtual FloatingType** getRawData(const unsigned id){return &data[id];}
	virtual int *getRawLabel(const int id){return &labels[id];}
	virtual void trim(const int numberToTrim) = 0;
protected:
    unsigned nrTrainingSamples;
	FloatingType **data;
	int *labels;
	int *indices;
    unsigned sizeOfData;
};
#endif
