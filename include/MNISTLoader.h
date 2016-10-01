#ifndef MNISTLOADER_H
#define MNISTLOADER_H
#include "DataLoader.h"
class MNISTLoader : public DataLoader
{
public:
    MNISTLoader(){}
    MNISTLoader(MNISTLoader& other, const unsigned sizeOfData);
	RESULT loadData(const char * dataFileName, const char * labelFileName);
    FloatingType* getTrainingDataById(const unsigned id);
    const int getLabelById(const unsigned id);
	void shuffleData();
	FloatingType* getDataById(const int id);
    FloatingType** getMiniBatch(const unsigned miniBatchSize, const unsigned batchNr);
    int* getMiniBatchLabels(const unsigned miniBatchSize, const unsigned batchNr);
	~MNISTLoader();
	void setNrTrainingSamples(const int nr);
    void setTrainingData(FloatingType **data, int *labels, const unsigned numberOfSamples, const unsigned sizeOfData);
	void trim(const int numberToTrim);
	const unsigned getNrRows(){ return nrRows; }
	const unsigned getNrCols(){ return nrCols; }
private:
	unsigned nrRows, nrCols;
};
#endif
