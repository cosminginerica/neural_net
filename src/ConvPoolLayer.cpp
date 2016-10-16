#include "ConvPoolLayer.h"
ConvPoolLayer::ConvPoolLayer() : hasFormattedOutput(false), formattedOutput(NULL)
{
}

void ConvPoolLayer::calculateOutputs()
{
    hasFormattedOutput = false;
    for (unsigned i = 0; i < featureMaps.size(); ++i)
    {
        featureMaps[i].calculateOutputs();
        maxPools[i].calculateOutputs();
    }
}

const unsigned ConvPoolLayer::getNumberOfNeurons()
{
    return numberOfNeurons;
}

FloatingType** ConvPoolLayer::formatOutput()
{
    if (!hasFormattedOutput)
    {
        const unsigned newRows = maxPools[0].getRows();
        const unsigned newCols = maxPools.size() * maxPools[0].getCols();
        if (formattedOutput)
        {
            for (unsigned i = 0; i < newRows; ++i)
            {
                delete []formattedOutput[i];
            }
            delete []formattedOutput;
        }

        formattedOutput = new FloatingType*[newRows];
        const unsigned colsStep = maxPools[0].getCols();
        for (unsigned i = 0; i < newRows; ++i)
        {
            formattedOutput[i] = new FloatingType[newCols];
        }
        FloatingType **currentMaxPool;
        for (unsigned i = 0; i < newRows; ++i)
        {
            for (unsigned j = 0; j < newCols; ++j)
            {
                if (j % colsStep == 0)
                {
                    currentMaxPool = maxPools[j / colsStep].getOutputs();
                }
                formattedOutput[i][j] = currentMaxPool[i][j % colsStep];
            }
        }
        hasFormattedOutput = true;
    }
    return formattedOutput;
}

void ConvPoolLayer::initializeWeights()
{
    for (unsigned i = 0; i < featureMaps.size(); ++i)
    {
        featureMaps[i].initWeights();
    }
}

void ConvPoolLayer::initializeBiases()
{
    for (unsigned i = 0; i < featureMaps.size(); ++i)
    {
        featureMaps[i].initBias();
    }

}

FloatingType **ConvPoolLayer::getOutputs()
{
    return formatOutput();
}

const int ConvPoolLayer::getOutputSize()
{
    const unsigned newRows = maxPools[0].getRows();
    const unsigned newCols = maxPools[0].getCols() * maxPools.size();
    return newRows * newCols;
}

void ConvPoolLayer::backPropagate(const int label)
{
    for (unsigned i = 0; i < featureMaps.size(); ++i)
    {
        maxPools[i].backPropagate(label);
        featureMaps[i].backPropagate();
    }
}

FloatingType **ConvPoolLayer::getWeights()
{
    unsigned newRows = numberOfNeurons;
    unsigned newCols = inputLayer->getNumberOfNeurons();

}

void ConvPoolLayer::updateNablaB()
{

}

void ConvPoolLayer::updateNablaW()
{

}

void ConvPoolLayer::resetNablaB()
{

}

void ConvPoolLayer::resetNablaW()
{

}

void ConvPoolLayer::updateWeights(const FloatingType eta, const FloatingType lambda, const int numberOfSamples, const int miniBatchSize)
{

}

void ConvPoolLayer::updateBiases(const FloatingType eta, const int miniBatchSize)
{

}

void ConvPoolLayer::init()
{
    for (unsigned i = 0; i < featureMaps.size(); ++i)
    {
        maxPools[i].init();
        featureMaps[i].init();
    }
    numberOfNeurons = featureMaps.size() * featureMaps[0].getNumberOfNeurons();
}

const string ConvPoolLayer::getClassId()
{
    return std::string("ConvPool");
}
