#include "ConvPoolLayer.h"


void ConvPoolLayer::calculateOutputs()
{
    for (unsigned i = 0; i < featureMaps.size(); ++i)
    {
        featureMaps[i].calculateOutputs();
        maxPools[i].calculateOutputs();
    }
}

const unsigned ConvPoolLayer::getNumberOfNeurons()
{

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
    return maxPools[maxPools.size() - 1].getOutputs();
}

const int ConvPoolLayer::getOutputSize()
{
    return maxPools[maxPools.size() - 1].getOutputSize();
}

void ConvPoolLayer::backPropagate(const int label)
{
    for (unsigned i = 0; i < featureMaps.size(); ++i)
    {
        maxPools[i].backPropagate(label);
        featureMaps[i].backPropagate(label);
    }
}

FloatingType **ConvPoolLayer::getWeights()
{

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

}

const string ConvPoolLayer::getClassId()
{
    return std::string("ConvPoolLayer");
}
