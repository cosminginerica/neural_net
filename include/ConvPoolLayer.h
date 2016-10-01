#ifndef CONVPOOLLAYER_H
#define CONVPOOLLAYER_H
#include "NeuronLayer.h"
#include "MaxPoolLayer.h"
#include "FeatureMapLayer.h"

class ConvPoolLayer : public NeuronLayer
{
public:
    void calculateOutputs();
    void initializeWeights();
    void initializeBiases();
    FloatingType** getOutputs();
    const int getOutputSize();
    void backPropagate(const int label);
    FloatingType **getWeights();
    void updateNablaB();
    void updateNablaW();
    void resetNablaB();
    void resetNablaW();
    void updateWeights(const FloatingType eta, const FloatingType lambda, const int numberOfSamples, const int miniBatchSize);
    void updateBiases(const FloatingType eta, const int miniBatchSize);
    void init();
    std::string const getClassId();
private:
    std::vector<FeatureMapLayer> featureMaps;
    std::vector<MaxPoolLayer> maxPools;

};

#endif
