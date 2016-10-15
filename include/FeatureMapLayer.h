#ifndef FEATUREMAP_H
#define FEATUREMAP_H
#include <assert.h>
#include "Types.h"
#include "CostFunction.h"
#include "MatrixOperation.h"
#include "CrossEntropyCostFunction.h"
#include "QuadraticCostFunction.h"
#include "Activations.h"

class FeatureMapLayer
{
public:
    FeatureMapLayer::FeatureMapLayer(const unsigned _localReceptiveFieldSize, 
                                     const unsigned _strideSize,
                                     const int _costFunction,
                                     const int _activation);
    const unsigned getNumberOfNeurons() const {return numberOfNeurons;}
    void init();
    void calculateOutputs();
    void backPropagate(const int label);
    void updateNablaB();
    void updateNablaW();
    void resetNablaB();
    void resetNablaW();
    void updateWeights(const FloatingType eta, 
                       const FloatingType lambda, 
                       const int numberOfSamples, 
                       const int miniBatchSize);
    void updateBias(const FloatingType eta, const int miniBatchSize);
    void initDelta();
    void initZs();
    void initActivations();
    FloatingType ** mp2fm();
    FloatingType** getExpandedWeightMatrix();
    
    
    
private:
    unsigned localReceptiveFieldSize;
    unsigned strideSize;
    unsigned rows, cols;
    unsigned numberOfNeurons;
    NeuronLayer* inputLayer;
    NeuronLayer* outputLayer;
    FloatingType** weights;
    FloatingType bias;

};
#endif
