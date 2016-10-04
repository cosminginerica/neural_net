#ifndef FEATUREMAP_H
#define FEATUREMAP_H
#include <assert.h>
#include "Types.h"
#include "NeuronLayer.h"

class FeatureMapLayer
{
public:
    FeatureMapLayer(unsigned _localReceptiveFieldSize, unsigned _strideSize);
    const unsigned getNumberOfNeurons() const;
private:
    unsigned localReceptiveFieldSize;
    unsigned strideSize;
    NeuronLayer* inputLayer;
    NeuronLayer* outputLayer;
    FloatingType** weights;
    FloatingType bias;

};
#endif
