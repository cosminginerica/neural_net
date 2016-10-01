#include <iostream>
#include "FullyConnectedLayer.h"
#include "NeuralNetwork.h"
#include "ConvPoolLayer.h"
void help()
{
    std::cout << "Usage: NEURAL_NET train_data train_labels test_data test_labels" << std::endl;
    std::cout << "train_data - path to training data file (currently supported MNIST dataset)" << std::endl;
    std::cout << "train_labels - path to training data label file (currently supported MNIST dataset)" << std::endl;
    std::cout << "test_data - path to test data file (currently supported MNIST dataset)" << std::endl;
    std::cout << "test_labels - path to test data label file (currently supported MNIST dataset)" << std::endl;

}

int main(int argc, char** argv)
{
    if (argc != 5)
    {
        help();
        return EXIT_FAILURE;
    }
    NeuralNetwork *network = new NeuralNetwork();
    network->loadTrainingData(DATA_MNIST, argv[1], argv[2]);
    network->loadTestData(DATA_MNIST, argv[3], argv[4]);
    network->addFullyConnectedLayer(28, 28, true, false, CROSS_ENTROPY, 0, SIGMOID_ACTIVATION);
    network->addFullyConnectedLayer(3, 10, false, false, CROSS_ENTROPY, 1, SIGMOID_ACTIVATION);
    network->addFullyConnectedLayer(3, 10, false, false, CROSS_ENTROPY, 2, SIGMOID_ACTIVATION);
    network->addFullyConnectedLayer(5, 2, false, true, CROSS_ENTROPY, 3, SIGMOID_ACTIVATION);
    network->initNeuronLayers();
    network->SGD(0.1, 0.1, 30, 10);

    return EXIT_SUCCESS;
}
