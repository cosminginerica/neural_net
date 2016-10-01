cmake_minimum_required(VERSION 2.8) 
PROJECT (NEURAL_NET)
if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-std=c++11)
endif()
set( NEURAL_NET_SRC
		src/MNISTLoader.cpp
		src/NeuralNetwork.cpp
		src/CrossEntropyCostFunction.cpp
		src/QuadraticCostFunction.cpp
		src/tu_neural_net.cpp
		src/FullyConnectedLayer.cpp
		src/Activations.cpp
		src/FeatureMapLayer.cpp
		src/MaxPoolLayer.cpp
		src/ConvPoolLayer.cpp
)

set( NEURAL_NET_HEADERS       
	 include/Types.h
	 include/DataLoader.h
	 include/MNISTLoader.h
	 include/CostFunction.h
	 include/CrossEntropyCostFunction.h
	 include/QuadraticCostFunction.h
	 include/NeuralNetwork.h
	 include/FullyConnectedLayer.h
	 include/NeuronLayer.h
	 include/Activations.h
	 include/FeatureMapLayer.h
	 include/MaxPoolLayer.h
	 include/ConvPoolLayer.h
)

INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/include )
link_directories( ${CMAKE_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
add_executable( NEURAL_NET ${NEURAL_NET_SRC} ${NEURAL_NET_HEADERS})
