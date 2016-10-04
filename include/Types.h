#ifndef TYPES_HH
#define TYPES_HH
#include <iostream>
#include <math.h>
#include <vector>
#include <ctime>
#include <algorithm>
#include <float.h>

//#define DEBUG_NETWORK
#ifdef DEBUG_NETWORK
#define NUM_DBG_SAMPLES 1000
#endif
typedef double FloatingType;
typedef int RESULT;
#define FEPS 1e-5
using namespace std;

enum results
{
    E_NOT_OK = 0,
    E_OK,
};

enum dataSource
{
    DATA_MNIST = 0,
};

enum costFunctions
{
    QUADRATIC_COST,
    CROSS_ENTROPY,
};

enum evaluationData
{
    TRAINING_DATA,
    VALIDATION_DATA,
    TEST_DATA
};

enum typesOfLayers
{
    FULLY_CONNECTED_LAYER,
    CONV_POOL_LAYER,
    SOFTMAX_LAYER
};
enum activationTypes
{
    SIGMOID_ACTIVATION,
    RELU_ACTIVATION
};

enum neuronsBasedOnLayer
{
    NEURONS_FULLY_CONNECTED,
    NEURONS_FEATURE_MAP,
    NEURONS_MAX_POOLING,
};

namespace HelperFunctions
{
    static const FloatingType sigmoid(const FloatingType z){return 1.0 / (1.0 + exp(-z));}
    static const FloatingType relu(const FloatingType z){ return (z >= 0)? z : 0; }
    inline FloatingType** vec2mat(FloatingType * vec, const unsigned dim, const unsigned rows, const unsigned cols)
    {
        if (rows * cols != dim)
        {
            std::cout << "Can not transform vec to mat. New dimensions don't match" << std::endl;
            return NULL;
        }
        FloatingType ** res = new FloatingType*[rows];
        for (unsigned i = 0; i < rows; ++i)
        {
            res[i] = new FloatingType[cols];
        }
        unsigned crtIdx = 0;
        for (unsigned i = 0; i < rows; ++i)
        {
            for (unsigned j = 0; j < cols; ++j)
            {
                res[i][j] = vec[crtIdx++];
            }
        }
        return res;
    }
    inline FloatingType* mat2vec(FloatingType** mat, const unsigned rows, const unsigned cols)
    {
        FloatingType* res = new FloatingType[rows * cols];
        unsigned crtIdx = 0;
        for (unsigned i = 0; i < rows; ++i)
        {
            for (unsigned j = 0; j < cols; ++j)
            {
                res[crtIdx++] = mat[i][j];
            }
        }

        return res;
    }
    inline FloatingType* sigmoid_vec(FloatingType* a, int size)
    {
        FloatingType *res = new FloatingType[size];
        for (int i = 0; i < size; ++i)
        {
            res[i] = sigmoid(a[i]);
        }
        return res;
    }

    inline FloatingType* relu_vec(FloatingType* a, int size)
    {
        FloatingType *res = new FloatingType[size];
        for (int i = 0; i < size; ++i)
        {
            res[i] = relu(a[i]);
        }
        return res;
    }

    inline FloatingType** sigmoid_mat(FloatingType** mat, const unsigned rows, const unsigned cols)
    {
        FloatingType** res = new FloatingType*[rows];
        for (unsigned i = 0; i < rows; ++i)
        {
            res[i] = new FloatingType[cols];
        }
        for (unsigned i = 0; i < rows; ++i)
        {
            for (unsigned j = 0; j < cols; ++j)
            {
                res[i][j] = sigmoid(mat[i][j]);
            }
        }
        return res;
    }

    inline FloatingType** relu_mat(FloatingType** mat, const unsigned rows, const unsigned cols)
    {
        FloatingType** res = new FloatingType*[rows];
        for (unsigned i = 0; i < rows; ++i)
        {
            res[i] = new FloatingType[cols];
        }
        for (unsigned i = 0; i < rows; ++i)
        {
            for (unsigned j = 0; j < cols; ++j)
            {
                res[i][j] = relu(mat[i][j]);
            }
        }
        return res;
    }

    inline FloatingType** softmax_mat(FloatingType** mat, const unsigned rows, const unsigned cols)
    {
        FloatingType sum = 0;
        FloatingType** res = new FloatingType*[rows];
        for (unsigned i = 0; i < rows; ++i)
        {
            res[i] = new FloatingType[cols];
        }
        for (unsigned i = 0; i < rows; ++i)
        {
            for (unsigned j = 0; j < cols; ++j)
            {
                sum += exp(mat[i][j]);
            }
        }

        for (unsigned i = 0; i < rows; ++i)
        {
            for (unsigned j = 0; j < cols; ++j)
            {
                res[i][j] = exp(mat[i][j]) / sum;
            }
        }

        return res;
    }

    static const FloatingType sigmoid_prime(const FloatingType z){return sigmoid(z) * (1 - sigmoid(z));}
    static const FloatingType relu_prime(const FloatingType z){ return relu(z); }

    static const FloatingType softmax_prime(FloatingType* a, const unsigned idx, const unsigned size)
    {
        FloatingType sum = 0;
        for (unsigned i = 0; i < size; ++i)
        {
            sum += exp(a[i]);
        }
        return exp(a[idx]) / sum;
    }
    inline FloatingType* sigmoid_prime_vec(FloatingType* a, int size)
    {
        FloatingType *res = new FloatingType[size];
        for (int i = 0; i < size; ++i)
        {
            res[i] = sigmoid_prime(a[i]);
        }
        return res;
    }

    inline FloatingType* relu_prime_vec(FloatingType* a, int size)
    {
        FloatingType *res = new FloatingType[size];
        for (int i = 0; i < size; ++i)
        {
            res[i] = relu(a[i]);
        }
        return res;
    }

    inline FloatingType* softmax_prime_vec(FloatingType* a, int size)
    {
        FloatingType *res = new FloatingType[size];
        for (int i = 0; i < size; ++i)
        {
            res[i] = softmax_prime(a, i, size);
        }
        return res;
    }

    inline FloatingType** sigmoid_prime_mat(FloatingType** a, unsigned rows, unsigned cols)
    {
        FloatingType **res = new FloatingType*[rows];
        for (unsigned i = 0; i < rows; ++i)
        {
            res[i] = new FloatingType[cols];
        }
        for (unsigned i = 0; i < rows; ++i)
        {
            for (unsigned j = 0; j < cols; ++j)
            {
                res[i][j] = sigmoid_prime(a[i][j]);
            }
        }
        return res;
    }

    inline FloatingType** relu_prime_mat(FloatingType** a, unsigned rows, unsigned cols)
    {
        FloatingType **res = new FloatingType*[rows];
        for (unsigned i = 0; i < rows; ++i)
        {
            res[i] = new FloatingType[cols];
        }
        for (unsigned i = 0; i < rows; ++i)
        {
            for (unsigned j = 0; j < cols; ++j)
            {
                res[i][j] = relu_prime(a[i][j]);
            }
        }
        return res;
    }

    static const FloatingType randomNumber()
    {

        FloatingType val = (FloatingType)rand() / RAND_MAX;
        return val;
    }

    // flip the bytes of an 8 byte integer
    inline unsigned int reverseBytes(unsigned int number)
    {
        unsigned int flippedNumber = 0;
        unsigned int mask = 255;

        flippedNumber |= (number & mask) << 24;
        mask = 255 << 8;
        flippedNumber |= (number & mask) << 16;
        mask = 255 << 16;
        flippedNumber |= (number & mask) >> 8;
        mask = 255 << 24;
        flippedNumber |= (number & mask) >> 24;

        return flippedNumber;

    }

    inline FloatingType** matrixMultiplication(FloatingType **m1, int m1Rows, int m1Cols, FloatingType **m2, int m2Rows, int m2Cols)
    {
        if (m1Cols != m2Rows || m1Cols == 0 || m2Cols == 0 || m1Rows == 0 || m2Rows == 0)
        {
            return NULL;
        }
        FloatingType **result;
        result = new FloatingType *[m1Rows];
        for (int i = 0; i < m1Rows; ++i)
        {
            result[i] = new FloatingType[m1Cols];
        }

        for (int i = 0; i < m1Rows; ++i)
        {
            for (int j = 0; j < m2Cols; ++j)
            {
                result[i][j] = 0;
            }
        }

        for (int i = 0; i < m1Rows; ++i)
        {
            for (int j = 0; j < m2Cols; ++j)
            {
                FloatingType sum = 0.;
                for (int k = 0; k < m1Rows; ++k)
                {
                    sum += m1[i][k] * m2[k][j];
                }
                result[i][j] = sum;
            }
        }

        return result;
    }
    inline FloatingType** matrixTranspose(FloatingType **m, int mRows, int mCols)
    {	
        FloatingType **result = new FloatingType *[mCols];
        for (int i = 0; i < mCols; ++i)
        {
            result[i] = new FloatingType[mRows];
        }
        for (int i = 0; i < mRows; ++i)
        {
            for (int j = 0; j < mCols; ++j)
            {
                result[j][i] = m[i][j];
            }
        }

        return result;
    }

    inline FloatingType* matVecMul(FloatingType **m1, int m1Rows, int m1Cols, FloatingType *m2, int m2Rows)
    {
        FloatingType * result = new FloatingType[m1Rows];
        for (int i = 0; i < m1Rows; ++i)
        {
            result[i] = 0;
        }

        for (int i = 0; i < m1Rows; ++i)
        {
            FloatingType sum = 0.;
            for (int k = 0; k < m1Cols; ++k)
            {
                sum += m1[i][k] * m2[k];
            }
            result[i] = sum;
        }

        return result;
    }

    inline FloatingType* vectorAddition(FloatingType *m1, int m1Rows, FloatingType *m2, int m2Rows)
    {
        FloatingType *res;
        res = new FloatingType[m1Rows];
        for (int i = 0; i < m1Rows; ++i)
        {
            res[i] = m1[i] + m2[i];
        }
        return res;
    }
    inline FloatingType* hadamardProduct(FloatingType* m1, FloatingType* m2, int size)
    {
        FloatingType *res = new FloatingType[size];
        for (int i  = 0; i < size; ++i)
        {
            res[i] = m1[i] * m2[i];
        }

        return res;
    }
    inline FloatingType* label2Vec(const int label, const int maxSize)
    {
        FloatingType* res = new FloatingType[maxSize];
        for (int i = 0; i < maxSize; ++i)
        {
            res[i] = 0;
        }
        res[(int)label] = 1;
        return res;
    }
    inline FloatingType** label2Mat(const int label, const unsigned rows, const unsigned cols)
    {
        FloatingType **res = new FloatingType*[rows];
        for (unsigned i = 0; i < rows; ++i)
            res[i] = new FloatingType[cols];
        unsigned crtIdx = 0;
        for (unsigned i = 0; i < rows; ++i)
        {
            for (unsigned j = 0; j < cols; ++j)
            {
                if (crtIdx == label)
                    res[i][j] = 1;
                else
                    res[i][j] = 0;
                crtIdx++;
            }
        }
        return res;
    }

    inline const FloatingType cost_derivative(const FloatingType a, const FloatingType y)
    {
        return a - y;
    }
    // derivative for cross-entropy cost function
    inline const FloatingType CE_cost_derivative(const FloatingType a, const FloatingType y)
    {
        return y / a - (1 - y) / (1 - a);
    }

    inline bool isPerfectSquare(long n)
    {
        if (n < 0)
            return false;

        long tst = (long)(sqrt((FloatingType)n) + 0.5);
        return tst*tst == n;
    }

    inline FloatingType getMax(FloatingType *vector, int size)
    {
        FloatingType max = FLT_MIN;
        for (int i = 0; i < size; ++i)
        {
            if (vector[i] > max)
            {
                max = vector[i];
            }
        }
        return max;
    }
    inline FloatingType sumVec(FloatingType *vector, int size)
    {
        FloatingType sum = 0;
        for (int i = 0; i < size; ++i)
        {

            sum += vector[i];

        }
        return sum;
    }
    inline FloatingType const sumMat(FloatingType** mat, const unsigned rows, const unsigned cols)
    {
        FloatingType sum = 0;
        for (unsigned i = 0; i < rows; ++i)
        {
            for (unsigned j = 0; j < cols; ++j)
            {
                sum += mat[i][j];
            }
        }
        return sum;
    }


}

#endif
