#ifndef MATRIXOPERATIONS_H
#define MATRIXOPERATIONS_H
#include "Types.h"

namespace HelperFunctions
{
    FloatingType** conv(FloatingType** m, 
                        const unsigned mRows, 
                        const unsigned mCols, 
                        FloatingType** filter, 
                        const unsigned filterRows, 
                        const unsigned filterCols, 
                        const unsigned strideSize);

    FloatingType** vec2mat(FloatingType * vec, const unsigned dim, const unsigned rows, const unsigned cols);

    FloatingType** matrixMultiplication(FloatingType **m1, 
                                        const unsigned m1Rows, 
                                        const unsigned m1Cols, 
                                        FloatingType **m2, 
                                        const unsigned m2Rows, 
                                        const unsigned m2Cols);

    FloatingType** matrixTranspose(FloatingType **m, const unsigned mRows, const unsigned mCols);

    FloatingType* matVecMul(FloatingType **m1, 
                            const unsigned m1Rows, 
                            const unsigned m1Cols, 
                            FloatingType *m2, 
                            const unsigned m2Rows);
};

#endif
