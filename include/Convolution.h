#ifndef CONVOLUTION_H
#define CONVOLUTION_H
#include "Types.h"

namespace HelperFunctions
{
    static inline FloatingType** convolve(const FloatingType** inputMat, 
                                          const unsigned m, 
                                          const unsigned n, 
                                          const FloatingType** filter, 
                                          const unsigned fM, 
                                          const unsigned fN, 
                                          const unsigned strideSize)
    {
        assert(fM % 2 == 1);
        assert(fN % 2 == 1);
        assert(m >= fM);
        assert(n >=fN);
        const unsigned resRows = (m - fM + 1) / strideSize;
        const unsigned resCols = (n - fN + 1) / strideSize;
        FloatingType** resMat = new FloatingType*[resRows];
        for (unsigned i = 0; i < resRows; ++i)
        {
            resMat[i] = new FloatingType[resCols];
        }

        for (unsigned i = 0; i < resRows; ++i)
        {
            for (unsigned j = 0; j < resCols; ++j)
            {
                resMat[i][j] = 0;
            }
        }
        
        for (unsigned i = 0; i < resRows; ++i)
        {
            for (unsigned j = 0; j < resCols; ++j)
            {
                for (unsigned k = 0; k < fM; ++k)
                {
                    for (unsigned l = 0; l < fN; ++l)
                    {
                        resMat[i][j] += inputMat[][] * filter[][];
                    }
                }
            }
        }

        return resMat;
    }
};
#endif //CONVOLUTION_H