#include "MatrixOperations.h"

FloatingType** HelperFunctions::conv(FloatingType** m, 
                                     const unsigned mRows, 
                                     const unsigned mCols, 
                                     FloatingType** filter, 
                                     const unsigned filterRows, 
                                     const unsigned filterCols, 
                                     const unsigned strideSize)
{
    if (!m || !filter)
    {
        return NULL;
    }
    const unsigned resRows = (mRows - filterRows + 1) / strideSize;
    const unsigned resCols = (mCols - filterCols + 1) / strideSize;
    FloatingType** result = new FloatingType*[resRows];
    for (unsigned i = 0; i < resRows; ++i)
        result[i] = new FloatingType[resCols];

    for (unsigned i = 0; i < resRows; i += strideSize)
    {
        for (unsigned j = 0; j < resCols; j += strideSize)
        {
            FloatingType sum = 0;
            for (unsigned k = 0; k < filterRows; ++k)
            {
                for (unsigned l = 0; l < filterCols; ++l)
                    {
                        sum += m[i + k][j + l] * filter[k][l];
                    }
            }
            result[i][j] = sum;
        }
    }
    return result;
}

FloatingType** HelperFunctions::vec2mat(FloatingType * vec, const unsigned dim, const unsigned rows, const unsigned cols)
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

FloatingType** HelperFunctions::matrixMultiplication(FloatingType **m1, 
                                                     const unsigned m1Rows, 
                                                     const unsigned m1Cols, 
                                                     FloatingType **m2, 
                                                     const unsigned m2Rows, 
                                                     const unsigned m2Cols)
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

FloatingType** HelperFunctions::matrixTranspose(FloatingType **m, const unsigned mRows, const unsigned mCols)
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

inline FloatingType* HelperFunctions::matVecMul(FloatingType **m1, 
                                                const unsigned m1Rows, 
                                                const unsigned m1Cols, 
                                                FloatingType *m2, 
                                                const unsigned m2Rows)
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

















