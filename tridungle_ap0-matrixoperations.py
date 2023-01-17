import random

import pprint

rows = 2

columns = 3

pp = pprint.PrettyPrinter(indent=0)

matrix = [[random.random() for e in range(columns)] for e in range(rows)]

pp.pprint(matrix)
import numpy as np

import pprint



pp = pprint.PrettyPrinter(indent=0)



def randomMatrix(cols,rowa):

    return np.random.rand(rows,cols)



matrix = randomMatrix(3,2);

pp.pprint(matrix)
import numpy as np

import matplotlib.pylab as pl



nx = 2

ny = 2

data = np.random.randint(0,2,size=(ny,nx))



pl.figure()

tb = pl.table(cellText=data, loc=(0,0), cellLoc='center')



tc = tb.properties()['child_artists']

for cell in tc: 

    cell.set_height(1/ny)

    cell.set_width(1/nx)



ax = pl.gca()

ax.set_xticks([])

ax.set_yticks([])
import numpy as np

import matplotlib.pylab as pl



nx = 3

ny = 3

data = np.random.randint(0,2,size=(ny,nx))



pl.figure()

tb = pl.table(cellText=data, loc=(0,0), cellLoc='center')



tc = tb.properties()['child_artists']

for cell in tc: 

    cell.set_height(1/ny)

    cell.set_width(1/nx)



ax = pl.gca()

ax.set_xticks([])

ax.set_yticks([])
import numpy as np

import matplotlib.pylab as pl



nx = 4

ny = 4

data = np.random.randint(0,3,size=(ny,nx))



pl.figure()

tb = pl.table(cellText=data, loc=(0,0), cellLoc='center')



tc = tb.properties()['child_artists']

for cell in tc: 

    cell.set_height(1/ny)

    cell.set_width(1/nx)



ax = pl.gca()

ax.set_xticks([])

ax.set_yticks([])
import numpy as np



def randomMatrix(cols,rows):

    return np.random.rand(rows,cols)



def createZeroMatrix(rows, cols):

    """

    @description: Creates a matrix filled with zeros.

    @param rows: the number of rows the matrix should have

    @param cols: the number of columns the matrix should have

    @return: list of lists that form the matrix.

    """

    matrix = []

    while len(matrix) < rows:

        matrix.append([])

        while len(matrix[-1]) < cols:

            matrix[-1].append(0.0)

    return matrix



def copyMatrix(matrix):

    """

    @description: Creates and returns a copy of a matrix.

    @param matrix: The matrix to be copied

    @return: The copy of the given matrix

    """

    # get matrix dimensions

    rows = len(matrix)

    cols = len(matrix[0])



    # create a new matrix of zeros

    zeroMatrix = createZeroMatrix(rows, cols)



    # copy values of matrix into the copy

    for row in range(rows):

        for col in range(cols):

            zeroMatrix[row][col] = matrix[row][col]

    return zeroMatrix





def printMatrix(matrix, decimals=3):

    """

    @description: Print matrix

    @param matrix: The matrix to be printed

    @param decimals: the indent

    """

    for row in matrix:

        print([round(x, decimals)+0 for x in row])



def determinant(matrix):

    """

    @description: calculate determinant of matrix.

    @param matrix: The matrix to be calculated

    """

    numOfColumn = len(matrix)

    matrixM = copyMatrix(matrix)

    # Row ops on A to get in upper triangle form

    for focusDiagonal in range(numOfColumn):

        # only use rows below focus diagonal row

        for row in range(focusDiagonal+1,numOfColumn): 

            # if diagonal is zero ...

            if matrixM[focusDiagonal][focusDiagonal] == 0: 

                # change to ~zero

                matrixM[focusDiagonal][focusDiagonal] == 1.0e-18 

            currentScaler = matrixM[row][focusDiagonal] / matrixM[focusDiagonal][focusDiagonal] 

            # current - currentScaler * focusDiagonalRow, one element at a time

            for col in range(numOfColumn): 

                matrixM[row][col] = matrixM[row][col] - currentScaler * matrixM[focusDiagonal][col]

     

    # Once matrixM is in upper triangle form product of diagonals is determinant

    product = 1.0

    for i in range(numOfColumn):

        product *= matrixM[i][i] 

    return product

 

matrix = randomMatrix(3,3);

printMatrix(matrix)

print('-----------------')

det = determinant(matrix);

print(det)

   
def randomMatrix(cols, rows, tol = None):

    if tol is None:

        return np.random.rand(rows, cols)

    else:

        return np.random.randint(tol, size = (rows, cols))



def zerosMatrix(rows, cols):

    """

    @description: Creates a matrix filled with zeros.

    @param rows: the number of rows the matrix should have

    @param cols: the number of columns the matrix should have

    @return: list of lists that form the matrix.

    """

    matrix = []

    while len(matrix) < rows:

        matrix.append([])

        while len(matrix[-1]) < cols:

            matrix[-1].append(0.0)

    return matrix



def checkSquareness(matrix):

    """

    @description: Makes sure that a matrix is square

    @param matrix: The matrix to be checked.

    """

    if len(matrix) != len(matrix[0]):

        raise ArithmeticError("Matrix must be square to inverse.")

def checkNonSingular(matrix):

    det = determinant(matrix)

    if det != 0:

        return det

    else:

        raise ArithmeticError("Singular Matrix!")

        

def identityMatrix(matrixSize):

    """

    @description: creates and returns an identity matrix.

    @param matrixSize: the square size of the matrix

    @returns: a square identity matrix

    """

    identity = zerosMatrix(matrixSize, matrixSize)

    for i in range(matrixSize):

        identity[i][i] = 1.0



    return identity



def matrixMultiply(matrixA, matrixB):

    """

    @description: Returns the product of the matrix matrixA * matrixB

    @param matrixA: The first matrix

    @param matrixB: The second matrix

    @return: The product of the two matrices

    """

    rowsA = len(matrixA)

    colsA = len(matrixA[0])

    rowsB = len(matrixB)

    colsB = len(matrixB[0])

    if colsA != rowsB:

        raise ArithmeticError('Number of A columns must equal number of B rows.')

    matrixC = zerosMatrix(rowsA, colsB)

    for row in range(rowsA):

        for col in range(colsB):

            total = 0

            for colA in range(colsA):

                total += matrixA[row][colA] * matrixB[colA][col]

            matrixC[row][col] = total

    return matrixC



def checkMatrixEquality(matrixA, matrixB, tolerance=None):

    """

    @description: Checks the equality of two matrices.

    @param matrixA: The first matrix

    @param matrixB: The second matrix

    @param tolerance: The decimal place tolerance of the check

    @return: The boolean result of the equality check

    """

    if len(matrixA) != len(matrixB) or len(matrixA[0]) != len(matrixB[0]):

        return False

    for row in range(len(matrixA)):

        for col in range(len(matrixA[0])):

            if tolerance == None:

                if matrixA[row][col] != matrixB[row][col]:

                    return False

            else:

                if round(matrixA[row][col],tolerance) != round(matrixB[row][col],tolerance):

                    return False

    return True 



def invertMatrix(matrix, tol=None):

    """

    Returns the inverse of the passed in matrix.

    @param matrix: The matrix to be inversed

    @return: The inverse of the matrix

     """

    # Make sure matrix can be inverted.

    checkSquareness(matrix)

    checkNonSingular(matrix)

 

    # Make copies of matrix & identity, matrixM & identityM, to use for row ops

    matrixSize = len(matrix)

    matrixM = copyMatrix(matrix)

    identity = identityMatrix(matrixSize)

    identityM = copyMatrix(identity)

 

    # Perform row operations to allow flexible row referencing

    indices = list(range(matrixSize))

    for focusDiagonal in range(matrixSize):

        focusDiagonalScaler = 1.0 / matrixM[focusDiagonal][focusDiagonal]

        # scale focus diagonal row with focus diagonal inverse. 

        # indicate column looping.

        for column in range(matrixSize): 

            matrixM[focusDiagonal][column] *= focusDiagonalScaler

            identityM[focusDiagonal][column] *= focusDiagonalScaler

        # operate on all rows except focus diagonal row as follows: skip row with focus diagonal in it.

        for row in indices[0:focusDiagonal] + indices[focusDiagonal+1:]: 

            currentScaler = matrixM[row][focusDiagonal]

            for column in range(matrixSize):

                # current - currentScaler * focusDiagonalRow, one element at a time

                matrixM[row][column] = matrixM[row][column] - currentScaler * matrixM[focusDiagonal][column]

                identityM[row][column] = identityM[row][column] - currentScaler * identityM[focusDiagonal][column]

 

    # Make sure IM is an inverse of matrix with specified tolerance

    multiply = matrixMultiply(matrix, identityM)

    if checkMatrixEquality(identity, multiply, tol):

        return identityM

    else:

        raise ArithmeticError("Matrix inverse out of tolerance.")

        

        

matrix = randomMatrix(3,3, 5)

printMatrix(matrix)

invert = invertMatrix(matrix)

print('----------------------')

printMatrix(invert)
