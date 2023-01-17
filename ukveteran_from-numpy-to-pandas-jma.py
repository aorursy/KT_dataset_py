import numpy as np



oneDim = np.array([1.0,2,3,4,5])   # a 1-dimensional array (vector)

print(oneDim)

print("#Dimensions =", oneDim.ndim)

print("Dimension =", oneDim.shape)

print("Size =", oneDim.size)

print("Array type =", oneDim.dtype)



twoDim = np.array([[1,2],[3,4],[5,6],[7,8]])  # a two-dimensional array (matrix)

print(twoDim)

print("#Dimensions =", twoDim.ndim)

print("Dimension =", twoDim.shape)

print("Size =", twoDim.size)

print("Array type =", twoDim.dtype)



arrFromTuple = np.array([(1,'a',3.0),(2,'b',3.5)])  # create ndarray from tuple

print(arrFromTuple)

print("#Dimensions =", arrFromTuple.ndim)

print("Dimension =", arrFromTuple.shape)

print("Size =", arrFromTuple.size)
print(np.random.rand(5))      # random numbers from a uniform distribution between [0,1]

print(np.random.randn(5))     # random numbers from a normal distribution

print(np.arange(-10,10,2))    # similar to range, but returns ndarray instead of list

print(np.arange(12).reshape(3,4))  # reshape to a matrix

print(np.linspace(0,1,10))    # split interval [0,1] into 10 equally separated values

print(np.logspace(-3,3,7))    # create ndarray with values from 10^-3 to 10^3
print(np.zeros((2,3)))        # a matrix of zeros

print(np.ones((3,2)))         # a matrix of ones

print(np.eye(3))              # a 3 x 3 identity matrix
x = np.array([1,2,3,4,5])



print(x + 1)      # addition

print(x - 1)      # subtraction

print(x * 2)      # multiplication

print(x // 2)     # integer division

print(x ** 2)     # square

print(x % 2)      # modulo  

print(1 / x)      # division
x = np.array([2,4,6,8,10])

y = np.array([1,2,3,4,5])



print(x + y)

print(x - y)

print(x * y)

print(x / y)

print(x // y)

print(x ** y)
x = np.arange(-5,5)

print(x)



y = x[3:5]     # y is a slice, i.e., pointer to a subarray in x

print(y)

y[:] = 1000    # modifying the value of y will change x

print(y)

print(x)



z = x[3:5].copy()   # makes a copy of the subarray

print(z)

z[:] = 500          # modifying the value of z will not affect x

print(z)

print(x)
my2dlist = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]   # a 2-dim list

print(my2dlist)

print(my2dlist[2])        # access the third sublist

print(my2dlist[:][2])     # can't access third element of each sublist

# print(my2dlist[:,2])    # this will cause syntax error



my2darr = np.array(my2dlist)

print(my2darr)

print(my2darr[2][:])      # access the third row

print(my2darr[2,:])       # access the third row

print(my2darr[:][2])      # access the third row (similar to 2d list)

print(my2darr[:,2])       # access the third column

print(my2darr[:2,2:])     # access the first two rows & last two columns
my2darr = np.arange(1,13,1).reshape(3,4)

print(my2darr)



divBy3 = my2darr[my2darr % 3 == 0]

print(divBy3, type(divBy3))



divBy3LastRow = my2darr[2:, my2darr[2,:] % 3 == 0]

print(divBy3LastRow)
my2darr = np.arange(1,13,1).reshape(4,3)

print(my2darr)



indices = [2,1,0,3]    # selected row indices

print(my2darr[indices,:])



rowIndex = [0,0,1,2,3]     # row index into my2darr

columnIndex = [0,2,0,1,2]  # column index into my2darr

print(my2darr[rowIndex,columnIndex])
print(y)



print(np.abs(y))          # convert to absolute values

print(np.sqrt(abs(y)))    # apply square root to each element

print(np.sign(y))         # get the sign of each element

print(np.exp(y))          # apply exponentiation

print(np.sort(y))         # sort array
x = np.arange(-2,3)

y = np.random.randn(5)

print(x)

print(y)



print(np.add(x,y))           # element-wise addition       x + y

print(np.subtract(x,y))      # element-wise subtraction    x - y

print(np.multiply(x,y))      # element-wise multiplication x * y

print(np.divide(x,y))        # element-wise division       x / y

print(np.maximum(x,y))       # element-wise maximum        max(x,y)
y = np.array([-3.2, -1.4, 0.4, 2.5, 3.4])    # generate a random vector

print(y)



print("Min =", np.min(y))             # min 

print("Max =", np.max(y))             # max 

print("Average =", np.mean(y))        # mean/average

print("Std deviation =", np.std(y))   # standard deviation

print("Sum =", np.sum(y))             # sum 
X = np.random.randn(2,3)    # create a 2 x 3 random matrix

print(X)

print(X.T)             # matrix transpose operation X^T



y = np.random.randn(3) # random vector 

print(y)

print(X.dot(y))        # matrix-vector multiplication  X * y

print(X.dot(X.T))      # matrix-matrix multiplication  X * X^T

print(X.T.dot(X))      # matrix-matrix multiplication  X^T * X
X = np.random.randn(5,3)

print(X)



C = X.T.dot(X)               # C = X^T * X is a square matrix



invC = np.linalg.inv(C)      # inverse of a square matrix

print(invC)

detC = np.linalg.det(C)      # determinant of a square matrix

print(detC)

S, U = np.linalg.eig(C)      # eigenvalue S and eigenvector U of a square matrix

print(S)

print(U)
from pandas import Series



s = Series([3.1, 2.4, -1.7, 0.2, -2.9, 4.5])   # creating a series from a list

print(s)

print('Values=', s.values)     # display values of the Series

print('Index=', s.index)       # display indices of the Series
import numpy as np



s2 = Series(np.random.randn(6))  # creating a series from a numpy ndarray

print(s2)

print('Values=', s2.values)   # display values of the Series

print('Index=', s2.index)     # display indices of the Series
s3 = Series([1.2,2.5,-2.2,3.1,-0.8,-3.2], 

            index = ['Jan 1','Jan 2','Jan 3','Jan 4','Jan 5','Jan 6',])

print(s3)

print('Values=', s3.values)   # display values of the Series

print('Index=', s3.index)     # display indices of the Series
capitals = {'MI': 'Lansing', 'CA': 'Sacramento', 'TX': 'Austin', 'MN': 'St Paul'}



s4 = Series(capitals)   # creating a series from dictionary object

print(s4)

print('Values=', s4.values)   # display values of the Series

print('Index=', s4.index)     # display indices of the Series
s3 = Series([1.2,2.5,-2.2,3.1,-0.8,-3.2], 

            index = ['Jan 1','Jan 2','Jan 3','Jan 4','Jan 5','Jan 6',])

print(s3)



# Accessing elements of a Series



print('\ns3[2]=', s3[2])        # display third element of the Series

print('s3[\'Jan 3\']=', s3['Jan 3'])   # indexing element of a Series 



print('\ns3[1:3]=')             # display a slice of the Series

print(s3[1:3])

print('s3.iloc([1:3])=')      # display a slice of the Series

print(s3.iloc[1:3])
print('shape =', s3.shape)  # get the dimension of the Series

print('size =', s3.size)    # get the # of elements of the Series
print(s3[s3 > 0])   # applying filter to select elements of the Series
print(s3 + 4)       # applying scalar operation on a numeric Series

print(s3 / 4)    
print(np.log(s3 + 4))    # applying numpy math functions to a numeric Series
%matplotlib inline



s3 = Series([1.2,2.5,-2.2,3.1,-0.8,-3.2,1.4], 

            index = ['Jan 1','Jan 2','Jan 3','Jan 4','Jan 5','Jan 6','Jan 7'])

s3.plot(kind='line', title='Line plot')
s3.plot(kind='bar', title='Bar plot')
s3.plot(kind='hist', title = 'Histogram')