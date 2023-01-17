import numpy as np #importing numpy library as pn
numbers = np.array([1,2,3,4,5]) #Creating an Array
numbers # Displaying an array
matrix = np.random.randn(3,3) #Creating an Multidimensional array
matrix
numbers * 2
numbers + 1
matrix/2
matrix%3
arr = np.array(range(10)) 
arr
arr[0]
arr[3:5] #displaying values from index 4 to 5 where last is element is not taken(n-1)
matrix[1,2]
matrix[2,2]
matrix[1:2,1:2] # Second row , Second column value
matrix[0:2,1:3]
names = np.array(['Bob','Joe','Will','Bob','Will','Joe','Joe'])
names
data = np.random.randn(7,4)
data
names == 'Bob'
data[names == 'Bob'] #displays values according to boolean logic passed
arr = np.empty((5,5))

for i in range(5):

    arr[i] = i
arr
arr[[0,1,3]] #which to be displayed Fancy indexing
arr[[4,2]]