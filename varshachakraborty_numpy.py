# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from numpy.random import rand

from numpy.linalg import solve, inv

a = np.array([[1, 2, 3], [3, 4, 6.7], [5, 9.0, 5]])

a.transpose()

               

               

    
c=rand(3,3)

print (c)

c.transpose()
from numpy.random import rand

c=rand(3,3)

print (c)

c.transpose()

from numpy.random import rand

c=rand(3,3)

print ("first matrix",c)

d=rand(3,3)

print ("second matrix",d)

print ("sum of two matrices is",c+d)

print ("difference of two matrices",c-d)

print ("multiplication of two matrices",c*d)



import numpy as np



a = np.zeros((2,2))   # Create an array of all zeros

print(a)              # Prints "[[ 0.  0.]

                      #          [ 0.  0.]]"



b = np.ones((2,2))    # Create an array of all ones

print(b)              # Prints "[[ 1.  1.]]"

e = np.random.random((2,2))  # Create an array filled with random values

print(e)         
import numpy as np

from scipy import stats



# X is a Python List

X = [32.32, 56.98, 21.52, 44.32, 55.63, 13.75, 43.47, 43.34]



# Sorting the data and printing it.

X.sort()

print(X)

# [13.75, 21.52, 32.32, 43.34, 43.47, 44.32, 55.63, 56.98]



# Using NumPy's built-in functions to Find Mean, Median, SD and Variance

mean = np.mean(X)

median = np.median(X)

mode=stats.mode(X)

sd = np.std(X)

variance = np.var(X)



# Printing the values

print("Mean", mean) # 38.91625

print("Median", median) # 43.405

print("Standard Deviation", sd) # 14.3815654029

print("Variance", variance) # 206.829423437

print("mode",mode)
import numpy as np



array = np.array([1,2,2,3,3,4,5,6,6,6,6,7,8,9])



unique = np.unique(array, axis=0)

print (unique)



import numpy as np

array1 = np.array([0, 10, 20, 40, 60, 80])

print("Array1: ",array1)

array2 = [10, 30, 40, 50, 70]

print("Array2: ",array2)

print("Unique sorted array of values that are in either of the two input arrays:")

print(np.union1d(array1, array2))

print("Common values between two arrays:")

print(np.intersect1d(array1, array2))

import numpy as np

a = [[11,2,4],[4,5,6],[10,8,-12]]

b = np.asarray(a)

print ('Diagonal (sum): ', np.trace(b))

print ('Diagonal (elements): ', np.diagonal(b))