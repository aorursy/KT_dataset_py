# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Basic Array Operations

# Defining Array 1

a = np.array([[1, 2],

              [3, 4]])

 

# Defining Array 2

b = np.array([[4, 3],

              [2, 1]])

               

# Adding 1 to every element

print ("Adding 1 to every element:", a + 1)

 

# Subtracting 2 from each element

print ("\nSubtracting 2 from each element:", b - 2)

 

# sum of array elements

# Performing Unary operations

print ("\nSum of all array "

       "elements: ", a.sum())

 

# Adding two arrays

# Performing Binary operations

print ("\nArray sum:\n", a + b)
# Splitting

a = np.array([[1, 3, 5, 7, 9, 11], 

              [2, 4, 6, 8, 10, 12]]) 



# horizontal splitting 

print("Splitting along horizontal axis into 2 parts:\n", np.hsplit(a, 2)) 



# vertical splitting 

print("\nSplitting along vertical axis into 2 parts:\n", np.vsplit(a, 2))
# Date and Time

# creating a date 

today = np.datetime64('2017-02-12') 

print("Date is:", today) 

print("Year is:", np.datetime64(today, 'Y')) 

  

# creating array of dates in a month 

dates = np.arange('2017-02', '2017-03', dtype='datetime64[D]') 

print("\nDates of February, 2017:\n", dates) 

print("Today is February:", today in dates) 

  

# arithmetic operation on dates 

dur = np.datetime64('2017-05-22') - np.datetime64('2016-05-22') 

print("\nNo. of days:", dur) 

print("No. of weeks:", np.timedelta64(dur, 'W')) 

  

# sorting dates 

a = np.array(['2017-02-12', '2016-10-13', '2019-05-22'], dtype='datetime64') 

print("\nDates in sorted order:", np.sort(a))
# Linear Algebra 

A = np.array([[6, 1, 1], 

              [4, -2, 5], 

              [2, 8, 7]]) 

  

print("Rank of A:", np.linalg.matrix_rank(A)) 

  

print("\nTrace of A:", np.trace(A)) 

  

print("\nDeterminant of A:", np.linalg.det(A)) 

  

print("\nInverse of A:\n", np.linalg.inv(A)) 

  

print("\nMatrix A raised to power 3:\n", np.linalg.matrix_power(A, 3))
# String Operation by join()

# splitting a string and joining by '-'

print(np.char.join('-', 'kaggle'))
# Line Spacing

# restep set to True 

print("B\n", np.linspace(2.0, 3.0, num=5, retstep=True), "\n") 

  

# To evaluate sin() in long range  

x = np.linspace(0, 2, 10) 

print("A\n", np.sin(x)) 
# Byte swaping   

# a is an array of integers. 

a = np.array([1, 256, 100], dtype = np.int16) 

print(a.byteswap(True)) 
# Numpy Filter

arr = np.array([41, 42, 43, 44])

x = arr[[True, False, True, False]]

print(x)