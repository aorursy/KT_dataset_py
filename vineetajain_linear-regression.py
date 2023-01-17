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
import numpy as np 

import matplotlib.pyplot as plt 

x = np.array([1,2,4,3,5])

y = np.array([1,3,3,2,5])

y_predict = []

x_mean = np.mean(x)

y_mean = np.mean(y)

n = np.size(x)

multiply = []

for i in range(5):

    multiply.append((x[i]-x_mean)*(y[i]-y_mean))

    



square = []

for i in range(5):

    square.append((x[i]-x_mean) * (x[i]-x_mean))

#print(square)

new_slope = np.sum(multiply)/np.sum(square)

print(new_slope)

new_c = y_mean - (x_mean*new_slope)

print(new_c)

for i in range(5):

    y_predict.append(new_slope*x[i] + new_c)

    

print(y_predict)



inter_y = []

for i in range(5):

    inter_y.append((y_predict[i]-y[i])*(y_predict[i]-y[i]))

    

print(inter_y)

print(np.sum(inter_y))

print((np.sum(inter_y))/(np.size(inter_y)))

print(np.sqrt((np.sum(inter_y))/(np.size(inter_y))))