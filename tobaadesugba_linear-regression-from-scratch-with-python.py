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
#importing other necessary libraries

%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (20.0, 10.0) #for changing the figure size parameter of the plot



#reading the data

data = pd.read_csv('../input/headbrain/headbrain.csv')

print(data.shape) #print the dimensions of the data matrix (height, width)

data.head() #show the first 5 values of the data
#collecting X(independent var) and Y(dependent variable)

X = data['Head Size(cm^3)'].values

Y = data['Brain Weight(grams)'].values
#finding mean ȳ and mean x̅

mean_x = np.mean(X)

mean_y = np.mean(Y)



n = len(X) #total number of values in x; used in the for loop



#using the formular, let's find m

numer = 0

denum = 0



for i in range(n):

    numer = (X[i] - mean_x) * (Y[i] - mean_y) + numer

    denum = ((X[i] - mean_x) ** 2) + denum

m = numer / denum

print (m)
#finding c

c = mean_y - (m * mean_x)

print("c is equal to", c, "and m is equal to", m)
#plotting values and regression line

#setting the limits of the x and y axis of the plot, the 100 is added to make all the points visible

max_x = np.max(X) + 100 

min_x = np.min(X) - 100



#make a linear space of 1000 evenly spaced numbers from min_x to max_x

x = np.linspace(min_x, max_x, 1000) #this is used to plot the x axis of the regression line

y = c + (m * x) #this is used to plot the y axis of the regression line
#plotting Scatter points

plt.scatter(X, Y, color='#ef4423', label='Scatter points')

#plotting line

plt.plot(x, y, color='#52b920', label='Regression line')

plt.xlabel('Head size in cm3')

plt.ylabel('Brain Weight in grams')

plt.legend()

plt.show()
ss_denum = 0

ss_numer = 0

for i in range(n): #n is the len of X from above

    y_pred = (m * X[i]) + c

    ss_numer += (Y[i] - y_pred) ** 2

    ss_denum += (Y[i] - mean_y) ** 2

r_squared = 1 - (ss_numer / ss_denum)

print(r_squared * 100, "percent")
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error



X = X.reshape(n, 1) #we cannot use rank 1 matrices in sklearn so we reshape



#model definition

brain_model = LinearRegression()



#model fitting

brain_model.fit(X, Y)



#model prediction

pred = brain_model.predict(X)



#model evaluation

r_squared_score = brain_model.score(X, Y)



print((r_squared_score * 100), "percent")