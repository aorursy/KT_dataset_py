# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Importing data

train = pd.read_csv('/kaggle/input/random-linear-regression/train.csv')

test = pd.read_csv('/kaggle/input/random-linear-regression/test.csv')



# dropping NaN rows

test = test.dropna()

train = train.dropna()



# Train set of X and y

X_train = np.array(train.iloc[:, :-1].values)

y_train = np.array(train.iloc[:, 1:2].values)



# Test set of X and y

X_test = np.array(test.iloc[:, :-1].values)

y_test = np.array(test.iloc[:, 1:2].values)

# Performing LinearRegression

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)



y_pred = regressor.predict(X_test)
# Plotting the training points and the best fit line

plt.scatter(X_train, y_train, color='red')

plt.plot(X_train, regressor.predict(X_train), color='blue')

plt.title("X_train vs y_train")

plt.xlabel("X_train")

plt.ylabel("y_train")

plt.show()
# Plotting the test points and also the best fit line

plt.scatter(X_test, y_test, color='red')

plt.plot(X_train, regressor.predict(X_train), color='blue')

plt.title("X_test vs y_test")

plt.xlabel("X_test")

plt.ylabel("y_test")

plt.show()
# Accuracy of the prediction

accuracy = regressor.score(X_test, y_test)

print(accuracy)