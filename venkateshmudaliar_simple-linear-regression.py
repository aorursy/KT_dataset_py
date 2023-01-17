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
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
dataset = pd.read_csv('/kaggle/input/random-linear-regression/train.csv')
dataset[dataset['x'].isnull()]

dataset[dataset['y'].isnull()]
dataset["y"].fillna(dataset["y"].mean(), inplace = True) 

dataset[dataset['x'].isnull()]

dataset[dataset['y'].isnull()]
dataset.nlargest(10,'x')
dataset = dataset.drop([213])

dataset.nlargest(10,'x')
X = dataset.iloc[:, :-1].values

y = dataset.iloc[:,-1:].values



print(type(X))

print (X.shape)

#print(X)
print(type(y))

print (y.shape)

#print(y)
print(X[0:5])

print(y[0:5])
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()## Train Simple Regression on Training Dataset

regressor.fit(X, y)
#plotting data points for dataset

plt.scatter(X, y, color = 'red')



#plotting prediction line on Training dataset

plt.plot(X, regressor.predict(X), color = 'blue')



plt.title('X vs y (Training set)')

plt.xlabel('X')

plt.ylabel('y')

plt.show()
test_dataset = pd.read_csv('/kaggle/input/random-linear-regression/test.csv')
test_dataset
test_dataset[test_dataset['x'].isnull()]

test_dataset[test_dataset['y'].isnull()]
X_test = test_dataset.iloc[:, :-1].values

y_test = test_dataset.iloc[:,-1:].values

print(type(X_test))

print (X_test.shape)
print(type(y_test))

print (y_test.shape)
y_pred = regressor.predict(X_test)



test_df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

test_df
plt.scatter(X_test, y_test, color = 'red')

plt.plot(X, regressor.predict(X), color = 'blue')

plt.title('X vs y (Test set)')

plt.xlabel('X')

plt.ylabel('y')

plt.show()
from sklearn import metrics



print('Mean Absolute Error:\t\t', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:\t\t', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:\t', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))



accuracy = regressor.score(X_test,y_test)

print("Accuracy:\t\t",accuracy*100,'%')