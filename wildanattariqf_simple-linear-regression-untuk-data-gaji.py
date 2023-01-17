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
#import all the lib

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
#read the dataset using pandas

data = pd.read_csv('/kaggle/input/salary-data-simple-linear-regression/Salary_Data.csv')
#this displays the top 5 rows of the data

data.head()
#Provides some information regarding the colums in the data

data.info()
#this describe the basic stat behind the data set used 

data.describe()
#these plots help to explin the values and how they are scattered

plt.figure(figsize=(12,6))

sns.pairplot(data, x_vars=['YearExperience'],y_vars=['Salary'],size=7,kind='scatter')

plt.xlabel('Years')

plt.ylabel('Salary')

plt.title('Salary Prediction')

plt.show()
#cooking the data

x=data['YearsExperience']

x.head()
#cooking the data

y=data['Salary']

y.head()
#import segregating data from scikit learn

from sklearn.model_selection import train_test_split
#split the data for train and test

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7, random_state=0)
#Create New axis for x column

x_train= x_train[:,np.newaxis]

x_test = x_test[:,np.newaxis]
#importing Linear Regression Model From Scikit learn

from sklearn.linear_model import LinearRegression
#fitting the model

lr = LinearRegression()

lr.fit(x_train,y_train)
#predicting the Salary for the Test Values

y_pred= lr.predict(x_test)
#Plotting the actual and predicted values

c = [i for i in range(1,len(y_test)+1,1)]

plt.plot(c,y_test,color= 'r',linestyle= '-')

plt.plot(c,y_test,color= 'b',linestyle= '-')

plt.xlabel('Salary')

plt.ylabel('index')

plt.title('Prediction')

plt.show()
#plotting the error

c =[i for i in range(1,len(y_test)+1,1)]

plt.plot(c,y_test-y_pred,color='green', linestyle='-')

plt.xlabel('index')

plt.ylabel('error')

plt.title('Error Value')

plt.show()
#importing metrics for the evaluation of the model

from sklearn.metrics import r2_score,mean_squared_error
#calculate mean square error

mse= mean_squared_error(y_test,y_pred)
#calculate R square vale

rsq = r2_score(y_test,y_pred)
print('mean squared error :',mse)

print('r square:',rsq)
#just plot actual and predicted values for more insights

plt.figure(figsize=(12,6))

plt.scatter(y_test,y_pred,color='r',linestyle='-')

plt.show()
#intecept and coeff off the line

print('Intercept of the model:',lr.intercept_)

print('Coefficient of the line:',lr.coef_)