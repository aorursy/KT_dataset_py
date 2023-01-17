# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/linear-regression-dataset.csv'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importing the dataset



data = pd.read_csv('../input/linearregressiondataset3/linear-regression-dataset.csv')
#printing the top 5 columns of dataset

data.head()
#selecting the respective columns for depenedent and independent variables

# x--> Independent

# y--> Dependent



x=data.iloc[:,:-1].values

y=data.iloc[:,-1].values

# Remove # from the line below to print the values of x

#x 
y

# Remove # from the line below to print the values of y

#y
#Splitting the model into training and testing set using SkLearn



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7,random_state=0)
#Importing LinearRegression Class from sklearn

#Creating a regression object from the LinearRegression Class



from sklearn.linear_model import LinearRegression

regression = LinearRegression()
#prediction Stage

regression.fit(x_train,y_train)

y_pred = regression.predict(x_test)
import matplotlib.pyplot as plt

plt.scatter(x_train,y_train,color='blue')

plt.plot(x_train,regression.predict(x_train),color='orange')

plt.title('Training Model')

plt.xlabel('Deneyim')

plt.ylabel('Mass')

plt.show()
plt.scatter(x_test,y_test,color='green')

plt.plot(x_train,regression.predict(x_train),color='red')

plt.title('Testing Model')

plt.xlabel('Deneyim')

plt.ylabel('Mass')

plt.show()
#Optional Step

plt.scatter(x_test,y_test,color='blue',label='Train')

plt.scatter(x_train,y_train,color='red',label='Test')

plt.plot(x_train,regression.predict(x_train),color='black',label='Best Fit Line')

plt.xlabel('Deneyim')

plt.ylabel('Mass')

plt.legend()

plt.show()
from sklearn.metrics import r2_score

r2_score(y_test,y_pred)