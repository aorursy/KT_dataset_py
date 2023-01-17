#importing all the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns



%matplotlib inline



#importing dataset using pandas

dataset = pd.read_csv('../input/kc_house_data.csv')

#to see what my dataset is comprised of

dataset.head()
#data cleaning



dataset.sample(5)
dataset.isnull().sum()
#breaking dataset into independent variable space, and dependent variable price.



space=dataset.iloc[:,5].values

space=space.reshape(-1,1)

price=dataset.iloc[:,2].values



#creating training set and test set

from sklearn.model_selection import train_test_split

space_train,space_test,price_train, price_test= train_test_split(space,price,test_size=1/3,random_state=0)



#creating Linear Regression model

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

regressor.fit(space_train,price_train)



price_pred=regressor.predict(space_test)



#visualizing training set results

import matplotlib.pyplot as plt

plt.scatter(space_train,price_train,color='red')

plt.plot(space_train,regressor.predict(space_train),color='blue')

plt.xlabel('space in sqft')

plt.ylabel('price')

plt.show()



#visualizing test set results

import matplotlib.pyplot as plt

plt.scatter(space_test,price_test,color='red')

plt.plot(space_test,regressor.predict(space_test),color='blue')

plt.xlabel('space in sqft')

plt.ylabel('price')

plt.show()