# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

dataframe = pd.read_csv("../input/kc_house_data.csv")

# Any results you write to the current directory are saved as output.

dataframe.head(5)

#find out the correlation matrix .features reduction is a important part

#finding the correlation with price.

dataframe.corr()['price']



#from the correlation matrix we can identify sqft_living has more impact

space=dataframe['sqft_living']

price=dataframe['price']

x = np.array(space).reshape(-1, 1)

y = np.array(price)

#plotting the diagram.

plt.scatter(x,y)

plt.title("House price prediction")

plt.xlabel("sqft_living")

plt.ylabel("price")

plt.show()

#spliting the data and train the data using scikit  learn

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest =train_test_split(x,y,test_size=2/3,random_state=0)

#training the model.

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

regressor.fit(xtrain,ytrain)

# Predict Test set

y_pred = regressor.predict(xtest)

# Visualise Training set result

plt.scatter(xtrain, ytrain)

plt.plot(xtrain, regressor.predict(xtrain),color="red")

plt.title('House price prediction')

plt.xlabel('sqft_living')

plt.ylabel('Price')

plt.show()
