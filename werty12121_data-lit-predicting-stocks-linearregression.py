# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
all_data=pd.read_csv("../input/GAFA Stock Prices.csv")

all_data.head()
all_data.info()
all_data.Date=pd.to_datetime(all_data.Date)

all_data.head()
all_data=all_data[all_data.Stock=="Facebook"]

all_data=all_data.drop(['Stock','High','Low','Adj Close','Volume','Open'],axis=1)

all_data.head()

all_data.isnull().sum()
all_data['id']=all_data.index

all_data.head()
f, ax = plt.subplots(figsize=(20, 20))

sns.lineplot(x="Date", y="Close", data=all_data,label="Close")
Y=all_data.Close

X=all_data[['id']]

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,shuffle=True)

model = LinearRegression()

model.fit(X_train,y_train)

mean_squared_error(model.predict(X_test),y_test)
f, ax = plt.subplots(figsize=(20, 20))

plt.scatter(X.id, Y, color='blue', label= 'Actual Price') #plotting the initial datapoints

plt.plot(X.id, model.predict(X), color='red', linewidth=3, label = 'Predicted Price') #plotting the line made by linear regression

plt.title('Linear Regression | Time vs. Price')

plt.legend()

plt.xlabel('Date Integer')

plt.show()