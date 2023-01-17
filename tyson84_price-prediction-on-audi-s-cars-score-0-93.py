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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



audi = pd.read_csv('/kaggle/input/used-car-dataset-ford-and-mercedes/audi.csv')

audi.head()
audi.info()
audi.describe()
fig = plt.figure(figsize=(20,20))

sns.violinplot(x='model',y='price',data=audi,scale='width')
audi_one = pd.get_dummies(audi,columns=['model', 'transmission','fuelType'])
audi_one.head()
audi_one.corrwith(audi_one['price'])
audi_nofueltype = audi_one.drop(['fuelType_Diesel','fuelType_Hybrid','fuelType_Petrol'], axis=1)
from sklearn.model_selection import train_test_split

X = audi_nofueltype.drop(['price'],axis=1)

y = audi_nofueltype['price']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train,y_train)

lr.score(X_test,y_test)
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(max_depth=2,criterion='mse')

gbr.fit(X_train,y_train)

gbr.score(X_test,y_test)
from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor()

dtr.fit(X_train,y_train)

dtr.score(X_test,y_test)
results = X_test.copy()

results['predicted_lr'] = lr.predict(X_test)

results['predicted_gbr'] = gbr.predict(X_test)

results['predicted_dtr'] = dtr.predict(X_test)

results['Real']= y_test

results = results[['predicted_lr', 'predicted_gbr', 'predicted_dtr', 'Real']]

results
sns.pairplot(results,x_vars=['predicted_lr', 'predicted_gbr','predicted_dtr'],y_vars=['Real'],kind='reg',height=7,aspect=1)