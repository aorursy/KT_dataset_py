# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/kc_house_data.csv')



df
import seaborn as sns



df.isna().any()



years = list(map(lambda x: x[:4], df['date']))



df['date'] = list(map(int, years))
df['age']  = df['date'] - df['yr_built']



basement_present = [0 if i == 0 else 1 for i in df['sqft_basement']]

rennovated = [0 if i == 0 else 1 for  i in df['yr_renovated']]



df['base'] = basement_present

df['rennovated'] = rennovated
corr_with_price = df.corr()['price']
from sklearn.linear_model import LinearRegression as lr

from sklearn.model_selection import train_test_split as tts



y = df['price']



corr_with_price.drop('price', inplace = True)
corr_with_price = corr_with_price.sort_values(ascending = False)



corr_with_price
X = list(corr_with_price[:].index) #+ ['age'] + ['base'] + ['rennovated']

X

#X.remove('sqft_basement')
from sklearn import metrics



x = df[X]



x_train, x_test, y_train, y_test = tts(x,y)



regressor = lr()



regressor.fit(x_train, y_train)



predicted_values = regressor.predict(x_test)



print (metrics.mean_squared_error(y_test, predicted_values)**(1.0/2.0))

print (metrics.r2_score(y_test, predicted_values))



test = regressor.predict(x_train)



print (metrics.mean_squared_error(y_train, test)**(1.0/2.0))