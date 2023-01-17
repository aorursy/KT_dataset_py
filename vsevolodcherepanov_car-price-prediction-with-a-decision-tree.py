import numpy as np 

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_excel('/kaggle/input/car-sales/CarSales.xlsx')

missing = df.isnull().sum()

print('missing data:',missing[missing > 0])

print('shape:', df.shape)
from sklearn.preprocessing import LabelEncoder

df = df.applymap(str) # consider all the data as strings, I am not sure that this approach is appropiate but then I can skip imputing 



s = (df.dtypes == 'object')

object_cols = list(s[s].index)

label= df.copy()

label_encoder = LabelEncoder()

for col in object_cols:

    label[col] = label_encoder.fit_transform(df[col])   



missing = label.isnull().sum()

print('missing values:',missing[missing > 0])

print('the head:\n',label.head())
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



dftrain, dftest = train_test_split(label, test_size=0.2) # randomly split df to train/test data with 80/20 proportion



#Let's check if prices distribution is same in test and train data

plt.figure(figsize=(10,5))

plt.subplot(1,2,1), dftrain['Price'].hist(bins=10)



plt.figure(figsize=(10,5))

plt.subplot(1,2,2),dftest['Price'].hist(bins=10)

from sklearn.tree import DecisionTreeRegressor

features = list(label.columns)

features.remove('Price') 

train_y = dftrain['Price'] 

train_x = dftrain[features] 

test_x=dftest[features] 

test_y=dftest['Price'] 

model = DecisionTreeRegressor(random_state=1)

model.fit(train_x, train_y)

predicted_prices = model.predict(test_x)

predicted_prices=pd.DataFrame(predicted_prices)
#Calculate mae 

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(predicted_prices, test_y)

print('mean absolute error: ',mae)

print('mean price in df:', label['Price'].mean(), '\nmae/mean, %:',100*mae/label['Price'].mean())