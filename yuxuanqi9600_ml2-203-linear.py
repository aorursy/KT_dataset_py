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
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
base_data = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')
sns.pairplot(base_data, y_vars= ['sqft_living'],x_vars = ['price', 'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long', 'sqft_living15', 'sqft_lot15']) 
base_data.drop(base_data[base_data['sqft_living']>10000].index,inplace=True)
base_data.drop(base_data[base_data['bedrooms']>20].index,inplace=True)
base_data.drop(base_data[base_data['grade']<5].index,inplace=True)
base_data.drop(base_data[base_data['sqft_lot']>750000].index,inplace=True)
base_data.drop(base_data[base_data['sqft_lot15']>400000].index,inplace=True)
base_data.drop(base_data[base_data['sqft_living15']>6000].index,inplace=True)
base_data.drop(base_data[base_data['sqft_basement']>4000].index,inplace=True)
base_data[['sqft_living', 'price', 'bedrooms', 'bathrooms', 
       'sqft_lot', 'floors', 'waterfront', 'view', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'sqft_living15', 'sqft_lot15']].corr()
sns.pairplot(base_data, y_vars= ['sqft_living'],x_vars = ['price', 'bedrooms', 'bathrooms', 'floors', 'view', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'sqft_living15']) 
select_volum = base_data[['sqft_living', 'price', 'bedrooms', 'bathrooms', 'floors', 'view', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'sqft_living15']]
x = select_volum[[ 'price', 'bedrooms', 'bathrooms', 'floors', 'view', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'sqft_living15']].values
y = select_volum['sqft_living'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
model = LinearRegression()
output_model=model.fit(x_train, y_train)
y_pred = output_model.predict(x_test)
print('Coefficients: \n', output_model.coef_)
print('MSE: '+ str(mean_squared_error(y_test, y_pred)))
print('R2: '+str(r2_score(y_test, y_pred)))
