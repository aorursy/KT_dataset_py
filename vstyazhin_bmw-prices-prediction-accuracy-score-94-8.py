# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/used-car-dataset-ford-and-mercedes/bmw.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
bmw = pd.read_csv(r'/kaggle/input/used-car-dataset-ford-and-mercedes/bmw.csv')
bmw.head()
bmw.tail()
bmw.describe()
bmw.query('engineSize == 0').head()
bmw.isnull().sum()
object_cols = [col for col in bmw.columns if bmw[col].dtype == "object"]
object_cols
bmw[object_cols].nunique()
y=bmw.price
X_=bmw.drop('price',axis=1)
X=pd.get_dummies(X_)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.33, random_state=42)
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score
my_model = XGBRegressor(n_estimators=500, learning_rate=0.1, n_jobs=-1)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_test, y_test)], 
             verbose=False)
y_pred = my_model.predict(X_test)
my_model.score(X_test,y_test)
prices = pd.DataFrame(data={'Actual':y_test,'Predicted':y_pred}).round(0).astype(int)
prices
