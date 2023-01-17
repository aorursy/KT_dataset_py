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

import pandas as pd

melbourne_file_path="../input/melb_data.csv"

melbourne_data=pd.read_csv(melbourne_file_path)

melbourne_data.describe()
melbourne_data.columns

#deleting missing values
melbourne_data=melbourne_data.dropna(axis=0)

melbourne_data.columns

melbourne_data.describe()

#Setting features
y=melbourne_data.Price

melbourne_features=['Rooms','Bathroom','Landsize','Lattitude','Longtitude']

melbourne_data.columns

x=melbourne_data[melbourne_features]

x.describe()

x.head()
#Building Model
from sklearn.tree import DecisionTreeRegressor
#define
melbourne_model=DecisionTreeRegressor(random_state=1)
#fit
melbourne_model.fit(x,y)
#predict
print("The predictions are")
print(melbourne_model.predict(x.head()))
#Wihtout Model Validation(in-sample score)
from sklearn.metrics import mean_absolute_error
predicted_home_prices=melbourne_model.predict(x)
mean_absolute_error(y,predicted_home_prices)

#With Model Validation(out-of-sample)
from sklearn.model_selection import train_test_split
train_x,val_x,train_y,val_y=train_test_split(x,y,random_state=0)
val_predictions=melbourne_model.predict(val_x)
print(mean_absolute_error(val_y,val_predictions))
