# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
home_data_path = '../input/train.csv';
test_data_path = '../input/test.csv';
sample = pd.read_csv('../input/sample_submission.csv')
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
sample
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

home_data = pd.read_csv(home_data_path)
test_data = pd.read_csv(test_data_path)

#Create target object
train_y = home_data.SalePrice
#create X
features = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
train_X = home_data[features]
test_X = test_data[features]
home_model = RandomForestRegressor(random_state=1)
#fit the data
home_model.fit(train_X,train_y)
#predict the data
test_predict = home_model.predict(test_X)
print(test_predict)

output= pd.DataFrame({'ID': test_data.Id, "SalePrice": test_predict})
output.to_csv('submission.csv', index=False)


