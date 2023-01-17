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

# Path of the file to read
iowa_file_path = '../input/train.csv'

# Fill in the line below to read the file into a variable home_data
home_data = pd.read_csv(iowa_file_path)


# Print summary statistics in next line
home_data.describe()
# What is the average lot size (rounded to nearest integer)?
avg_lot_size = home_data['LotArea'].mean()

import datetime
now = datetime.datetime.now()
# As of today, how old is the newest home (current year - the date in which it was built)
newest_home_age = now.year-home_data['YrSold'].min()

print(avg_lot_size)
print(newest_home_age)
#Selecionar a coluna que queremos prever
y = home_data['SalePrice']
#Choosing "Features"
home_features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath','BedroomAbvGr','TotRmsAbvGrd']
X = home_data[home_features]
X.describe()
# Import the train_test_split function
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)

#Building Model
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# Define model. Specify a number for random_state to ensure same results each run
iowa_model = DecisionTreeRegressor(random_state=1)

# Fit model
iowa_model.fit(X, y)
val_predictions = iowa_model.predict(X)
print('In sample in the training         %8.2f'% mean_absolute_error(y, val_predictions))



# Define model
iowa_model = DecisionTreeRegressor()
# Fit model
iowa_model.fit(train_X, train_y)

from sklearn.metrics import mean_absolute_error
# get predicted prices on validation data
val_predictions = iowa_model.predict(val_X)
print('Segregate training and validation %8.2f'% mean_absolute_error(val_y, val_predictions))



#Check the best number of nodes
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X,val_X,train_y,val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,random_state=0)
    model.fit(train_X,train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y,preds_val)
    return(mae)

# Compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5,25,50,100,250,500]:
    mae = get_mae(max_leaf_nodes, train_X,val_X,train_y,val_y)
    print('Max leaf nodes: %d \t\t Mean Absolute Error: %d'%(max_leaf_nodes,mae))
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
best_tree_size = min(scores, key=scores.get)
best_tree_size
#Compare DecisionTrees with RandonForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_mae = rf_model.predict(val_X)
print('Randon Forest MAE %d'%mean_absolute_error(val_y, rf_val_mae))


model = DecisionTreeRegressor(random_state=1)
model.fit(train_X,train_y)
preds_val = model.predict(val_X)
mae = mean_absolute_error(val_y,preds_val)
print('Decision Tree MAE %d'%mae)


#XGBoost
data = pd.read_csv('../input/train.csv')
#Drop row that has NA in the SalePrice column
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = data.SalePrice

#Drop SalePrice Column, exclude not number columns
X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)

from sklearn.preprocessing import Imputer
my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)
from xgboost import XGBRegressor

my_model = XGBRegressor()
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, verbose=False)
# make predictions
predictions = my_model.predict(test_X)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))

# n_estimators specifies how many times to go through the modeling cycle described above.


# early_stopping_rounds offers a way to automatically find the ideal value. 
# Early stopping causes the model to stop iterating when the validation score stops improving, even if we aren't at the hard stop for n_estimators. 
# It's smart to set a high value for n_estimators and then use early_stopping_rounds to find the optimal time to stop iterating.

# ince random chance sometimes causes a single round where validation scores don't improve, you need to specify a number for how many rounds of 
# straight deterioration to allow before stopping
my_model = XGBRegressor(n_estimators=1000)
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=False)
# make predictions
predictions = my_model.predict(test_X)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))

# learning_rate

# Instead of getting predictions by simply adding up the predictions from each component model, we will multiply the predictions from 
# each model by a small number before adding them in. This means each tree we add to the ensemble helps us less. In practice, this reduces 
# the model's propensity to overfit.

# So, you can use a higher value of n_estimators without overfitting. If you use early stopping, the appropriate number of trees will be set automatically.

# In general, a small learning rate (and large number of estimators) will yield more accurate XGBoost models, though it will also take the model longer 
# to train since it does more iterations through the cycle.

# Modifying the example above to include a learing rate would yield the following code:
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=False)
# make predictions
predictions = my_model.predict(test_X)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
# n_jobs
# On larger datasets where runtime is a consideration, you can use parallelism to build your models faster. 
# It's common to set the parameter n_jobs equal to the number of cores on your machine. On smaller datasets, this won't help.

# The resulting model won't be any better, so micro-optimizing for fitting time is typically nothing but a distraction. 
# But, it's useful in large datasets where you would otherwise spend a long time waiting during the fit command.




