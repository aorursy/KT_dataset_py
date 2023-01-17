import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import os
data_path = '../input/192954129_T_ONTIME_REPORTING.csv'

data = pd.read_csv(data_path)
data = data.fillna({'DEP_DELAY_NEW' : 0})

#df_dummies_origin = pd.get_dummies(data['ORIGIN'],prefix='ORIGIN',drop_first=True)

# df_dummies_dest = pd.get_dummies(data['DEST'], prefix='DEST', drop_first=True)

features = ["DAY_OF_MONTH", "DAY_OF_WEEK"]

# data = pd.concat([data[features], df_dummies_origin, df_dummies_dest], axis = 1)
data_to_plot = data[["DAY_OF_MONTH", "DEP_DELAY_NEW"]]

# data_to_plot.head()

data_to_plot = data_to_plot.set_index('DAY_OF_MONTH')

# data_to_plot.head()

plt.figure(figsize=(12,6)) # Your code here

plt.title("Delay Duration Based on Day of Month")

sns.lineplot(data=data_to_plot['DEP_DELAY_NEW'])

plt.xlabel("Day of the Month")
data_to_plot = data[["DAY_OF_WEEK", "DEP_DELAY_NEW"]]

data_to_plot = data_to_plot.set_index('DAY_OF_WEEK')

plt.figure(figsize=(12,6)) # Your code here

plt.title("Delay Duration Based on Day of Week")

sns.lineplot(data=data_to_plot['DEP_DELAY_NEW'])

plt.xlabel("Day of the Week")
data_to_plot = data[["DAY_OF_WEEK", "DEP_DELAY_NEW"]]

data_to_plot = data_to_plot.set_index('DEP_DELAY_NEW')

plt.figure(figsize=(12,6)) # Your code here

plt.title("Delay Duration Based on Day of Week")

sns.heatmap(data_to_plot.head(100), annot=True)

plt.xlabel("Day of the Week")
data_to_plot = data[["DAY_OF_MONTH", "DEP_DELAY_NEW"]]

data_to_plot = data_to_plot.set_index('DAY_OF_MONTH')

plt.figure(figsize=(12,6)) # Your code here

plt.title("Delay Duration Based on Day of Month")

sns.lineplot(data=data_to_plot['DEP_DELAY_NEW'])

plt.xlabel("Day of the Month")
y = data.DEP_DELAY_NEW #dependiente

X = data[["DAY_OF_WEEK", "DAY_OF_MONTH"]] #independiente



train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=1)



airline_model = DecisionTreeRegressor(random_state=1)

airline_model.fit(train_X, train_y)



val_predictions = airline_model.predict(test_X)

val_mae = mean_absolute_error(val_predictions, test_y)



print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))
y = data.DEP_DELAY_NEW #dependiente

X = data[["DAY_OF_WEEK", "DAY_OF_MONTH"]] #independiente



train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=1)



airline_model = DecisionTreeRegressor(max_leaf_nodes=20,random_state=1)

airline_model.fit(train_X, train_y)



val_predictions = airline_model.predict(test_X)

val_mae = mean_absolute_error(val_predictions, test_y)



print("Validation MAE when 100 max_leaf_nodes: {:,.0f}".format(val_mae))
airline_model.predict([[1, 5]])