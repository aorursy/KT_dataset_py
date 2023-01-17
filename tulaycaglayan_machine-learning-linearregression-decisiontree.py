# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score #  residual square score .

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/column_2C_weka.csv")
data.info()
data.head()
print(data['class'].unique())
data.groupby(['class']).count()

(data.iloc[:,0:5].values)
data.corr(method='pearson')

pd.plotting.scatter_matrix(data.loc[:, data.columns != 'class'],
                                       c=['red' if i=='Abnormal' else 'green' for i in data.loc[:,'class']],
                                       figsize= [15,15],
                                       diagonal='hist',
                                       alpha=0.5,
                                       s = 200,
                                       marker = '.',
                                       edgecolor= "black")
plt.show()
x = data.loc[:,'pelvic_incidence'].values.reshape(-1,1)
y = data.loc[:,'sacral_slope'].values.reshape(-1,1)
# LinearRegression
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(x,y)

# Predict data 
x_pred = np.arange(min(x), max(x), 0.1).reshape(-1,1)
y_pred = linear_reg.predict (x_pred)

plt.figure(figsize=[8,5])
plt.scatter(x=x,y=y, color ='blue')
plt.plot(x_pred,y_pred, color='red')
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
plt.show()


# predict success
# R^2 
y_head = linear_reg.predict(x)
print("Linear Regression R^2 score: ",r2_score(y, y_head) )

# R^2 
print('Linear Regression R^2 score: ',linear_reg.score(x, y))
# TreeRegression
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(x,y)

# Predict data 
x_pred = np.arange(min(x), max(x), 0.1).reshape(-1,1)
y_pred = tree_reg.predict (x_pred)

plt.figure(figsize=[8,5])
plt.scatter(x=x,y=y, color ='blue')
plt.plot(x_pred,y_pred, color='red')
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
plt.show()
# predict success
# R^2 
print("Decision tree R^2 score: ",r2_score(y, tree_reg.predict(x)) )

# R^2 
print('Decision tree R^2 score: ',tree_reg.score(x, y))
# RandomForestRegression
from sklearn.ensemble import RandomForestRegressor
random_forest_reg = RandomForestRegressor()
random_forest_reg.fit(x,y)

# Predict data 
x_pred = np.arange(min(x), max(x), 0.1).reshape(-1,1)
y_pred = random_forest_reg.predict (x_pred)

plt.figure(figsize=[8,5])
plt.scatter(x=x,y=y, color ='blue')
plt.plot(x_pred,y_pred, color='red')
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
plt.show()
# predict success
# R^2 
print("Random Forest R^2 score: ",r2_score(y, random_forest_reg.predict(x)) )

# R^2 
print('Random Forest R^2 score: ',random_forest_reg.score(x, y))
