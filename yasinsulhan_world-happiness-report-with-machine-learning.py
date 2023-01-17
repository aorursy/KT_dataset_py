# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/world-happiness/2019.csv")
data.head()
data.columns
data.tail()
data.info()
data.drop(["Overall rank","Country or region"],axis=1,inplace=True)
#done
data.head()
x_data = data.iloc[:,1:]
y_data = data.loc[:,["Score"]]
x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data)).values
y = (y_data - np.min(y_data))/(np.max(y_data) - np.min(y_data)).values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=1)
# LinearRegression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)
#Predict Score
print("print accuracy of dt algo:",reg.score(x_test,y_test))
# CV
from sklearn.model_selection import cross_val_score
reg = LinearRegression()
k = 3
cv_result = cross_val_score(reg,x_train,y_train,cv=k) # uses R^2 as score 
print('CV Scores: ',cv_result)
print('CV scores average: ',np.sum(cv_result)/k)
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=7) #n_neighbors=k
knn.fit(x,y)
prediction = knn.predict(x_test) #prediction test ile alakalidir.

print("{} nn score: {}".format(3,knn.score(x,y)))
# grid search cross validation with 1 hyperparameter
from sklearn.model_selection import GridSearchCV
grid = {'n_neighbors': np.arange(1,50)}
knn = KNeighborsRegressor()
knn_cv = GridSearchCV(knn, grid, cv=3) # GridSearchCV
knn_cv.fit(x_train,y_train)# Fit

# Print hyperparameter
print("Tuned hyperparameter k: {}".format(knn_cv.best_params_)) 
print("Best score: {}".format(knn_cv.best_score_))
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train,y_train) #x_train e gore fit ettik

#Predict
print("print accuracy of dt algo:",dt.score(x_test,y_test))
# grid search cross validation with 1 hyperparameter
from sklearn.model_selection import GridSearchCV
grid = {'min_samples_leaf': np.arange(1,25),"criterion":["mse","friedman_mse","mae"]}
det = DecisionTreeRegressor()
det_cv = GridSearchCV(det, grid, cv=3) # GridSearchCV
det_cv.fit(x_train,y_train)# Fit

# Print hyperparameter
print("Tuned hyperparameter k: {}".format(det_cv.best_params_)) 
print("Best score: {}".format(det_cv.best_score_))
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(x_train,y_train)
print("print accuracy of rf algo:",rf.score(x_test,y_test))
from sklearn.linear_model import SGDRegressor 

svm = SGDRegressor()
svm.fit(x_train,y_train)

#test
print("print accuracy of svm algo:",svm.score(x_test,y_test))
# grid search cross validation with 1 hyperparameter
from sklearn.model_selection import GridSearchCV
grid = {'learning_rate':["constant","optimal","invscaling","adaptive"]}
svm = SGDRegressor()
svm_cv = GridSearchCV(svm, grid, cv=3) # GridSearchCV
svm_cv.fit(x_train,y_train)# Fit

# Print hyperparameter
print("Tuned hyperparameter k: {}".format(svm_cv.best_params_)) 
print("Best score: {}".format(svm_cv.best_score_))
from sklearn.linear_model import Ridge
ridge = Ridge(alpha = 0.1, normalize = True)
ridge.fit(x_train,y_train)
print('Ridge score: ',ridge.score(x_test,y_test))