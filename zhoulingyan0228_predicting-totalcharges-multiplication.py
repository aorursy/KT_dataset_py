import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import sklearn.metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import math

data = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")
data.head()
data = data.loc[data['TotalCharges'] != ' ']
data['TotalCharges'] = data['TotalCharges'].apply(float)
y = data['TotalCharges'].values
X = data[['tenure', 'MonthlyCharges']].values

y_pred = X[:,0] * X[:,1]

rmse = math.sqrt(sklearn.metrics.mean_squared_error(y, y_pred))
r2 = sklearn.metrics.r2_score(y, y_pred)
print('rmse=', rmse, ', r2=', r2)
plt.scatter(y, y_pred);
X_extended = np.append(X, (X[:,0]*X[:,1]).reshape(-1,1), axis=1)
X_extended = np.append(X_extended, (X[:,0]*X[:,0]).reshape(-1,1), axis=1)
X_extended = np.append(X_extended, (X[:,1]*X[:,1]).reshape(-1,1), axis=1)
kfold = KFold(5)

for train_index, test_index in kfold.split(X_extended):
    X_ext_train, X_ext_test, y_train, y_test = X_extended[train_index], X_extended[test_index], y[train_index], y[test_index]
    lr = LinearRegression()

    lr.fit(X_ext_train, y_train)
    y_pred = lr.predict(X_ext_test)

    rmse = math.sqrt(sklearn.metrics.mean_squared_error(y_test, y_pred))
    r2 = sklearn.metrics.r2_score(y_test, y_pred)
    print('rmse=', rmse, ', r2=', r2)