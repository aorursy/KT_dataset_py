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
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.svm import LinearSVR
df = pd.read_csv('../input/Admission_Predict.csv', delimiter=',')
X = df.iloc[:,:8].values
y = df.iloc[:, 8].values
X_train, X_test, y_train, y_test = train_test_split(X, y)
lr = LinearRegression()
lr.fit(X_train, y_train)
lr.predict(X_test)
lr.score(X_test, y_test)
lr.coef_
lr.intercept_
standardScaler = StandardScaler()
standardScaler.fit(X_train)
X_train_standard = standardScaler.transform(X_train)
X_test_standard = standardScaler.transform(X_test)
sgd_reg = SGDRegressor(n_iter = 50)
sgd_reg.fit(X_train_standard, y_train)
sgd_reg.predict(X_test_standard)
sgd_reg.coef_
sgd_reg.intercept_
sgd_reg.score(X_test_standard, y_test)
svr = LinearSVR()
svr.fit(X_train_standard, y_train)
svr.predict(X_test_standard)
svr.score(X_test_standard, y_test)
svr.coef_
svr.intercept_