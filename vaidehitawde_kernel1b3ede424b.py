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
df = pd.read_csv("../input/headbrain/headbrain.csv")
df.head
df.columns
df.shape
df.isnull().sum()
df.corr()
df.columns
X = df.drop('Brain Weight(grams)',axis = 1 )
y = df['Brain Weight(grams)']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
model.coef_
X_train.columns
model.intercept_
pred = model.predict(X_test)
print(pred)
from sklearn.metrics import mean_absolute_error, mean_squared_error
mean_absolute_error(y_test, pred)
#root mean sqr error
np.sqrt(mean_squared_error(y_test, pred))
#R2 score (outdated)
model.score(X_train, y_train)
print("Mean Absolute Percentage Error = ", mape)
accuracy = 1- mape
print("accuracy = ",accuracy)