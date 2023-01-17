# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score, mean_squared_error
df = pd.read_csv("/kaggle/input/graduate-admissions/Admission_Predict.csv")
df.head()
df.shape
df.describe()
df.info()
df.Research.value_counts().plot(kind = "bar")

plt.show()
plt.figure(figsize=(8,8))

sns.heatmap(abs(df.corr()), annot = True)

plt.show()
x = df.iloc[:,0:7]

y = df.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)
x_train.shape, x_test.shape, y_train.shape, y_test.shape
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
y_pred = pd.Series(y_pred)
y_test = y_test.reset_index(drop=True)
plt.figure(figsize=(8,6))

y_pred.plot(label = "Predicted")

y_test.reset_index(drop = True).plot(label = "Original")

plt.legend()

plt.show()
mse = mean_squared_error(y_test, y_pred)
mse ** 0.5
r2 = r2_score(y_test, y_pred)
r2
sns.regplot(y_test.reset_index(drop=True), y_pred)

plt.show()