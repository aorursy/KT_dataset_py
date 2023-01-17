import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")

df.head(10) 
df = df.drop(['Province/State', 'Country/Region', 'Lat', 'Long'], axis=1).T

df.head(10)
df['y']= df.sum(axis=1)

df
df.shape
len(df)
# df.insert(0, 'X', range(1, len(df)+1))

df['X'] = range(1, len(df)+1)

df
df = df[['X', 'y']]
df
from sklearn.model_selection import train_test_split

X = df.X.values.reshape(-1, 1)

y = df.y.values.reshape(-1, 1)

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

sc_y = StandardScaler()

X = sc_X.fit_transform(X)

y = sc_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=42)
from sklearn.svm import SVR

SupportVectorRegModel = SVR()

SupportVectorRegModel.fit(X_train, y_train)
y_pred = SupportVectorRegModel.predict(X_test)

y_pred
from sklearn.metrics import r2_score,mean_squared_error

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

rmse
plt.scatter(X, y, color = 'magenta')

plt.plot(X, SupportVectorRegModel.predict(X), color = 'green')

plt.title('Support Vector Regression Model')

plt.xlabel('Day')

plt.ylabel('Cases')

plt.show()