import numpy as np

import seaborn as sns

import pandas as pd

import matplotlib.pyplot as plt
df = pd.read_csv("../input/starup/50_Startups.csv")
df.head()
df.info
df.shape
df.isna().sum() 
df.corr()
corr = df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values);
sns.set(rc={'figure.figsize':(10,7)}) # oluşacak grafiklerin uzunluğunu ve genişliğini belirliyorum.



sns.scatterplot(df["R&D Spend"], df["Profit"]);  # maaş ve tecrübe değerlerinin dağılımını görüntüleyelim.
plt.hist(df["R&D Spend"])

plt.hist(df["Administration"])

plt.hist(df["Marketing Spend"])

plt.hist(df["Profit"])
df.describe().T
df["State"].unique()
pd.get_dummies(df['State']).mean()

y = df['Profit']

X = df.drop(['Profit'], axis=1)

X = X["Marketing Spend"]
X
y
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size = 0.25, 

                                                    random_state = 42)
X_train
X_test
y_train
y_test
from sklearn.linear_model import LinearRegression

linear_regresyon = LinearRegression()
linear_regresyon.fit(X, y)
y_pred = linear_regresyon.predict(X_test)

df2 = pd.DataFrame({'Gercek': y_test, 'Tahmini': y_pred})

df2
from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error



MAE = mean_absolute_error(Y_test, Y_pred)

MSE = mean_squared_error(Y_test, Y_pred)

RMSE = math.sqrt(MSE)

MAE

MSE

RMSE