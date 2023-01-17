import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectFromModel

from sklearn.preprocessing import StandardScaler, scale

from sklearn.linear_model import LinearRegression, Lasso, RANSACRegressor

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
dataset = pd.read_csv("../input/real-estate-price-prediction/Real estate.csv")
dataset.head()
dataset.isnull().sum()
x = dataset.iloc[:, :-1] #Independent Features

y = dataset.iloc[:, -1] #Dependent/Target Features
lasso = Lasso(random_state=7).fit(x,y)

model = SelectFromModel(lasso, prefit=True)

x_new = model.transform(x)

selected_feat = pd.DataFrame(model.inverse_transform(x_new), columns=x.columns, index=x.index)

selected_feat.head()
selected_col = selected_feat.columns[selected_feat.var() != 0]

X = dataset[selected_col]

X.head()
plt.figure(figsize=(10,8))

plt.subplot(221, title="X2 House age")

sns.distplot(X['X2 house age'])



plt.subplot(222, title="Distance Nearest MRT Station", facecolor = 'y')

sns.distplot(X['X3 distance to the nearest MRT station'])



plt.subplot(223, title="Convenience Stores", facecolor = 'w')

sns.distplot(X['X4 number of convenience stores'])



plt.subplot(224, title="Serial Number", facecolor = 'y')

sns.distplot(X['No'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 0)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
def reg_score(y_test, y_pred):

    print(f'RMSE Score \t - \t{np.sqrt(mean_squared_error(y_test, y_pred))}')

    print(f'R2 Score \t - \t{r2_score(y_test, y_pred)}')

    print(f'MAE Score \t - \t{mean_absolute_error(y_test, y_pred)}')
reg = LinearRegression() 

reg.fit(X_train, y_train)

reg.score(X_test, y_test)
y_predLN = reg.predict(X_test)

reg_score(y_test, y_predLN)
rfm = RandomForestRegressor()

rfm.fit(X_train, y_train)

rfm.score(X_test, y_test)
y_predRF = rfm.predict(X_test)

reg_score(y_test, y_predRF)
gbm = GradientBoostingRegressor()

gbm.fit(X_train, y_train)

gbm.score(X_test, y_test)
y_predGB = gbm.predict(X_test)

reg_score(y_test, y_predGB)
rnr = RANSACRegressor()

rnr.fit(X_train, y_train)

rnr.score(X_test, y_test)
y_predRN = rnr.predict(X_test)

reg_score(y_test, y_predRN)
plt.figure(figsize=(10,5))

plt.scatter(y_test, y_predLN)

plt.xlabel('Target', size=20)

plt.ylabel('Prediction', size=20)

plt.show()
comparedDF = pd.DataFrame(y_test)

comparedDF.head()
comparedDF['Prediction'] = y_predRN #Using the RANSAC Predicted Model

comparedDF.reset_index(drop=True, inplace=True) #Reset the index to count sequentially

comparedDF['Prediction'] = comparedDF['Prediction'].apply(lambda x: f'{x:.1f}') #Change the Prediction feature to 1dp
comparedDF.head(10)