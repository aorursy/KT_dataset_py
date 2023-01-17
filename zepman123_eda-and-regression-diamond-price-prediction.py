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
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
df = pd.read_csv('/kaggle/input/diamonds/diamonds.csv')
df.head()
df.drop('Unnamed: 0',axis=1, inplace=True)
df = df.reindex(columns=["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z", "price"])
len(df)
df.describe()
# check for missing values 
df.isnull().any()
sns.distplot(df['price'])
# Skewness 
print("The skewness of the Price in the dataset is {}".format(df['price'].skew()))
# Transforming the target variable
target = np.log(df['price'])
print("Skewness: {}".format(target.skew()))
sns.distplot(target)
df['carat'].hist()
df['cut'].unique()
sns.countplot(x='cut', data=df)
df['color'].unique()
sns.countplot(x='color', data=df)
df['clarity'].unique()
sns.countplot(df['clarity'])
fig, ax = plt.subplots(2, figsize=(10,10))
df['depth'].hist(ax=ax[0])
df['table'].hist(ax=ax[1])
ax[0].set_title("Distribution of depth")
ax[1].set_title("Distribution of table")
fig, ax = plt.subplots(3, figsize=(10,10))
df['x'].hist(ax=ax[0])
df['y'].hist(ax=ax[1])
df['z'].hist(ax=ax[2])
ax[0].set_title("Distribution of x")
ax[1].set_title("Distribution of y")
ax[2].set_title("Distribution of z")
df['price'].hist()
# Using Pearson Correlation 
plt.figure(figsize=(12,10))
cor = df.corr()
sns.heatmap(cor, annot=True,cmap=plt.cm.Reds)
plt.show()
# correlation with output variable 
cor_target = abs(cor["price"])

# Selecting highly correlated features 
relevent_features = cor_target[cor_target>0.5]
relevent_features
df.drop(['depth', 'table'], axis=1, inplace=True)
df.head()
# Encoding the categorical data 
# Encoding the independent variables
dummy_cut = pd.get_dummies(df['cut'],drop_first=True)   # drop_first to avoid the dummy variable trap
df = pd.concat([df, dummy_cut], axis=1)
df = df.drop('cut',axis=1)
df.head()
dummy_color = pd.get_dummies(df['color'], drop_first=True)   
df = pd.concat([df, dummy_color], axis=1)
df = df.drop('color',axis=1)
df.head()
dummy_clarity = pd.get_dummies(df['clarity'], drop_first=True)
df = pd.concat([df, dummy_clarity], axis=1)
df = df.drop('clarity', axis=1)
df.head()
order = df.columns.to_list()
order
order = ['carat',
 'x',
 'y',
 'z',
 'Good',
 'Ideal',
 'Premium',
 'Very Good',
 'E',
 'F',
 'G',
 'H',
 'I',
 'J',
 'IF',
 'SI1',
 'SI2',
 'VS1',
 'VS2',
 'VVS1',
 'VVS2',
  'price']
df = df[order]
df.head()
X = df.iloc[:,:-1].values
y = df.iloc[:,21].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
from sklearn.metrics import r2_score, mean_squared_error

regressor = LinearRegression()
regressor.fit(X_train, y_train)
# making predictions
y_pred = regressor.predict(X_test)
y_pred
mlr_score = regressor.score(X_test, y_test)
from sklearn import preprocessing, svm

X_svm = X.copy()
X_svm = preprocessing.scale(X_svm)

X_svm_train, X_svm_test, y_svm_train, y_svm_test = train_test_split(X_svm, y, test_size=0.2, random_state=0)

clf = svm.SVR(kernel='linear')
clf.fit(X_svm_train, y_svm_train)

svr_score = clf.score(X_svm_test,y_svm_test)
from sklearn.tree import DecisionTreeRegressor
regressor_dt = DecisionTreeRegressor(random_state=0)
regressor_dt.fit(X_train, y_train)
regressor_dt.predict(X_test)
dt_score = regressor_dt.score(X_test, y_test)
from sklearn.ensemble import RandomForestRegressor
regressor_rf = RandomForestRegressor(n_estimators=100, random_state=0)
regressor_rf.fit(X_train, y_train)
rf_score = regressor_rf.score(X_test, y_test)
print('Multiple Linear Regression accuracy:', mlr_score)
print('SVR score: ', svr_score)
print('Decision Tree Regression score: ', dt_score)
print('Random Forest Regression score: ', rf_score)
