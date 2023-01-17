# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/train.csv",index_col=0)

test_df = pd.read_csv("../input/test.csv",index_col=0)



train_df.head()
test_df.shape
price=pd.DataFrame({"origin price":train_df.SalePrice,"log1p price":np.log1p(train_df.SalePrice)})

price.hist()
y_train = np.log1p(train_df.pop("SalePrice"))

all_df = pd.concat((train_df,test_df),axis=0)

y_train.shape

all_df.shape


all_df.MSSubClass=all_df.MSSubClass.astype(str)

all_df['MSSubClass'].dtypes
all_df.MSSubClass.value_counts()
all_dummy_df = pd.get_dummies(all_df)

all_dummy_df.shape
all_dummy_df.isnull().sum().sort_values(ascending=False).head(10)
mean_cols = all_dummy_df.mean()

all_dummy_df = all_dummy_df.fillna(mean_cols)

all_dummy_df.isnull().sum().sum()
numeric_cols = all_df.columns[all_df.dtypes != 'object']

numeric_cols
numeric_mean=all_dummy_df.loc[:,numeric_cols].mean()

numeric_std = all_dummy_df.loc[:,numeric_cols].std()

all_dummy_df.loc[:,numeric_cols] = (all_dummy_df.loc[:,numeric_cols]-numeric_mean)/numeric_std
dummy_train_df  = all_dummy_df.loc[train_df.index]

dummy_test_df  = all_dummy_df.loc[test_df.index]

dummy_train_df.shape,dummy_test_df.shape
from sklearn.linear_model import Ridge

from sklearn.model_selection import cross_val_score
X_train = dummy_train_df.values

X_test = dummy_test_df.values
alphas = np.logspace(-3,2,60)

test_scores = []

for alpha in alphas:

    clf = Ridge(alpha)

    score=np.sqrt(-cross_val_score(clf,X_train,y_train,cv=10, scoring='neg_mean_squared_error'))

    test_scores.append(np.mean(score))
alphas
import matplotlib.pyplot as plt

%matplotlib inline

plt.plot(alphas, test_scores)

plt.title("Alpha vs CV Error");
from sklearn.ensemble import RandomForestRegressor
max_features = [.1, .3, .5, .7, .9, .99]

test_scores = []

for max_feature in max_features:

    clf=RandomForestRegressor(n_estimators=200,max_features=max_feature)

    score=np.sqrt(-cross_val_score(clf,X_train,y_train,cv=5, scoring='neg_mean_squared_error'))

    test_scores.append(np.mean(score))
plt.plot(max_features, test_scores)

plt.title("Max Features vs CV Error");
final_ridge = Ridge(15)

final_forest = RandomForestRegressor(n_estimators=200,max_features=0.3)

final_ridge.fit(X_train, y_train)

final_forest.fit(X_train, y_train)
y_ridge=np.expm1(final_ridge.predict(X_test))

y_forest=np.expm1(final_forest.predict(X_test))



y_final=(y_ridge+y_forest)/2.0



submission_df = pd.DataFrame(data= {'Id' : test_df.index, 'SalePrice': y_final})

submission_df.head(10)