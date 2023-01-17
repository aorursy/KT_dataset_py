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



import seaborn as sns



from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
# No.2

# データの読み込み

df = pd.read_csv("../input/train.csv")
df.head()
len(df)
df.describe()
corrmat = df.corr()

corrmat.head()
# ヒートマップに表示させるカラムの数

k = 10



# SalesPriceとの相関が大きい上位10個のカラム名を取得

cols = corrmat.abs().nlargest(k, 'SalePrice')['SalePrice'].index



# SalesPriceとの相関が大きい上位10個のカラムを対象に相関を算出

# .T(Trancepose[転置行列])を行う理由は、corrcoefで相関を算出する際に、各カラムの値を行毎にまとめなければならない為

cm = np.corrcoef(df[cols].values.T)



# ヒートマップのフォントサイズを指定

sns.set(font_scale=1.25)



# 算出した相関データをヒートマップで表示

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
df.isnull().sum().sort_values().tail(15)
df_tidy = df.fillna({'PoolQC':'Nothing','MiscFeature':'Nothing','Alley':'Nothing','Fence': 'Nothing', 'FireplaceQu': 'Nothing', 'LotFrontage': 0}).dropna()

df_tidy.isnull().sum().sort_values().tail(15)
df_tidy.describe()
df_onehot = pd.get_dummies(df_tidy, columns=['MSZoning','LotFrontage','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','MasVnrArea','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','Heating','HeatingQC','CentralAir','Electrical','BsmtFullBath','BsmtHalfBath','KitchenQual','Functional','FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','SaleType','SaleCondition'], drop_first=True)

df_onehot.columns
# 説明変数と目的変数

# XにOverallQual、yにSalePriceをセット

X = df_onehot[["OverallQual"]]

y = df_onehot["SalePrice"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

print(X_train.count())

print(X_test.count())
# アルゴリズムに線形回帰(Linear Regression)を採用

slr = LinearRegression()



# fit関数でモデル作成

slr.fit(X_train,y_train)
# 散布図を描画

plt.scatter(X_train,y_train)



# 折れ線グラフを描画

plt.plot(X_train,slr.predict(X_train),color='red')



# 表示

plt.show()
from sklearn.metrics import r2_score

y_train_pred = slr.predict(X_train)

y_test_pred = slr.predict(X_test)

print ('Accuracy on Training Set: {:.3f}'.format(r2_score(y_train, y_train_pred)))

print ('Accuracy on Validation Set: {:.3f}'.format(r2_score(y_test, y_test_pred)))
from sklearn.model_selection import train_test_split, cross_val_score

scores = cross_val_score(slr, X, y, cv=5)

print ('Scores:', scores)

print ('Mean Score: {:f} ± {:.3}'.format(scores.mean(), scores.std()))
# No.7

# テストデータの読込

df_test = pd.read_csv("../input/test.csv")
# No.8

# テストデータの内容確認(評価用のデータなので、SalePriceはない)

df_test.head()
# No.9

# テストデータの OverallQual の値をセット

X_test = df_test[["OverallQual"]].values



# 学習済みのモデルから予測した結果をセット

y_test_pred = slr.predict(X_test)

y_test_pred
# No.11

# df_testに SalePrice カラムを追加し、学習済みのモデルから予測した結果をセット

df_test["SalePrice"] = y_test_pred
df_test.head()
df_test[["Id","SalePrice"]].head()
# Id, SalePriceの2列だけのファイルに変換

df_test[["Id","SalePrice"]].to_csv("submission.csv",index=False)
# 説明変数と目的変数

# XにOverallQual、yにSalePriceをセット

X = df_onehot[["GrLivArea"]]

y = df_onehot["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)



# アルゴリズムに線形回帰(Linear Regression)を採用

slr = LinearRegression()



# fit関数でモデル作成

slr.fit(X_train,y_train)



from sklearn.model_selection import train_test_split, cross_val_score

scores = cross_val_score(slr, X, y, cv=5)

print ('Scores:', scores)

print ('Mean Score: {:f} ± {:.3}'.format(scores.mean(), scores.std()))
# 説明変数と目的変数

# XにOverallQual、yにSalePriceをセット

X = df_onehot.drop('SalePrice',axis="columns")

y = df_onehot["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)



# アルゴリズムに線形回帰(Linear Regression)を採用

slr = LinearRegression()



# fit関数でモデル作成

slr.fit(X_train,y_train)



from sklearn.model_selection import train_test_split, cross_val_score

scores = cross_val_score(slr, X, y, cv=5)

print ('Scores:', scores)

print ('Mean Score: {:f} ± {:.3}'.format(scores.mean(), scores.std()))
df_tidy = df.fillna({'PoolQC':'Nothing','MiscFeature':'Nothing','Alley':'Nothing','Fence': 'Nothing', 'FireplaceQu': 'Nothing', 'LotFrontage': 0,'GarageYrBlt':'Nothing','GarageYrBlt':'Nothing','GarageQual':'Nothing','GarageFinish':'Nothing','GarageType':'Nothing','GarageCond':'Nothing','BsmtFinType2':'Nothing','BsmtExposure':'Nothing','BsmtFinType1':'Nothing','BsmtQual':'Nothing','BsmtCond':'Nothing','MasVnrType':'Nothing','Electrical':'Nothing','MasVnrArea':0})
# SalesPriceとの相関が大きい上位10個のカラム名を取得

corrmat = df_tidy.corr()

cols = corrmat.abs().nlargest(10, 'SalePrice')['SalePrice'].index



sns.set()

sns.pairplot(df_tidy,y_vars=['SalePrice'],x_vars=cols,size = 2.5)

plt.show()
# 数値の大きい上位2位のデータを表示

df_tidy.sort_values(by = 'GrLivArea', ascending = False)[:2]
df_tidy = df_tidy.drop(index = df_tidy[df_tidy['Id'] == 1299].index)

df_tidy = df_tidy.drop(index = df_tidy[df_tidy['Id'] == 524].index)

df_tidy.sort_values(by = 'GrLivArea', ascending = False)[:2]
df_onehot = pd.get_dummies(df_tidy, columns=['MSZoning','LotFrontage','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','MasVnrArea','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','Heating','HeatingQC','CentralAir','Electrical','BsmtFullBath','BsmtHalfBath','KitchenQual','Functional','FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','SaleType','SaleCondition'], drop_first=True)

df_onehot.columns
from sklearn.model_selection import GridSearchCV, KFold

from sklearn.svm import SVC



p_grid = {'fit_intercept': [True,False], 'normalize': [True,False], 'copy_X':[True,False]}

#svc = SVC(kernel='rbf')

slr = LinearRegression()

cv = KFold(n_splits=5, shuffle=True, random_state=1)

#clf_cv = GridSearchCV(estimator=svc, param_grid=p_grid, cv=cv, scoring='f1')

slr_cv = GridSearchCV(estimator=slr, param_grid=p_grid, cv=cv, scoring='r2')



X = df_onehot.drop('SalePrice',axis="columns")

y = df_onehot["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

rtn = slr_cv.fit(X_train, y_train)



print(slr_cv.best_params_)

print(slr_cv.best_score_)
# 説明変数と目的変数

# XにOverallQual、yにSalePriceをセット

X = df_onehot.drop('SalePrice',axis="columns")

y = df_onehot["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)



# アルゴリズムに線形回帰(Linear Regression)を採用

slr = LinearRegression(copy_X=slr_cv.best_params_['copy_X'], fit_intercept=slr_cv.best_params_['fit_intercept'], 

                       normalize=slr_cv.best_params_['normalize'],)



from sklearn.model_selection import train_test_split, cross_val_score

scores = cross_val_score(slr, X, y, cv=5)

print ('Scores:', scores)

print ('Mean Score: {:f} ± {:.3}'.format(scores.mean(), scores.std()))

# fit関数でモデル作成

slr.fit(X_train,y_train)
# テストデータの読込

df_test = pd.read_csv("../input/test.csv")



# TODO

df_test_tidy = df_test.fillna({'PoolQC':'Nothing','MiscFeature':'Nothing','Alley':'Nothing','Fence': 'Nothing', 'FireplaceQu': 'Nothing', 'LotFrontage': 0,'GarageYrBlt':'Nothing','GarageYrBlt':'Nothing','GarageQual':'Nothing','GarageFinish':'Nothing','GarageType':'Nothing','GarageCond':'Nothing','BsmtFinType2':'Nothing','BsmtExposure':'Nothing','BsmtFinType1':'Nothing','BsmtQual':'Nothing','BsmtCond':'Nothing','MasVnrType':'Nothing','Electrical':'Nothing','MasVnrArea':0})



#print(df_test_tidy[df_test_tidy.isnull().any(axis="columns")])

df_test_tidy = df_test_tidy.fillna({'BsmtFinSF1':'Nothing','BsmtFinSF2':0,'BsmtUnfSF':0,'BsmtHalfBath':0,

'TotalBsmtSF':0,'BsmtFullBath':0,'GarageCars':0,'GarageArea':0,'KitchenQual':'TA',

'Exterior1st':'Nothing','Exterior2nd':'Nothing','SaleType':'WD','Utilities':'AllPub','Functional':'Typ',

'MSZoning':'RL'})

df_test_tidy.isnull().sum().sort_values().tail(15)



df_tidy['WhatIsData'] = 'Train'

df_test_tidy['WhatIsData'] = 'Test'

df_test_tidy['SalePrice'] = 9999999999

alldata = pd.concat([df_tidy,df_test_tidy],axis=0).reset_index(drop=True)
alldata_onehot = pd.get_dummies(alldata, columns=['MSZoning','LotFrontage','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','MasVnrArea','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','Heating','HeatingQC','CentralAir','Electrical','BsmtFullBath','BsmtHalfBath','KitchenQual','Functional','FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','SaleType','SaleCondition'], drop_first=True)

print(alldata_onehot.columns)

X_train = alldata_onehot[alldata_onehot['WhatIsData']=='Train'].drop(['WhatIsData','SalePrice'], axis=1).reset_index(drop=True)

y_train = alldata_onehot[alldata_onehot['WhatIsData']=='Train']["SalePrice"]



# アルゴリズムに線形回帰(Linear Regression)を採用

slr = LinearRegression(copy_X=slr_cv.best_params_['copy_X'], fit_intercept=slr_cv.best_params_['fit_intercept'], 

                       normalize=slr_cv.best_params_['normalize'],)



scores = cross_val_score(slr, X_train, y_train, cv=5)

print ('Scores:', scores)

print ('Mean Score: {:f} ± {:.3}'.format(scores.mean(), scores.std()))



# fit関数でモデル作成

slr.fit(X_train,y_train)



X_test = alldata_onehot[alldata_onehot['WhatIsData']=='Test'].drop(['WhatIsData','SalePrice'], axis=1).reset_index(drop=True)



# 学習済みのモデルから予測した結果をセット

y_test_pred = slr.predict(X_test)



df_test["SalePrice"] = y_test_pred



df_test[["Id","SalePrice"]].to_csv("submission.csv",index=False)