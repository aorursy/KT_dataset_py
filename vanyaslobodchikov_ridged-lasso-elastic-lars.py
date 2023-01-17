%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt



import seaborn as sns



from sklearn import linear_model



from sklearn import preprocessing



from sklearn.model_selection import cross_val_score
train = pd.read_csv("../input/train.csv")
train.head()
"""

Getting a list of cols with NaN values

"""

train.isnull().sum()[train.isnull().sum() > 0]
"""

Getting a list of object type columns

"""

obj_cols = list(train.select_dtypes(include=["object"]).columns)

obj_cols
"""

Getting a list of numeric type columns

"""

numeric_cols = list(train.select_dtypes(exclude=["object"]).columns)

del numeric_cols[-1]

numeric_cols
train["FinishedArea"] = train["TotalBsmtSF"] - train["BsmtUnfSF"]
"""

Below there are some graph to get the feel for the data and get rid of outliers

"""
with sns.axes_style("whitegrid"):

    train.SalePrice.hist(bins=50)
train = train[train["LotArea"] < 100000]



feat = "LotArea"

def scat(train_2std, feat, res):

    plt.scatter(train_2std[feat], train_2std[res], lw=1, alpha=.1)

    plt.xlabel(feat)

    plt.ylabel(res)

        

scat(train, feat, "SalePrice")
train = train[train["TotalBsmtSF"] < 6000]

scat(train, "TotalBsmtSF", "SalePrice")
scat(train, "OverallQual", "SalePrice")
#train = train[train["TotRmsAbvGrd"] < 14]

scat(train, "TotRmsAbvGrd", "SalePrice")
scat(train, "1stFlrSF", "SalePrice")
scat(train, "2ndFlrSF", "SalePrice")
scat(train, "FinishedArea", "SalePrice")
train = train[train["GrLivArea"] < 4600]

scat(train, "GrLivArea", "SalePrice")
scat(train, "GarageArea", "SalePrice")
scat(train, "GarageYrBlt", "SalePrice")
train["QualYear"] = train["OverallQual"] / train["YearBuilt"]
plt.scatter(train["QualYear"], train["SalePrice"], lw=1, alpha=.1)
scat(train, "GarageCars", "SalePrice")
scat(train, "FullBath", "SalePrice")
#Prepping data for linear regression



test = pd.read_csv("../input/test.csv")



"""

Below is a part to create a dummy columns for the object type columns

"""

dummy_df_train = pd.get_dummies(train[obj_cols])

ls_dummy_cols = list(dummy_df_train.columns)



dummy_df_test = pd.get_dummies(test[obj_cols])



ls_dummy_cols = list(set(list(dummy_df_train.columns)) & set(list(dummy_df_test.columns)))



ls_dummy_empty_cols = list(set(list(dummy_df_train.columns)) ^ set(list(dummy_df_test.columns)))



train = pd.concat([train, dummy_df_train], axis=1)



test = pd.concat([test, dummy_df_test, pd.DataFrame(columns=ls_dummy_empty_cols)], axis=1)



numeric_cols.append("QualYear")

numeric_cols.append("FinishedArea")

numeric_cols.extend(ls_dummy_cols)

ls_params = numeric_cols



"""

Just a little try to create some new features

"""

test["FinishedArea"] = test["TotalBsmtSF"] - test["BsmtUnfSF"]



test["QualYear"] = test["OverallQual"] / test["YearBuilt"]



test_X = test[ls_params].fillna(0)

train_X = train[ls_params].fillna(0)

train_Y = np.log(train["SalePrice"])
"""

Function to get the scoring like in the competition. Taken from Alexandru Papiu's kernel

"""

def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, train_X, train_Y, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)
linreg = linear_model.LinearRegression()
linreg.fit(train_X, train_Y)
pred_Y = linreg.predict(test_X)
linreg.score(train_X, train_Y)
rmse_cv(linreg).mean()
ridgereg = linear_model.RidgeCV(alphas=[0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75], cv=5)

ridgereg.fit(train_X, train_Y)

pred_Y_ridge = ridgereg.predict(test_X)

ridgereg.score(train_X, train_Y)
rmse_cv(ridgereg).mean()
scores_ridge = cross_val_score(ridgereg, train_X, train_Y, cv=10)

scores_ridge
lasso = linear_model.LassoCV(alphas=[1, 0.1, 0.001, 0.0005], cv=5)

lasso.fit(train_X, train_Y)

pred_Y_lasso = lasso.predict(test_X)

lasso.score(train_X, train_Y)
rmse_cv(lasso).mean()
elastic = linear_model.ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], 

                                    alphas=[0.001, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75], cv=5)

elastic.fit(train_X, train_Y)

pred_Y_elastic = lasso.predict(test_X)

elastic.score(train_X, train_Y)
rmse_cv(elastic).mean()
df_out_elastic = pd.DataFrame({"Id":test["Id"], "SalePrice":np.exp(pred_Y_elastic)})
df_out_elastic.to_csv("elastic.csv", index=False)
lars = linear_model.LassoLarsCV(cv=5)

lars.fit(train_X, train_Y)

pred_Y_lars = lars.predict(test_X)

lars.score(train_X, train_Y)
rmse_cv(lars).mean()
df_out_lars = pd.DataFrame({"Id":test["Id"], "SalePrice":np.exp(pred_Y_lars)})

df_out_lars.to_csv("lars.csv", index=False)
"""

Plotting residuals

"""

pred_Y_res = ridgereg.predict(train_X)

plt.scatter(pred_Y_res, pred_Y_res - train_Y)
df_out_ridge = pd.DataFrame({"Id":test["Id"], "SalePrice":np.exp(pred_Y_ridge)})

df_out_ridge.to_csv("ridged.csv", index=False)


df_out_lasso = pd.DataFrame({"Id":test["Id"].astype("int"), "SalePrice":np.exp(pred_Y_lasso)})

df_out_lasso.to_csv("lasso.csv", index=False)