%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt



import seaborn as sns



from sklearn import linear_model
train = pd.read_csv("../input/train.csv")
train.head()
with sns.axes_style("whitegrid"):

    train.LotArea.hist(bins=50)
plt.scatter(train.OverallQual, train.SalePrice, lw=1)

plt.xlabel("Overall Quality")

plt.ylabel("Sale Price")
plt.scatter(train.OverallCond, train.SalePrice, lw=1)

plt.xlabel("Overall Condition")

plt.ylabel("Sale Price")
plt.scatter(train.TotalBsmtSF, train.SalePrice, lw=1)

plt.xlabel("Total Basement Area")

plt.ylabel("Sale Price")
ls_feature = ["LotArea", "TotalBsmtSF", "OverallQual", "TotRmsAbvGrd", "GarageArea", 

              "1stFlrSF", "2ndFlrSF", "YearBuilt", "YearRemodAdd"]

for i in ls_feature:

    print(np.corrcoef(train[i].fillna(0), train["SalePrice"]))
train["FinishedArea"] = train["TotalBsmtSF"] - train["BsmtUnfSF"]
#Prepping data for linear regression

#ls_feature = ["LotArea", "TotalBsmtSF", "OverallQual", "TotRmsAbvGrd", "GarageArea", "GarageCars",

#              "1stFlrSF", "2ndFlrSF", "YearBuilt", "YearRemodAdd"]

ls_feature =["LotArea","TotalBsmtSF", "OverallQual", "TotRmsAbvGrd", "GarageArea",

             "1stFlrSF", "2ndFlrSF", "YearBuilt", "YearRemodAdd", "FinishedArea"]

test = pd.read_csv("../input/test.csv")



test["FinishedArea"] = test["TotalBsmtSF"] - test["BsmtUnfSF"]



test_X = test[ls_feature].fillna(0)



train_X = train[ls_feature].fillna(0)

train_Y = train["SalePrice"].fillna(0)
linreg = linear_model.LinearRegression()
linreg.fit(train_X, train_Y)
pred_Y = linreg.predict(test_X)
pred_Y[pred_Y < 0] = 0

linreg.score(train_X, train_Y)
df_out = pd.DataFrame({"Id":test["Id"], "SalePrice":pred_Y})
df_out.to_csv("output_wolotFinishedArea.csv", index=False)