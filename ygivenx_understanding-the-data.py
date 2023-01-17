%matplotlib inline

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings

from IPython.core.display import HTML 
from IPython.display import Image
from scipy import stats

sns.set_style("darkgrid")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
data = pd.read_csv("../input/train.csv")
data.describe()
(mu, sigma) = stats.norm.fit(data["SalePrice"])
ax = sns.distplot(data["SalePrice"])
ax.set(xlabel='Sale Price of houses')
plt.title("Sale Prices Distribution ")
plt.legend(["normal dist. fit ($\mu=${0:.2g}, $\sigma=${1:.2f})".format(mu, sigma)],)
(mu, sigma) = stats.norm.fit(np.log(data["SalePrice"]))
ax = sns.distplot(np.log(data["SalePrice"]))
ax.set(xlabel='Sale Price of houses')
plt.title("Sale Prices Distribution ")
plt.legend(["normal dist. fit ($\mu=${0:.2g}, $\sigma=${1:.2f})".format(mu, sigma)],)
print(data.shape)
cols = [c for c in data.columns]
print(cols)
data["SaleType"].describe()
def mapMSSubClass(x):
    if(x==20):
        return 1
    elif(x==60):
        return 2
    elif(x==50):
        return 3
    else:
        return 4
    
def get_features(df):
    le = LabelEncoder()
#     df = df.fillna(0)
#     features = df.apply(LabelEncoder().fit_transform)
#     return features
    features = pd.DataFrame()
    features["MSSubClass"] = df['MSSubClass'].apply(mapMSSubClass)
    features["LotFrontage"] = df.LotFrontage.fillna(0)
    features["LotArea"] = df.LotArea
    features["Age"] = df.YearBuilt.max() - df.YearBuilt
    features["Remodeled"] = df.YearRemodAdd.max() - df.YearRemodAdd
    features["MasVnrArea"] = df.MasVnrArea.fillna(0)
    features["MasVnrType"] = le.fit_transform(df.MasVnrType.fillna("None"))
    features["Street"] = le.fit_transform(df.Street)
    features["Alley"] = le.fit_transform(df.Alley.fillna("NA"))
    features["LotShape"] = le.fit_transform(df.LotShape)
    features["GrLivArea"] = df.GrLivArea
    features["LotArea"] = df.LotArea
    features["LandContour"] = le.fit_transform(df.LandContour)
    features["LotConfig"] = le.fit_transform(df.LotConfig)
    features["LandSlope"] = le.fit_transform(df.LandSlope)
    features["Neighborhood"] = le.fit_transform(df.Neighborhood)
    return features
data.MasVnrType.describe()
g = sns.scatterplot(x="LotArea",y="SalePrice",data=data,hue="SaleType")
g.set_title("LotArea Vs SalePrice with SaleType")
g.legend(bbox_to_anchor=(1, 1), loc=2)
g = sns.scatterplot(x="GrLivArea",y="SalePrice",data=data,hue="SaleType")
g.set_title("LotArea Vs SalePrice with SaleType")
g.legend(bbox_to_anchor=(1, 1), loc=2)
# MSSubClass 
#         20	1-STORY 1946 & NEWER ALL STYLES
#         30	1-STORY 1945 & OLDER
#         40	1-STORY W/FINISHED ATTIC ALL AGES
#         45	1-1/2 STORY - UNFINISHED ALL AGES
#         50	1-1/2 STORY FINISHED ALL AGES
#         60	2-STORY 1946 & NEWER
#         70	2-STORY 1945 & OLDER
#         75	2-1/2 STORY ALL AGES
#         80	SPLIT OR MULTI-LEVEL
#         85	SPLIT FOYER
#         90	DUPLEX - ALL STYLES AND AGES
#        120	1-STORY PUD (Planned Unit Development) - 1946 & NEWER
#        150	1-1/2 STORY PUD - ALL AGES
#        160	2-STORY PUD - 1946 & NEWER
#        180	PUD - MULTILEVEL - INCL SPLIT LEV/FOYER
#        190	2 FAMILY CONVERSION - ALL STYLES AND AGES
sns.countplot(data.MSSubClass)
sns.violinplot(x="MSSubClass", y="GrLivArea", data=data)
# create a features vector
features = pd.DataFrame()
features["MSSubClass"] = data['MSSubClass'].apply(mapMSSubClass)
features["LotFrontage"] = data.LotFrontage.fillna(0)
features.head()
features["LotArea"] = data.LotArea
f = {'GrLivArea':['mean'], 'SalePrice':['mean']}
year_data = data.groupby("YearBuilt").agg(f).reset_index()
year_data["GrLivArea"]["mean"]
g = sns.scatterplot(x=year_data["YearBuilt"],y=year_data["GrLivArea"]["mean"],hue=year_data["SalePrice"]["mean"])
g.legend(bbox_to_anchor=(1, 1), loc=2)
g.set_title("Year Built vs Living Area with SalePrice")
features["Age"] = data.YearBuilt.max() - data.YearBuilt
f = {'GrLivArea':['mean'], 'SalePrice':['mean']}
re_year_data = data.groupby("YearRemodAdd").agg(f).reset_index()
re_year_data["GrLivArea"]["mean"]
g = sns.scatterplot(x=re_year_data["YearRemodAdd"],y=re_year_data["GrLivArea"]["mean"],hue=re_year_data["SalePrice"]["mean"])
g.legend(bbox_to_anchor=(1, 1), loc=2)
g.set_title("Year Remodeled vs Living Area with SalePrice")
features["Remodeled"] = data.YearRemodAdd.max() - data.YearRemodAdd
g = sns.scatterplot(x="MasVnrArea",y="SalePrice", hue="MasVnrType", data=data.loc[data["MasVnrType"] != "None"])
g.legend(bbox_to_anchor=(1, 1), loc=2)
sns.violinplot(x="MasVnrType", y="SalePrice", data=data)
le = LabelEncoder()
features["MasVnrArea"] = data.MasVnrArea.fillna(0)
features["MasVnrType"] = le.fit_transform(data.MasVnrType.fillna("None"))
features["Street"] = le.fit_transform(data.Street)
features["Alley"] = le.fit_transform(data.Alley.fillna("NA"))
sns.violinplot(x="LotShape", y="SalePrice", data=data)
features["LotShape"] = le.fit_transform(data.LotShape)
features["GrLivArea"] = data.GrLivArea
features["LotArea"] = data.LotArea
sns.violinplot(x="LandContour", y="SalePrice", data=data)
features["LandContour"] = le.fit_transform(data.LandContour)
sns.violinplot(x="LotConfig", y="SalePrice", data=data)
features["LotConfig"] = le.fit_transform(data.LotConfig)
features["LandSlope"] = le.fit_transform(data.LandSlope)
features["Neighborhood"] = le.fit_transform(data.Neighborhood)

y = data["SalePrice"]

X = get_features(data)
X_train, X_test, y_train, y_test = train_test_split(X,y)

# X_train = X_train.reshape(-1,1)
# X_test = X_test.reshape(-1,1)
print(X_train.shape)
print(X_test.shape)

lm = LinearRegression()
model = lm.fit(X_train, y_train)
predictions = model.predict(X_test)
predictions[:5]
print(model.coef_)
print(model.score(X_test,y_test))
submission = pd.DataFrame()
full_X = pd.read_csv("../input/test.csv",)
features_full = get_features(full_X)
full_pred = model.predict(features_full)
print(full_pred.shape)
print(features_full.shape)
submission["Id"] = full_X["Id"]
submission["SalePrice"] = full_pred
submission.to_csv("submission.csv",index=False)