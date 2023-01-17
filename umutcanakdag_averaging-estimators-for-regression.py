import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats

from scipy.special import log1p

from matplotlib.pyplot import figure



from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder



from sklearn.impute import SimpleImputer

import warnings; warnings.simplefilter('ignore')

pd.options.display.max_rows = 100



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')



len(test_data), len(train_data)
train_data.isna().sum()[train_data.isna().sum() != 0]
test_data.isna().sum()[test_data.isna().sum() != 0]
train_data[["MSSubClass", "YrSold", "MoSold"]]
train_data["MSSubClass"] = train_data["MSSubClass"].astype(str)

test_data["MSSubClass"] = test_data["MSSubClass"].astype(str)



train_data["YrSold"] = train_data["YrSold"].astype(str)

test_data["YrSold"] = test_data["YrSold"].astype(str)



train_data["MoSold"] = train_data["MoSold"].astype(str)

test_data["MoSold"] = test_data["MoSold"].astype(str)
all_data = pd.concat([train_data, test_data])



all_data["Alley"].fillna("null", inplace=True)

all_data["BsmtQual"].fillna("null", inplace=True)

all_data["BsmtCond"].fillna("null", inplace=True)

all_data["BsmtExposure"].fillna("null", inplace=True)

all_data["BsmtFinType1"].fillna("null", inplace=True)

all_data["BsmtFinType2"].fillna("null", inplace=True)

all_data["FireplaceQu"].fillna("null", inplace=True)

all_data["GarageType"].fillna("null", inplace=True)

all_data["GarageFinish"].fillna("null", inplace=True)

all_data["GarageQual"].fillna("null", inplace=True)

all_data["GarageCond"].fillna("null", inplace=True)

all_data["PoolQC"].fillna("null", inplace=True)

all_data["Fence"].fillna("null", inplace=True)

all_data["MiscFeature"].fillna("null", inplace=True)



train_data = all_data.iloc[:1459]

test_data = all_data.iloc[1460:]



figure(figsize = (20,15))

corr = train_data.corr()



sns.heatmap(corr, 

        xticklabels=corr.columns,

        yticklabels=corr.columns,

            )

print("Most correlated features with SalePrice")

print(corr.SalePrice.sort_values(ascending = False))

numerical_features = list(train_data.select_dtypes(exclude = 'O').columns)

del numerical_features[0]





fig, axs = plt.subplots(7,5, figsize=(40,40))



for i, column in enumerate(numerical_features):

    j = i//5

    k = i%5

    sns.scatterplot(x=column, y='SalePrice', data=train_data, ax=axs[j][k])

train_data.drop(train_data[train_data["TotalBsmtSF"] > 5000].index, inplace = True)

train_data.drop(train_data[train_data["1stFlrSF"] > 4000].index, inplace = True)

train_data.drop(train_data[(train_data["GrLivArea"] > 4000) & (train_data["SalePrice"] < 300000)].index, inplace = True)

train_data.drop(train_data[(train_data["GarageArea"] > 1200) & (train_data["SalePrice"] < 300000)].index, inplace = True)
categorical_features = list(train_data.select_dtypes(include = 'O').columns)



fig, axs = plt.subplots(8,6, figsize=(40,40))



for i, column in enumerate(categorical_features):

    j = i//6

    k = i%6

    sns.boxplot(x=column, y='SalePrice', data=train_data, ax=axs[j][k])

train_data["SoldAge"] = train_data["YrSold"].astype(int) - train_data["YearBuilt"] 

test_data["SoldAge"] = test_data["YrSold"].astype(int) - test_data["YearBuilt"]



train_data.drop(columns = ["YrSold"], inplace = True)

test_data.drop(columns = ["YrSold"], inplace = True)
def drop_unbalanced_features(train_data, test_data):

    columns = list(train_data.select_dtypes(include = 'O').columns)

    for column in columns:

        variable_perc = train_data[column].value_counts(normalize=True)[0]

        if variable_perc >= 0.95:

            train_data.drop(columns = [column], inplace = True)

            test_data.drop(columns = [column], inplace = True)



drop_unbalanced_features(train_data, test_data)
pd.concat([test_data, train_data]).isna().sum()[pd.concat([test_data, train_data]).isna().sum() != 0]
# https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9



def cramers_v(x, y):

    confusion_matrix = pd.crosstab(x,y)

    chi2 = stats.chi2_contingency(confusion_matrix)[0]

    n = confusion_matrix.sum().sum()

    phi2 = chi2/n

    r,k = confusion_matrix.shape

    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))

    rcorr = r-((r-1)**2)/(n-1)

    kcorr = k-((k-1)**2)/(n-1)

    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

def custom_fill_categorical_na(train_data, test_data, x):

    cat_columns = list(train_data.select_dtypes(include = 'O').columns)

    corr_df = []

    for column in cat_columns:

        corr_val = cramers_v(train_data[column], train_data[x])

        corr_df.append({"column": column, "corr": corr_val})

    corr_df = pd.DataFrame(corr_df).sort_values(by = "corr", ascending = False)

    most_correlated = corr_df.iloc[1].column

    second_most_correlated = corr_df.iloc[2].column

    most_corr_dict = dict(train_data.groupby(most_correlated)[x].agg(lambda df: df.mode()[0]))

    if train_data[(train_data[most_correlated].isna()) & (train_data[x].isna())].empty and test_data[(test_data[most_correlated].isna()) & (test_data[x].isna())].empty:

        train_data[x].fillna(train_data[most_correlated].map(most_corr_dict), inplace = True)

        test_data[x].fillna(test_data[most_correlated].map(most_corr_dict), inplace = True)

        print("most correlated: {} null feature : {}".format(most_correlated, x))

    else:

        most_corr_dict = dict(train_data.groupby(second_most_correlated)[x].agg(lambda df: df.mode()[0]))

        train_data[x].fillna(train_data[second_most_correlated].map(most_corr_dict), inplace = True)

        test_data[x].fillna(test_data[second_most_correlated].map(most_corr_dict), inplace = True)

        print("most correlated: {} null feature : {}".format(second_most_correlated, x))

    return corr_df, most_corr_dict



na_features = ["MSZoning", "Exterior1st", "Exterior2nd", "MasVnrType", "Electrical", "KitchenQual", "Functional", "SaleType" ]

for feature in na_features:

    custom_fill_categorical_na(train_data, test_data, feature)
pd.concat([test_data, train_data]).isna().sum()[pd.concat([test_data, train_data]).isna().sum() != 0]
cols = ["1stFlrSF", "LotArea", "LotFrontage"]

imp = IterativeImputer(random_state=0)

imp.fit(train_data[cols])

train_data["LotFrontage"] = imp.transform(train_data[cols])[:,2]



test_data["LotFrontage"] = imp.transform(test_data[cols])[:,2]



cols = ["YearBuilt", "GarageYrBlt"]



imp = IterativeImputer(random_state=0)

imp.fit(train_data[cols])

train_data["GarageYrBlt"] = imp.transform(train_data[cols])[:,1]



test_data["GarageYrBlt"] = imp.transform(test_data[cols])[:,1]



cols = ["OverallQual", "YearBuilt", "MasVnrArea"]

imp = IterativeImputer(random_state=0)

imp.fit(train_data[cols])

train_data["MasVnrArea"] = imp.transform(train_data[cols])[:,2]



test_data["MasVnrArea"] = imp.transform(test_data[cols])[:,2]



na_cols = ["BsmtFinSF1", "BsmtUnfSF", "BsmtFinSF2", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath", "GarageCars", "GarageArea"]

for col in na_cols:

    avg = train_data[col].mean()

    train_data[col].fillna(avg, inplace = True)

    test_data[col].fillna(avg, inplace = True)

sns.distplot(train_data['SalePrice'],fit=stats.norm)



print(train_data["SalePrice"].describe())

print("skewness: ",train_data["SalePrice"].skew())
fig, axs = plt.subplots(3, figsize = (10,20))

x = train_data["SalePrice"]

stats.probplot(x, plot=axs[0])

axs[0].set_title("Probability Plot before transformation")



x = log1p(x)

stats.probplot(x, plot=axs[1])

axs[1].set_title("Probability Plot after log transformation")



sns.distplot(x,fit=stats.norm, ax = axs[2])

axs[2].set_title("Distplot after log transformation")



train_data["SalePrice"] = x

print("skewness : ", train_data["SalePrice"].skew())
numerical_features = list(train_data.select_dtypes(exclude = 'O').columns)

del numerical_features[0]



skewness_list = train_data[numerical_features].apply(lambda x: x.skew())

skew_df = pd.DataFrame({"column": numerical_features, "skewness": skewness_list})



print(skew_df.loc[abs(skew_df["skewness"]) > 1])



transform_columns = skew_df.loc[(abs(skew_df["skewness"]) > 1), "column"]
for idx, val in enumerate(transform_columns): 

    

    train_data[val] = log1p(train_data[val])

    test_data[val] = log1p(test_data[val])
all_data = pd.concat([train_data, test_data])



all_data["LotShape"].replace(["Reg", "IR1", "IR2", "IR3"], [3,2,1,0], inplace = True)

all_data["LandSlope"].replace(["Gtl", "Mod", "Sev"], [2,1,0], inplace = True)

all_data["HouseStyle"].replace(["1Story", "1.5Fin", "1.5Unf", "2Story", "2.5Fin", "2.5Unf", "SFoyer", "SLvl"], [7,6,5,4,3,2,1,0], inplace = True)

all_data["OverallQual"].astype(int)

all_data["OverallCond"].astype(int)

all_data["ExterQual"].replace(["Ex", "Gd", "TA", "Fa", "Po"], [4,3,2,1,0], inplace = True)

all_data["ExterCond"].replace(["Ex", "Gd", "TA", "Fa", "Po"], [4,3,2,1,0], inplace = True)

all_data["BsmtQual"].replace(["Ex", "Gd", "TA", "Fa", "Po", "null"], [5,4,3,2,1,0], inplace = True)

all_data["BsmtCond"].replace(["Ex", "Gd", "TA", "Fa", "Po", "null"], [5,4,3,2,1,0], inplace = True)

all_data["BsmtExposure"].replace(["Gd", "Av", "Mn", "No", "null"], [4,3,2,1,0], inplace = True)

all_data["BsmtFinType1"].replace(["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "null"], [6,5,4,3,2,1,0], inplace = True)

all_data["BsmtFinType2"].replace(["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "null"], [6,5,4,3,2,1,0], inplace = True)

all_data["HeatingQC"].replace(["Ex", "Gd", "TA", "Fa", "Po"], [4,3,2,1,0], inplace = True)

all_data["CentralAir"].replace(["N", "Y"], [0,1], inplace = True)

all_data["KitchenQual"].replace(["Ex", "Gd", "TA", "Fa", "Po"], [4,3,2,1,0], inplace = True)

all_data["Functional"].replace(["Typ", "Min1", "Min2", "Mod", "Maj1", "Maj2", "Sev", "Sal"], [7,6,5,4,3,2,1,0], inplace = True)

all_data["FireplaceQu"].replace(["Ex", "Gd", "TA", "Fa", "Po", "null"], [5,4,3,2,1,0], inplace = True)

all_data["GarageFinish"].replace(["Fin", "RFn", "Unf", "null"], [3,2,1,0], inplace = True)

all_data["GarageQual"].replace(["Ex", "Gd", "TA", "Fa", "Po", "null"], [5,4,3,2,1,0], inplace = True)

all_data["GarageCond"].replace(["Ex", "Gd", "TA", "Fa", "Po", "null"], [5,4,3,2,1,0], inplace = True)

all_data["Fence"].replace(["GdPrv", "MnPrv", "GdWo", "MnWw","null"], [4,3,2,1,0], inplace = True)







encoded_columns = all_data.select_dtypes(include = "O").columns



encoded_features = []

for column in encoded_columns:

    encoded_arr = OneHotEncoder().fit_transform(all_data[column].values.reshape(-1, 1)).toarray()

    n = all_data[column].nunique()

    cols = ['{}_{}'.format(column, n) for n in range(1, n + 1)]

    encoded_df = pd.DataFrame(encoded_arr, columns=cols)

    encoded_df.index = all_data.index

    encoded_features.append(encoded_df)

all_data = pd.concat([all_data, *encoded_features[:38]], axis=1)



all_data.drop(columns = encoded_columns , inplace = True)



train_data = all_data[all_data["SalePrice"].notnull()]

test_data = all_data[all_data["SalePrice"].isna()]



train_data.set_index(train_data["Id"], inplace = True)

test_data.set_index(test_data["Id"], inplace = True)



train_data.drop(columns= ["Id"], inplace = True)

test_data.drop(columns = ["Id"], inplace = True)
X = train_data.drop(columns = ["SalePrice"])

y = train_data["SalePrice"]



test_data.drop(columns = ["SalePrice"], inplace = True)

from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

n_folds = 5

kf = StratifiedKFold(n_folds, shuffle=True, random_state=42).get_n_splits(train_data)

def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train_data)

    rmse= np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler, StandardScaler

import xgboost as xgb

import lightgbm as lgb

lin_reg = LinearRegression()



score = rmsle_cv(lin_reg)

print(score.mean(), score.std())



lin_reg.fit(X, y)
lasso = make_pipeline(RobustScaler(), LassoCV(cv=kf, random_state=0))



score = rmsle_cv(lasso)

print(score.mean(), score.std())

lasso.fit(X, y)
elastic = make_pipeline(RobustScaler(),ElasticNetCV(cv = kf, random_state = 0))



score = rmsle_cv(elastic)

print(score.mean(), score.std())



elastic.fit(X, y)
ker_ridge = make_pipeline(RobustScaler(),KernelRidge(alpha=0.55, kernel='polynomial', degree=3, coef0=2))



score = rmsle_cv(ker_ridge)

print(score.mean(), score.std())



ker_ridge.fit(X, y)
model_xgb = xgb.XGBRegressor(verbosity = 0, eta = 0.3, max_depth= 3, colsample_bytree = 0.095, reg_lambda =  0.1, num_parallel_tree = 30

                             , tree_method = 'exact', random_state = 0 ) 



model_xgb = make_pipeline(StandardScaler(),model_xgb)



score = rmsle_cv(model_xgb)

print(score.mean(), score.std())

model_xgb.fit(X, y)
paired_preds = pd.DataFrame(data = {

"elastic_pred" : np.expm1(elastic.predict(test_data)),

"lasso_pred" : np.expm1(lasso.predict(test_data)),

"ker_ridge_pred" : np.expm1(ker_ridge.predict(test_data)),

"model_xgb_pred" : np.expm1(model_xgb.predict(test_data)),

"lin_reg_pred" : np.expm1(lin_reg.predict(test_data))

})



sns.pairplot(paired_preds)
def blended_predictions(X):

    return (

            (0.6 * model_xgb.predict(X)) +\

            (0.4 * ker_ridge.predict(X)) 

)
y_pred = np.expm1(blended_predictions(test_data))



submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

submission["SalePrice"] = y_pred

submission.set_index("Id", inplace = True)

submission.to_csv('submission.csv')