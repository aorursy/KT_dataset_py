import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
pd.options.display.max_columns = 100

pd.options.display.max_rows = 100
data = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
data.tail()
data.shape
data.info()
data.columns
data.describe()
mis_val_features = [feature for feature in data.columns if data[feature].isnull().any()]

len(mis_val_features)
def Missing_Values(data):

    nan_value = data.isnull().sum()

    nan_value_percent = 100*nan_value / len(data)

    nan_Dataframe = pd.concat([nan_value , nan_value_percent], axis = 1)

    DataFrame = nan_Dataframe.rename(columns = {0:"Missing Values" , 1:"Missing Values %"}).sort_values(by = "Missing Values %" ,ascending = False)

    return DataFrame
Missing_Values(data).head(19)
plt.style.use("ggplot")

for feature in mis_val_features:

    datacopy = data.copy()

    datacopy[feature] = np.where(datacopy[feature].isnull() , "Miss" , "Real")

    datacopy.groupby(feature)["SalePrice"].median().plot.bar()

    plt.title(feature)

    plt.show()
num_features = [feature for feature in data.columns if data[feature].dtypes != "O" and feature not in ["Id"]]

print(len(num_features))

miss_num = [feature for feature in num_features if data[feature].isnull().any()]

print(miss_num)
year_features = [feature for feature in num_features if "Yr" in feature or "Year" in feature]

print(len(year_features))

miss_year = [feature for feature in year_features if data[feature].isnull().any()]

print(miss_year)
cat_features = [feature for feature in data.columns if data[feature].dtypes == 'O']

print(len(cat_features))

miss_cat = [feature for feature in cat_features if data[feature].isnull().any()]

print(miss_cat)
#len(num_features) - len(year_features) + len(cat_features) + len(year_features) + len(["Id"] == len(data.columns)
data[year_features].nunique()
for feature in year_features:

    datacopy = data.copy()

    datacopy.groupby(feature)["SalePrice"].median().plot()

    plt.show()
datacopy = data.copy()

datacopy["Yrold"] = datacopy.YrSold - datacopy.YearBuilt
sns.scatterplot( x = datacopy["Yrold"] , y = datacopy["SalePrice"] , data = datacopy)

plt.show()
discrete_features = [feature for feature in num_features if (data[feature].nunique() < 25)

                    and feature not in year_features]

print(len(discrete_features))

miss_discrete = [feature for feature in discrete_features if data[feature].isnull().any()]

print(miss_discrete)
data[discrete_features].head()
plt.style.use("fivethirtyeight")

for feature in discrete_features:

    datacopy = data.copy()

    datacopy.groupby(feature)["SalePrice"].median().plot.bar()

    plt.xlabel(feature)

    plt.ylabel('SalePrice')

    plt.title(feature)

    plt.show()
conti_features = [feature for feature in num_features if feature not in discrete_features+year_features]

print(len(conti_features))

miss_conti = [feature for feature in conti_features if data[feature].isnull().any()]

print(miss_conti)
data[conti_features].head()
for feature in conti_features:

    datacopy = data.copy()

    sns.distplot(datacopy[feature] , hist = True , bins= 50 , kde = False , rug = False)

    plt.xlabel(feature)

    plt.ylabel("Count")

    plt.title(feature)

    plt.show()
#len(discrete_features) + len(conti_features) + len(year_features) + len(["Id"]) == len(num_features)
#log transformation
for feature in conti_features:

    datacopy = data.copy()

    datacopy[feature] = np.log1p(datacopy[feature])

    sns.distplot(datacopy[feature] , hist = True , bins = 50 , kde = False)

    plt.xlabel(feature)

    plt.ylabel("Count")

    plt.title(feature)

    plt.show()

    
for feature in conti_features:

    datacopy = data.copy()

    datacopy[feature] = np.log1p(datacopy[feature])

    sns.scatterplot(x = datacopy[feature] ,y = datacopy["SalePrice"])

    plt.xlabel(feature)

    plt.ylabel("Sale Price")

    plt.title(feature)

    plt.show()
#Outliers
for feature in conti_features:

    datacopy = data.copy()

    datacopy[feature]=np.log1p(datacopy[feature])

    sns.boxplot(datacopy[feature])

    plt.xlabel(feature)

    plt.ylabel(feature)

    plt.title(feature)

    plt.show()
data[cat_features].head()
def Categorical_Detail(data , cat_features):

    for feature in cat_features:

        print("The feature {} has {} categories".format(feature , data[feature].nunique()))
Categorical_Detail(data , cat_features)
for feature in cat_features:

    datacopy = data.copy()

    datacopy.groupby(feature)["SalePrice"].median().plot.bar()

    plt.xlabel(feature)

    plt.ylabel("Sale Price")

    plt.title(feature)

    plt.show()
Missing_Values(data).head(19)
#miss_cat , miss_num , miss_year , miss_discrete , miss_conti
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
com_data = data.append(test)
com_data
def replace_cat_features(com_data , miss_cat):

    datacopy = com_data.copy()

    datacopy[miss_cat] = datacopy[miss_cat].fillna("Missing")

    return datacopy

com_data = replace_cat_features(com_data , miss_cat)

com_data[miss_cat].isnull().sum()
for feature in miss_num:

    median_val = com_data[feature].median()

    

    com_data[feature+'NaN'] = np.where(com_data[feature].isnull() , 1 ,0)

    com_data[feature].fillna(median_val , inplace = True)

com_data[miss_num].isnull().sum()
for feature in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:

    com_data[feature] = com_data["YrSold"] - com_data[feature]
print(com_data.shape)

com_data.head()
#log normal distribution

num_features_x =['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea']



for feature in num_features_x:

    data[feature] = np.log(data[feature])

    
np.isfinite(com_data[num_features_x]).sum()
com_data.head(2)
data["SalePrice"].corr(data["GrLivArea"])      
#Handling Rare Variables

for feature in cat_features:

    var = data.groupby(feature)["GrLivArea"].count() / len(data)

    var_data = var[var > 0.01].index

    data[feature] = np.where(data[feature].isin(var_data) , data[feature] , "Rare_Var")
com_data.head(50)
com_data.tail(10)
cat_features
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for feature in cat_features:

    if feature in com_data.columns:

        i = com_data.columns.get_loc(feature)

        com_data.iloc[:,i] = com_data.apply(lambda i:le.fit_transform(i.astype(str)), axis=0, result_type='expand')   
com_data.head(5)
com_data.Alley.value_counts()
scaling_features = [feature for feature in com_data.columns if feature not in ["Id" , "SalePrice"]]

len(scaling_features)
com_data.shape
(com_data.isnull().sum()).nlargest(10)
for feature in ["BsmtFinSF1", "BsmtFinSF2"  , "BsmtUnfSF"  , "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath", "GarageCars" , "GarageArea"]:

    median = com_data[feature].median()

    

    com_data[feature] = com_data[feature].fillna(median)
X = com_data.iloc[:1460, : ].drop(columns = ["Id" ,"SalePrice"])

Y = com_data[:1460]["SalePrice"].values
X.shape , Y.shape
from sklearn.model_selection import train_test_split

xtrain , xtest , ytrain , ytest = train_test_split(X , Y , test_size = 0.15 , random_state = 0)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit_transform(X)
xtrain
X.shape , Y.shape
from sklearn.linear_model import LassoCV , Lasso
from sklearn.feature_selection import SelectFromModel
reg = LassoCV()

reg.fit((scaler.transform(X)) ,Y)

print("Best alpha using built-in LassoCV: %f" % reg.alpha_)

print("Best score using built-in LassoCV: %f" %reg.score(xtrain,ytrain))
sel_ = SelectFromModel(Lasso(alpha = 122.214606 , random_state = 1))

sel_.fit((scaler.transform(X)), Y)
sel_.get_support()
selected_feat = X.columns[(sel_.get_support())]

print('total features: {}'.format((X.shape[1])))

print('selected features: {}'.format(len(selected_feat)))

print('features with coefficients shrank to zero: {}'.format(

      np.sum(sel_.estimator_.coef_ == 0)))
X_ = X[selected_feat].values
from sklearn.preprocessing import MinMaxScaler

scalar = MinMaxScaler()

X_ = scalar.fit_transform(X_)
Y
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score , mean_squared_error
# Using Gridsearch Cv performed below

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',

                      max_depth=None, max_features=11, max_leaf_nodes=None,

                      max_samples=None, min_impurity_decrease=0.0,

                      min_impurity_split=None, min_samples_leaf=1,

                      min_samples_split=2, min_weight_fraction_leaf=0.0,

                      n_estimators=100, n_jobs=None, oob_score=False,

                      random_state=None, verbose=0, warm_start=False)

rfr.fit( X_ , Y)

#pred = rfr.predict(xtest_)
#mean_squared_error(ytest , pred , squared = False )
score = cross_val_score(rfr , X_ , Y , cv = 6)
score.mean()
#uncomment to run below Cells
parameters = {"max_features" : np.arange(7 ,13 ,2) , "n_estimators" : np.arange(100 , 500 , 50)}
grid_r = GridSearchCV(rfr , param_grid = parameters , cv = 5 )

grid_r.fit(X_ , Y)
grid_r.best_score_
grid_r.best_estimator_
import xgboost as xgb
# selected using RandomisedSearcv

clf = xgb.XGBRegressor(max_depth= 5,

    n_estimators =  3000, 

    learning_rate=  0.1,

    subsample =  0.5,

    colsample_bytree = 0.7,

    min_child_weight= 1.5,

    reg_alpha =  0.75,

    reg_lambda= 0.4,

    seed =  42,)

clf.fit( X_ , Y)
#pred = clf.predict(xtest_)
#mean_squared_error(ytest , pred , squared = False)
scoreg = cross_val_score(clf , X_ , Y , cv = 6)
scoreg.mean()
XTEST = com_data[1460:].drop(columns = ["SalePrice" , "Id"])
XTEST = XTEST[selected_feat].values
XTEST_ = scalar.transform(XTEST)
XTEST_
ypred = rfr.predict(XTEST_) #randomforest
y_pred = clf.predict(XTEST_) #xgboost
#submission_l = pd.DataFrame(pd.read_csv("house_test.csv")['Id'])

#submission_l['SalePrice'] = y_pred.astype('int32')

#submission_l.to_csv("Submission_l1.csv", index = False)  #xgboost
# xgboost performs better than others with this kind of feature engineering , would love to improve further.