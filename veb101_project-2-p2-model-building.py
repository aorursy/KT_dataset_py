!pip install -qU pip
!pip install -qU xgboost

!pip install -qU lightgbm
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from scipy import stats

from scipy.stats import norm, skew

import os



from sklearn.model_selection import KFold

from sklearn.model_selection import cross_validate

from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import Lasso

from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.metrics import r2_score

from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline

from sklearn.ensemble import GradientBoostingRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from sklearn.preprocessing import RobustScaler



plt.style.use("fivethirtyeight")

pd.pandas.set_option('display.max_columns', None)

sns.set_style('darkgrid')

%matplotlib inline
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
print(train.shape)

print(test.shape)
test['Id'].values
# dropping ID



train.drop(['Id'], axis=1, inplace=True)



test_id = test['Id'].values # for submission

test.drop(['Id'], axis=1, inplace=True)
fig, ax = plt.subplots()

ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
#Deleting outliers



train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

fig, ax = plt.subplots()

ax.scatter(train['GrLivArea'], train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
sns.distplot(train['SalePrice'] , fit=norm);



(mu, sigma) = norm.fit(train['SalePrice'])

print(f'mu = {mu:.2f} and sigma = {sigma:.2f}')



#Now plot the distribution

plt.legend([f'Normal dist. ($\mu=$ {mu:.2f} and $\sigma=$ {sigma:.2f} )'],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)

plt.show()
train["SalePrice"] = np.log1p(train["SalePrice"])



sns.distplot(train['SalePrice'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(train['SalePrice'])

print(f'mu = {mu:.2f} and sigma = {sigma:.2f}')



#Now plot the distribution

plt.legend([f'Normal dist. ($\mu=$ {mu:.2f} and $\sigma=$ {sigma:.2f} )'],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)

plt.show()
ntrain = train.shape[0]

ntest = test.shape[0]

y_train = train.SalePrice.values

all_data = pd.concat((train, test)).reset_index(drop=True)

all_data.drop(['SalePrice'], axis=1, inplace=True)

print(f"all_data size is : {all_data.shape}")
features_with_na = {feature: all_data[feature].isnull().sum() for feature in all_data.columns 

                    if all_data[feature].isnull().sum() > 0}



size = all_data.shape[0]

a = pd.DataFrame({

    'features': list(features_with_na.keys()),

    'Total': list(features_with_na.values()),

    'Missing_PCT': [np.round((features_with_na[i] / size) * 100, 3) for i in features_with_na.keys()]

}).sort_values(by='Missing_PCT', ascending=False).reset_index(drop=True)

a.style.background_gradient(cmap='Reds') 
print(f"Total number of missing values: {all_data.isna().sum().sum()}")
num_with_nan = [feature for feature in features_with_na.keys() if train[feature].dtypes != 'O']

pd.DataFrame({

    'feature': num_with_nan,

    'Count': [all_data[i].isna().sum() for i in num_with_nan]

})
# LotFrontage



all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].apply(

    lambda x: x.fillna(x.median()))

all_data["LotFrontage"].isna().sum()
# MasVnrArea



all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

all_data["MasVnrArea"].isna().sum()
# BsmtX



for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 

            'BsmtFullBath', 'BsmtHalfBath'):

    all_data[col] = all_data[col].fillna(0)



all_data[['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 

         'BsmtFullBath', 'BsmtHalfBath']].isna().sum()
# GarageX



for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    all_data[col] = all_data[col].fillna(0)



all_data[['GarageYrBlt', 'GarageArea', 'GarageCars']].isna().sum()
sns.heatmap(pd.DataFrame(

    {

        'BsmtFinSF1': train['BsmtFinSF1'],

        'BsmtFinSF2': train['BsmtFinSF2'], 

        'BsmtUnfSF': train['BsmtUnfSF'],

        'TotalBsmtSF': train['TotalBsmtSF'], 

        'BsmtFullBath': train['BsmtFullBath'], 

        'BsmtHalfBath': train['BsmtHalfBath'],

        'SalePrice': train['SalePrice'],

    }

).corr(), cmap='coolwarm', annot=True) 

plt.title("Bsmt numerical features - train set")

plt.show()
sns.heatmap(pd.DataFrame(

    {

        'GarageYrBlt': train['GarageYrBlt'], 

        'GarageArea': train['GarageArea'],

        'GarageCars': train['GarageCars'],

        'SalePrice': train['SalePrice'],

    }).corr(), annot=True, cmap='coolwarm'

)

plt.title("Garage numerical features - train set")

plt.show()
to_remove_ = ['BsmtFinSF2', 'BsmtUnfSF', 'BsmtFullBath', 'BsmtHalfBath', 

             'GarageYrBlt', 'GarageArea']
all_data[num_with_nan].isna().sum()
cat_nan = [feature for feature in features_with_na if all_data[feature].dtypes == "O"]

pd.DataFrame({

    'feature': cat_nan,

    'Count': [all_data[i].isna().sum() for i in cat_nan]

}).sort_values(by="Count", ascending = False).reset_index(drop=True)
to_remove_.extend(['PoolQC','MiscFeature','Alley','Fence', 'Utilities'])
plt.figure(figsize=(10, 8))

basement_variables = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 

                      'BsmtFinType1', 'BsmtFinType2']



for i, feature in enumerate(basement_variables, 1):

    plt.subplot(3, 2, i)

    sns.boxenplot(data=train, x=feature, y=train['SalePrice'])



plt.tight_layout(h_pad=1.2)

plt.show()
plt.figure(figsize=(10, 8))

garage_variables = ['GarageType', 'GarageFinish', 'GarageQual', 

                    'GarageCond']



for i, feature in enumerate(garage_variables, 1):

    plt.subplot(2, 2, i)

    sns.boxenplot(data=train, x=feature, y=train['SalePrice'])



plt.tight_layout(h_pad=1.2)

plt.show()
fill_none = ["FireplaceQu",  "GarageType", "GarageFinish", "GarageQual", "GarageCond", 

             "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "MasVnrType"]



for feature in fill_none:

    all_data[feature].fillna("None", inplace=True)



all_data[fill_none].isna().sum().sum()
fill_mode = ["MSZoning", "Electrical", "KitchenQual", "Exterior1st", 

             "Exterior2nd", "SaleType"]



for feature in fill_mode:

    all_data[feature].fillna(all_data[feature].mode()[0], inplace=True)



all_data[fill_mode].isna().sum().sum()
all_data["Functional"] = all_data["Functional"].fillna("Typ")
print(f"Number of missing values: {all_data.isna().sum().sum()}")
# dropping features



all_data.drop(to_remove_, axis=1, inplace=True)

all_data.shape
all_data_cat = all_data.copy()

all_data_cat.shape
all_data_free = all_data.copy()

remove_cat_garage_bsmt = ["GarageType", "GarageFinish", "GarageQual", "GarageCond", 

             "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]

all_data_free.drop(remove_cat_garage_bsmt, axis=1, inplace=True)

all_data_free.shape
# MSSubClass=The building class

all_data_cat['MSSubClass'] = all_data_cat['MSSubClass'].apply(str)





# Changing OverallCond into a categorical variable

all_data_cat['OverallCond'] = all_data_cat['OverallCond'].astype(str)





# Year and month sold are transformed into categorical features.

all_data_cat['YrSold'] = all_data_cat['YrSold'].astype(str)

all_data_cat['MoSold'] = all_data_cat['MoSold'].astype(str)



# same transformation to other dataframe

# MSSubClass=The building class

all_data_free['MSSubClass'] = all_data_free['MSSubClass'].apply(str)





# Changing OverallCond into a categorical variable

all_data_free['OverallCond'] = all_data_free['OverallCond'].astype(str)





# Year and month sold are transformed into categorical features.

all_data_free['YrSold'] = all_data_free['YrSold'].astype(str)

all_data_free['MoSold'] = all_data_free['MoSold'].astype(str)
from sklearn.preprocessing import LabelEncoder





cols = ['BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 

        'CentralAir', 'ExterCond', 'ExterQual', 'FireplaceQu', 'Functional', 

        'GarageCond', 'GarageFinish', 'GarageQual', 'HeatingQC', 'KitchenQual', 

        'LandSlope', 'LotShape', 'MSSubClass', 'MoSold', 'OverallCond', 'PavedDrive', 

         'Street', 'YrSold']



for feature in cols:

    encoder = LabelEncoder()

    encoder.fit(all_data_cat[feature].values)

    all_data_cat[feature] = encoder.transform(all_data_cat[feature].values)





print(f'Shape all_data_cat: {all_data_cat.shape}')



# Same for all_data_free



cols = ['CentralAir', 'ExterCond', 'ExterQual', 'FireplaceQu', 'Functional', 

        'HeatingQC', 'KitchenQual', 'LandSlope', 'LotShape', 'MSSubClass', 

        'MoSold', 'OverallCond', 'PavedDrive', 'Street', 'YrSold']



for feature in cols:

    encoder = LabelEncoder()

    encoder.fit(all_data_free[feature].values)

    all_data_free[feature] = encoder.transform(all_data_free[feature].values)







print(f'Shape all_data_free: {all_data_free.shape}')
all_data_cat['TotalSF'] = all_data_cat['TotalBsmtSF'] + all_data_cat['1stFlrSF'] + all_data_cat['2ndFlrSF']



all_data_free['TotalSF'] = all_data_free['TotalBsmtSF'] + all_data_free['1stFlrSF'] + all_data_free['2ndFlrSF']
# all_data_cat



numeric_feature_cat = all_data_cat.select_dtypes("number").columns



# Check the skew of all numerical features

skewed_feats_cat = all_data_cat[numeric_feature_cat].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness_cat = pd.DataFrame({'Skew' :skewed_feats_cat})

skewness_cat.head(10)
# all_data_free



numeric_feature_free = all_data_free.select_dtypes("number").columns



# Check the skew of all numerical features

skewed_feats_free = all_data_free[numeric_feature_free].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness_free = pd.DataFrame({'Skew' :skewed_feats_free})

skewness_free.head(10)

# for all_data_cat



from scipy.special import boxcox1p



skewness_cat = skewness_cat[abs(skewness_cat) > 0.75]

print("There are {} skewed numerical features to Box Cox transform".format(skewness_cat.shape[0]))



skewed_features_cat = skewness_cat.index

lam = 0.15

for feat in skewed_features_cat:

    all_data_cat[feat] = boxcox1p(all_data_cat[feat], lam)
# all_data_free



skewness_free = skewness_free[abs(skewness_free) > 0.75]

print("There are {} skewed numerical features to Box Cox transform".format(skewness_free.shape[0]))



skewed_features_free = skewness_free.index

lam = 0.15

for feat in skewed_features_free:

    #all_data[feat] += 1

    all_data_free[feat] = boxcox1p(all_data_free[feat], lam)

 
print(f"all_data_cat.shape: {all_data_cat.shape}")

print(f"all_data_free.shape: {all_data_free.shape}")
all_data_free.head()
# apply min max scaler first
all_data_cat = pd.get_dummies(all_data_cat)

print(all_data_cat.shape)
all_data_free = pd.get_dummies(all_data_free)

print(all_data_free.shape)
all_data_free[numeric_feature_free].head()
train_cat = all_data_cat[:ntrain]

test_cat = all_data_cat[ntrain:]
train_free = all_data_free[:ntrain]

test_free = all_data_free[ntrain:]
train_cat.shape, test_cat.shape
train_free.shape, test_free.shape
# Using MinMaxScaler
X_cat = train_cat.copy()

cols_cat = X_cat.columns



scaler_cat = MinMaxScaler()



# fitting MinaMaxScaler to training data

scaler_cat.fit(X_cat)



# transforming training and test data

X_cat = scaler_cat.transform(X_cat)

test_cat = scaler_cat.transform(test_cat)





X_cat = pd.DataFrame(X_cat, columns=[cols_cat])

test_cat = pd.DataFrame(test_cat, columns=[cols_cat])

X_cat.shape, test_cat.shape
X_free = train_free.copy()

cols_free = X_free.columns



scaler_free = MinMaxScaler()



# fitting MinaMaxScaler to training data

scaler_free.fit(X_free)



# transforming training and test data

X_free = scaler_free.transform(X_free)

test_free = scaler_free.transform(test_free)





X_free = pd.DataFrame(X_free, columns=[cols_free])

test_free = pd.DataFrame(test_free, columns=[cols_free])

X_free.shape, test_free.shape
y = y_train
# Using SelectFromModel with lasso for selecting best features
# for X_cat

lasso = Pipeline([

    ("scaler", RobustScaler()), 

    ("ls", Lasso(alpha =0.0005, random_state=1))

])



feature_sel_model = SelectFromModel(lasso).fit(X_cat, y)



coefficiets = feature_sel_model.estimator_['ls'].coef_

X_cat_cols = []



for i, j in enumerate(coefficiets):

    if j != 0:

        X_cat_cols.append(X_cat.columns[i])





# selected_feat_cat = X_cat.columns[(feature_sel_model.get_support())]

print(len(X_cat_cols))



X_cat_lasso = X_cat[X_cat_cols].reset_index(drop=True)

test_cat_lasso = test_cat[X_cat_cols].reset_index(drop=True)



print(X_cat_lasso.shape, test_cat_lasso.shape)
# for X_free

lasso = Pipeline([

    ("scaler", RobustScaler()), 

    ("ls", Lasso(alpha =0.0005, random_state=1))

])



feature_sel_model = SelectFromModel(lasso).fit(X_free, y)



coefficiets = feature_sel_model.estimator_['ls'].coef_

X_free_cols = []



for i, j in enumerate(coefficiets):

    if j != 0:

        X_free_cols.append(X_free.columns[i])





print(len(X_free_cols))



X_free_lasso = X_free[X_free_cols].reset_index(drop=True)

test_free_lasso = test_free[X_free_cols].reset_index(drop=True)



print(X_free_lasso.shape, test_free_lasso.shape)
tree_models = {

    "Light_GBM": LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=1200,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf=6, min_sum_hessian_in_leaf = 11),

    

    "XGBoost": XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, learning_rate=0.05, 

                 max_depth=5, min_child_weight=1.7817, n_estimators=2200,

                 reg_alpha=0.4640, reg_lambda=0.8571, subsample=0.5213,

                 nthread = -1, objective="reg:squarederror", random_state=42), 

    

    "Gradient_boosting": GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, 

                                                   max_depth=6, min_samples_split=17, 

                                                   max_features='sqrt', min_samples_leaf=13, 

                                                   loss='huber', random_state=42),

               

    "Extra_trees": ExtraTreesRegressor(n_estimators=2000, max_depth=9, min_samples_split= 13, 

                        max_leaf_nodes=11, min_weight_fraction_leaf=0.39, max_features='sqrt', 

                        n_jobs=-1, random_state=42),

}
# for X_cat



print("For X_cat - Extra Trees")

selection_extra_cat = SelectFromModel(tree_models["Extra_trees"]).fit(X_cat, y_train)



selected_feat_extra_cat = X_cat.columns[(selection_extra_cat.get_support())]

print("Number of selected features:", len(selected_feat_extra_cat))



X_cat_extra = X_cat[selected_feat_extra_cat].reset_index(drop=True)

test_cat_extra = test_cat[selected_feat_extra_cat].reset_index(drop=True)

print("Transformed shape: ", X_cat_extra.shape, test_cat_extra.shape)



# uncomment next line to print selected features

# print(X_cat_extra.columns)

# for X_free



print("\nFor X_free - Extra Trees")



selection_extra_free = SelectFromModel(tree_models["Extra_trees"]).fit(X_free, y_train)



selected_feat_extra_free = X_free.columns[(selection_extra_free.get_support())]

print("Number of selected features:", len(selected_feat_extra_free))



X_free_extra = X_free[selected_feat_extra_free].reset_index(drop=True)

test_free_extra = test_free[selected_feat_extra_free].reset_index(drop=True)

print("Transformed shape: ", X_free_extra.shape, test_free_extra.shape)



# uncomment next line to print selected features

# print(X_cat_extra.columns)
def fit_model(model_name, model, X, y, n_folds=7, show=10):

    X = X.copy()

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X.values)

    output = cross_validate(model, X.values, y, scoring="neg_mean_squared_error", cv=kf, return_estimator=True)

    

    feat_scores = pd.DataFrame(index=X.columns)



    for idx,estimator in enumerate(output['estimator']):

        temp = pd.Series(estimator.feature_importances_, index=X.columns)

        feat_scores[str(idx)] = temp



    feat_scores.reset_index(inplace=True)



    feat_scores['Importance'] = feat_scores.mean(axis=1)



    feat_scores = feat_scores.sort_values(by="Importance", ascending=False)[['level_0', "Importance"]].head(show)



    plt.figure(figsize=(15, 10))

    plt.gca()

    sns.barplot(x=feat_scores['Importance'], y=feat_scores['level_0'])

    plt.ylabel("Features")

    plt.title(f"Feature Importance: {model_name}")



    print(f"{model_name} score => RMSE: {np.round(np.mean(np.sqrt(-output['test_score'])), 4)}, std: {np.round(np.std(-output['test_score']), 4)}")

    

    plt.show()

    print("-------------------------------------------------------------------------------------------------")
# print("Running boosting tree on X_cat_lasso, X_cat_free, X_cat, X_free")



# training_data = {

#     "X_cat_lasso": X_cat_lasso, "X_free_lasso": X_free_lasso,

#     "X_cat": X_cat, "X_free": X_free, 

# }





# for X_name, X in training_data.items():

#     print(f"X => {X_name}, shape: {X.shape}\n")

#     for model_name, model in tree_models.items():

#         fit_model(model_name, model, X, y, n_folds=7, show=35)

#     print()
from sklearn.ensemble import StackingRegressor

from sklearn.linear_model import LinearRegression

from sklearn.neural_network import MLPRegressor

from sklearn.svm import LinearSVR

from sklearn.metrics import mean_squared_error

from sklearn.base import clone



def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



level0 = list()

# level0.extend(('LGB', tree_models["Light_GBM"]))

level0.append(('xgb', tree_models["XGBoost"]))

level0.append(('GBT', tree_models["Gradient_boosting"]))

level0.append(('xtra', tree_models["Extra_trees"]))



training_data = {"X_cat_lasso": X_cat_lasso, "X_free_lasso": X_free_lasso,

                 "X_cat": X_cat, "X_free": X_free, 

                 }





level1_models = {

    "Linear": LinearRegression(), 

#     "Linear_SVR":  LinearSVR(epsilon=1.5),

#     "MLP": MLPRegressor(hidden_layer_sizes=(2), activation='logistic', solver='sgd', 

#                         max_iter=1000, learning_rate='adaptive', random_state=42)

}
# To see outputs, uncomment the next lines 



# for X_name, X in training_data.items():

#     print("==========================================")

#     X = X.copy()

#     print(f"X => {X_name}, shape: {X.shape}\n")



#     for name, level1 in level1_models.items():

#         level1_ = clone(level1)

#         model_ = StackingRegressor(estimators=level0, final_estimator=level1_, cv=7, n_jobs=-1)

#         model_.fit(X, y)

#         y_preds = model_.predict(X)

#         print(f"Model: {name}, R2: {r2_score(y_preds, y)}, RMSE: {rmsle(y_preds, y)}")
model_ = StackingRegressor(estimators=level0, final_estimator=level1_models['Linear'], cv=7, n_jobs=-1)

model_.fit(X_cat_lasso, y)

y_preds = model_.predict(X_cat_lasso)

print(f"Model: Linear, R2: {r2_score(y_preds, y)}, RMSE: {rmsle(y_preds, y)}")
test_predictions = model_.predict(test_cat_lasso)





sub = pd.DataFrame()





sub['Id'] = test_id

sub['SalePrice'] = test_predictions

sub.to_csv('submission.csv',index=False)