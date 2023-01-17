# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#import some necessary librairies



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt  # Matlab-style plotting

import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)





from scipy import stats

from scipy.stats import norm, skew #for some statistics





pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points





from subprocess import check_output

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
print(check_output(["ls", "../input"]).decode("utf8")) #check the files available in the directory

df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
df_train.columns
#check the numbers of samples and features

print("The train data size before dropping Id feature is : {} ".format(df_train.shape))

print("The test data size before dropping Id feature is : {} ".format(df_test.shape))



#Save the 'Id' column

train_ID = df_train['Id']

test_ID = df_test['Id']



#Now drop the  'Id' colum since it's unnecessary for  the prediction process.

df_train.drop("Id", axis = 1, inplace = True)

df_test.drop("Id", axis = 1, inplace = True)



#check again the data size after dropping the 'Id' variable

print("\nThe train data size after dropping Id feature is : {} ".format(df_train.shape)) 

print("The test data size after dropping Id feature is : {} ".format(df_test.shape))
sns.distplot(df_train['SalePrice'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(df_train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)

plt.show()



print("Skewness before log transform = " + str(df_train['SalePrice'].skew()))

print("Kurtosis before log transform = " + str(df_train['SalePrice'].kurt()))
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column

df_train["SalePrice"] = np.log1p(df_train["SalePrice"])



#Check the new distribution 

sns.distplot(df_train['SalePrice'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(df_train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)

plt.show()



print("Skewness after after log transform = " + str(df_train['SalePrice'].skew()))

print("Kurtosis after after log transform = " + str(df_train['SalePrice'].kurt()))
df_train = df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index)

ntrain = df_train.shape[0]

ntest = df_test.shape[0]

y_train = df_train.SalePrice.values

all_data = pd.concat((df_train, df_test)).reset_index(drop=True)

all_data.drop(['SalePrice'], axis=1, inplace=True)

print("all_data size is : {}".format(all_data.shape))
plt.style.use('seaborn')

sns.set_style('whitegrid')



plt.subplots(0,0,figsize=(15,3))





all_data.isnull().mean().sort_values(ascending=False).plot.bar(color='black')

plt.axhline(y=0.1, color='r', linestyle='-')

plt.title('(Pre Removal)Missing values average per column --->', fontsize=20, weight='bold' )

plt.show()





all_data["PoolQC"] = all_data["PoolQC"].fillna("None")

all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")

all_data["Alley"] = all_data["Alley"].fillna("None")

all_data["Fence"] = all_data["Fence"].fillna("None")

all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))



plt.subplots(0,0,figsize=(15,3))

all_data.isnull().mean().sort_values(ascending=False).plot.bar(color='green')

plt.axhline(y=0.1, color='r', linestyle='-')

plt.title('(Post Removal)Missing values average per column --->', fontsize=20, weight='bold' )

plt.show()



NA=all_data[['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','GarageYrBlt','BsmtFinType2','BsmtFinType1','BsmtCond', 'BsmtQual','BsmtExposure', 'MasVnrArea','MasVnrType','Electrical','MSZoning','BsmtFullBath','BsmtHalfBath','Utilities','Functional','Exterior1st','BsmtUnfSF','Exterior2nd','TotalBsmtSF','GarageArea','GarageCars','KitchenQual','BsmtFinSF2','BsmtFinSF1','SaleType']]
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data.head(20)
NAcatTest=all_data.select_dtypes(include='object')

NAnumTest=all_data.select_dtypes(exclude='object')

print('We have :',NAcatTest.shape[1],'categorical features with missing values')

print('We have :',NAnumTest.shape[1],'numerical features with missing values')
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    all_data[col] = all_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    all_data[col] = all_data[col].fillna('None')

all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")

all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

all_data = all_data.drop(['Utilities'], axis=1)

all_data["Functional"] = all_data["Functional"].fillna("Typ")

all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data.head(20)
fig = plt.figure(figsize=(20,20))

ax1 = plt.subplot2grid((6,2),(0,0))

plt.scatter(x=df_train['GrLivArea'], y=df_train['SalePrice'], color=('yellowgreen'), alpha=0.5)

plt.axvline(x=4600, color='r', linestyle='-')

plt.title('Ground living Area- Price scatter plot', fontsize=15, weight='bold' )



ax1 = plt.subplot2grid((6,2),(0,1))

plt.scatter(x=df_train['TotalBsmtSF'], y=df_train['SalePrice'], color=('red'),alpha=0.5)

plt.axvline(x=5900, color='r', linestyle='-')

plt.title('Basement Area - Price scatter plot', fontsize=15, weight='bold' )



ax1 = plt.subplot2grid((6,2),(1,0))

plt.scatter(x=df_train['1stFlrSF'], y=df_train['SalePrice'], color=('deepskyblue'),alpha=0.5)

plt.axvline(x=4000, color='r', linestyle='-')

plt.title('First floor Area - Price scatter plot', fontsize=15, weight='bold' )



ax1 = plt.subplot2grid((6,2),(1,1))

plt.scatter(x=df_train['MasVnrArea'], y=df_train['SalePrice'], color=('gold'),alpha=0.9)

plt.axvline(x=1500, color='r', linestyle='-')

plt.title('Masonry veneer Area - Price scatter plot', fontsize=15, weight='bold' )



ax1 = plt.subplot2grid((6,2),(2,0))

plt.scatter(x=df_train['GarageArea'], y=df_train['SalePrice'], color=('orchid'),alpha=0.5)

plt.axvline(x=1230, color='r', linestyle='-')

plt.title('Garage Area - Price scatter plot', fontsize=15, weight='bold' )



ax1 = plt.subplot2grid((6,2),(2,1))

plt.scatter(x=df_train['TotRmsAbvGrd'], y=df_train['SalePrice'], color=('tan'),alpha=0.9)

plt.axvline(x=13, color='r', linestyle='-')

plt.title('TotRmsAbvGrd - Price scatter plot', fontsize=15, weight='bold' )



ax1 = plt.subplot2grid((6,2),(3,0))

plt.scatter(x=df_train['EnclosedPorch'], y=df_train['SalePrice'], color=('crimson'),alpha=0.5)

plt.axvline(x=400, color='r', linestyle='-')

plt.title('EnclosedPorch - Price scatter plot', fontsize=15, weight='bold' )



ax1 = plt.subplot2grid((6,2),(3,1))

plt.scatter(x=df_train['OpenPorchSF'], y=df_train['SalePrice'], color=('gray'),alpha=0.5)

plt.axvline(x=470, color='r', linestyle='-')

plt.title('OpenPorchSF - Price scatter plot', fontsize=15, weight='bold' )



ax1 = plt.subplot2grid((6,2),(4,0))

plt.scatter(x=df_train['LotArea'], y=df_train['SalePrice'], color=('skyblue'),alpha=0.9)

plt.axvline(x=90000, color='r', linestyle='-')

plt.title('LotArea  - Price scatter plot', fontsize=15, weight='bold' )



ax1 = plt.subplot2grid((6,2),(4,1))

plt.scatter(x=df_train['BedroomAbvGr'], y=df_train['SalePrice'], color=('mediumorchid'),alpha=0.5)

plt.axvline(x=7.5, color='r', linestyle='-')

plt.title('BedroomAbvGr - Price scatter plot', fontsize=15, weight='bold' )

plt.tight_layout(0.85)

plt.xticks(weight='bold')

plt.show()
#correlation matrix

corrmat = df_train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True)



#saleprice correlation matrix

k = 8#number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#scatterplot

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(df_train[cols], size = 2.5)

plt.show()
Num=corrmat['SalePrice'].sort_values(ascending=False).head(10).to_frame()

cm = sns.light_palette("cyan", as_cmap=True)



corrplt = Num.style.background_gradient(cmap=cm)

corrplt
#missing data

total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
from sklearn.preprocessing import LabelEncoder

cols = ('BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')

# process columns, apply LabelEncoder to categorical features

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(all_data[c].values)) 

    all_data[c] = lbl.transform(list(all_data[c].values))



# shape        

print('Shape all_data: {}'.format(all_data.shape))
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



# Check the skew of all numerical features

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(10)
skewness = skewness[abs(skewness) > 0.75]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))



from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.5

for feature in skewed_features:

    all_data[feature] = boxcox1p(all_data[feature], lam)
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



# Check the skew of all numerical features

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(10)
all_data = pd.get_dummies(all_data)

print(all_data.shape)
df_train = all_data[:ntrain]

df_test = all_data[ntrain:]
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb

#Validation function

n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(df_train.values)

    rmse= np.sqrt(-cross_val_score(model, df_train.values, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

score = rmsle_cv(lasso)

print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
score = rmsle_cv(ENet)

print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

score = rmsle_cv(KRR)

print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.005,

                                   max_depth=4, max_features='sqrt',

                                 min_samples_leaf=15, min_samples_split=5, 

                                   loss='huber', random_state =5)

score = rmsle_cv(GBoost)

print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)

score = rmsle_cv(model_xgb)

print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

score = rmsle_cv(model_lgb)

print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, models):

        self.models = models

        

    # we define clones of the original models to fit the data in

    def fit(self, X, y):

        self.models_ = [clone(x) for x in self.models]

        

        # Train cloned base models

        for model in self.models_:

            model.fit(X, y)



        return self

    

    #Now we do the predictions for cloned models and average them

    def predict(self, X):

        predictions = np.column_stack([

            model.predict(X) for model in self.models_

        ])

        return np.mean(predictions, axis=1)   

    
averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))



score = rmsle_cv(averaged_models)

print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, base_models, meta_model, n_folds=5):

        self.base_models = base_models

        self.meta_model = meta_model

        self.n_folds = n_folds

   

    # We again fit the data on clones of the original models

    def fit(self, X, y):

        self.base_models_ = [list() for x in self.base_models]

        self.meta_model_ = clone(self.meta_model)

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        

        # Train cloned base models then create out-of-fold predictions

        # that are needed to train the cloned meta-model

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):

            for train_index, holdout_index in kfold.split(X, y):

                instance = clone(model)

                self.base_models_[i].append(instance)

                instance.fit(X[train_index], y[train_index])

                y_pred = instance.predict(X[holdout_index])

                out_of_fold_predictions[holdout_index, i] = y_pred

                

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature

        self.meta_model_.fit(out_of_fold_predictions, y)

        return self

   

    #Do the predictions of all base models on the test data and use the averaged predictions as 

    #meta-features for the final prediction which is done by the meta-model

    def predict(self, X):

        meta_features = np.column_stack([

            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)

            for base_models in self.base_models_ ])

        return self.meta_model_.predict(meta_features)
stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),

                                                 meta_model = lasso)



score = rmsle_cv(stacked_averaged_models)

print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
stacked_averaged_models.fit(df_train.values, y_train)

stacked_train_pred = stacked_averaged_models.predict(df_train.values)

stacked_pred = np.expm1(stacked_averaged_models.predict(df_test.values))

print(rmsle(y_train, stacked_train_pred))
model_xgb.fit(df_train, y_train)

xgb_train_pred = model_xgb.predict(df_train)

xgb_pred = np.expm1(model_xgb.predict(df_test))

print(rmsle(y_train, xgb_train_pred))
model_lgb.fit(df_train, y_train)

lgb_train_pred = model_lgb.predict(df_train)

lgb_pred = np.expm1(model_lgb.predict(df_test.values))

print(rmsle(y_train, lgb_train_pred))
ensemble = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15
sub = pd.DataFrame()

sub['Id'] = test_ID

sub['SalePrice'] = ensemble

sub.to_csv('submission.csv',index=False)

print("Submitted Successfully ...!")