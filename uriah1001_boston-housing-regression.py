import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
%matplotlib inline

import warnings 
warnings.filterwarnings("ignore")
#Loading the DataSet

train = pd.read_csv("/kaggle/input/train.csv")
test = pd.read_csv("/kaggle/input/test.csv")


print(train.columns)

print("="*100)

print(train.info())

print("="*100)

print(train.describe())
Id = test.Id
plt.figure(figsize=(20,20))

sns.heatmap(train.corr() , cmap = "RdGy")
plt.scatter(train["SalePrice"] , train["GrLivArea"])
plt.title("PLoting SalePrice(x) against GrlivArea(y)")
plt.xlabel("SalePrice")
plt.ylabel("GrLivArea")

print("GrlivArea -> Space of the living area (sqr feet)")

print("Cutoff Outliers that are above 4000 GrLivArea")
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

plt.scatter(train["SalePrice"] , train["GrLivArea"])
plt.title("PLoting SalePrice(x) against GrlivArea(y)")
plt.xlabel("SalePrice")
plt.ylabel("GrLivArea")
plt.scatter(train["TotalBsmtSF"] , train["SalePrice"])
plt.title("PLoting SalePrice(y) against TotalBsmtSF(x)")
plt.ylabel("SalePrice")
plt.xlabel("TotalBsmtSF")
plt.scatter(train["OverallQual"] , train["SalePrice"])
sns.distplot(train["SalePrice"])

from scipy.stats import boxcox
transformed_sales = boxcox(train["SalePrice"])
sns.distplot(transformed_sales[0])
features = pd.concat((train, test) , keys = ["Train" , "Test"] , axis = 0)

train_number = train.shape[0]

test_number = test.shape[0]

y = boxcox(train["SalePrice"])[0]

lambda_ = boxcox(train["SalePrice"])[1]

features.pop("SalePrice")

Missing_Values_Train = train.isnull().sum()

Missing_Values_Test = test.isnull().sum()

Missing_Values = pd.concat([Missing_Values_Train , Missing_Values_Test], axis = 1 , keys=["Train" , "Test"])

Missing_Values = Missing_Values[Missing_Values.sum(axis=1)>0]

Missing_Ratio = (features.isnull().sum()/features.shape[0])

Missing_Ratio = Missing_Ratio.drop(Missing_Ratio[Missing_Ratio == 0 ].index)
Missing_Ratio.sort_values(ascending=False)

features["PoolQC"] = features["PoolQC"].fillna("No Pool")

features["MiscFeature"] = features["MiscFeature"].fillna("No Features")

features["Alley"] = features["Alley"].fillna("No Alley")

features["Fence"] = features["Fence"].fillna("No Fence")

features["FireplaceQu"] = features["FireplaceQu"].fillna("No FirePlace")

features["LotFrontage"] = features.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.mean()))

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    features[col] = features[col].fillna('None')
    
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    features[col] = features[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    features[col] = features[col].fillna(0)
    
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    features[col] = features[col].fillna('None')

features["MasVnrType"] = features["MasVnrType"].fillna("None")

features["MasVnrArea"] = features["MasVnrArea"].fillna(0)

features['MSZoning'] = features['MSZoning'].fillna(features['MSZoning'].mode()[0])

features = features.drop(['Utilities'], axis=1)

features["Functional"] = features["Functional"].fillna("Typical")

features['Electrical'] = features['Electrical'].fillna(features['Electrical'].mode()[0])

features['KitchenQual'] = features['KitchenQual'].fillna(features['KitchenQual'].mode()[0])

features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])

features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])

features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])

features['MSSubClass'] = features['MSSubClass'].fillna("None")
Missing_Values_Train = train.isnull().sum()

Missing_Values_Test = test.isnull().sum()

Missing_Values = pd.concat([Missing_Values_Train , Missing_Values_Test], axis = 1 , keys=["Train" , "Test"])

Missing_Values = Missing_Values[Missing_Values.sum(axis=1)>0]

Missing_Ratio = (features.isnull().sum()/features.shape[0])

Missing_Ratio = Missing_Ratio.drop(Missing_Ratio[Missing_Ratio == 0 ].index)

Missing_Ratio.sort_values(ascending = False)

features['MSSubClass'] = features['MSSubClass'].apply(str)

features['OverallCond'] = features['OverallCond'].astype(str)

features['YrSold'] = features['YrSold'].astype(str)

features['MoSold'] = features['MoSold'].astype(str)
from sklearn.preprocessing import LabelEncoder


cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')

for i in cols:
    print(i + " is doing LabelEncoding")
    le = LabelEncoder()
    le.fit(list(features[i].values))
    
    features[i] = le.transform(list(features[i].values))
  

from scipy.special import boxcox1p

numeric_feats = features.dtypes[features.dtypes != "object"].index
features[numeric_feats] = boxcox1p(features[numeric_feats] , 0.2)

features = pd.get_dummies(features)

print(features.shape)

features['TotalSF'] = features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']

features['Total_sqr_footage'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +
                                 features['1stFlrSF'] + features['2ndFlrSF'])

features['Total_Bathrooms'] = (features['FullBath'] + (0.5*features['HalfBath']) + 
                               features['BsmtFullBath'] + (0.5*features['BsmtHalfBath']))

features['Total_porch_sf'] = (features['OpenPorchSF'] + features['3SsnPorch'] +
                              features['EnclosedPorch'] + features['ScreenPorch'] +
                             features['WoodDeckSF'])

features["-_TotalHouse_LotArea"] = features["TotalSF"] + features["LotArea"]

features["+_TotalHouse_OverallQual"] = features["TotalSF"] * features["OverallQual"]

features["+_GrLivArea_OverallQual"] = features["GrLivArea"] * features["OverallQual"]

features["-_LotArea_OverallQual"] = features["LotArea"] * features["OverallQual"]


loglist = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
                 'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea',
                 'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',
                 'TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF',
                 'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','YearRemodAdd','TotalSF']

m = features.shape[1]

for l in loglist:
        features = features.assign(newcol=pd.Series(np.log(1.01+features[l])).values)   
        features.columns.values[m] = l + '_log'
        m += 1
training = features[:train.shape[0]]
testing = features[train.shape[0]:]
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold , cross_val_score ,train_test_split
from sklearn.pipeline import make_pipeline
from sklearn import linear_model
from sklearn.linear_model import ElasticNetCV , LassoCV , RidgeCV 
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor 
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.kernel_ridge import KernelRidge
from sklearn.base import BaseEstimator , RegressorMixin , TransformerMixin , clone
from sklearn.metrics import mean_squared_error
X_train , X_test , y_train , y_test = train_test_split(training, y , test_size= 0.25 , random_state = 7)
folds = 5

def score_indicator(model):
    kf = KFold(folds, shuffle=True, random_state=42).get_n_splits(training.values)
    rmse= np.sqrt(-cross_val_score(model, training.values, y[:train.shape[0]], scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
Ridge_model = make_pipeline(RobustScaler() , linear_model.RidgeCV(alphas=[0.001 ,0.01 , 0.05 , 0.1 , 0.5 , 1 , 10 ],scoring="neg_mean_squared_error")).fit(X_train , y_train)
score_indicator(Ridge_model).mean()
Lass = linear_model.LassoCV(alphas=[0.0001, 0.0005,0.001, 0.01, 0.1, 1, 10]).fit(X_train , y_train)
score_indicator(Lass).mean()
ENSTest = make_pipeline(RobustScaler() , linear_model.ElasticNetCV(alphas=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10]  , l1_ratio= [0.01 ,0.05 ,0.1 , 0.5 , 0.99 ] )).fit(X_train , y_train)

score_indicator(ENSTest).mean()
KRR = KernelRidge(alpha=0.05, kernel='polynomial', degree=1, coef0=2.5)

score_indicator(KRR).mean()
GBoost = GradientBoostingRegressor(n_estimators=10000, learning_rate=0.05,
                                   max_depth= 3 , max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
score_indicator(GBoost).mean()
lgbm_model = LGBMRegressor(objective='regression',num_leaves= 3 ,
                              learning_rate=0.05, n_estimators= 2000 ,
                              max_bin = 55, bagging_fraction = 0.8 ,
                              bagging_freq = 5, feature_fraction = 0.2319 ,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

score_indicator(lgbm_model).mean()
regr = XGBRegressor(learning_rate = 0.01 ,  n_estimators=4500 , max_depth = 4 , gamma = 0 ,
                     min_child_weight=0 , subsample=0.7,
                     colsample_bytree=0.7,objective= 'reg:linear',
                     nthread=4,scale_pos_weight=1, seed=27, reg_alpha=0.00006)

score_indicator(regr).mean()
class StackingModel(BaseEstimator, RegressorMixin, TransformerMixin):
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
                
        # Now train the cloned meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)
AverageModel = StackingModel(base_models=(Lass , KRR , GBoost , Ridge_model , regr), meta_model= ENSTest)
AverageModel.fit(X_train.values, y_train)

train_prediction = AverageModel.predict(X_test.values)

print("RMSE : {} " .format(np.sqrt(mean_squared_error(train_prediction , y_test))))
plt.scatter(train_prediction , y_test)
plt.plot([7.0,8.5],[7.0,8.5], 'red', linewidth=2)
from scipy.special import inv_boxcox

results = AverageModel.predict(testing.values)

re = inv_boxcox(results , lambda_)
re
sub = pd.DataFrame()
sub["ID"] = Id 
sub["SalePrice"] = re 
#q1 = sub["SalePrice"].quantile(0.04)
#q2 = sub['SalePrice'].quantile(0.99)

#sub['SalePrice'] = sub['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)
#sub['SalePrice'] = sub['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)

sub.to_csv("Results_Simple2.csv" , index= False)
  