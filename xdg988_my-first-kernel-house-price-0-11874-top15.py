import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from mlxtend.regressor import StackingRegressor
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.shape
fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
train=train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
train.shape
all_data=pd.concat([train,test],ignore_index=True)
all_data=all_data.drop(['SalePrice','Id'],axis=1)
all_data.head(5)
all_data.info()
sns.set_style("whitegrid")
sns.distplot(train['SalePrice'])
plt.show()
print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())
y_train=np.log1p(train['SalePrice'])
y_train.astype(float)
sns.set_style("whitegrid")
sns.distplot(y_train)
plt.show()
print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())
%matplotlib inline
corrmat = all_data.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)
cat_features1={'BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtQual','GarageCond','GarageFinish','GarageQual','GarageType','Alley'
              ,'FireplaceQu','Fence','MiscFeature','PoolQC','MasVnrType'}
num_features1={'BsmtFinSF1','BsmtFinSF2','BsmtFullBath','BsmtHalfBath','BsmtUnfSF','TotalBsmtSF','GarageArea','GarageCars',
               'MasVnrArea'}
for c in cat_features1:
    all_data[c]=all_data[c].fillna('None')
    
for n in num_features1:
    all_data[n]=all_data[n].fillna(0)
cat_features2={'Functional','Electrical','Exterior1st','Exterior2nd','KitchenQual','MSZoning','SaleType','Utilities'}
for c in cat_features2:
    all_data[c]=all_data[c].fillna(all_data[c].mode()[0])
#num_features2={'GarageYrBlt'}
all_data['GarageYrBlt']=all_data['GarageYrBlt'].fillna(all_data['GarageYrBlt'].median())
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
all_data.info()
num_f=all_data.dtypes[all_data.dtypes!=object].index
cat_f=all_data.dtypes[all_data.dtypes==object].index
new_train=all_data[:train.shape[0]]
new_test=all_data[train.shape[0]:]
for c in cat_f:
    sns.boxplot(x=new_train[c],y=train['SalePrice'])
    plt.show()
for n in num_f:
    sns.jointplot(x=new_train[n],y=train['SalePrice'])
    plt.show()
new_features2={'MSSubClass','MoSold','YrSold'}
for n in new_features2:
    all_data[n]=all_data[n].astype(str)
cat_features2={'FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 
               'KitchenQual'}
for c in cat_features2:
    lbl=LabelEncoder()
    lbl.fit(['None','No','Po','Fa','TA','Gd','Ex']) 
    all_data[c] = lbl.transform(list(all_data[c].values))

all_data = all_data.replace({
                "BsmtFinType1": {'None':0 ,'Unf':1 ,'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6},
                "BsmtFinType2": {'None':0 ,'Unf':1 ,'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6},
                "BsmtExposure" : {"None":0,"No" : 1, "Mn" : 2, "Av": 3, "Gd" : 4},
                            })
num_f2=all_data.dtypes[all_data.dtypes!=object].index
skewed=all_data[num_f2].skew()
skewed=skewed[skewed>0.75]
skewed_feat=skewed.index

all_data[skewed.index]=np.log1p(all_data[skewed.index])
all_data=pd.get_dummies(all_data)
print(all_data.shape)
rbs_x=RobustScaler()
all_data=rbs_x.fit_transform(all_data)
#y_train=rbs_y.fit_transform(y_train.reshape(-1,1))
#y_train=y_train.ravel()
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
def cv_score(model):
    mse=-cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
    rmse=np.sqrt(mse)
    return rmse
base_models=[LinearRegression(),Ridge(),Lasso(), DecisionTreeRegressor(),SVR(),LinearSVR(),RandomForestRegressor(),
            GradientBoostingRegressor(),XGBRegressor(), LGBMRegressor()]
names=['lr','ridge','lasso','dtr','svr','lsvr','rfg','gdbt','xgbr','lgbr']
for name, model in zip(names, base_models):
    score = cv_score(model)
    print("{}: {:.6f}, {:.4f}".format(name,score.mean(),score.std()))
ridge=Ridge(alpha=32.9363)

score = cv_score(ridge)
print("{}: {:.6f}, {:.4f}".format(ridge,score.mean(),score.std()))
lasso=Lasso(alpha=0.0001)

score = cv_score(lasso)
print("{}: {:.6f}, {:.4f}".format(lasso,score.mean(),score.std()))
linearSVR=LinearSVR(C=1.8644, epsilon=0.0047)

score = cv_score(linearSVR)
print("{}: {:.6f}, {:.4f}".format(linearSVR,score.mean(),score.std()))
svr=SVR(C=100, epsilon=0.001, gamma=0.0001)

score = cv_score(svr)
print("{}: {:.6f}, {:.4f}".format(svr,score.mean(),score.std()))
rfr=RandomForestRegressor(max_features=0.3938, min_samples_split=2, n_estimators=249)

score = cv_score(rfr)
print("{}: {:.6f}, {:.4f}".format(rfr,score.mean(),score.std()))
gbdt=GradientBoostingRegressor(n_estimators=249, min_samples_split=21, max_features=0.3671,alpha=0.6999,max_depth=5)

score = cv_score(gbdt)
print("{}: {:.6f}, {:.4f}".format(gbdt,score.mean(),score.std()))
xgbr=XGBRegressor(min_child_weight=19,
                 colsample_bytree=0.3502,
                 max_depth=5,
                 subsample=0.9616,
                 gamma=0.0455,
                 reg_alpha=0.0307)

score = cv_score(xgbr)
print("{}: {:.6f}, {:.4f}".format(xgbr,score.mean(),score.std()))
lgbr= LGBMRegressor(min_child_weight=9,
                 colsample_bytree=0.6885,
                 num_leaves=327,
                 subsample=0.6977,
                 min_split_gain=0.0042,
                 reg_alpha=0.3788)

score = cv_score(lgbr)
print("{}: {:.6f}, {:.4f}".format(lgbr,score.mean(),score.std()))
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
average_models=AveragingModels(models=(ridge, lasso, svr, gbdt))

score = cv_score(average_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
stregr1 = StackingRegressor(regressors=[ridge, gbdt, svr,lasso], 
                           meta_regressor=xgbr)

score1 = cv_score(stregr1)
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score1.mean(), score1.std()))
stregr2 = StackingRegressor(regressors=[ridge, gbdt, svr,lasso], 
                           meta_regressor=lgbr)

score2 = cv_score(stregr2)
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score2.mean(), score2.std()))
lasso.fit(X_train,y_train)
lasso_pred=lasso.predict(X_test)
xgbr.fit(X_train,y_train)
xgbr_y_predict=xgbr.predict(X_test)
average=0.5*lasso_pred+0.5*xgbr_y_predict
average=np.expm1(average)
average=pd.DataFrame({'Id':test['Id'],'SalePrice':average})
#average.to_csv('D:/machine-learning/kaggle/houseprice/average1_submission.csv',index=False)
stregr2.fit(X_train,y_train)
stregr2_pred=stregr2.predict(X_test).astype(float)
stregr2_pred=np.expm1(stregr2_pred)
stregr2_pred=pd.DataFrame({'Id':test['Id'],'SalePrice':stregr2_pred})
#stregr2_pred.to_csv('D:/machine-learning/kaggle/houseprice/stregr2_submission.csv',index=False)
#xgbr=XGBRegressor()
xgbr.fit(X_train,y_train)
xgbr_y_predict=xgbr.predict(X_test).astype(float)
xgbr_y_predict=np.expm1(xgbr_y_predict)
xgbr_submission=pd.DataFrame({'Id':test['Id'],'SalePrice':xgbr_y_predict})
#xgbr_submission.to_csv('D:/machine-learning/kaggle/houseprice/xgbr_submission2.csv',index=False)
gbdt.fit(X_train,y_train)
gbdt_pred=gbdt.predict(X_test).astype(float)
gbdt_pred=np.expm1(gbdt_pred)
gbdt_pred=pd.DataFrame({'Id':test['Id'],'SalePrice':gbdt_pred})
#gbdt_pred.to_csv('D:/machine-learning/kaggle/houseprice/gbdt_submission1.csv',index=False)
#ridge=Ridge(alpha=35)
ridge.fit(X_train,y_train)
ridge_pred=ridge.predict(X_test).astype(float)
ridge_pred=np.expm1(ridge_pred)
ridge_pred=pd.DataFrame({'Id':test['Id'],'SalePrice':ridge_pred})
#ridge_pred.to_csv('D:/machine-learning/kaggle/houseprice/ridge_submission.csv',index=False)
#svr=SVR(kernel='rbf',C=15, epsilon=0.009, gamma=0.0004)
svr.fit(X_train,y_train)
svr_pred=svr.predict(X_test).astype(float)
svr_pred=np.expm1(svr_pred)
svr_pred=pd.DataFrame({'Id':test['Id'],'SalePrice':svr_pred})
#svr_pred.to_csv('D:/machine-learning/kaggle/houseprice/svr_submission.csv',index=False)
#lasso=Lasso(alpha=0.0005, max_iter=10000)
lasso.fit(X_train,y_train)
lasso_pred=lasso.predict(X_test).astype(float)
lasso_pred=np.expm1(lasso_pred)
lasso_pred=pd.DataFrame({'Id':test['Id'],'SalePrice':lasso_pred})
#lasso_pred.to_csv('D:/machine-learning/kaggle/houseprice/lasso_submission.csv',index=False)
lgbr.fit(X_train,y_train)
lgbr_y_predict=lgbr.predict(X_test).astype(float)
lgbr_y_predict=np.expm1(lgbr_y_predict)
lgbr_submission=pd.DataFrame({'Id':test['Id'],'SalePrice':lgbr_y_predict})
#lgbr_submission.to_csv('D:/machine-learning/kaggle/houseprice/lgbr_submission.csv',index=False)
%store X_train
%store y_train