
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import ElasticNetCV,LassoCV,RidgeCV
from sklearn.ensemble import RandomForestRegressor

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense,Dropout,LeakyReLU

from keras.callbacks import ModelCheckpoint
from keras.models import load_model


from sklearn import metrics
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df_train.head()
df_train.drop('Id',axis=1,inplace=True)
def visualNA(df,perc=0):
    #Percentage of NAN Values 
    NAN = [(c, df[c].isna().mean()*100) for c in df]
    NAN = pd.DataFrame(NAN, columns=["column_name", "percentage"])
    NAN = NAN[NAN.percentage > perc]
    print(NAN.sort_values("percentage", ascending=False))
    

visualNA(df_train)
def handleNA(df):
    df['Alley'].fillna(value='No alley access',inplace=True)    
    df['BsmtQual'].fillna(value='No Basement',inplace=True)
    df['BsmtCond'].fillna(value='No Basement',inplace=True)
    df['BsmtExposure'].fillna(value='No Basement',inplace=True)
    df['BsmtFinType1'].fillna(value='No Basement',inplace=True)    
    df['BsmtFinType2'].fillna(value='No Basement',inplace=True)    
    df['FireplaceQu'].fillna(value='No Fireplace',inplace=True)    
    df['GarageType'].fillna(value='No Garage',inplace=True)  
    df['GarageYrBlt'].fillna(value=0,inplace=True)
    df['GarageFinish'].fillna(value='No Garage',inplace=True)
    df['GarageQual'].fillna(value='No Garage',inplace=True)
    df['GarageCond'].fillna(value='No Garage',inplace=True)
    df['MasVnrType'].fillna(value='None',inplace=True)
    df['MasVnrArea'].fillna(value=0.0,inplace=True)
    df['PoolQC'].fillna(value='No Pool',inplace=True)    
    df['Fence'].fillna(value='No Fence',inplace=True)
    df['MiscFeature'].fillna(value='None',inplace=True)
    
    
handleNA(df_train)

visualNA(df_train)
df_train[df_train['Electrical'].isnull()]
df_train[(df_train['Electrical'].notnull()) & (df_train['MSSubClass']==80) 
        & (df_train['LotFrontage']==73) ]['Electrical']
df_train.loc[df_train['Electrical'].isnull(),'Electrical'] = 'SBrkr'
dict_neighbor = {
'NAmes'  :{'lat': 42.045830,'lon': -93.620767},
'CollgCr':{'lat': 42.018773,'lon': -93.685543},
'OldTown':{'lat': 42.030152,'lon': -93.614628},
'Edwards':{'lat': 42.021756,'lon': -93.670324},
'Somerst':{'lat': 42.050913,'lon': -93.644629},
'Gilbert':{'lat': 42.060214,'lon': -93.643179},
'NridgHt':{'lat': 42.060357,'lon': -93.655263},
'Sawyer' :{'lat': 42.034446,'lon': -93.666330},
'NWAmes' :{'lat': 42.049381,'lon': -93.634993},
'SawyerW':{'lat': 42.033494,'lon': -93.684085},
'BrkSide':{'lat': 42.032422,'lon': -93.626037},
'Crawfor':{'lat': 42.015189,'lon': -93.644250},
'Mitchel':{'lat': 41.990123,'lon': -93.600964},
'NoRidge':{'lat': 42.051748,'lon': -93.653524},
'Timber' :{'lat': 41.998656,'lon': -93.652534},
'IDOTRR' :{'lat': 42.022012,'lon': -93.622183},
'ClearCr':{'lat': 42.060021,'lon': -93.629193},
'StoneBr':{'lat': 42.060227,'lon': -93.633546},
'SWISU'  :{'lat': 42.022646,'lon': -93.644853}, 
'MeadowV':{'lat': 41.991846,'lon': -93.603460},
'Blmngtn':{'lat': 42.059811,'lon': -93.638990},
'BrDale' :{'lat': 42.052792,'lon': -93.628820},
'Veenker':{'lat': 42.040898,'lon': -93.651502},
'NPkVill':{'lat': 42.049912,'lon': -93.626546},
'Blueste':{'lat': 42.010098,'lon': -93.647269}
}
df_train['Lat'] = df_train['Neighborhood'].map(lambda neighbor: dict_neighbor[neighbor]['lat'])
df_train['Lon'] = df_train['Neighborhood'].map(lambda neighbor: dict_neighbor[neighbor]['lon'])
Categorical_features = df_train.select_dtypes(include=['object'])
Numerical_features = df_train.select_dtypes(exclude=['object'])
Numerical_features.columns
def plotdist(data):
    try:
        sns.distplot(data)
    except RuntimeError as re:
        if str(re).startswith("Selected KDE bandwidth is 0. Cannot estimate density."):
            sns.distplot(data, kde_kws={'bw': 0.1})
        else:
            raise re
for feature in Numerical_features.columns:
    plotdist(Numerical_features[feature])
    plt.show()
for feature in Categorical_features.columns:
    
    if Categorical_features[feature].nunique() > 12 :
        continue
    if Categorical_features[feature].nunique() > 5 :
        plt.figure(figsize=(10,6))
        plt.xticks(rotation=45)
    sns.violinplot(x=feature,y='SalePrice',data=df_train)
    plt.tight_layout()
    plt.show()
X = df_train.drop('SalePrice',axis=1)
y = df_train['SalePrice']
LotFrontageX = X['LotFrontage']
GarageYrBltX = X['GarageYrBlt']
ohe = OneHotEncoder(sparse=False,drop="if_binary")
Categorical_Encoded = ohe.fit_transform(Categorical_features.astype(str))
Categorical_Encoded_Frame = pd.DataFrame(Categorical_Encoded, columns= ohe.get_feature_names(Categorical_features.columns))
Categorical_Encoded_Frame.head()
Numerical_features_X = Numerical_features.drop(['SalePrice','LotFrontage','GarageYrBlt'],axis=1)
X = Categorical_Encoded_Frame.join(Numerical_features_X).join(LotFrontageX).join(GarageYrBltX)
Xcolumns = X.columns
X.head()
X = KNNImputer(n_neighbors=5).fit_transform(X)
X = pd.DataFrame(X,columns=Xcolumns)
X.head()
sns.distplot(y)
y.skew()
y_log = np.log(y)
y_log.skew()
sns.distplot(y_log)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.3, random_state=42)
print("X_train Shape: ",X_train.shape)
print("X_test Shape: ",X_test.shape)
print("y_train Shape: ",y_train.shape)
print("y_test Shape: ",y_test.shape)

xStandardScaler = StandardScaler()
yStandardScaler = StandardScaler()
X_train = xStandardScaler.fit_transform(X_train)
X_test = xStandardScaler.transform(X_test)
y_train = yStandardScaler.fit_transform(y_train.ravel().reshape(-1, 1))
y_test = yStandardScaler.transform(y_test.ravel().reshape(-1, 1))
ModelCompList = list()
alphas = np.linspace(0,.1,num=21)
lgLasso = LassoCV(cv=10,alphas=alphas)
lgLasso.fit(X_train,y_train)
# print the intercept
print(lgLasso.alpha_)
coeff_df = pd.DataFrame(lgLasso.coef_.T,X.columns,columns=['Coefficient'])
coeff_df.sort_values(by='Coefficient').head(10).plot(kind='bar')
plt.xlabel('Features')
plt.title('Top 10 Negative Features')

coeff_df.sort_values(by='Coefficient',ascending=False).head(10).plot(kind='bar')
plt.xlabel('Features')
plt.title('Top 10 Positive Features')

predictionsLinear = lgLasso.predict(X_test)
plt.scatter(np.exp(yStandardScaler.inverse_transform(y_test)),np.exp(yStandardScaler.inverse_transform(predictionsLinear)))
sns.distplot((yStandardScaler.inverse_transform(y_test)-yStandardScaler.inverse_transform(predictionsLinear)),bins=50);
def evaluateModel(name,y_test,Predictions):
    modelComp = dict()
    modelComp['Name'] = name
    modelComp['MAE'] = metrics.mean_absolute_error(np.exp(yStandardScaler.inverse_transform(y_test)),
                                                   np.exp(yStandardScaler.inverse_transform(Predictions)))
    modelComp['MSE'] = metrics.mean_squared_error(np.exp(yStandardScaler.inverse_transform(y_test)),
                                                  np.exp(yStandardScaler.inverse_transform(Predictions)))
    modelComp['RMSE'] = np.sqrt(metrics.mean_squared_error(np.exp(yStandardScaler.inverse_transform(y_test)),
                                                           np.exp(yStandardScaler.inverse_transform(Predictions))))
    
    print('MAE:', modelComp['MAE'])
    print('MSE:', modelComp['MSE'])
    print('RMSE:', modelComp['RMSE'])
    
    ModelCompList.append(modelComp)
evaluateModel('Lasso',y_test,predictionsLinear)
alphas = np.linspace(0,20,num=21)
lgRidge = RidgeCV(cv=10,alphas=alphas)
lgRidge.fit(X_train,y_train)
# print the intercept
print(lgRidge.alpha_)
coeff_df = pd.DataFrame(lgRidge.coef_.T,X.columns,columns=['Coefficient'])
coeff_df.sort_values(by='Coefficient').head(10).plot(kind='bar')
plt.xlabel('Features')
plt.title('Top 10 Negative Features')

coeff_df.sort_values(by='Coefficient',ascending=False).head(10).plot(kind='bar')
plt.xlabel('Features')
plt.title('Top 10 Positive Features')

predictionsLinear = lgRidge.predict(X_test)
plt.scatter(np.exp(yStandardScaler.inverse_transform(y_test)),np.exp(yStandardScaler.inverse_transform(predictionsLinear)))
sns.distplot((yStandardScaler.inverse_transform(y_test)-yStandardScaler.inverse_transform(predictionsLinear)),bins=50);
evaluateModel('Ridge',y_test,predictionsLinear)
alphas = np.linspace(0,10,num=21)
l1_ratio = np.linspace(0,1,num=21)
lgNet = ElasticNetCV(cv=10,alphas=alphas,l1_ratio=l1_ratio)
lgNet.fit(X_train,y_train)
# print the intercept
print(lgNet.alpha_)
print(lgNet.l1_ratio_)
coeff_df = pd.DataFrame(lgNet.coef_.T,X.columns,columns=['Coefficient'])
coeff_df.sort_values(by='Coefficient').head(10).plot(kind='bar')
plt.xlabel('Features')
plt.title('Top 10 Negative Features')

coeff_df.sort_values(by='Coefficient',ascending=False).head(10).plot(kind='bar')
plt.xlabel('Features')
plt.title('Top 10 Positive Features')

predictionsLinear = lgNet.predict(X_test)
plt.scatter(np.exp(yStandardScaler.inverse_transform(y_test)),np.exp(yStandardScaler.inverse_transform(predictionsLinear)))
sns.distplot((yStandardScaler.inverse_transform(y_test)-yStandardScaler.inverse_transform(predictionsLinear)),bins=50);
evaluateModel('Elastic Net',y_test,predictionsLinear)
rfr = RandomForestRegressor(n_estimators=2500)
parameters = {'min_samples_split': [2],
              'max_depth': [8],
              'min_samples_leaf' :[2],
              'max_samples': [.7]}

rfr_grid = GridSearchCV(rfr,
                        parameters,
                        cv = 2)
rfr_grid.fit(X_train,y_train)
print(rfr_grid.best_score_)
print(rfr_grid.best_params_)
coeff_df = pd.DataFrame(rfr_grid.best_estimator_.feature_importances_,X.columns,columns=['Importances'])

coeff_df.sort_values(by='Importances',ascending=False).head(10).plot(kind='bar')
plt.xlabel('Features')
plt.title('Top 10 Positive Features')

predictionsForest = rfr_grid.best_estimator_.predict(X_test)
plt.scatter(yStandardScaler.inverse_transform(y_test),yStandardScaler.inverse_transform(predictionsForest))
sns.distplot((np.exp(yStandardScaler.inverse_transform(y_test))-np.exp(yStandardScaler.inverse_transform(predictionsForest))),bins=50);
evaluateModel('Random Forest',y_test,predictionsForest)
xgb = XGBRegressor()
parameters = {'objective':['reg:squarederror'],
            'learning_rate': [.1],
              'max_depth': [3],
              'min_child_weight': [2],
              'subsample': [0.7],
              'colsample_bytree': [.7],
              'colsample_bylevel':[.7],
              'alpha' : [.05],
              'lambda' : [.3],
              'n_estimators': [2500]}

xgb_grid = GridSearchCV(xgb,
                        parameters,
                        cv = 2)
xgb_grid.fit(X_train,
         y_train)
print(xgb_grid.best_score_)
print(xgb_grid.best_params_)
coeff_df = pd.DataFrame(xgb_grid.best_estimator_.feature_importances_,X.columns,columns=['Coefficient'])

coeff_df.sort_values(by='Coefficient',ascending=False).head(10).plot(kind='bar')
plt.xlabel('Features')
plt.title('Top 10 Positive Features')

predictionsLinear = xgb_grid.predict(X_test)
plt.scatter(np.exp(yStandardScaler.inverse_transform(y_test)),np.exp(yStandardScaler.inverse_transform(predictionsLinear)))
sns.distplot((yStandardScaler.inverse_transform(y_test)-yStandardScaler.inverse_transform(predictionsLinear)),bins=50);
evaluateModel('XG Boost',y_test,predictionsLinear)
xgb = LGBMRegressor()
parameters = {'learning_rate': [.1],
              'max_depth': [6],
              'min_child_weight': [1],
              'subsample': [0.7],
              'colsample_bytree': [.7],
              'colsample_bylevel': [.7],
              'reg_alpha' : [.1],
              'reg_lambda' : [.3],
              'n_estimators': [12500]}

xgb_grid = GridSearchCV(xgb,
                        parameters,
                        cv = 2)
xgb_grid.fit(X_train,
         y_train)
print(xgb_grid.best_score_)
print(xgb_grid.best_params_)
coeff_df = pd.DataFrame(xgb_grid.best_estimator_.feature_importances_,X.columns,columns=['Coefficient'])

coeff_df.sort_values(by='Coefficient',ascending=False).head(10).plot(kind='bar')
plt.xlabel('Features')
plt.title('Top 10 Positive Features')

predictionsLinear = xgb_grid.predict(X_test)
plt.scatter(np.exp(yStandardScaler.inverse_transform(y_test)),np.exp(yStandardScaler.inverse_transform(predictionsLinear)))
sns.distplot((yStandardScaler.inverse_transform(y_test)-yStandardScaler.inverse_transform(predictionsLinear)),bins=50);
evaluateModel('Light GBM',y_test,predictionsLinear)
from sklearn.model_selection import train_test_split
X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(X, y, test_size=0.3, random_state=42)
print("X_train Shape: ",X_train_nn.shape)
print("X_test Shape: ",X_test_nn.shape)
print("y_train Shape: ",y_train_nn.shape)
print("y_test Shape: ",y_test_nn.shape)

xStandardScaler = StandardScaler()
yStandardScaler = StandardScaler()

X_train_nn = xStandardScaler.fit_transform(X_train_nn)
X_test_nn = xStandardScaler.transform(X_test_nn)

y_train_nn = yStandardScaler.fit_transform(y_train_nn.ravel().reshape(-1, 1))
y_test_nn= yStandardScaler.transform(y_test_nn.ravel().reshape(-1, 1))
model = Sequential()
model.add(Dense(256, input_dim=301))
model.add(LeakyReLU(alpha=.1))
model.add(Dropout(.1))
model.add(Dense(192, activation='relu'))
model.add(Dropout(.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(.1))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.summary()
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['mse'])
mcp = ModelCheckpoint(
    'model.h5',
    monitor="val_mse",
    verbose=2,
    save_best_only=True,
    mode="min"
)
history = model.fit(X_train_nn,y_train_nn,epochs=100,batch_size=32,callbacks=mcp,validation_data=(X_test_nn,y_test_nn))
# summarize history for accuracy
plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('model MSE')
plt.ylabel('MSE')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
model = load_model('model.h5')
predictionsNeural = model.predict(X_test_nn)
modelComp = dict()
modelComp['Name'] = 'Neural Network'
modelComp['MAE'] = metrics.mean_absolute_error(yStandardScaler.inverse_transform(y_test_nn),
                                               yStandardScaler.inverse_transform(predictionsNeural))
modelComp['MSE'] = metrics.mean_squared_error(yStandardScaler.inverse_transform(y_test_nn),
                                              yStandardScaler.inverse_transform(predictionsNeural))
modelComp['RMSE'] = np.sqrt(metrics.mean_squared_error(yStandardScaler.inverse_transform(y_test_nn),
                                                       yStandardScaler.inverse_transform(predictionsNeural)))

print('MAE:', modelComp['MAE'])
print('MSE:', modelComp['MSE'])
print('RMSE:', modelComp['RMSE'])

ModelCompList.append(modelComp)
df_models = pd.DataFrame(ModelCompList)
df_models.head()
for i in df_models.columns[1:]:
    sns.barplot(x='Name',y=i,data=df_models)
    plt.xlabel('Model')
    plt.xticks(rotation=-45)
    plt.show()
df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
df_test.head()
IdTest = df_test['Id']
df_test.drop('Id',axis=1,inplace=True)
visualNA(df_test)
handleNA(df_test)
visualNA(df_test)
df_test.loc[(df_test['MSZoning'].isnull()) & (df_test['MSSubClass'].astype(int)<70),'MSZoning'] = 'RL'
df_test.loc[(df_test['MSZoning'].isnull()) & (df_test['MSSubClass'].astype(int)>=70),'MSZoning'] = 'RM'
df_test.loc[df_test['Utilities'].isnull(),'Utilities'] = 'AllPub'
df_test.loc[df_test['Exterior1st'].isnull(),'Exterior1st'] = 'Wd Sdng'
df_test.loc[df_test['Exterior2nd'].isnull(),'Exterior2nd'] = 'Wd Sdng'
df_test.loc[df_test['BsmtFinSF1'].isnull(),'BsmtFinSF1'] = 180
df_test.loc[df_test['BsmtFinSF2'].isnull(),'BsmtFinSF2'] = 374
df_test.loc[df_test['BsmtUnfSF'].isnull(),'BsmtUnfSF'] = 340
df_test.loc[df_test['TotalBsmtSF'].isnull(),'TotalBsmtSF'] = 894
df_test.loc[(df_test['BsmtFullBath'].isnull()) & (df_test['FullBath']==1),'BsmtFullBath'] = 0
df_test.loc[(df_test['BsmtFullBath'].isnull()) & (df_test['FullBath']==3),'BsmtFullBath'] = 1
df_test.loc[df_test['BsmtHalfBath'].isnull(),'BsmtHalfBath'] = 0
df_test.loc[df_test['KitchenQual'].isnull(),'KitchenQual'] = 'TA'
df_test.loc[(df_test['Functional'].isnull()) & (df_train['ScreenPorch']<1), 'Functional'] = 'Typ'
df_test.loc[(df_test['Functional'].isnull()) & (df_train['ScreenPorch']>1) , 'Functional'] = 'Mod'
df_test.loc[df_test['GarageCars'].isnull(),'GarageCars']= 2
df_test.loc[df_test['GarageArea'].isnull(),'GarageArea']= 324
df_test.loc[df_test['SaleType'].isnull(),'SaleType'] = 'WD'
visualNA(df_test)
df_test['Lat'] = df_test['Neighborhood'].map(lambda neighbor: dict_neighbor[neighbor]['lat'])
df_test['Lon'] = df_test['Neighborhood'].map(lambda neighbor: dict_neighbor[neighbor]['lon'])
Categorical_features_test = df_test.select_dtypes(include=['object'])
Numerical_features_test = df_test.select_dtypes(exclude=['object'])
Categorical_Encoded_test = ohe.transform(Categorical_features_test)

Categorical_Encoded_Frame_test = pd.DataFrame(Categorical_Encoded_test, columns= ohe.get_feature_names(Categorical_features_test.columns))
Categorical_Encoded_Frame_test.head()
LotFrontageTestX = Numerical_features_test['LotFrontage']
GarageYrBltTestX = Numerical_features_test['GarageYrBlt']
Numerical_features_test.drop(['LotFrontage','GarageYrBlt'],axis=1,inplace=True)
X_test_Full = Categorical_Encoded_Frame_test.join(Numerical_features_test).join(LotFrontageTestX).join(GarageYrBltTestX)
X_testcolumns = X_test_Full.columns
X_test_Full.head()
X_test_Full = KNNImputer(n_neighbors=5).fit_transform(X_test_Full)
X_test_Frame = pd.DataFrame(X_test_Full,columns=X_testcolumns)
X_test_Frame.head()
Xtest = StandardScaler()
Ytest = StandardScaler()

X_Full = Xtest.fit_transform(X)
y_Full = Ytest.fit_transform(y_log.ravel().reshape(-1,1))
lgLassoFinal = LassoCV(cv=5,alphas =[.02])
lgLassoFinal.fit(X_Full,y_Full)
X_test_Frame = Xtest.transform(X_test_Frame)
ypredLasso = lgLassoFinal.predict(X_test_Frame)
lgRidgeFinal = RidgeCV(cv=5,alphas =[20])
lgRidgeFinal.fit(X_Full,y_Full)
ypredRidge = lgRidgeFinal.predict(X_test_Frame)
ypredRidge = ypredRidge.reshape(-1,)
xgbFinal = XGBRegressor(objective='reg:squarederror',
            learning_rate= .1,
              max_depth= 3,
              min_child_weight= 2,
              subsample= 0.7,
              colsample_bytree= .7,
              colsample_bylevel=.7,
              alpha = .05,
              reg_lambda = .3,
              n_estimators= 2500)
xgbFinal.fit(X_Full,y_Full)
ypredXG = xgbFinal.predict(X_test_Frame)
Final_Pred = -4447 + np.exp(Ytest.inverse_transform((ypredRidge*.20309379 + ypredLasso*.46724816 + ypredXG*.37162723)))
Final_Pred[np.exp(Ytest.inverse_transform(ypredRidge))>y.max()] = np.exp(Ytest.inverse_transform(ypredXG))[np.exp(Ytest.inverse_transform(ypredRidge))>y.max()]
submission = pd.DataFrame({
        "Id": IdTest,
        "SalePrice": Final_Pred
    })
submission.to_csv('submission.csv', index=False)