import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train=pd.concat([train,test])
Sales_Price = train['SalePrice']
Sales_Price = Sales_Price.iloc[0:1460]
train.head()
train.drop(['Alley', 'FireplaceQu','PoolQC','Fence', 'MiscFeature'],  axis=1, inplace=True)
train.dropna(axis=1, inplace=True)
train.info()
train['Street'].unique()
train['Street'].replace({'Pave': 1, 'Grvl': 2}, inplace=True)
train['LotShape'].unique()
train['LotShape'].replace({'Reg': 1, 'IR1': 2, 'IR2': 3, 'IR3': 4}, inplace=True)
train['LandContour'].unique()
train['LandContour'].replace({'Lvl': 1, 'Bnk': 2, 'Low': 3, 'HLS': 4}, inplace=True)
train['LotConfig'].unique()
train['LotConfig'].replace({'Inside': 1, 'FR2': 2, 'Corner': 3, 'CulDSac': 4, 'FR3': 5}, inplace=True)
train['LandSlope'].unique()
train['LandSlope'].replace({'Gtl': 1, 'Mod': 2, 'Sev': 3}, inplace=True)
train['Neighborhood'].unique()
train['Neighborhood'].replace({'CollgCr':1, 'Veenker':2, 'Crawfor':3, 'NoRidge':4, 'Mitchel':5, 'Somerst':6,
       'NWAmes':7, 'OldTown':8, 'BrkSide':9, 'Sawyer':10, 'NridgHt':11, 'NAmes':12,
       'SawyerW':13, 'IDOTRR':14, 'MeadowV':15, 'Edwards':16, 'Timber':17, 'Gilbert':18,
       'StoneBr':19, 'ClearCr':20, 'NPkVill':21, 'Blmngtn':22, 'BrDale':23, 'SWISU':24,
       'Blueste':25}, inplace=True)
train['Condition1'].unique()
train['Condition1'].replace({'Norm':1, 'Feedr':2, 'PosN':3, 'Artery':4, 'RRAe':5, 'RRNn':6, 'RRAn':7, 'PosA':8,
       'RRNe':9}, inplace=True)
train['Condition2'].unique()
train['Condition2'].replace({'Norm':1, 'Artery':2, 'RRNn':3, 'Feedr':4, 'PosN':5, 'PosA':6, 'RRAn':7, 'RRAe':8}, inplace=True)

train['BldgType'].unique()
train['BldgType'].replace({'1Fam': 1, '2fmCon': 2, 'Duplex': 3, 'TwnhsE': 4, 'Twnhs': 5}, inplace=True)
train['HouseStyle'].unique()
train['HouseStyle'].replace({'2Story': 1, '1Story':2, '1.5Fin':3, '1.5Unf':4, 'SFoyer':5, 'SLvl':6, '2.5Unf':7, '2.5Fin':8}, inplace=True)
train['RoofStyle'].unique()
train['RoofStyle'].replace({'Gable': 1, 'Hip':2, 'Gambrel':3, 'Mansard':4, 'Flat':5, 'Shed':6}, inplace=True)
train['RoofMatl'].unique()
train['RoofMatl'].replace({'CompShg':1, 'WdShngl':2, 'Metal':3, 'WdShake':4, 'Membran':5, 'Tar&Grv':6,
       'Roll':7, 'ClyTile':8}, inplace=True)
train['ExterQual'].unique()
train['ExterQual'].replace({'Gd':1, 'TA':2, 'Ex':3, 'Fa':4}, inplace=True)
train['ExterCond'].unique()
train['ExterCond'].replace({'TA':1, 'Gd':2, 'Fa':3, 'Po':4, 'Ex':5}, inplace=True)
train['Foundation'].unique()
train['Foundation'].replace({'PConc':1, 'CBlock':2, 'BrkTil':3, 'Wood':4, 'Slab':5, 'Stone':6}, inplace=True)
train['Heating'].unique()
train['Heating'].replace({'GasA':1, 'GasW':2, 'Grav':3, 'Wall':4, 'OthW':5, 'Floor':6}, inplace=True)
train['HeatingQC'].unique()
train['HeatingQC'].replace({'Ex':1, 'Gd':2, 'TA':3, 'Fa':4, 'Po':5}, inplace=True)
train['CentralAir'].unique()
train['CentralAir'].replace({'Y':0, 'N':1}, inplace=True)
train['PavedDrive'].unique()
train['PavedDrive'].replace({'Y':1, 'N':0, 'P':2}, inplace=True)
train['SaleCondition'].unique()
train['SaleCondition'].replace({'Normal':1, 'Abnorml':2, 'Partial':3, 'AdjLand':4, 'Alloca':5, 'Family':6}, inplace=True)
train.info()
train_set = train.iloc[0:1460]
test_set = train.iloc[1460:2919]
train_set = pd.concat([train_set, Sales_Price],axis=1)
X = train_set.drop('SalePrice', axis=1)
y = train_set['SalePrice']
from sklearn.feature_selection import SelectKBest, f_regression, chi2
test=SelectKBest(score_func=chi2,k=20)
fit=test.fit(X,y)
score = fit.scores_.reshape(1,46)
score = pd.DataFrame(score, columns=X.columns)
score = score.transpose()
score[score > 190].count()
w = score[score > 190].dropna().reset_index()['index']
X = X[w]
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,y, test_size=0.1)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_validate = scaler.transform(X_test)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
model = Sequential()
model.add(Dense(units = 46, activation = 'relu'))
#model.add(Dropout(0.5))
#model.add(Dense(units = 92, activation = 'relu'))
#model.add(Dropout(0.5))
model.add(Dense(units = 23, activation = 'relu'))
#model.add(Dropout(0.5))
#model.add(Dense(units = 10, activation = 'relu'))
#model.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))



model.add(Dense(units = 1))
model.compile(optimizer = 'adam', loss = 'mse')
early_stop = EarlyStopping(monitor='val_loss', mode= 'min', verbose= 1, patience=25)
model.fit(x=X_train, y=y_train, epochs = 10000, validation_data=(X_validate, y_test), batch_size=20, callbacks=[early_stop])
loss = pd.DataFrame(model.history.history)
loss.plot()
check = model.predict(X_validate)
from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(y_test, check))
final_test=scaler.transform(test_set[w])
pred = model.predict(final_test)
#from sklearn.metrics import mean_squared_error
#np.sqrt(mean_squared_error(y_test, pred))
pred = pd.DataFrame(pred)
pred=pd.concat([test_set['Id'], pred], axis=1)
pred.rename(columns={0: 'SalePrice'},inplace=True)
pred.to_csv('prediction.csv',index=False)
pred