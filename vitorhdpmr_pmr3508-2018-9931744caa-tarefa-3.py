import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt 
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
os.listdir("../input")
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.shape
train.head()
train['longitude'].value_counts().plot(kind = 'bar')
train['longitude'].value_counts()
train['latitude'].value_counts().plot(kind = 'bar')
train['latitude'].value_counts()
train.describe()
train.corr()
M = np.array([['longitude',train['longitude'].mean()],['latitude',train['latitude'].mean()],['median age',train['median_age'].mean()],
              ['total rooms',train['total_rooms'].mean()],['total bedrooms',train['total_bedrooms'].mean()],['population',train['population'].mean()],
              ['households',train['households'].mean()],['median income',train['median_income'].mean()],['median house value',train['median_house_value'].mean()]])
print(M)
from sklearn.neighbors import KNeighborsRegressor
KnnXtrain = train[['longitude','latitude','median_age','total_rooms','population','households','median_income']]
KnnYtrain = train.median_house_value

KnnXtest = test[['longitude','latitude','median_age','total_rooms','population','households','median_income']]
KNNreg = KNeighborsRegressor(n_neighbors=50)
KNNreg.fit(KnnXtrain, KnnYtrain)
KnnYpred = KNNreg.predict(KnnXtest)
Xid = test['Id']
array = np.vstack((Xid,KnnYpred)).T
final = pd.DataFrame(columns=['Id', 'median_house_value'], data=array)
final['Id']=final['Id'].astype('Int32')
final.to_csv('resultsKnn_50.csv', index = False)

KNNreg = KNeighborsRegressor(n_neighbors=70)
KNNreg.fit(KnnXtrain, KnnYtrain)
KnnYpred = KNNreg.predict(KnnXtest)
array = np.vstack((Xid,KnnYpred)).T
final = pd.DataFrame(columns=['Id', 'median_house_value'], data=array)
final['Id']=final['Id'].astype('Int32')
final.to_csv('resultsKnn_70.csv', index = False)

KNNreg30 = KNeighborsRegressor(n_neighbors=30)
KNNreg30.fit(KnnXtrain, KnnYtrain)
KnnYpred = KNNreg30.predict(KnnXtest)
array = np.vstack((Xid,KnnYpred)).T
final = pd.DataFrame(columns=['Id', 'median_house_value'], data=array)
final['Id']=final['Id'].astype('Int32')
final.to_csv('resultsKnn_30.csv', index = False)

KNNreg = KNeighborsRegressor(n_neighbors=100)
KNNreg.fit(KnnXtrain, KnnYtrain)

KnnYpred = KNNreg.predict(KnnXtest)

array = np.vstack((Xid,KnnYpred)).T
final = pd.DataFrame(columns=['Id', 'median_house_value'], data=array)
final['Id']=final['Id'].astype('Int32')
final.to_csv('resultsKnn_100.csv', index = False)
KNNreg = KNeighborsRegressor(n_neighbors=125)
KNNreg.fit(KnnXtrain, KnnYtrain)

KnnYpred = KNNreg.predict(KnnXtest)

array = np.vstack((Xid,KnnYpred)).T
final = pd.DataFrame(columns=['Id', 'median_house_value'], data=array)
final['Id']=final['Id'].astype('Int32')
final.to_csv('resultsKnn_125.csv', index = False)
KnnXtrain = train[['longitude','latitude','median_age','total_rooms','population','median_income']]
KnnXtest = test[['longitude','latitude','median_age','total_rooms','population','median_income']]

KNNreg = KNeighborsRegressor(n_neighbors=70)
KNNreg.fit(KnnXtrain, KnnYtrain)

KnnYpred = KNNreg.predict(KnnXtest)

array = np.vstack((Xid,KnnYpred)).T
final = pd.DataFrame(columns=['Id', 'median_house_value'], data=array)
final['Id']=final['Id'].astype('Int32')
final.to_csv('resultsKnn_70X1.csv', index = False)
KnnXtrain = train[['longitude','latitude','total_rooms','population','households','median_income']]
KnnXtest = test[['longitude','latitude','total_rooms','population','households','median_income']]

KNNreg = KNeighborsRegressor(n_neighbors=70)
KNNreg.fit(KnnXtrain, KnnYtrain)

KnnYpred = KNNreg.predict(KnnXtest)

array = np.vstack((Xid,KnnYpred)).T
final = pd.DataFrame(columns=['Id', 'median_house_value'], data=array)
final['Id']=final['Id'].astype('Int32')
final.to_csv('resultsKnn_70X2.csv', index = False)
KnnXtrain = train[['longitude','latitude','median_age','total_rooms','households','median_income']]
KnnXtest = test[['longitude','latitude','median_age','total_rooms','households','median_income']]

KNNreg = KNeighborsRegressor(n_neighbors=70)
KNNreg.fit(KnnXtrain, KnnYtrain)

KnnYpred = KNNreg.predict(KnnXtest)

array = np.vstack((Xid,KnnYpred)).T
final = pd.DataFrame(columns=['Id', 'median_house_value'], data=array)
final['Id']=final['Id'].astype('Int32')
final.to_csv('resultsKnn_70X3.csv', index = False)
from sklearn import linear_model
RidgeReg = linear_model.Ridge(alpha = 0.5)
RidgeRegXtrain = train[['longitude','latitude','median_age','total_rooms','population','households','median_income']]
RidgeRegYtrain = train.median_house_value

RidgeRegXtest = test[['longitude','latitude','median_age','total_rooms','population','households','median_income']]
RidgeReg.fit(RidgeRegXtrain, RidgeRegYtrain)
RidgeReg.coef_
RidgeRegYpred = RidgeReg.predict(RidgeRegXtest)

array = np.vstack((Xid,abs(RidgeRegYpred))).T
final = pd.DataFrame(columns=['Id', 'median_house_value'], data=array)
final['Id']=final['Id'].astype('Int32')
final.to_csv('resultsRR_X1.csv', index = False)
RidgeReg = linear_model.Ridge(alpha = 10.04)
RidgeReg.fit(RidgeRegXtrain, RidgeRegYtrain)

RidgeReg.coef_
RidgeReg = linear_model.Ridge(alpha = 25)
RidgeReg.fit(RidgeRegXtrain, RidgeRegYtrain)

RidgeReg.coef_
RidgeReg = linear_model.Ridge(alpha = .0003)
RidgeReg.fit(RidgeRegXtrain, RidgeRegYtrain)

RidgeReg.coef_
RidgeRegXtrain = train[['longitude','latitude','median_age','total_rooms','households','median_income']]
RidgeRegYtrain = train.median_house_value

RidgeRegXtest = test[['longitude','latitude','median_age','total_rooms','households','median_income']]

RidgeReg = linear_model.Ridge(alpha = .5)
RidgeReg.fit(RidgeRegXtrain, RidgeRegYtrain)

RidgeReg.coef_

RidgeReg.fit(RidgeRegXtrain, RidgeRegYtrain)

RidgeRegYpred = RidgeReg.predict(RidgeRegXtest)

array = np.vstack((Xid,abs(RidgeRegYpred))).T
final = pd.DataFrame(columns=['Id', 'median_house_value'], data=array)
final['Id']=final['Id'].astype('Int32')
final.to_csv('resultsRR_X2.csv', index = False)
RidgeRegXtrain = train[['longitude','latitude','total_rooms','population','households','median_income']]
RidgeRegYtrain = train.median_house_value

RidgeRegXtest = test[['longitude','latitude','total_rooms','population','households','median_income']]

RidgeReg = linear_model.Ridge(alpha = .5)
RidgeReg.fit(RidgeRegXtrain, RidgeRegYtrain)

RidgeReg.coef_

RidgeReg.fit(RidgeRegXtrain, RidgeRegYtrain)

RidgeRegYpred = RidgeReg.predict(RidgeRegXtest)

array = np.vstack((Xid,abs(RidgeRegYpred))).T
final = pd.DataFrame(columns=['Id', 'median_house_value'], data=array)
final['Id']=final['Id'].astype('Int32')
final.to_csv('resultsRR_X3.csv', index = False)
from sklearn.ensemble import RandomForestRegressor
RFRegXtrain = train[['longitude','latitude','median_age','total_rooms','population','households','median_income']]
RFRegYtrain = train.median_house_value

RFRegXtest = test[['longitude','latitude','median_age','total_rooms','population','households','median_income']]
RFReg = RandomForestRegressor(n_estimators = 50)
RFReg.fit(RFRegXtrain, RFRegYtrain)
RFRegYpred = RFReg.predict(RFRegXtest)

array = np.vstack((Xid,abs(RFRegYpred))).T
final = pd.DataFrame(columns=['Id', 'median_house_value'], data=array)
final['Id']=final['Id'].astype('Int32')
final.to_csv('resultsRFR_50.csv', index = False)
RFReg = RandomForestRegressor(n_estimators = 10)

RFReg.fit(RFRegXtrain, RFRegYtrain)

RFRegYpred = RFReg.predict(RFRegXtest)

array = np.vstack((Xid,abs(RFRegYpred))).T
final = pd.DataFrame(columns=['Id', 'median_house_value'], data=array)
final['Id']=final['Id'].astype('Int32')
final.to_csv('resultsRFR_10.csv', index = False)
RFReg = RandomForestRegressor(n_estimators = 100)

RFReg.fit(RFRegXtrain, RFRegYtrain)

RFRegYpred = RFReg.predict(RFRegXtest)

array = np.vstack((Xid,abs(RFRegYpred))).T
final = pd.DataFrame(columns=['Id', 'median_house_value'], data=array)
final['Id']=final['Id'].astype('Int32')
final.to_csv('resultsRFR_100u.csv', index = False)
RFReg = RandomForestRegressor(n_estimators = 200)

RFReg.fit(RFRegXtrain, RFRegYtrain)

RFRegYpred = RFReg.predict(RFRegXtest)

array = np.vstack((Xid,abs(RFRegYpred))).T
final = pd.DataFrame(columns=['Id', 'median_house_value'], data=array)
final['Id']=final['Id'].astype('Int32')
final.to_csv('resultsRFR_200.csv', index = False)
RFReg = RandomForestRegressor(n_estimators = 125)

RFReg.fit(RFRegXtrain, RFRegYtrain)

RFRegYpred = RFReg.predict(RFRegXtest)

array = np.vstack((Xid,abs(RFRegYpred))).T
final = pd.DataFrame(columns=['Id', 'median_house_value'], data=array)
final['Id']=final['Id'].astype('Int32')
final.to_csv('resultsRFR_125.csv', index = False)
RFReg = RandomForestRegressor(n_estimators = 150)

RFReg.fit(RFRegXtrain, RFRegYtrain)

RFRegYpred = RFReg.predict(RFRegXtest)

array = np.vstack((Xid,abs(RFRegYpred))).T
final = pd.DataFrame(columns=['Id', 'median_house_value'], data=array)
final['Id']=final['Id'].astype('Int32')
final.to_csv('resultsRFR_150.csv', index = False)
RFReg = RandomForestRegressor(n_estimators = 175)

RFReg.fit(RFRegXtrain, RFRegYtrain)

RFRegYpred = RFReg.predict(RFRegXtest)

array = np.vstack((Xid,abs(RFRegYpred))).T
final = pd.DataFrame(columns=['Id', 'median_house_value'], data=array)
final['Id']=final['Id'].astype('Int32')
final.to_csv('resultsRFR_175.csv', index = False)
RFRegXtrain = train[['longitude','latitude','median_age','total_bedrooms','households','median_income']]
RFRegXtest = test[['longitude','latitude','median_age','total_bedrooms','households','median_income']]

RFReg = RandomForestRegressor(n_estimators = 100)

RFReg.fit(RFRegXtrain, RFRegYtrain)

RFRegYpred = RFReg.predict(RFRegXtest)

array = np.vstack((Xid,abs(RFRegYpred))).T
final = pd.DataFrame(columns=['Id', 'median_house_value'], data=array)
final['Id']=final['Id'].astype('Int32')
final.to_csv('resultsRFR_X32.csv', index = False)
RFRegXtrain = train[['longitude','latitude','total_bedrooms','households','median_income']]
RFRegXtest = test[['longitude','latitude','total_bedrooms','households','median_income']]

RFReg = RandomForestRegressor(n_estimators = 100)

RFReg.fit(RFRegXtrain, RFRegYtrain)

RFRegYpred = RFReg.predict(RFRegXtest)

array = np.vstack((Xid,abs(RFRegYpred))).T
final = pd.DataFrame(columns=['Id', 'median_house_value'], data=array)
final['Id']=final['Id'].astype('Int32')
final.to_csv('resultsRFR_X33.csv', index = False)