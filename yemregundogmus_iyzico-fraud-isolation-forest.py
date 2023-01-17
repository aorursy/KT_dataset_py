import numpy as np 

import pandas as pd 



from sklearn.ensemble import IsolationForest

from sklearn.model_selection import train_test_split
train = pd.read_csv('/kaggle/input/iyzico-projesi/train.csv')

dropset = ['BASKETREGISTERCARD']

train.drop(dropset,axis=1,inplace=True)

train.head()
test = pd.read_csv('/kaggle/input/iyzico-projesi/test.csv')

dropset = ['BASKETREGISTERCARD','ID']

test.drop(dropset,axis=1,inplace=True)

test.head()
bank_islem_sayisi = train['CARDBANKID'].value_counts()

bank_fraud_islem_sayisi = train[train['ISFRAUD']==1]['CARDBANKID'].value_counts()



islemler2 = pd.concat([bank_fraud_islem_sayisi,bank_islem_sayisi],axis=1)

islemler2.columns = ["bank_fraud_islem","bank_islem"]

islemler2.bank_fraud_islem = islemler2.bank_fraud_islem.fillna(0)

islemler2['bank_fraud_proba'] = islemler2['bank_fraud_islem'] / islemler2['bank_islem']

islemler2['bank_mean_basket_price'] = train.groupby(['CARDBANKID']).mean()['BASKETPAIDPRICE'].values

islemler2['bank_new_fraud_proba'] = islemler2['bank_fraud_proba'] * islemler2['bank_mean_basket_price']

islemler2['CARDBANKID'] = islemler2.index

islemler2.head()
channel_islem_sayisi = train['BASKETPAYMENTCHANNEL'].value_counts()

channel_fraud_islem_sayisi = train[train['ISFRAUD']==1]['BASKETPAYMENTCHANNEL'].value_counts()



islemler3 = pd.concat([channel_fraud_islem_sayisi,channel_islem_sayisi],axis=1)

islemler3.columns = ["channel_fraud_islem","channel_islem"]

islemler3.channel_fraud_islem = islemler3.channel_fraud_islem.fillna(0)

islemler3['channel_fraud_proba'] = islemler3['channel_fraud_islem'] / islemler3['channel_islem']

islemler3['channel_mean_basket_price'] = train.groupby(['BASKETPAYMENTCHANNEL']).mean()['BASKETPAIDPRICE'].values

islemler3['channel_new_fraud_proba'] = islemler3['channel_fraud_proba'] * islemler3['channel_mean_basket_price']

islemler3['BASKETPAYMENTCHANNEL'] = islemler3.index

islemler3.head()
sourcetype_islem_sayisi = train['BASKETPAYMENTSOURCETYPE'].value_counts()

sourcetype_fraud_islem_sayisi = train[train['ISFRAUD']==1]['BASKETPAYMENTSOURCETYPE'].value_counts()



islemler4 = pd.concat([sourcetype_fraud_islem_sayisi,sourcetype_islem_sayisi],axis=1)

islemler4.columns = ["sourcetype_fraud_islem","sourcetype_islem"]

islemler4.sourcetype_fraud_islem = islemler4.sourcetype_fraud_islem.fillna(0)

islemler4['sourcetype_fraud_proba'] = islemler4['sourcetype_fraud_islem'] / islemler4['sourcetype_islem']

islemler4['sourcetype_mean_basket_price'] = train.groupby(['BASKETPAYMENTSOURCETYPE']).mean()['BASKETPAIDPRICE'].values

islemler4['sourcetype_new_fraud_proba'] = islemler4['sourcetype_fraud_proba'] * islemler4['sourcetype_mean_basket_price']

islemler4['BASKETPAYMENTSOURCETYPE'] = islemler4.index

islemler4.head()


train = pd.merge(train,islemler2,on="CARDBANKID",how="left")

test = pd.merge(test,islemler2,on="CARDBANKID",how="left")



train = pd.merge(train,islemler3,on="BASKETPAYMENTCHANNEL",how="left")

test = pd.merge(test,islemler3,on="BASKETPAYMENTCHANNEL",how="left")



train = pd.merge(train,islemler4,on="BASKETPAYMENTSOURCETYPE",how="left")

test = pd.merge(test,islemler4,on="BASKETPAYMENTSOURCETYPE",how="left")



train.head()
# Kaynak: https://www.veribilimiokulu.com/uctan-uca-makine-ogrenmesi-ornegi-titanik-gemi-kazasi-uygulamasi/



columns=['BASKETPAIDPRICE','BASKETINSTALLMENT','BASKETPAYMENTSOURCETYPE']



obj_cols=['BASKETHASVIRTUALITEM','CARDTYPE','CARDASSOCIATION','CARDBANKID','BASKETPAYMENTCHANNEL','BASKETINSTALLMENT', 'BASKETPAYMENTSOURCETYPE' ,'BASKETISTHREEDS','MERCHANT_ID']



for col in columns:

    for feat in obj_cols:

        train[f'{col}_mean_group_{feat}']=train[col]/train.groupby(feat)[col].transform('mean')

        train[f'{col}_max_group_{feat}']=train[col]/train.groupby(feat)[col].transform('max')

        train[f'{col}_min_group_{feat}']=train[col]/train.groupby(feat)[col].transform('min')

        train[f'{col}_count_group_{feat}']=train[col]/train.groupby(feat)[col].transform('count')



for col in columns:

    for feat in obj_cols:

        test[f'{col}_mean_group_{feat}']=test[col]/test.groupby(feat)[col].transform('mean')

        test[f'{col}_max_group_{feat}']=test[col]/test.groupby(feat)[col].transform('max')

        test[f'{col}_min_group_{feat}']=test[col]/test.groupby(feat)[col].transform('min')

        test[f'{col}_count_group_{feat}']=test[col]/test.groupby(feat)[col].transform('count')

     
train.drop('sourcetype_islem',axis=1,inplace=True)

test.drop('sourcetype_islem',axis=1,inplace=True)
train = train.drop_duplicates()
yck = train.groupby(['EMAIL','BASKETPAIDPRICE','CARDBANKID'])['CARDASSOCIATION'].count()

yck = pd.DataFrame(yck)

yck.reset_index(inplace=True)

yck.columns  = ['EMAIL','BASKETPAIDPRICE','CARDBANKID','NEW_UNIQ']

yck['NEW_UNIQ'] = (yck['NEW_UNIQ'] > 1).map({True:1, False:0})

train = pd.merge(train, yck, on=['EMAIL','BASKETPAIDPRICE','CARDBANKID'],how='left')



ypk = train.groupby(['EMAIL','MERCHANT_ID','BASKETPAIDPRICE'])['CARDASSOCIATION'].count()

ypk = pd.DataFrame(ypk)

ypk.reset_index(inplace=True)

ypk.columns  = ['EMAIL','MERCHANT_ID','BASKETPAIDPRICE','NEW_UNIQ2']

ypk['NEW_UNIQ2'] = (ypk['NEW_UNIQ2'] > 1).map({True:1, False:0})

train = pd.merge(train, ypk, on=['EMAIL','MERCHANT_ID','BASKETPAIDPRICE'],how='left')
yck = test.groupby(['EMAIL','BASKETPAIDPRICE','CARDBANKID'])['CARDASSOCIATION'].count()

yck = pd.DataFrame(yck)

yck.reset_index(inplace=True)

yck.columns  = ['EMAIL','BASKETPAIDPRICE','CARDBANKID','NEW_UNIQ']

yck['NEW_UNIQ'] = (yck['NEW_UNIQ'] > 1).map({True:1, False:0})

test = pd.merge(test, yck, on=['EMAIL','BASKETPAIDPRICE','CARDBANKID'],how='left')



ypk = test.groupby(['EMAIL','MERCHANT_ID','BASKETPAIDPRICE'])['CARDASSOCIATION'].count()

ypk = pd.DataFrame(ypk)

ypk.reset_index(inplace=True)

ypk.columns  = ['EMAIL','MERCHANT_ID','BASKETPAIDPRICE','NEW_UNIQ2']

ypk['NEW_UNIQ2'] = (ypk['NEW_UNIQ2'] > 1).map({True:1, False:0})

test = pd.merge(test, ypk, on=['EMAIL','MERCHANT_ID','BASKETPAIDPRICE'],how='left')
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler((0,1))

train['BASKETPAIDPRICE'] = scaler.fit_transform(train['BASKETPAIDPRICE'].values.reshape(-1,1))
#X = train.iloc[:, train.columns != 'ISFRAUD']

#y = train.ISFRAUD



#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 25)
#from sklearn.ensemble import IsolationForest

#from sklearn.metrics import make_scorer, f1_score, fbeta_score

#from sklearn import model_selection

#from sklearn.datasets import make_classification



#clf = IsolationForest(random_state=47, behaviour='new')



#param_grid = {'n_estimators': list(range(400, 500, 50)), 

#              'max_samples': list(range(400, 500, 50)), 

#              'contamination': [0.3, 0.4, 0.5], 

#              'max_features': [5,15], 

#              'n_jobs': [5, 10, 20, 30]}







#f1sc = make_scorer(fbeta_score, beta=1, average='micro')



#grid_dt_estimator = model_selection.GridSearchCV(clf, 

#                                                 param_grid,

#                                                 scoring=f1sc, 

#                                                 refit=True,

#                                                 cv=2, 

#                                                 return_train_score=True,

#                                                verbose=3)

#grid_dt_estimator.fit(X_train, y_train)
data_normal = train[train.ISFRAUD == 0].drop('ISFRAUD',axis=1)

data_fraud = train[train.ISFRAUD == 1].drop('ISFRAUD',axis=1)



normal_train, normal_test = train_test_split(data_normal, test_size=0.30, random_state=42)
model = IsolationForest(contamination=0.275, max_features=5, max_samples=350, n_estimators=375, n_jobs=15)

model.fit(normal_train)

inlier_pred_test = model.predict(normal_test)

outlier_pred = model.predict(data_fraud)



print("Accuracy in Detecting Legit Cases:", list(inlier_pred_test).count(1)/inlier_pred_test.shape[0])

print("Accuracy in Detecting Fraud Cases:", list(outlier_pred).count(-1)/outlier_pred.shape[0])
y_pred = model.predict(test)

submission = test.copy()



submission['ID'] = submission.index

submission['ISFRAUD'] = y_pred

submission = submission[['ID','ISFRAUD']]

submission.ISFRAUD = submission.ISFRAUD.map({-1:0, 1:1})

submission.ISFRAUD.value_counts()
submission.to_csv('isolation_final.csv',index=False)