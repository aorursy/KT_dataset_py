import numpy as np

import pandas as pd

from scipy import stats

import gc, datetime, random

import os



import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

from category_encoders.target_encoder import TargetEncoder

from scipy.stats import norm, skew #for some statistics, if needed

from math import sqrt



def seed_everything(seed=0):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)



SEED = 42

seed_everything(SEED)





for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        
def resumetable(df):

    #print(f"Dataset Shape: {df.shape}")

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    summary['First Value'] = df.loc[0].values

    summary['Second Value'] = df.loc[1].values

    summary['Third Value'] = df.loc[2].values



    for name in summary['Name'].value_counts().index:

        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 



    return summary
train = pd.read_csv('/kaggle/input/ml-challenge-tr-is-bankasi/train.csv')

test = pd.read_csv('/kaggle/input/ml-challenge-tr-is-bankasi/test.csv')

sample_submission = pd.read_csv('/kaggle/input/ml-challenge-tr-is-bankasi/sampleSubmission.csv')
print("Train shape: ", train.shape)

print("Test shape: ", test.shape)

print("Submission shape: ", sample_submission.shape)
summary = resumetable(train)

summary
musteri_harcama = train.groupby(['CUSTOMER'])['ISLEM_TUTARI'].sum().reset_index()

musteri_harcama.head()
def davranis_ekle(df):

    df.loc[df['ISLEM_TUTARI'] <= 5000, 'DAVRANIS'] = 'CAT_1'

    df.loc[(df['ISLEM_TUTARI'] > 5000) & (df['ISLEM_TUTARI'] <= 100000), 'DAVRANIS'] = 'CAT_2'

    df.loc[(df['ISLEM_TUTARI'] > 100000) & (df['ISLEM_TUTARI'] <= 200000), 'DAVRANIS'] = 'CAT_3'

    df.loc[(df['ISLEM_TUTARI'] > 200000) & (df['ISLEM_TUTARI'] <= 300000), 'DAVRANIS'] = 'CAT_4'

    df.loc[(df['ISLEM_TUTARI'] > 300000) & (df['ISLEM_TUTARI'] <= 400000), 'DAVRANIS'] = 'CAT_5'

    df.loc[(df['ISLEM_TUTARI'] > 400000) & (df['ISLEM_TUTARI'] <= 500000), 'DAVRANIS'] = 'CAT_6'

    df.loc[(df['ISLEM_TUTARI'] > 500000) & (df['ISLEM_TUTARI'] <= 600000), 'DAVRANIS'] = 'CAT_7'

    df.loc[(df['ISLEM_TUTARI'] > 600000) & (df['ISLEM_TUTARI'] <= 700000), 'DAVRANIS'] = 'CAT_8'

    df.loc[(df['ISLEM_TUTARI'] > 700000) & (df['ISLEM_TUTARI'] <= 800000), 'DAVRANIS'] = 'CAT_9'

    

    df.loc[(df['ISLEM_TUTARI'] > 800000) & (df['ISLEM_TUTARI'] <= 900000), 'DAVRANIS'] = 'CAT_10'

    df.loc[(df['ISLEM_TUTARI'] > 900000) & (df['ISLEM_TUTARI'] <= 1000000), 'DAVRANIS'] = 'CAT_11'

    

    df.loc[df['ISLEM_TUTARI'] > 1000000, 'DAVRANIS'] = 'CAT_12'



davranis_ekle(musteri_harcama)
train = pd.merge(train, musteri_harcama[['CUSTOMER','DAVRANIS']], on='CUSTOMER')

test = pd.merge(test, musteri_harcama[['CUSTOMER','DAVRANIS']], on='CUSTOMER')
customer_sektor_islem_turu_mean = train.groupby(['CUSTOMER','SEKTOR','ISLEM_TURU']).mean()

customer_sektor_islem_turu_mean['TUTAR_PER_ADET_SCT'] = customer_sektor_islem_turu_mean['ISLEM_TUTARI']/customer_sektor_islem_turu_mean['ISLEM_ADEDI']

customer_sektor_islem_turu_mean.reset_index(level=customer_sektor_islem_turu_mean.index.names, inplace=True)

customer_sektor_islem_turu_mean = customer_sektor_islem_turu_mean.drop(columns=['YIL_AY','Record_Count','ISLEM_TUTARI','ISLEM_ADEDI'])
train = pd.merge(train, customer_sektor_islem_turu_mean, how='left', on=['CUSTOMER','SEKTOR','ISLEM_TURU'])

test = pd.merge(test, customer_sektor_islem_turu_mean, how='left', on=['CUSTOMER','SEKTOR','ISLEM_TURU'])
customer_mean = train.groupby('CUSTOMER').mean()

customer_mean=customer_mean[['ISLEM_ADEDI','ISLEM_TUTARI']]

customer_mean=customer_mean.rename(columns={'ISLEM_ADEDI':'ADET_CUSTOMER','ISLEM_TUTARI':'TUTAR_CUSTOMER'})
train = pd.merge(train,customer_mean, how='inner',on='CUSTOMER')

test = pd.merge(test,customer_mean, how='inner',on='CUSTOMER')
test["TUTAR_PER_ADET_SCT"] = test["TUTAR_PER_ADET_SCT"].fillna(test['TUTAR_PER_ADET_SCT'].mean())
#test.isnull().sum().max()
sektor = train.groupby(['SEKTOR'])['ISLEM_TUTARI'].sum().reset_index().sort_values(by='ISLEM_TUTARI', ascending=False)
f, axe = plt.subplots(1,1,figsize=(12,12))

sns.barplot(x = 'ISLEM_TUTARI', y = 'SEKTOR', data = sektor, ax = axe)

axe.set_xlabel('Toplam İşlem Tutarı', fontsize=14)

axe.set_ylabel('Sektör', fontsize=14)

axe.set_xticklabels(axe.get_xticklabels(), rotation=90)

plt.show()
def sektor_davranis(df):

    df.loc[df['SEKTOR'].str.contains('MARKET / ALISVERIS MERKEZLERI', na=False), 'SEKTOR_DAVRANIS'] = 'SD_1'

    

    df.loc[df['SEKTOR'].str.contains('BENZIN VE YAKIT ISTASYONLARI', na=False), 'SEKTOR_DAVRANIS'] = 'SD_2'

    df.loc[df['SEKTOR'].str.contains('GIYIM / AKSESUAR', na=False), 'SEKTOR_DAVRANIS'] = 'SD_2'

    df.loc[df['SEKTOR'].str.contains('CESITLI GIDA', na=False), 'SEKTOR_DAVRANIS'] = 'SD_2'

    df.loc[df['SEKTOR'].str.contains('ELEKTRIK-ELEKTRONIK ESYA / BILGISAYAR', na=False), 'SEKTOR_DAVRANIS'] = 'SD_2'

    

    df.loc[df['SEKTOR'].str.contains('RESTORAN / CATERING', na=False), 'SEKTOR_DAVRANIS'] = 'SD_3'

    df.loc[df['SEKTOR'].str.contains('TURIZM / KONAKLAMA', na=False), 'SEKTOR_DAVRANIS'] = 'SD_3'

    df.loc[df['SEKTOR'].str.contains('ARAC BAKIM / SERVIS', na=False), 'SEKTOR_DAVRANIS'] = 'SD_3'

    df.loc[df['SEKTOR'].str.contains('MOBILYA / DEKORASYON', na=False), 'SEKTOR_DAVRANIS'] = 'SD_3'

    df.loc[df['SEKTOR'].str.contains('HIZMET SEKTORLERI', na=False), 'SEKTOR_DAVRANIS'] = 'SD_3'

    df.loc[df['SEKTOR'].str.contains('BIREYSEL EMEKLILIK', na=False), 'SEKTOR_DAVRANIS'] = 'SD_3'

    df.loc[df['SEKTOR'].str.contains('SAGLIK URUNLERI SATISI', na=False), 'SEKTOR_DAVRANIS'] = 'SD_3'

    

    df.loc[df['SEKTOR'].str.contains('HAVAYOLLARI', na=False), 'SEKTOR_DAVRANIS'] = 'SD_4'

    df.loc[df['SEKTOR'].str.contains('KITAP-DERGI / KIRTASIYE / OFIS MALZEMELERI', na=False), 'SEKTOR_DAVRANIS'] = 'SD_4'

    df.loc[df['SEKTOR'].str.contains('KOZMETIK / GUZELLIK', na=False), 'SEKTOR_DAVRANIS'] = 'SD_4'

    df.loc[df['SEKTOR'].str.contains('MUTEAHHIT ISLERI', na=False), 'SEKTOR_DAVRANIS'] = 'SD_4'

    df.loc[df['SEKTOR'].str.contains('KUYUMCULAR', na=False), 'SEKTOR_DAVRANIS'] = 'SD_4'

    df.loc[df['SEKTOR'].str.contains('DIGER', na=False), 'SEKTOR_DAVRANIS'] = 'SD_4'

    df.loc[df['SEKTOR'].str.contains('TASIMACILIK', na=False), 'SEKTOR_DAVRANIS'] = 'SD_4'

    df.loc[df['SEKTOR'].str.contains('DOGRUDAN PAZARLAMA', na=False), 'SEKTOR_DAVRANIS'] = 'SD_4'

    df.loc[df['SEKTOR'].str.contains('KULUP / DERNEK / SOSYAL HIZMETLER', na=False), 'SEKTOR_DAVRANIS'] = 'SD_4'

    df.loc[df['SEKTOR'].str.contains('EGLENCE / SPOR / HOBI', na=False), 'SEKTOR_DAVRANIS'] = 'SD_4'

    df.loc[df['SEKTOR'].str.contains('DIJITAL URUNLER', na=False), 'SEKTOR_DAVRANIS'] = 'SD_4'



sektor_davranis(train)

sektor_davranis(test)
def yil_ekle(df):

    df.loc[df['YIL_AY'].str.contains('2017', na=False), 'Yil'] = '2017'

    df.loc[df['YIL_AY'].str.contains('2018', na=False), 'Yil'] = '2018'

    df.loc[df['YIL_AY'].str.contains('2019', na=False), 'Yil'] = '2019'

    

def ay_ekle(df):

    df.loc[df['YIL_AY'].str.contains('201801', na=False), 'Ay'] = '1'

    df.loc[df['YIL_AY'].str.contains('201901', na=False), 'Ay'] = '1'

    df.loc[df['YIL_AY'].str.contains('02', na=False), 'Ay'] = '2'

    df.loc[df['YIL_AY'].str.contains('03', na=False), 'Ay'] = '3'

    df.loc[df['YIL_AY'].str.contains('04', na=False), 'Ay'] = '4'

    df.loc[df['YIL_AY'].str.contains('05', na=False), 'Ay'] = '5'

    df.loc[df['YIL_AY'].str.contains('06', na=False), 'Ay'] = '6'

    df.loc[df['YIL_AY'].str.contains('07', na=False), 'Ay'] = '7'

    df.loc[df['YIL_AY'].str.contains('08', na=False), 'Ay'] = '8'

    df.loc[df['YIL_AY'].str.contains('09', na=False), 'Ay'] = '9'

    df.loc[df['YIL_AY'].str.contains('10', na=False), 'Ay'] = '10'

    df.loc[df['YIL_AY'].str.contains('11', na=False), 'Ay'] = '11'

    df.loc[df['YIL_AY'].str.contains('12', na=False), 'Ay'] = '12'
train['YIL_AY'] = train['YIL_AY'].astype(str)

test['YIL_AY'] = test['YIL_AY'].astype(str)



yil_ekle(train)

yil_ekle(test)

ay_ekle(train)

ay_ekle(test)



train['Yil'] = train['Yil'].astype(int)

test['Yil'] = test['Yil'].astype(int)

train['Ay'] = train['Ay'].astype(int)

test['Ay'] = test['Ay'].astype(int)
def ceyrek_ekle(df):

    df.loc[df['Ay'] == 1, 'ceyrek'] = 'Q1'

    df.loc[df['Ay'] == 2, 'ceyrek'] = 'Q1'

    df.loc[df['Ay'] == 3, 'ceyrek'] = 'Q1'

    

    df.loc[df['Ay'] == 4, 'ceyrek'] = 'Q2'

    df.loc[df['Ay'] == 5, 'ceyrek'] = 'Q2'

    df.loc[df['Ay'] == 6, 'ceyrek'] = 'Q2'

    

    df.loc[df['Ay'] == 7, 'ceyrek'] = 'Q3'

    df.loc[df['Ay'] == 8, 'ceyrek'] = 'Q3'

    df.loc[df['Ay'] == 9, 'ceyrek'] = 'Q3'

    

    df.loc[df['Ay'] == 10, 'ceyrek'] = 'Q4'

    df.loc[df['Ay'] == 11, 'ceyrek'] = 'Q4'

    df.loc[df['Ay'] == 12, 'ceyrek'] = 'Q4'



ceyrek_ekle(train)

ceyrek_ekle(test)
# Concatenating train and test data

test['ISLEM_TUTARI'] = 'test'

df = pd.concat([train, test], axis=0, sort=False)

print("Data shape:", df.shape)
def dolar_alis(df):

    df.loc[df['YIL_AY'] == "201711", 'DOLAR_ALIS'] = 3.88

    df.loc[df['YIL_AY'] == "201712", 'DOLAR_ALIS'] = 3.85

    df.loc[df['YIL_AY'] == "201801", 'DOLAR_ALIS'] = 3.77

    df.loc[df['YIL_AY'] == "201802", 'DOLAR_ALIS'] = 3.78

    df.loc[df['YIL_AY'] == "201803", 'DOLAR_ALIS'] = 3.88

    df.loc[df['YIL_AY'] == "201804", 'DOLAR_ALIS'] = 4.05

    df.loc[df['YIL_AY'] == "201805", 'DOLAR_ALIS'] = 4.41

    df.loc[df['YIL_AY'] == "201806", 'DOLAR_ALIS'] = 4.63

    df.loc[df['YIL_AY'] == "201807", 'DOLAR_ALIS'] = 4.75

    df.loc[df['YIL_AY'] == "201808", 'DOLAR_ALIS'] = 5.73

    df.loc[df['YIL_AY'] == "201809", 'DOLAR_ALIS'] = 6.37

    df.loc[df['YIL_AY'] == "201810", 'DOLAR_ALIS'] = 5.86

    df.loc[df['YIL_AY'] == "201811", 'DOLAR_ALIS'] = 5.37

    df.loc[df['YIL_AY'] == "201812", 'DOLAR_ALIS'] = 5.31

    df.loc[df['YIL_AY'] == "201901", 'DOLAR_ALIS'] = 5.37

    df.loc[df['YIL_AY'] == "201902", 'DOLAR_ALIS'] = 5.26



dolar_alis(df)
dummy_cols = ['DAVRANIS', 'SEKTOR','SEKTOR_DAVRANIS','ceyrek','YIL_AY']
print(f'Shape before dummy transformation: {df.shape}')

df = pd.get_dummies(df, columns=[dummy_cols[0]],\

                          prefix=['DAVRANIS'], drop_first=True)

print(f'Shape after dummy transformation: {df.shape}')
print(f'Shape before dummy transformation: {df.shape}')

df = pd.get_dummies(df, columns=[dummy_cols[1]],\

                          prefix=['SEKTOR'], drop_first=True)

print(f'Shape after dummy transformation: {df.shape}')
print(f'Shape before dummy transformation: {df.shape}')

df = pd.get_dummies(df, columns=[dummy_cols[2]],\

                          prefix=['SEKTOR_DAVRANIS'], drop_first=True)

print(f'Shape after dummy transformation: {df.shape}')
print(f'Shape before dummy transformation: {df.shape}')

df = pd.get_dummies(df, columns=[dummy_cols[3]],\

                          prefix=['CEYREK'], drop_first=True)

print(f'Shape after dummy transformation: {df.shape}')
print(f'Shape before dummy transformation: {df.shape}')

df = pd.get_dummies(df, columns=[dummy_cols[4]],\

                          prefix=['YIL_AY'], drop_first=True)

print(f'Shape after dummy transformation: {df.shape}')
bin_dict = {'PESIN': 1, 'TAKSITLI': 0}

df['ISLEM_TURU'] = df['ISLEM_TURU'].map(bin_dict)
X=df[df.columns.difference(["ISLEM_TUTARI","ID","Record_Count"])]

X.head(3)
num_train=len(train)



X_train = X[:num_train]

X_test = X[num_train:]

y_train = train["ISLEM_TUTARI"].values



#X_train = X_train.astype(float)

#y_train = y_train.astype(float)
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

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

from math import sqrt
params = {

    'boosting_type': 'gbdt',

    'objective': 'regression',

    'metric': {'rmse'},

    'num_leaves': 64,

    'max_depth': 63,

    'learning_rate': 0.009,

    'min_data_in_leaf': 2,

    'bagging_freq': 1,

}
%%time



NFOLDS = 5

folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)



columns = X_train.columns

splits = folds.split(X_train, y_train)

y_preds = np.zeros(X_test.shape[0])

y_oof = np.zeros(X_train.shape[0])

score = 0



feature_importances = pd.DataFrame()

feature_importances['feature'] = columns



for fold_n, (train_index, valid_index) in enumerate(splits):

    X_tr, X_val = pd.DataFrame(X_train).loc[train_index], pd.DataFrame(X_train).loc[valid_index]

    y_tr, y_val = pd.DataFrame(y_train).loc[train_index], pd.DataFrame(y_train).loc[valid_index]

    

    dtrain = lgb.Dataset(X_tr, label=y_tr)

    dvalid = lgb.Dataset(X_val, label=y_val)



    clf = lgb.train(params, dtrain, 10000, valid_sets = [dtrain, dvalid], verbose_eval=100, early_stopping_rounds=100)

    

    feature_importances[f'fold_{fold_n + 1}'] = clf.feature_importance()

    

    y_pred_valid = clf.predict(X_val)

    y_oof[valid_index] = y_pred_valid

    print(f"Fold {fold_n + 1} | RMSE: {sqrt(mean_squared_error(y_val, y_pred_valid))}")

    

    score += sqrt(mean_squared_error(y_val.astype(float), y_pred_valid.astype(float))) / NFOLDS

    y_preds += clf.predict(X_test) / NFOLDS

    

    del X_tr, X_val, y_tr, y_val

    gc.collect()

    

print(f"\nMean RMSE = {score}")

print(f"Out of folds RMSE = {sqrt(mean_squared_error(y_train, y_oof))}")
feature_importances['average'] = feature_importances[[f'fold_{fold_n + 1}' for fold_n in range(folds.n_splits)]].mean(axis=1)

feature_importances.to_csv('feature_importances.csv')



plt.figure(figsize=(16, 16))

sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature');

plt.title('50 TOP feature importance over {} folds average'.format(folds.n_splits));
y_pred_train = clf.predict(X_train)

print(sqrt(mean_squared_error(y_train, y_pred_train)))
y_preds = clf.predict(X_test.astype(float))
y_preds = np.clip(y_preds, train['ISLEM_TUTARI'].min(), train['ISLEM_TUTARI'].max())
sample_submission['Predicted'] = y_preds
sample_submission.head()
sample_submission.to_csv('submission.csv', index=False)
from IPython.display import FileLink

FileLink(r'submission.csv')