import os

import scipy as sp

import numpy as np

import sklearn

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

from sklearn.linear_model import Lasso

from sklearn import svm

from sklearn.ensemble import RandomForestRegressor

from sklearn.datasets import make_regression

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error



import warnings

warnings.filterwarnings('ignore')
df_train=pd.read_csv('./train_set.csv')

df_test=pd.read_csv('./test_set.csv')
df_train.head()
df_train.shape
df_test.head()
df_test.shape
df_train_index = df_train['Id']

df_test_index = df_test['Id']

df_train.drop(['Id'], axis =1, inplace=True)

df_test.drop(['Id'], axis =1, inplace=True)
df_train.head()
df_test.head()
y_train = df_train['PRICE'].values

df_train.drop(['PRICE'], axis =1, inplace=True)
df_train.head()
df_all = pd.concat((df_train, df_test)).reset_index(drop=True)
df_all.head()
df_all.shape
df_all.isnull().sum()[df_all.isnull().sum()!=0]. sort_values(ascending=False)
df_all['LIVING_GBA'].fillna('NA',inplace=True)

df_all['CMPLX_NUM'].fillna('NA',inplace=True)

df_all['FULLADDRESS'].fillna('NA',inplace=True)

df_all['CENSUS_BLOCK'].fillna('NA',inplace=True)

df_all['NATIONALGRID'].fillna('NA',inplace=True)

df_all['STATE'].fillna('NA',inplace=True)

df_all['CITY'].fillna('NA',inplace=True)

df_all['YR_RMDL'].fillna('NA',inplace=True)

df_all['STORIES'].fillna('NA',inplace=True)

df_all['KITCHENS'].fillna('NA',inplace=True)

df_all['INTWALL'].fillna('NA',inplace=True)

df_all['GRADE'].fillna('NA',inplace=True)

df_all['GBA'].fillna('NA',inplace=True)

df_all['STYLE'].fillna('NA',inplace=True)

df_all['STRUCT'].fillna('NA',inplace=True)

df_all['NUM_UNITS'].fillna('NA',inplace=True)

df_all['CNDTN'].fillna('NA',inplace=True)

df_all['EXTWALL'].fillna('NA',inplace=True)

df_all['ROOF'].fillna('NA',inplace=True)

df_all['ASSESSMENT_SUBNBHD'].fillna('NA',inplace=True)

df_all['AYB'].fillna('NA',inplace=True)

df_all['Y'].fillna('NA',inplace=True)

df_all['X'].fillna('NA',inplace=True)

df_all['QUADRANT'].fillna('NA',inplace=True)

df_all['SALEDATE'].fillna('NA',inplace=True)
df_all.isnull().sum()[df_all.isnull().sum()!=0]. sort_values(ascending=False)
df_all.dtypes
for i in range(df_all.shape[1]):

    if df_all.iloc[:,i].dtypes == object:

        lbl= LabelEncoder()

        lbl.fit(list(df_all.iloc[:,i]))

        df_all.iloc[:,i] = lbl.transform(list(df_all.iloc[:,i].values))
df_all.head()
ntrain = df_train.shape[0]



train = df_all[:ntrain]

test = df_all[ntrain:]

y = y_train

x = train.loc[:,train.columns!='PRICE']

ylog = np.log(y)
def RMSE(y_train, y_pred):

    return np.sqrt(mean_squared_error(y_train,y_pred))



rf = RandomForestRegressor()

rf.fit(x, ylog)

y_predlog_rf = rf.predict(x)

y_pred_rf = np.exp(y_predlog_rf)

print('RFでRMSE', RMSE(y_train,y_pred_rf))
from sklearn import cross_validation, preprocessing, linear_model #機械学習用のライブラリを利用(Lasso)

Lasso= linear_model.Lasso(alpha=1.0)

Lasso.fit(x, df_ylog)

y_pred_lasso_log = Lasso.predict(x)

y_pred_lasso = np.exp(y_pred_lasso_log)

print('LassoでRMSE', RMSE(y_train,y_pred_lasso))
from sklearn.model_selection import cross_val_score



# 交差検証

scores_rf = cross_val_score(rf, x , y_train, cv=5, scoring='neg_mean_squared_error')

scores_corrected_rf = np.sqrt(-1 * scores_rf)

# 各分割におけるスコア

print('Cross-Validation scores: {}'.format(scores_corrected_rf))

# スコアの平均値

import numpy as np

print('Average score: {}'.format(np.mean(scores_corrected_rf)))
from sklearn.model_selection import cross_val_score



# 交差検証

scores_lasso = cross_val_score(Lasso, x , y_train, cv=5, scoring='neg_mean_squared_error')

scores_corrected_lasso = np.sqrt(-1 * scores_lasso)

# 各分割におけるスコア

print('Cross-Validation scores: {}'.format(scores_corrected_lasso))

# スコアの平均値

import numpy as np

print('Average score: {}'.format(np.mean(scores_corrected_lasso)))
y1 = np.array(y_pred_rf)

y2 = np.array(y_pred_lasso)

y_pred_combined = 0.9 * y1 + 0.1 * y2
y_train_log =  np.log(y_train)

y_pred_combined_log = np.log(y_pred_combined)



plt.figure(figsize=(16,3))

plt.hist(y_train_log,bins=100,rwidth=0.8)

plt.hist(y_pred_combined_log,bins=100,rwidth=0.8)
plt.scatter(y_train_log, y_pred_combined_log)
y_pred_rf_test_log = rf.predict(test)

y_pred_rf_test = np.exp(y_pred_rf_test_log)



y_pred_lasso_test_log = Lasso.predict(test)

y_pred_lasso_test = np.exp(y_pred_lasso_test_log)



y1 = np.array(y_pred_rf_test)

y2 = np.array(y_pred_lasso_test)

y_pred_combined_test = 0.9 * y1 + 0.1 * y2
submission = pd.concat((df_test_index, pd.DataFrame(y_pred_combined_test)),axis=1)

submission.columns = ['id','PRICE']
submission.head()
submission.to_csv('RF_Lasso_combined_allpara.csv', sep=',', index=False)
y_pred_combined_2_test = 0.8 * y1 + 0.2 * y2
submission = pd.concat((df_test_index, pd.DataFrame(y_pred_combined_2_test)),axis=1)

submission.columns = ['id','PRICE']
submission.head()
submission.to_csv('RF_Lasso_combined_allpara_2.csv', sep=',', index=False)
y_pred_combined_3_test = 0.85 * y1 + 0.15 * y2
submission = pd.concat((df_test_index, pd.DataFrame(y_pred_combined_3_test)),axis=1)

submission.columns = ['id','PRICE']
submission.head()
submission.to_csv('RF_Lasso_combined_allpara_3.csv', sep=',', index=False)
y_pred_combined_4_test = 0.95 * y1 + 0.05 * y2
submission = pd.concat((df_test_index, pd.DataFrame(y_pred_combined_4_test)),axis=1)

submission.columns = ['id','PRICE']
submission.head()
submission.to_csv('RF_Lasso_combined_allpara_4.csv', sep=',', index=False)