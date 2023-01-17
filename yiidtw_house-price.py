import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
df_train_orig = pd.read_csv('../input/train.csv')
df_train_orig['SalePrice'].describe()
sns.distplot(df_train_orig['SalePrice'], fit=norm);
fig = stats.probplot(df_train_orig['SalePrice'], plot=plt)
print("Skewness: %f" % df_train_orig['SalePrice'].skew())
print("Kurtosis: %f" % df_train_orig['SalePrice'].kurt())
# 下面這行只是在調整圖片大小
f, ax = plt.subplots(figsize=(12, 9))

# 畫 heatmap
sns.heatmap(df_train_orig.corr(), vmax=.8, square=True);
# 相似的指令 plt.matshow(df_train.corr())
df_train_orig.corr().nlargest(20, 'SalePrice')['SalePrice']
df_train = pd.concat([df_train_orig['SalePrice'], \
                      df_train_orig['OverallQual'], \
                      df_train_orig['GrLivArea'], \
                      df_train_orig['GarageCars'], \
                      df_train_orig['TotalBsmtSF'], \
                      df_train_orig['MasVnrArea'], \
                      df_train_orig['Fireplaces']], axis=1).copy()
df_train.isnull().sum()
# 我們前面已經畫過這張圖了，看起來可以取個 log
sns.distplot(df_train['SalePrice'], fit=norm);
sns.distplot(np.log(df_train['SalePrice']), fit=norm);
df_train['log_SalePrice'] = np.log(df_train['SalePrice'])
sns.distplot(df_train['GrLivArea'], fit=norm);
sns.distplot(np.log(df_train['GrLivArea']), fit=norm);
df_train['log_GrLivArea'] = np.log(df_train['GrLivArea'])
sns.distplot(df_train['TotalBsmtSF'], fit=norm);
totalbsmtzero_idx = df_train[df_train['TotalBsmtSF'] == 0].index.tolist()
print("共有 %d 組為 0 ，所佔比例 %.2f " % (len(totalbsmtzero_idx), len(totalbsmtzero_idx) / df_train['TotalBsmtSF'].count() * 100))
# 沒有地下室或沒有地下室面積資料 (TotalBsmtSF == 0) 所佔比率約為 2.5 %
# 去掉 0 做 log transformation
tmp_totalbsmtsf = df_train['TotalBsmtSF'].copy()
sns.distplot(np.log(tmp_totalbsmtsf.drop(totalbsmtzero_idx)), fit=norm);
# 如果用 binning 的話
tmp2_totalbsmtsf = df_train['TotalBsmtSF'].copy()
tmp2_mean = df_train['TotalBsmtSF'].mean()
tmp2_std = df_train['TotalBsmtSF'].std()
bins = [tmp2_totalbsmtsf.min()-1, tmp2_mean - 2*tmp2_std, tmp2_mean - tmp2_std, tmp2_mean, tmp2_mean + tmp2_std, tmp2_mean + 2*tmp2_std, tmp2_totalbsmtsf.max()+1]
labels = [1,2,3,4,5,6]
tmp2_totalbsmtsf_cut = pd.cut(tmp2_totalbsmtsf, bins=bins, labels=labels)
sns.distplot(tmp2_totalbsmtsf_cut, fit=norm);
# 決定直接去掉為 0 的 row，做 log transformation
df_train.count()
df_train = df_train.drop(totalbsmtzero_idx)
df_train.count()
df_train['log_TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
# 前面已經知道有 null 值了
masvnrnull_idx = df_train[df_train['MasVnrArea'].isnull()].index.tolist()
masvnrnull_idx
# 先看一下圖長怎樣，再決定怎麼填 null 值
tmp_masvnrarea = df_train['MasVnrArea'].copy()
sns.distplot(tmp_masvnrarea.drop(masvnrnull_idx), fit=norm);
masvnrarea_ratio = df_train[df_train['MasVnrArea'] == 0]['MasVnrArea'].count() / df_train['MasVnrArea'].count() * 100
print("有石造物/全部 = %.2f " % masvnrarea_ratio)
# 決定直接分成有石造物與沒有的兩組，先把 null 填 0
df_train['MasVnrArea'].loc[masvnrnull_idx] = 0
df_train['has_MasVnsArea'] = (df_train['MasVnrArea'] > 0).astype(float)
df_train[['MasVnrArea', 'has_MasVnsArea']].head()
#先看一下目前處理的狀況
df_train.columns
dummy_fields = ['OverallQual', 'GarageCars', 'Fireplaces', 'has_MasVnsArea']
riders = df_train.copy()
for field in dummy_fields:
    dummies = pd.get_dummies( riders.loc[:, field], prefix=field)
    riders = pd.concat([riders, dummies], axis = 1)
riders['GarageCars_5'] = 0
riders['Fireplaces_4'] = 0
drop_fields = ['OverallQual', 'GarageCars', 'Fireplaces', 'has_MasVnsArea', 'GrLivArea', 'TotalBsmtSF', 'MasVnrArea']
df_data = riders.drop(drop_fields, axis = 1)
df_data.shape
df_data.head()
train_x = df_data.iloc[:,[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,26]].values
train_y = df_data.iloc[:,[1]].values
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(train_x, train_y)
print(lm.intercept_)
print(lm.coef_)
train_yy = lm.predict(train_x)
from sklearn.metrics import mean_squared_error
mean_squared_error(train_yy, train_y)
sns.distplot(train_y - train_yy)
sns.distplot(df_data.iloc[:,[0]].values - np.exp(train_yy))
np.exp(train_yy)
np.exp(train_y)
df_test_orig = pd.read_csv('../input/test.csv')
df_test_orig.head()

df_test = df_test_orig[['Id', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'MasVnrArea', 'Fireplaces']].copy()
df_test.head()
df_test.isnull().sum()
nullgaragecars_idx = df_test[df_test['GarageCars'].isnull()].index.tolist()
df_test['GarageCars'].loc[nullgaragecars_idx] = 2
nulltotalbsmtsf_test_idx = df_test[df_test['TotalBsmtSF'].isnull()].index.tolist()
df_test['TotalBsmtSF'].loc[nulltotalbsmtsf_test_idx] = df_test['TotalBsmtSF'].mean()
nullmasvnrarea_test_idx = df_test[df_test['MasVnrArea'].isnull()].index.tolist()
df_test['MasVnrArea'].loc[nullmasvnrarea_test_idx] = 0
df_test.isnull().sum()
df_test['log_GrLivArea'] = np.log(df_test['GrLivArea'])
df_test['log_TotalBsmtSF'] = np.log(df_test['TotalBsmtSF'])
df_test['has_MasVnsArea'] = (df_test['MasVnrArea'] > 0).astype(float)

dummy_fields = ['OverallQual', 'GarageCars', 'Fireplaces', 'has_MasVnsArea']
riders_test = df_test.copy()
for field in dummy_fields:
    dummies = pd.get_dummies( riders_test.loc[:, field], prefix=field)
    riders_test = pd.concat([riders_test, dummies], axis = 1)

drop_fields = ['OverallQual', 'GarageCars', 'Fireplaces', 'has_MasVnsArea', 'GrLivArea', 'TotalBsmtSF', 'MasVnrArea']
df_data_test = riders_test.drop(drop_fields, axis = 1)
df_data_test.shape
df_data_test.head()
test_x = df_data_test.iloc[:,[1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25]].values
test_yy = lm.predict(test_x)
res = pd.concat([df_data_test['Id'], pd.DataFrame(np.exp(test_yy))], axis=1)
res.columns = ['Id', 'SalePrice']
res.head()
res.to_csv('summission.csv', encoding='utf-8', index = False)
