import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sys import path

%matplotlib inline
train = pd.read_csv('../input/cs-training.csv')

test = pd.read_csv('../input/cs-test.csv')

df = pd.concat([train, test], ignore_index=1)
mask = df['SeriousDlqin2yrs'].notnull() # test部分SeriousDlqin2yrs值为空,可能影响部分变量的分布
df.shape
df.head()
df = df.drop('Unnamed: 0', axis=1) # 去掉重复的id列
# 样本平均违约率

df.SeriousDlqin2yrs.mean()
# 检查空值

df.info()
# 检查MonthlyIncome的分布

sns.distplot(df[df['MonthlyIncome'].notnull()]['MonthlyIncome'],

            bins=20)
# 可见MonthlyIncome存在大量的outlier,最好用众数填充空值

df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df.MonthlyIncome.median())
# 检查NumberOfDependents的分布

sns.distplot(df[df['NumberOfDependents'].notnull()]['NumberOfDependents'],

            bins=20)
# NumberOfDependents同样有大量outlier而且10以上的dependant显然不合常理

# 空值可以用平均数填充

df['NumberOfDependents'] = df['NumberOfDependents'].fillna(df.NumberOfDependents.mean())
# 检查age项的分布

g = sns.FacetGrid(df[mask], col='SeriousDlqin2yrs')

g.map(sns.distplot, 'age')
# 把age分组后检查各组平均违约率

bins = np.arange(0, 120, 10)

df['age_grouped'] = pd.cut(df['age'], bins, right=0)

gb = df[mask].groupby('age_grouped')['SeriousDlqin2yrs']

pd.concat([gb.count(), gb.mean()], axis=1)
# 重新分组,合并样本太少或者违约率过于接近的分组

bins = [0, 30, 40, 50, 60, 70, 110]

labels = ['0-29', '30-39', '40-49', '50-59', '60-69', '70+']

df['age_grouped'] = pd.cut(df['age'], bins, right=0, labels=labels)

gb = df[mask].groupby('age_grouped')['SeriousDlqin2yrs']

pd.concat([gb.count(), gb.mean()], axis=1)
sns.countplot(data=df, x='age_grouped', hue='SeriousDlqin2yrs')
# 检查RevolvingUtilizationOfUnsecuredLines,DebtRatio两项

df[['RevolvingUtilizationOfUnsecuredLines', 'DebtRatio']].describe()
# RevolvingUtilizationOfUnsecuredLines项离散化

bins = [0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.05, 

       df['RevolvingUtilizationOfUnsecuredLines'].max()]

labels = ['0-0.15', '0.15-0.30', '0.30-0.45', '0.45-0.60', '0.60-0.75', '0.75-0.90', '0.90-1.05', '1.05+']

# 以上分组是测试过多次的结果,测试过程这里省略

df['ru_grouped'] = pd.cut(df['RevolvingUtilizationOfUnsecuredLines'], bins, right=0, labels=labels)

gb = df[mask].groupby('ru_grouped')['SeriousDlqin2yrs']

pd.concat([gb.count(), gb.mean()], axis=1)
# DebtRatio项离散化

bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100,

       df['DebtRatio'].max()]

df['dr_grouped'] = pd.cut(df['DebtRatio'], bins, right=0)

gb = df[mask].groupby('dr_grouped')['SeriousDlqin2yrs']

pd.concat([gb.count(), gb.mean()], axis=1)
sns.distplot(df['DebtRatio'].apply(np.log1p)) # 取对数以后长尾部分一目了然
df['dr_log'] = df['DebtRatio'].apply(np.log1p)

bins = [0, 0.3, 0.6, 1.0, 3.0,  

       df['dr_log'].max()]

labels = ['0-0.3', '0.3-0.6', '0.6-1.0', '1.0-3.0', '3.0+']

# 以上分组同样是测试过多次的结果,测试过程省略

df['dr_grouped'] = pd.cut(df['dr_log'], bins, right=0, labels=labels)

gb = df[mask].groupby('dr_grouped')['SeriousDlqin2yrs']

pd.concat([gb.count(), gb.mean()], axis=1)
# 检查DaysPastDue的三列

pd_cols = ['NumberOfTime30-59DaysPastDueNotWorse',

          'NumberOfTime60-89DaysPastDueNotWorse',

          'NumberOfTimes90DaysLate']

df[pd_cols].describe()
# 计算有逾期记录的人数占比

df[pd_cols][df[pd_cols]!=0].count()/df.count().max()
# 检查这三列的相关性

df[pd_cols].corr()
# 尝试将这三列简化成一个变量:是否有90+以上的逾期记录

df['pd_90+'] = (df['NumberOfTimes90DaysLate']>0).astype(int)

df.groupby('pd_90+')['SeriousDlqin2yrs'].mean()
df['MonthlyIncome'].describe()
# 同样是极值很多,取对数后再看一下分布

df['income_log'] = (df['MonthlyIncome']/10000).apply(np.log1p)

sns.distplot(df['income_log'])
# 检查NumberOfOpenCreditLinesAndLoans, NumberRealEstateLoansOrLines两列

num_cols = ['NumberOfOpenCreditLinesAndLoans', 'NumberRealEstateLoansOrLines']

df[num_cols].describe()
sns.relplot(data=df, x=num_cols[0], y=num_cols[1], hue='SeriousDlqin2yrs')
# 对NumberOfOpenCreditLinesAndLoans分组

bins = [0, 2, 4, 6, 10, 14,

       df['NumberOfOpenCreditLinesAndLoans'].max()]

labels = ['0-1', '2-3', '4-5', '6-9', '10-13', '14+']

df['num_oc_grouped'] = pd.cut(df['NumberOfOpenCreditLinesAndLoans'], bins, right=0, labels=labels)

gb = df[mask].groupby('num_oc_grouped')['SeriousDlqin2yrs']

pd.concat([gb.count(), gb.mean()], axis=1)
# 对NumberRealEstateLoansOrLines分组

bins = [0, 1, 3, 

       df['NumberRealEstateLoansOrLines'].max()]

labels = ['0', '1-2', '3+']

df['num_re_grouped'] = pd.cut(df['NumberRealEstateLoansOrLines'], bins, right=0, labels=labels)

gb = df[mask].groupby('num_re_grouped')['SeriousDlqin2yrs']

pd.concat([gb.count(), gb.mean()], axis=1)
# 对NumberOfDependents分组

bins = [0, 1, 2, 4, 

       df['NumberOfDependents'].max()]

labels = ['0', '1', '2-3', '4+']

df['num_dep_grouped'] = pd.cut(df['NumberOfDependents'], bins, right=0, labels=labels)

gb = df[mask].groupby('num_dep_grouped')['SeriousDlqin2yrs']

pd.concat([gb.count(), gb.mean()], axis=1)
# 舍去没有用到的变量

df1 = df.drop(['RevolvingUtilizationOfUnsecuredLines',

              'age',

              'DebtRatio',

              'MonthlyIncome',

              'NumberOfDependents',

              'dr_log']

              + num_cols + pd_cols, axis=1)

df1.columns
df2 = pd.get_dummies(df1, drop_first=True)

df2.columns
# 由于整个样本中违约样本占比仅有6-7%,在训练模型时应该对违约样本up sample处理

X0 = df2[df['SeriousDlqin2yrs']==0].drop('SeriousDlqin2yrs', axis=1).values

X1 = df2[df['SeriousDlqin2yrs']==1].drop('SeriousDlqin2yrs', axis=1).values

y0 = df2[df['SeriousDlqin2yrs']==0]['SeriousDlqin2yrs'].values

y1 = df2[df['SeriousDlqin2yrs']==1]['SeriousDlqin2yrs'].values

print(X0.shape, X1.shape, y0.shape, y1.shape)
X1_upsample = np.ndarray(X0.shape)

y1_upsample = np.ndarray(y0.shape)

for i in np.arange(len(y0)):

    idx = np.random.randint(0, y1.shape[0])

    X1_upsample[i] = (X1[idx])

    y1_upsample[i] = (y1[idx])

print(X1_upsample.shape, y1_upsample.shape)
X = np.concatenate([X0, X1_upsample])

y = np.concatenate([y0, y1_upsample])

print(X.shape, y.shape)
#X = df2[mask].drop('SeriousDlqin2yrs', axis=1).values

#y = df2[mask]['SeriousDlqin2yrs'].values

#print(X.shape, y.shape)
y.sum()
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
#scaler = StandardScaler().fit(X)

#Xt = scaler.transform(X)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, shuffle=True)
# GridSearch运行时间过长,代码如下

#

# params_LR = {'C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10],

#             'solver': ['lbfgs', 'liblinear']}

# gs = GridSearchCV(LogisticRegression(max_iter=1000), 

#                   param_grid = params_LR,

#                   scoring = 'f1',

#                   cv=5).fit(X_train, y_train)

# gs.best_params_
model_LR = LogisticRegression(C=3, solver='lbfgs').fit(X_train, y_train)
print('F1 Score: %.6f' % f1_score(y_valid, model_LR.predict(X_valid)))

print('AUC Score: %.6f' %  roc_auc_score(y_valid, model_LR.predict(X_valid)))

print('Confusion Matrix: \n', confusion_matrix(y_valid, model_LR.predict(X_valid)))
# 加入XGBoost测试

import xgboost as xgb

params_xgb = {'max_depth': 6,

              'eta': 1,

              'silent': 1,

              'objective': 'binary:logistic',

              'eval_matric': 'f1'}



# 借用GaryMulder的参数:

params_xgb2 = {'max_depth': 5,

               'eta': 0.025,

               'silent':1,

               'objective': 'binary:logistic',

               'eval_matric': 'auc',

               'minchildweight': 10.0,

               'maxdeltastep': 1.8,

               'colsample_bytree': 0.4,

               'subsample': 0.8,

               'gamma': 0.65,

               'numboostround' : 391}
dtrain = xgb.DMatrix(X_train, y_train, feature_names=df2.columns.drop('SeriousDlqin2yrs'))

dvalid = xgb.DMatrix(X_valid, y_valid, feature_names=df2.columns.drop('SeriousDlqin2yrs'))

evals = [(dtrain, 'train'), (dvalid, 'valid')]

model_xgb = xgb.train(params_xgb2, dtrain, 1000, evals, early_stopping_rounds=100);
model_xgb.dump_model('xgb_v1')
X_test = df2[df['SeriousDlqin2yrs'].isnull()].drop('SeriousDlqin2yrs', axis=1).values

X_test.shape
dtest = xgb.DMatrix(X_test, feature_names=df2.columns.drop('SeriousDlqin2yrs'))

y_test = model_xgb.predict(dtest)
entry = pd.DataFrame()

entry['ID'] = np.arange(1, len(y_test)+1)

entry['Probability'] = y_test

entry.to_csv('entry02.csv', header=True, index=False)