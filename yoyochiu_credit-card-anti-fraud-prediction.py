import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import xgboost as xgb

import lightgbm as lgbm

from sklearn.ensemble import GradientBoostingClassifier as gbdt

from sklearn.linear_model import LogisticRegression



from sklearn.preprocessing import LabelEncoder, MinMaxScaler, Imputer

from sklearn.model_selection import GridSearchCV, KFold, train_test_split

from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score

from datetime import datetime, timedelta



from imblearn.over_sampling import SMOTE



from heamy.dataset import Dataset

from heamy.estimator import Regressor, Classifier

from heamy.pipeline import ModelsPipeline



import warnings

warnings.filterwarnings('ignore')  



%matplotlib inline
#导入原始数据集

ini_2018q3 = pd.read_csv('../input/LoanStats_securev1_2018Q3.csv', skiprows=1)

ini_2018q4 = pd.read_csv('../input/LoanStats_securev1_2018Q4.csv', skiprows=1)
#数据规模

print('shape of 2018q3: ', ini_2018q3.shape)

print('shape of 2018q4: ', ini_2018q4.shape)
#数据形式

pd.options.display.max_columns = None

ini_2018q3
ini_2018q4
#删去每个表的最后两行

ini_2018q3.drop([128194, 128195], inplace=True)

ini_2018q4.drop([128412, 128413], inplace=True)
pd.merge(ini_2018q3, ini_2018q4, on='id').shape
ini_data = pd.concat([ini_2018q3, ini_2018q4], ignore_index=True)
ini_data.shape
ini_data.drop_duplicates(inplace=True)

ini_data.shape
ini_data.drop_duplicates('id', inplace=True)

ini_data.shape
# 查看每个特征的缺失值数量与比例



def missing_values_table(data):

    #计算缺失值数量与占比

    mis_val = data.isna().sum()

    mis_val_percent = 100 * data.isna().sum() / len(data)

    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    

    #重命名表列

    mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0:'missing values', 1:'% of total values'})

    

    #按照缺失值占比对表格进行排序

    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] !=0].sort_values(

    '% of total values', ascending = False).round(1)

    

    return mis_val_table_ren_columns
pd.options.display.max_rows = None

missing_values_table(ini_data)
#删除缺失值达50%以上的列

thresh_count = len(ini_data) * 0.5

data_dena = ini_data.dropna(thresh=thresh_count, axis=1)

data_dena.shape
#查看各个缺失特征的具体取值

pd.options.display.max_rows = 15



na_columns = missing_values_table(data_dena)

na_columns = data_dena[na_columns.index]

na_columns
na_columns.info()
#删去无意义列

na_columns2 = na_columns.drop(['next_pymnt_d', 'emp_title', 'last_pymnt_d', 'last_credit_pull_d'], axis=1)

na_columns2.shape
#处理特征emp_length

na_columns2['emp_length'] = na_columns2['emp_length'].replace({'< 1 year':0,

                                 '1 year':1,

                                 '2 years':2,

                                 '3 years':3,

                                 '4 years':4,

                                 '5 years':5,

                                 '6 years':6,

                                 '7 years':7,

                                 '8 years':8,

                                 '9 years':9,

                                 '10+ years':10}).astype('float64')
#处理特征revol_util

na_columns2['revol_util'] = na_columns2['revol_util'].str.rstrip('%').astype('float64')
#现在需要进行缺失值填充的都是数值型特征，因此用imputer进行中位数填充

imp = Imputer(missing_values='NaN', strategy='median', axis=0)

na_columns3 = imp.fit_transform(na_columns2)

na_columns3 = pd.DataFrame(na_columns3)

na_columns3.columns = na_columns2.columns
#把缺失值处理结果更新到总数据集中

data_dena2 = data_dena.drop(na_columns.columns, axis=1)

data_dena2 = pd.concat([data_dena2, na_columns3], axis=1)

data_dena3 = data_dena2.copy() #这份数据是为了之后数据探索使用
#先看看经过缺失值处理后的现存特征

data_dena2.head()
pd.options.display.max_rows = None

data_dena2.info()
#处理特征term

data_dena2['term'] = data_dena2['term'].map(lambda x: x.replace('months',''))

data_dena2['term'] = data_dena2['term'].astype('float64')
#处理特征int_rate

data_dena2['int_rate'] = data_dena2['int_rate'].str.rstrip('%').astype('float64')
#处理特征grade与sub_grade：二者都是表示用户等级，只取其中一个特征即可，因此删除sub_grade，并且把grade转化为float类型

data_dena2['grade'] = data_dena2['grade'].replace({'A':1,

                                                   'B':2,

                                                   'C':3,

                                                   'D':4,

                                                   'E':5,

                                                   'F':6,

                                                   'G':7}).astype('float')

del data_dena2['sub_grade']
pd.options.display.max_rows = 10
#处理特征home_ownership：这是无序性类别特征，用labelecoding

le = LabelEncoder()

data_dena2['home_ownership'] = le.fit_transform(data_dena2['home_ownership']).astype('float')
#处理特征verification_status：这是无序性类别特征，用labelecoding

data_dena2['verification_status'] = le.fit_transform(data_dena2['verification_status']).astype('float')
#处理特征issue_d：表示贷款发放月份，这个特征会给模型泄露信息，删去

del data_dena2['issue_d']
#处理特征pymnt_plan：表示是否已为贷款制定了付款计划，是无序类型变量，用labelencoding

data_dena2['pymnt_plan'] = le.fit_transform(data_dena2['pymnt_plan']).astype('float')
#处理特征url：无意义，直接删除

del data_dena2['url']
#处理特征purpose：借款人贷款目的，是无序类型变量，用labelencoding

data_dena2['purpose'] = le.fit_transform(data_dena2['purpose']).astype('float')
#处理特征title：借款人提供的贷款名目，与purpose高度相似，直接删去

del data_dena2['title']
#处理特征zip_code：借款人提供的邮政编码前三位，接近900个取值，对预测目标的意义不大，直接删去

del data_dena2['zip_code']
#处理特征addr_state：借款人所在州，是无序类型变量，用labelencoding

data_dena2['addr_state'] = le.fit_transform(data_dena2['addr_state']).astype('float')
#处理特征earliest_cr_line：时间特征，转换为datatime类型

data_dena2['earliest_cr_line'] = pd.to_datetime(data_dena2['earliest_cr_line'])
#处理特征initial_list_status：贷款的初始列表状态，可能的值为– W，F，进行labelencoding

data_dena2['initial_list_status'] = le.fit_transform(data_dena2['initial_list_status']).astype('float')
#处理特征application_type：贷款申请类型，是个人贷款还是共同贷款，进行labelencoding

data_dena2['application_type'] = le.fit_transform(data_dena2['application_type']).astype('float')
#处理特征hardship_flag：这是贷后信息，删去

del data_dena2['hardship_flag']
#处理特征debt_settlement_flag：这是贷后信息，删去

del data_dena2['debt_settlement_flag']
data_dena2.head()
data_dena2.info()
data_corr = data_dena2.drop(['id', 'loan_status'], axis=1)
corr_matrix = data_corr.corr()
fig = plt.figure(figsize=(25, 20))

sns.heatmap(corr_matrix, linewidths=0.1)

plt.xticks(rotation='vertical')
print(data_corr['policy_code'].value_counts())

print(data_corr['acc_now_delinq'].value_counts())

print(data_corr['num_tl_30dpd'].value_counts())

print(data_corr['tax_liens'].value_counts())

print(data_corr['num_tl_120dpd_2m'].value_counts())
del data_corr['policy_code']

del data_corr['acc_now_delinq']

del data_corr['num_tl_30dpd']

del data_corr['tax_liens']

del data_corr['num_tl_120dpd_2m']
pd.options.display.max_rows = None

corr_matrix2 = corr_matrix.abs()[corr_matrix.abs() >= 0.85]

corr_matrix2.stack()
data_corr = data_corr.drop(['funded_amnt',

                            'funded_amnt_inv',

                            'out_prncp',

                            'out_prncp_inv',

                            'fico_range_high',

                            'num_op_rev_tl',

                            'num_sats',

                            'pub_rec_bankruptcies',

                            'total_pymnt_inv',

                            'total_pymnt',

                            'total_rec_prncp',

                            'last_pymnt_amnt',

                            'collection_recovery_fee',

                            'tot_hi_cred_lim',

                            'total_bal_ex_mort',

                            'total_il_high_credit_limit',

                            'num_actv_rev_tl',

                            'revol_util',

                            'total_bc_limit',

                            'last_fico_range_high'], axis=1)
data_decorr = pd.concat([data_dena2[['id', 'loan_status']], data_corr], axis=1)
pd.options.display.max_rows = 10

data_decorr
del data_decorr['total_rec_int']

del data_decorr['total_rec_late_fee']
data1 = data_decorr.copy()

data1
#删除状态为‘default’的数据

data1 = data1[data1['loan_status'] != 'Default']
#对标签值进行0/1编码

data1['loan_status'] = data1['loan_status'].replace({'Current':0,

                                                     'Fully Paid':0,

                                                     'Charged Off':1,

                                                     'In Grace Period':1,

                                                     'Late (16-30 days)':1,

                                                     'Late (31-120 days)':1})
#查看标签值分布情况

plt.style.use('seaborn-whitegrid')

fig = plt.figure(figsize=(8, 6))

sns.countplot('loan_status', data=data1, palette='hls')
#正负样本比例

positive = data1[data1['loan_status'] == 1]['loan_status'].count()

negetive = data1[data1['loan_status'] == 0]['loan_status'].count()



positive_rate = positive / (positive + negetive)



print('正样本的数量为：', positive)

print('负样本的数量为：', negetive)

print('正样本占所有样本比例：', positive_rate)
#'earliest_cr_line'转换成距今时间

now = datetime(2019,10,15)

data1['earliest_cr_line'] = (now - data1['earliest_cr_line']).map(lambda x: x.days)
onehot_col = ['term', 'home_ownership', 'verification_status', 'purpose']



onehot = data1[['id', 'term', 'home_ownership', 'verification_status', 'purpose']]

onehot = pd.get_dummies(onehot, columns=onehot_col)



data_onehot = data1.drop(onehot_col, axis=1)

data_onehot = pd.merge(data_onehot, onehot, on='id', how='left')
#切分特征集与标签集

data_fea = data_onehot.drop(['id', 'loan_status'], axis=1)

data_label = data_onehot['loan_status']
#使用xgboost的输出结果构造新特征

xgb_getfea = xgb.XGBClassifier(objective='binary:logistic', eta=0.05, max_depth=3, reg_lambda=1, subsample=0.7, colsample_bytree=0.8, n_estimators=100)

xgb_getfea.fit(data_fea, data_label)

new_fea = xgb_getfea.apply(data_fea)
#把新特征与原特征集合并

new_fea = pd.DataFrame(new_fea)

data_add_fea = pd.concat([data_fea, new_fea], axis=1)
data_add_fea
#特征归一化

scaler = MinMaxScaler()

data_fea_scale = scaler.fit_transform(data_add_fea)
#切分数据集，用于输出特征重要性

xtrain, xtest, ytrain, ytest = train_test_split(data_add_fea, data_label, test_size=0.3, random_state=0)
#使用lightgbm得出特征重要性

lgb = lgbm.LGBMClassifier(objective='binary')

lgb.fit(xtrain, ytrain)
## 查看特征重要性

importance = lgb.feature_importances_

importance_table = {'fea':[], 'importance':[]}



for i in list(range(len(importance))):

    importance_table['fea'].append(data_add_fea.columns[i])

    importance_table['importance'].append(importance[i])

              

importance_table = pd.DataFrame(importance_table).sort_values(by='importance', ascending=False)



pd.options.display.max_rows = None

importance_table
pd.options.display.max_rows = 10
#划分训练集与测试集

xtrain, xtest, ytrain, ytest = train_test_split(data_add_fea, data_label, test_size=0.3, random_state=0)
#SMOTE过采样

sm = SMOTE(sampling_strategy=0.25, random_state=2)

xtrain_sm, ytrain_sm = sm.fit_sample(xtrain, ytrain)
xtrain_sm2 = pd.DataFrame(xtrain_sm)

xtrain_sm2.columns = xtest.columns
#模型1-GBDT

def GBDT(X_train, y_train, X_test, y_test=None):

    GBDT = gbdt(learning_rate=0.02, max_features=0.7, n_estimators=100 , max_depth=5)

    model = GBDT.fit(X_train, y_train)

    pre = model.predict(X_test)

    return pre
#模型2-XGBOOST

def XGBOOST(X_train, y_train, X_test, y_test=None):

    XGBOOST = xgb.XGBClassifier(objective='binary:logistic', eta=0.05, max_depth=5, 

                                 reg_lambda=1, subsample=0.7, colsample_bytree=0.8, n_estimators=200)

    model = XGBOOST.fit(X_train, y_train)

    pre = model.predict(X_test)

    return pre
#模型3-LIGHTGBM

def LIGHTGBM(X_train, y_train, X_test, y_test=None):

    LIGHTGBM = lgbm.LGBMClassifier(objective='binary', n_estimators=500, learning_rate=0.05,

                          num_leaves=40, max_depth=6, colsample_bytree=0.7, subsample=0.6, reg_lambda=1)

    model = LIGHTGBM.fit(X_train, y_train)

    pre = model.predict(X_test)

    return pre
dataset = Dataset(X_train=xtrain_sm2, y_train=ytrain_sm, X_test=xtest, y_test=None)
#各个基模型初始化

model_gbdt = Classifier(dataset=dataset, estimator=GBDT, name='gbdt', use_cache=False)

model_xgboost = Classifier(dataset=dataset, estimator=XGBOOST, name='xgb', use_cache=False)

model_lightgbm = Classifier(dataset=dataset, estimator=LIGHTGBM, name='lgb', use_cache=False)
#第一层stacking

pipeline = ModelsPipeline(model_gbdt, model_xgboost, model_lightgbm)

stack_ds = pipeline.stack(k=3,seed=111)
def logistic_model(X_train, y_train, X_test, y_test=None):

    model = LogisticRegression(penalty = 'l2').fit(X_train,y_train)

    return model.predict(X_test)
stacker = Classifier(dataset=stack_ds, estimator=logistic_model)

results = stacker.predict()
f1score = f1_score(ytest, results)

print('模型在测试集上的f1得分为：', f1score)