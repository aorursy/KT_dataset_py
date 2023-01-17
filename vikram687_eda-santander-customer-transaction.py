# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import datetime
%%time

train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
print('Train data Rows {} and Clomuns {}'.format(train_df.shape[0], train_df.shape[1]))



print('Test data Rows {} and Clomuns {}'.format(test_df.shape[0], test_df.shape[1]))
# load the random train data

train_df.sample(5)
#load the random test data

test_df.sample(5)
#Check the train data type

train_df.info()
#ID_code are object type 

#Traget values are in int64

#Variable data are in float
#Check the test data type

test_df.info()
#Now check for missing values

total = train_df.isnull().sum().sort_values(ascending = False)

percent = (train_df.isnull().sum()/train_df.isnull().count()*100).sort_values(ascending = False)

missing_df = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# As we can see there is No mising values in data set

missing_df.head()
#Now check for missing values

total = test_df.isnull().sum().sort_values(ascending = False)

percent = (test_df.isnull().sum()/test_df.isnull().count()*100).sort_values(ascending = False)

missing_df_test = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_df_test.head()
train_df.describe()
test_df.describe()
#Lets check for target data distribution

sns.set(style="darkgrid")

sns.countplot(train_df['target'],palette="Set3")

plt.show()
%%time

print('% values of Target 0 is {}'.format(100*train_df['target'].value_counts()[0]/train_df.shape[0]))

print('% values of Target 1 is {}'.format(100*train_df['target'].value_counts()[1]/train_df.shape[0]))
#90% of data contain 0 values & 10% is 1 ----> (Imbalance data)
from scipy.stats import norm

from scipy import stats

#histogram and normal probability plot

sns.distplot(train_df['target'], fit=norm);

fig = plt.figure()

res = stats.probplot(train_df['target'], plot=plt)
#train_df.drop(axis=1, columns=['ID_code'], inplace=True)
#X = train_df.drop(['target'], axis=1)

#y = train_df['target']
#from sklearn.feature_selection import RFE

#from sklearn.ensemble import RandomForestClassifier
#rfe_selector = RFE(estimator=RandomForestClassifier(), n_features_to_select=1, step=1, verbose=0)

#rfe_selector.fit(X, y)
#rfe_support = rfe_selector.get_support()

#rfe_feature = X.loc[:,rfe_support].columns.tolist()

#print(str(len(rfe_feature)), 'Selected Features')
#Rank=rfe_selector.ranking_

#Columns = X.columns
#REF_Fea = pd.DataFrame([Columns,Rank], index=['Name','Rank'])

#REF_Fea.transpose().sort_values('Rank',ascending=True)

#REF_Fea.transpose().sort_values('Rank',ascending=True).Name.tolist()
#pd.Series(Rank.Rank).plot.barh(color='red', figsize=(10, 8))
#EDA for selected Features

train_df1=train_df[['var_139', 'var_146', 'var_81', 'var_12', 'var_80', 'var_174', 'var_6', 'var_13', 'var_18', 'var_110', 'var_26',

 'var_166', 'var_133', 'var_198', 'var_21', 'var_127', 'var_22', 'var_86', 'var_190', 'var_40', 'var_99', 'var_75',

 'var_109', 'var_170', 'target']]
#taking Mean, Max, Std and Min values for checking distribtion

df=pd.DataFrame(train_df1.mean(axis=0),columns=["Mean"])

df['Min']=train_df1.min(axis=0)

df['Max']=train_df1.max(axis=0)

df['Std']=train_df1.std(axis=0)

df['Var_name']=df.index

df = df.reset_index(drop=True)

df
plt.subplots(figsize=(10,5))

df.boxplot(rot=90);
train_df1.head(2)
#Box flot of Selected variable for checking the distriution

plt.subplots(figsize=(18,8))

train_df1.boxplot(rot=90);
#Histogram for Selected Variables

train_df1.hist(figsize=(20,20));
Yes = train_df1[train_df1['target']==1]

No = train_df1[train_df1['target']==0]
z1=len(train_df1.columns)

z1
var=train_df1.columns

sns.set_style('whitegrid')

plt.figure()

fig, ax = plt.subplots(5,5,figsize=(15,15))

j=0

for i in var:

    j+=1

    plt.subplot(5,5,j)

    sns.kdeplot(Yes[i], label='1', color='r',alpha=0.75)

    sns.kdeplot(No[i], label='0', color='b',alpha=0.75)

    plt.title(i)
#for traget value=1

df_Yes=pd.DataFrame(Yes.mean(axis=0),columns=["Mean"])

df_Yes['Min']=Yes.min(axis=0)

df_Yes['Max']=Yes.max(axis=0)

df_Yes['Std']=Yes.std(axis=0)

df_Yes['Sum']=Yes.sum(axis=0)

df_Yes['Skew']=Yes.skew(axis=0)

df_Yes['Kurt']=Yes.kurt(axis=0)

df_Yes['Var_name']=df_Yes.index

df_Yes = df_Yes.reset_index(drop=True)
#for target value 0

df_No=pd.DataFrame(No.mean(axis=0),columns=["Mean"])

df_No['Min']=No.min(axis=0)

df_No['Max']=No.max(axis=0)

df_No['Std']=No.std(axis=0)

df_No['Sum']=No.sum(axis=0)

df_No['Skew']=No.skew(axis=0)

df_No['Kurt']=No.kurt(axis=0)

df_No['Var_name']=df_Yes.index

df_No = df_No.reset_index(drop=True)
var1=df_Yes.columns[:-1]

sns.set_style('whitegrid')

plt.figure()

fig, ax = plt.subplots(4,2,figsize=(10,10))

j=0

for i in var1:

    j+=1

    plt.subplot(4,2,j)

    sns.kdeplot(df_Yes[i], label='1', color='r',alpha=0.75)

    sns.kdeplot(df_No[i], label='0', color='b',alpha=0.75)

    plt.title(i)
#Distribution for Target 1 along axis 1

df_Y=pd.DataFrame(Yes.mean(axis=1),columns=["Mean"])

df_Y['Min']=Yes.min(axis=1)

df_Y['Max']=Yes.max(axis=1)

df_Y['Std']=Yes.std(axis=1)

df_Y['Sum']=Yes.sum(axis=1)

df_Y['Skew']=Yes.skew(axis=1)

df_Y['Kurt']=Yes.kurt(axis=1)

df_Y['Var_name']=df_Y.index

df_Y = df_Y.reset_index(drop=True)



#Distribution for Target 0 along axis 1

df_N=pd.DataFrame(No.mean(axis=1),columns=["Mean"])

df_N['Min']=No.min(axis=1)

df_N['Max']=No.max(axis=1)

df_N['Std']=No.std(axis=1)

df_N['Sum']=No.sum(axis=1)

df_N['Skew']=No.skew(axis=1)

df_N['Kurt']=No.kurt(axis=1)

df_N['Var_name']=df_N.index

df_N = df_N.reset_index(drop=True)
var2=df_N.columns[:-1]

sns.set_style('whitegrid')

plt.figure()

fig, ax = plt.subplots(4,2,figsize=(10,10))

j=0

for i in var2:

    j+=1

    plt.subplot(4,2,j)

    sns.kdeplot(df_Y[i], label='1', color='r',alpha=0.75)

    sns.kdeplot(df_N[i], label='0', color='b',alpha=0.75)

    plt.title(i)
plt.subplots(4,2,figsize=(20,15))

plt.subplot(4,2,1)

train_df1.std(axis=0).plot('hist')

plt.title('Std')

plt.subplot(4,2,2)

train_df1.std(axis=1).plot('hist')

plt.title('std')



plt.subplot(4,2,3)

train_df1.mean(axis=0).plot('hist')

plt.title('Mean')

plt.subplot(4,2,4)

train_df1.mean(axis=1).plot('hist')

plt.title('Mean')



plt.subplot(4,2,5)

train_df1.max(axis=0).plot('hist')

plt.title('Max')

plt.subplot(4,2,6)

train_df1.max(axis=1).plot('hist')

plt.title('Max')



plt.subplot(4,2,7)

train_df1.min(axis=0).plot('hist')

plt.title('Min')

plt.subplot(4,2,8)

train_df1.min(axis=1).plot('hist')

plt.title('Min')
from xgboost import XGBClassifier

xgb = XGBClassifier(learning_rate=0.02, n_estimators=10, objective='binary:logistic',silent=True)
#XGBOOST Parameters

params = {

        'min_child_weight': [12, 15, 20, 30, 40],

        'gamma': [0.1, 0.2, 0.3],

        'subsample': [0.5, 0.55, 0.6, 0.7],

        'colsample_bytree': [0.5, 0.6, 0.7],

        'max_depth': [4, 5, 7, 8],

        'learning_rate': [0.01,0.02],

        'n_estimators':[10,15],

        'reg_alpha': [0,1],

        'reg_lambda': [1,2]

        }
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score, RepeatedStratifiedKFold
X = train_df.drop(['target'], axis=1)

y = train_df['target']
X_fit, X_val, y_fit, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
X_test=test_df.drop("ID_code",axis=1)
X = train_df.drop(['ID_code','target'],axis=1)

y = train_df['target']
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=42)
clf = XGBClassifier(objective = "binary:logistic",

                    subsample= 0.5, 

                    reg_lambda= 1, 

                    reg_alpha= 0, 

                    n_estimators= 2500, 

                    min_child_weight=12, 

                    max_depth= 20, 

                    learning_rate= 0.02, 

                    gamma= 0.3,

                    colsample_bytree= 0.5,

                    eval_metric ="auc").fit(train_X, train_y)
y_pred_xgb = clf.predict(X_test)
clf.get_booster().get_score(importance_type='gain')
#DataFrame.from_dict(data, orient='columns', dtype=None)

feature_importance=pd.DataFrame.from_dict(clf.get_booster().get_score(importance_type='gain'), orient='index')

feature_importance['Var_name']=feature_importance.index

feature_importance = feature_importance.reset_index(drop=True)

plt.figure(figsize=(20,30))

sns.barplot(x=0, y="Var_name", data=feature_importance.sort_values(by=0,ascending=False));
submission_rfc = pd.DataFrame({

        "ID_code": test_df["ID_code"],

        "target": y_pred_xgb

    })

submission_rfc.to_csv('submission_gxb1.csv', index=False)
#Top 46 Feature selected on the basis of Gain

train_df3=train_df[['ID_code','target','var_81', 'var_12', 'var_53', 'var_174','var_139', 'var_22', 'var_80', 'var_166', 'var_26', 'var_177', 'var_146', 'var_78', 'var_198', 'var_6', 'var_110', 'var_133', 'var_164', 'var_109', 'var_99', 'var_190', 'var_13', 'var_94', 'var_165','var_7', 'var_0', 'var_76', 'var_2', 'var_178', 'var_108', 'var_44', 'var_179', 'var_5', 'var_145', 'var_194', 'var_33', 'var_137','var_154','var_40', 'var_18', 'var_1', 'var_121', 'var_92', 'var_75', 'var_14', 'var_173', 'var_34', 'var_50', 'var_127']]
#Top 46 Feature selected on the basis of Gain

test_df3=test_df[['ID_code','var_81', 'var_12', 'var_53', 'var_174','var_139', 'var_22', 'var_80', 'var_166', 'var_26', 'var_177', 'var_146', 'var_78', 'var_198', 'var_6', 'var_110', 'var_133', 'var_164', 'var_109', 'var_99', 'var_190', 'var_13', 'var_94', 'var_165','var_7', 'var_0', 'var_76', 'var_2', 'var_178', 'var_108', 'var_44', 'var_179', 'var_5', 'var_145', 'var_194', 'var_33', 'var_137','var_154','var_40', 'var_18', 'var_1', 'var_121', 'var_92', 'var_75', 'var_14', 'var_173', 'var_34', 'var_50', 'var_127']]
X = train_df3.drop(['ID_code','target'],axis=1)

y = train_df3['target']
#train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=42)

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
clf = XGBClassifier(objective = "binary:logistic",

                    subsample= 0.5, 

                    reg_lambda= 1, 

                    reg_alpha= 0, 

                    n_estimators= 2500, 

                    min_child_weight=12, 

                    max_depth= 10, 

                    learning_rate= 0.02, 

                    gamma= 0.3,

                    colsample_bytree= 0.5,

                    eval_metric ="auc").fit(train_X, train_y)
predictions_train = clf.predict_proba(train_X)

predictions_test = clf.predict_proba(val_X)



print('train',roc_auc_score(train_y, predictions_train[:,1]))

print('test',roc_auc_score(val_y, predictions_test[:,1]))
X_test=test_df3.drop("ID_code",axis=1)
Test_Prediction = clf.predict_proba(X_test)[:,1]
sub_df = pd.DataFrame({"ID_code":test_df["ID_code"].values})

sub_df["target"] = Test_Prediction

sub_df.to_csv("submission_final1.csv", index=False)