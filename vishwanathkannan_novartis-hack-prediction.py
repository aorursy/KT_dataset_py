import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as stats

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, classification_report,roc_curve,auc,confusion_matrix

from imblearn.over_sampling import SMOTE

from sklearn.impute import KNNImputer
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv('/kaggle/input/hacked/train.csv',parse_dates=['DATE'])

df=df.drop('INCIDENT_ID',axis=1)

df.head()
sns.countplot(df['MULTIPLE_OFFENSE'])
df['Day']=df['DATE'].dt.day_name()

df['Year']=df['DATE'].dt.year

df['Week']=df['DATE'].dt.week



le=LabelEncoder()

df['Day']=le.fit_transform(df['Day'])

df=df.drop(['DATE'],axis=1)
from scipy.stats import chi2_contingency

l=[]

categorical=['Day','Week','Year','X_1','X_4','X_5','X_9']

for i in categorical:

    pvalue  = chi2_contingency(pd.crosstab(df['MULTIPLE_OFFENSE'],df[i]))[1]

    l.append(1-pvalue)

plt.figure(figsize=(7,5))

sns.barplot(x=l, y=categorical)

plt.title('Best Categorical Features')

plt.axvline(x=(1-0.05),color='r')

plt.show()
print('Nan values in the columns')

df.isna().sum()
sns.boxplot(df['X_12'], orient='v')
cols=df.columns

imp=KNNImputer(n_neighbors=5, missing_values=np.nan)

df=pd.DataFrame(imp.fit_transform(df.values),columns=cols)
from scipy.stats import ttest_ind

num=['X_3', 'X_7', 'X_2', 'X_6', 'X_8', 'X_10', 'X_11', 'X_12', 'X_13', 'X_14']

p=[]



for i in num:

    df1=df.groupby('MULTIPLE_OFFENSE').get_group(0)

    df2=df.groupby('MULTIPLE_OFFENSE').get_group(1)

    t,pvalue=ttest_ind(df1[i],df2[i])

    p.append(1-pvalue)

plt.figure(figsize=(7,7))

sns.barplot(x=p, y=num)

plt.title('Best Numerical Features')

plt.axvline(x=(1-0.05),color='r')

plt.xlabel('1-p value')

plt.show()
df=df.drop(['X_4','X_5','Week','X_13','X_7','X_6'],axis=1)
X=df.drop('MULTIPLE_OFFENSE',axis=1)

y=df['MULTIPLE_OFFENSE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
smt=SMOTE(random_state=10) #FOR CLASS IMBALANCE

X_train, y_train = smt.fit_sample(X_train, y_train)
param_test1 = {

 'max_depth':range(7,17,2),

 'min_child_weight':range(1,6,2)

}

gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,

 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 

 param_grid = param_test1, scoring='f1',n_jobs=4,iid=False, cv=3)

gsearch1.fit(X_train,y_train)

gsearch1.best_params_, gsearch1.best_score_
param_test3 = {

 'gamma':[i/10.0 for i in range(0,5)]

}

gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=9,

 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 

 param_grid = param_test3, scoring='f1',n_jobs=4,iid=False, cv=5)

gsearch3.fit(X_train,y_train)

gsearch3.best_params_, gsearch3.best_score_
param_test5 = {

 'subsample':[i/100.0 for i in range(75,90,5)],

 'colsample_bytree':[i/100.0 for i in range(75,90,5)]

}

gsearch5 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=140, max_depth=9,

 min_child_weight=1, gamma=0.4, subsample=0.8, colsample_bytree=0.8,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 

 param_grid = param_test5, scoring='f1',n_jobs=4,iid=False, cv=5)

gsearch5.fit(X_train,y_train)

gsearch5.best_params_, gsearch5.best_score_
param_test7 = {

 'reg_lambda':[0, 0.001, 0.005, 0.01, 0.05]

}

gsearch7 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=9,

 min_child_weight=1, gamma=0.4, subsample=0.8, colsample_bytree=0.75,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 

 param_grid = param_test7, scoring='f1',n_jobs=4,iid=False, cv=5)

gsearch7.fit(X_train,y_train)

gsearch7.best_params_, gsearch7.best_score_
param_test8 = {

 'reg_lambda':[i/1000.0 for i in range(9,20,1)]

}

gsearch8 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=9,

 min_child_weight=1, gamma=0.4, subsample=0.8, colsample_bytree=0.75,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 

 param_grid = param_test7, scoring='f1',n_jobs=4,iid=False, cv=5)

gsearch8.fit(X_train,y_train)

gsearch8.best_params_, gsearch8.best_score_
xgb = XGBClassifier(

 learning_rate =0.01,

 n_estimators=800,

 max_depth=9,

 min_child_weight=1,

 gamma=0.4,

 subsample=0.8,

 colsample_bytree=0.75,

 reg_lambda=0.01,

 objective= 'binary:logistic',

 nthread=4)

xgb.fit(X_train,y_train)

ypred=xgb.predict(X_test)

recall_score(ypred,y_test)
plt.figure(figsize=(7,10))

sns.barplot(x=xgb.feature_importances_,y=X.columns)

plt.title('Significant Features of the Final Model')

plt.show()
df=df.drop(['X_1','X_9','X_14'],axis=1)
X=df.drop('MULTIPLE_OFFENSE',axis=1)

y=df['MULTIPLE_OFFENSE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
smt=SMOTE(random_state=10) #FOR CLASS IMBALANCE

X_train, y_train = smt.fit_sample(X_train, y_train)
xgb = XGBClassifier(

 learning_rate =0.01,

 n_estimators=800,

 max_depth=9,

 min_child_weight=1,

 gamma=0.4,

 subsample=0.8,

 colsample_bytree=0.75,

 reg_lambda=0.01,

 objective= 'binary:logistic',

 nthread=4)

xgb.fit(X_train,y_train)

ypred=xgb.predict(X_test)

recall_score(ypred,y_test)
plt.figure(figsize=(7,10))

sns.barplot(x=xgb.feature_importances_,y=X.columns)

plt.title('Significant Features of the Final Model, after all the feature Engineering')

plt.show()
probs = xgb.predict_proba(X_test)

preds = probs[:,1]

fpr, tpr, threshold = roc_curve(y_test, preds)

roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.title('ROC-AUC Curve')

plt.show()
confusion_matrix(y_test,ypred)
test=pd.read_csv('/kaggle/input/hacked/test.csv', parse_dates=['DATE'])

test.head()
test['Day']=test['DATE'].dt.day_name()

test['Year']=test['DATE'].dt.year

test['Week']=test['DATE'].dt.week
le=LabelEncoder()

test['Day']=le.fit_transform(test['Day'])

ids=test['INCIDENT_ID']

test=test.drop(['INCIDENT_ID','DATE'],axis=1)
cols=test.columns

imp=KNNImputer(n_neighbors=5, missing_values=np.nan)

test=imp.fit_transform(test)

test=pd.DataFrame(test,columns=cols)
test=test.drop(['X_1','X_4','X_5','Week','X_13','X_7','X_6','X_9','X_14'],axis=1)
x=test.values
x = scaler.transform(x)
pred=xgb.predict(x)
submission = pd.DataFrame({'INCIDENT_ID': ids,

                           'MULTIPLE_OFFENSE':pred

                           })



submission.to_csv("final_sub.csv",index=False)