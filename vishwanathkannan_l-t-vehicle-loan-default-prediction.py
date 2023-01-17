#This project was largely a learning opportunity to understand compromises between Domain-based inferences and model building

#The flow of the project will explain the train of thought to justify use of specific features for further model building,

#and defend the use of new features made on the way.
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import re

from sklearn.preprocessing import LabelEncoder,StandardScaler

from sklearn.linear_model import LogisticRegression

#ensemble

from xgboost import XGBClassifier

#metrics

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report,confusion_matrix



#warnings

import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv('/kaggle/input/lt-vehicle-loan-default-prediction/train.csv')



df=df.drop(['UniqueID', 'branch_id','supplier_id', 'Current_pincode_ID','State_ID', 'Employee_code_ID', 'MobileNo_Avl_Flag'],axis=1)
def credit_risk(df):

    d1=[]

    d2=[]

    for i in df:

        p = i.split("-")

        if len(p) == 1:

            d1.append(p[0])

            d2.append('unknown')

        else:

            d1.append(p[1])

            d2.append(p[0])



    return d2



sub_risk = {'unknown':-1, 'A':13, 'B':12, 'C':11,'D':10,'E':9,'F':8,'G':7,'H':6,'I':5,'J':4,'K':3, 'L':2,'M':1}

employment_map = {'Self employed':0, 'Salaried':1,np.nan:2}



df.loc[:,'credit_risk_grade']  = credit_risk(df["PERFORM_CNS.SCORE.DESCRIPTION"])

df.loc[:,'Credit Risk'] = df['credit_risk_grade'].apply(lambda x: sub_risk[x])



df.loc[:,'Employment Type'] = df['Employment.Type'].apply(lambda x: employment_map[x])



df=df.drop(['PERFORM_CNS.SCORE.DESCRIPTION', 'credit_risk_grade','Employment.Type'],axis=1)
def age(dur):

    yr = int(dur.split('-')[2])

    if yr >=0 and yr<=19:

        return yr+2000

    else:

         return yr+1900



df['Date.of.Birth'] = df['Date.of.Birth'].apply(age)

df['DisbursalDate'] = df['DisbursalDate'].apply(age)

df['Age']=df['DisbursalDate']-df['Date.of.Birth']

df=df.drop(['DisbursalDate','Date.of.Birth'],axis=1)
numerical=['disbursed_amount','asset_cost','PRI.NO.OF.ACCTS',

       'PRI.ACTIVE.ACCTS', 'PRI.OVERDUE.ACCTS', 'PRI.CURRENT.BALANCE',

       'PRI.SANCTIONED.AMOUNT', 'PRI.DISBURSED.AMOUNT', 'SEC.NO.OF.ACCTS',

       'SEC.ACTIVE.ACCTS', 'SEC.OVERDUE.ACCTS', 'SEC.CURRENT.BALANCE',

       'SEC.SANCTIONED.AMOUNT', 'SEC.DISBURSED.AMOUNT', 'PRIMARY.INSTAL.AMT',

       'SEC.INSTAL.AMT', 'NEW.ACCTS.IN.LAST.SIX.MONTHS',

       'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS','NO.OF_INQUIRIES','Age','NEW.ACCTS.IN.LAST.SIX.MONTHS', 

        'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS']

categorical=['manufacturer_id', 'Aadhar_flag', 'PAN_flag',

       'VoterID_flag', 'Driving_flag', 'Passport_flag', 'PERFORM_CNS.SCORE',

       'NEW.ACCTS.IN.LAST.SIX.MONTHS', 'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS',

       'AVERAGE.ACCT.AGE', 'NO.OF_INQUIRIES', 'Credit Risk','AVERAGE.ACCT.AGE','CREDIT.HISTORY.LENGTH',

       'Employment Type']
#T Test for numerical columns

p=[]

from scipy.stats import ttest_ind



for i in numerical:

    df1=df.groupby('loan_default').get_group(0)

    df2=df.groupby('loan_default').get_group(1)

    t,pvalue=ttest_ind(df1[i],df2[i])

    p.append(1-pvalue)

plt.figure(figsize=(7,7))

sns.barplot(x=p, y=numerical)

plt.title('Best Numerical Features')

plt.axvline(x=(1-0.05),color='r')

plt.xlabel('1-p value')

plt.show()
for i in numerical:

    df1=df.groupby('loan_default').get_group(0)

    df2=df.groupby('loan_default').get_group(1)

    print(np.std(df1[i],ddof=1),np.std(df2[i],ddof=1))
from sklearn.feature_selection import SelectKBest,f_classif

n = SelectKBest(score_func=f_classif, k=10)

numcols=n.fit(df[numerical],df['loan_default'])

plt.figure(figsize=(7,7))

sns.barplot(x=numcols.scores_,y=numerical)

plt.title('Best Numerical Features')

plt.show()
df.loc[:,'No of Accounts'] = df['PRI.NO.OF.ACCTS'] + df['SEC.NO.OF.ACCTS']

df.loc[:,'PRI Inactive accounts'] = df['PRI.NO.OF.ACCTS'] - df['PRI.ACTIVE.ACCTS']

df.loc[:,'SEC Inactive accounts'] = df['SEC.NO.OF.ACCTS'] - df['SEC.ACTIVE.ACCTS']

df.loc[:,'Total Inactive accounts'] = df['PRI Inactive accounts'] + df['SEC Inactive accounts']

df.loc[:,'Total Overdue Accounts'] = df['PRI.OVERDUE.ACCTS'] + df['SEC.OVERDUE.ACCTS']

df.loc[:,'Total Current Balance'] = df['PRI.CURRENT.BALANCE'] + df['SEC.CURRENT.BALANCE']

df.loc[:,'Total Sanctioned Amount'] = df['PRI.SANCTIONED.AMOUNT'] + df['SEC.SANCTIONED.AMOUNT']

df.loc[:,'Total Disbursed Amount'] = df['PRI.DISBURSED.AMOUNT'] + df['SEC.DISBURSED.AMOUNT']

df.loc[:,'Total Installment'] = df['PRIMARY.INSTAL.AMT'] + df['SEC.INSTAL.AMT']
#Chi Square test for Categorical Columns

from scipy.stats import chi2_contingency

l=[]

for i in categorical:

    pvalue  = chi2_contingency(pd.crosstab(df['loan_default'],df[i]))[1]

    l.append(1-pvalue)

plt.figure(figsize=(7,7))

sns.barplot(x=l, y=categorical)

plt.title('Best Categorical Features')

plt.axvline(x=(1-0.05),color='r')

plt.show()
def duration(dur):

    yrs = int(dur.split(' ')[0].replace('yrs',''))

    mon = int(dur.split(' ')[1].replace('mon',''))

    return yrs*12+mon
df['AVERAGE.ACCT.AGE'] = df['AVERAGE.ACCT.AGE'].apply(duration)

df['CREDIT.HISTORY.LENGTH'] = df['CREDIT.HISTORY.LENGTH'].apply(duration)

#df.drop(['AVERAGE.ACCT.AGE','CREDIT.HISTORY.LENGTH'],axis=1,inplace=True)
df=df.drop(['PRI.NO.OF.ACCTS','SEC.NO.OF.ACCTS','PRI.CURRENT.BALANCE','PRI Inactive accounts','SEC Inactive accounts',

            'PRI.SANCTIONED.AMOUNT','SEC.NO.OF.ACCTS','PRI.NO.OF.ACCTS','PRI.DISBURSED.AMOUNT','PRI.ACTIVE.ACCTS', 

            'PRI.OVERDUE.ACCTS','SEC.CURRENT.BALANCE','SEC.SANCTIONED.AMOUNT', 'SEC.OVERDUE.ACCTS',

            'SEC.DISBURSED.AMOUNT','PRIMARY.INSTAL.AMT','SEC.INSTAL.AMT','disbursed_amount','SEC.ACTIVE.ACCTS'],axis=1)
nums=['asset_cost', 'ltv','PERFORM_CNS.SCORE',

       'NEW.ACCTS.IN.LAST.SIX.MONTHS', 'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS',

       'AVERAGE.ACCT.AGE', 'CREDIT.HISTORY.LENGTH', 'NO.OF_INQUIRIES','No of Accounts', 'Total Inactive accounts',

       'Total Overdue Accounts', 'Total Current Balance', 'Total Sanctioned Amount',

       'Total Disbursed Amount', 'Total Installment','Age']
len(nums)
y=df.loan_default

X=df.drop("loan_default",axis=1)

from sklearn.model_selection import train_test_split,KFold,cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

lr=LogisticRegression()

lr.fit(X_train,y_train)

y_pred=lr.predict(X_test)

print('train accuracy :',lr.score(X_train,y_train))

print('test accuracy :',lr.score(X_test,y_test))

print("precision :",precision_score(y_test,y_pred),"\n")

print("recall :",recall_score(y_test,y_pred),"\n")

print("f1 score:",f1_score(y_test,y_pred),"\n")

print(classification_report(y_test,y_pred))
sns.countplot(df['loan_default'])
n=['PERFORM_CNS.SCORE','NO.OF_INQUIRIES','No of Accounts', 'Total Inactive accounts',

       'Total Overdue Accounts', 'Total Current Balance', 'Total Sanctioned Amount',

       'Total Disbursed Amount', 'Total Installment']

data=df[n]

fig, axes = plt.subplots(nrows=3, ncols=3,figsize=(20,10))

fig.subplots_adjust(hspace=0.5)

fig.suptitle('Distributions of Credit History Features')



for ax, feature, name in zip(axes.flatten(), data.values.T, data.columns):

    sns.distplot(feature, ax=ax)

    ax.set(title=str(name))

plt.show()
from sklearn.preprocessing import  RobustScaler

rob_scaler = RobustScaler()



df[nums] = rob_scaler.fit_transform(df[nums])
df['Missing Features'] = (df == 0).astype(int).sum(axis=1)
y=df.loan_default

X=df.drop("loan_default",axis=1)

from sklearn.model_selection import train_test_split,KFold,cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
from sklearn.model_selection import GridSearchCV
param_test1 = {

 'max_depth':range(3,10,2),

 'min_child_weight':range(1,6,2)

}

gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,

 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 

 param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=3)

gsearch1.fit(X_train,y_train)

gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_
param_test2b = {

 'max_depth':range(7,10,2)

}

gsearch2b = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=4,

 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 

 param_grid = param_test2b, scoring='roc_auc',n_jobs=4,iid=False, cv=3)

gsearch2b.fit(X_train,y_train)
gsearch2b.cv_results_, gsearch2b.best_params_, gsearch2b.best_score_
param_test3 = {

 'gamma':[i/10.0 for i in range(0,5)]

}

gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=9,

 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 

 param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch3.fit(X_train,y_train)

gsearch3.best_params_, gsearch3.best_score_
xgb4 = XGBClassifier(

 learning_rate =0.01,

 n_estimators=5000,

 max_depth=9,

 min_child_weight=1,

 gamma=0.4,

 subsample=0.8,

 colsample_bytree=0.8,

 reg_alpha=0.005,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27)
xgb4.fit(X_train,y_train)

y_pred=xgb4.predict(X_test)

print("accuracy train:",xgb4.score(X_train,y_train),"\n")

print("accuracy test:",xgb4.score(X_test,y_test),"\n")

print("precision :",precision_score(y_test,y_pred),"\n")

print("Recall :",recall_score(y_test,y_pred),"\n")

print("f1 score:",f1_score(y_test,y_pred),"\n")

print("Confusion Matrix \n",confusion_matrix(y_test,y_pred))
import sklearn.metrics as metrics
probs = xgb4.predict_proba(X_test)

preds = probs[:,1]

fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
print(fpr, tpr)
roc_auc = metrics.auc(fpr, tpr)



import matplotlib.pyplot as plt

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
plt.figure(figsize=(7,10))

sns.barplot(x=xgb4.feature_importances_,y=X.columns)

plt.title('Significant Features of the Final Model')

plt.show()