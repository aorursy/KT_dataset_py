import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
pd.set_option('float_format', '{:f}'.format)
train = pd.read_csv('../input/janata-hack-machine-learning-for-banking/train_fNxu4vz.csv',thousands=',')

test = pd.read_csv('../input/janata-hack-machine-learning-for-banking/test_fjtUOL8.csv',thousands=',')
print(train.shape)

train.head()
train_original = train.copy()

test_original = test.copy()
train.dtypes
category = []

for col in train.columns:

    if train[col].dtypes == 'object':

        category.append(col)

category
train['Interest_Rate'].value_counts()
#f,ax = plt.subplots(1,2,figsize = (18,8))

plt.subplot(121)

train['Interest_Rate'].value_counts().plot.pie(autopct = '%1.2f%%',figsize = (18,8),title = 'Target Variable classification')

plt.subplot(122)

train['Interest_Rate'].value_counts().plot.bar(figsize = (18,8))
cont_col = ['Loan_Amount_Requested','Annual_Income','Debt_To_Income','Months_Since_Deliquency','Number_Open_Accounts',

 'Total_Accounts']

ord_col = ['Length_Employed', 'Inquiries_Last_6Mo']

nom_col = ['Home_Owner','Income_Verified','Purpose_Of_Loan','Gender']
#Independent Variable (Categorical)

#'Home_Owner','Income_Verified','Purpose_Of_Loan','Gender'

print(plt.figure(1) )

ax = plt.subplot(221) 

#ax.xaxis.label.set_color('red')

ax.tick_params(axis='x', colors='red')

ax.tick_params(axis='y', colors='red')

train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,18), title= 'Gender',colormap = 'viridis',fontsize = 16) 

ax = plt.subplot(222)

ax.tick_params(axis='x', colors='red')

ax.tick_params(axis='y', colors='red')

train['Home_Owner'].value_counts(normalize=True).plot.bar(title= 'Home_Owner',colormap = 'Accent',fontsize = 16) 

plt.subplot(223) 

train['Income_Verified'].value_counts(normalize=True).plot.bar(title= 'Income_Verified',colormap = 'Set2_r',fontsize = 16) 

plt.subplot(224) 

train['Purpose_Of_Loan'].value_counts(normalize=True).plot.bar(title= 'Purpose_Of_Loan',colormap = 'Set1',fontsize = 16) 

plt.show()

ax.xaxis.label.set_color('red')

ax.tick_params(axis='x', colors='red')
## 'Loan_Amount_Requested','Length_Employed'

#Inquiries_Last_6Mo, Number_Open_Accounts,Total_Accounts               int64

#Independent Variable (Ordinal)

print(train['Length_Employed'].unique())

print(train['Inquiries_Last_6Mo'].unique())

print(train['Number_Open_Accounts'].unique())

print(train['Total_Accounts'].unique())
#Independent Variable (Ordinal)

#'Length_Employed', Inquiries_Last_6Mo

print(plt.figure(1) )

ax = plt.subplot(221) 

#ax.xaxis.label.set_color('red')

ax.tick_params(axis='x', colors='red')

ax.tick_params(axis='y', colors='red')

train['Length_Employed'].value_counts(normalize=True).plot.bar(figsize=(20,18), title= 'Length_Employed',colormap = 'spring',fontsize = 16) 

ax = plt.subplot(222) 

#ax.xaxis.label.set_color('red')

ax.tick_params(axis='x', colors='red')

ax.tick_params(axis='y', colors='red')

train['Inquiries_Last_6Mo'].value_counts(normalize=True).plot.bar(figsize=(20,18), title= 'Inquiries_Last_6Mo',colormap = 'cool',fontsize = 16) 



i = 1



for col in cont_col:

    plt.figure(figsize = (25,35))

    plt.subplot(7,2,i)

    sns.distplot(train[col]); 

    plt.subplot(7,2,i+1) 

    train[col].plot.box()

    i += 2

    plt.show()
for col in nom_col:

    Gender=pd.crosstab(train[col],train['Interest_Rate']) 

    #print(Gender)

    Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(15,4))
for col in ord_col:

    Gender=pd.crosstab(train[col],train['Interest_Rate']) 

    #print(Gender)

    Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(15,4))
train.groupby('Interest_Rate')['Annual_Income'].mean().plot.bar()
print(train['Annual_Income'].max())

print(train['Annual_Income'].min())

print(train['Annual_Income'].mean())
bins=[0,100000,500000,1000000,7600000] 

group=['Low','Average','High', 'Very high'] 

train['Annual_Income_bin']=pd.cut(train['Annual_Income'],bins,labels=group)
Annual_Income_bin=pd.crosstab(train['Annual_Income_bin'],train['Interest_Rate']) 

Annual_Income_bin.div(Annual_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 

plt.xlabel('Annual_Income') 

P = plt.ylabel('Percentage')
train.describe()
i = 1

for col in cont_col:

    plt.figure(figsize = (25,35))

    plt.subplot(len(cont_col),1,i)

    train.groupby('Interest_Rate')[col].mean().plot.bar(title = col)

    i += 1
bins=[0,8000,12000,20000,36000] 

group=['Low','Average','High','veryhigh'] 

train['LoanAmount_bin']=pd.cut(train['Loan_Amount_Requested'],bins,labels=group)

LoanAmount_bin=pd.crosstab(train['LoanAmount_bin'],train['Interest_Rate']) 

LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 

plt.xlabel('LoanAmount') 

P = plt.ylabel('Percentage')
train.columns
train['Length_Employed'].str.extract('(\d+)')

#train=train.drop(['Annual_Income_bin', 'LoanAmount_bin'], axis=1)

train['Length_Employed'] = train['Length_Employed'].str.extract('(\d+)').astype('int64',errors='ignore')

test['Length_Employed'] = test['Length_Employed'].str.extract('(\d+)').astype('int64',errors='ignore')

train
 

f, ax = plt.subplots(figsize=(28, 12)) 

sns.heatmap(train.corr(), vmax=.8, square=True, cmap="RdYlGn",annot = True);
train.isnull().sum()
test.isnull().sum()
train.drop('Months_Since_Deliquency',axis = 1,inplace = True)

test.drop('Months_Since_Deliquency',axis = 1,inplace = True)
train['Length_Employed'].value_counts()

train['Home_Owner'].value_counts()

train['Length_Employed'].fillna(train['Length_Employed'].mode()[0],inplace = True)

train['Home_Owner'].fillna(train['Home_Owner'].mode()[0],inplace = True)



test['Length_Employed'].fillna(train['Length_Employed'].mode()[0],inplace = True)

test['Home_Owner'].fillna(train['Home_Owner'].mode()[0],inplace = True)
train['Annual_Income'].fillna(train['Annual_Income'].median(),inplace = True)

test['Annual_Income'].fillna(train['Annual_Income'].median(),inplace = True)
minvalue=min(train['Annual_Income'])

maxvalue=max(train['Annual_Income'])

meanvalue=np.mean(train['Annual_Income'])

medianvalue=np.median(train['Annual_Income'])



points=[minvalue, maxvalue, meanvalue, medianvalue]

names=["min","max","mean","mid"]
#Lets see scatter plot

fig, ax = plt.subplots(figsize=(16,8))

ax.scatter(train.index,train['Annual_Income'],color = 'deeppink')

ax.axhline(y=minvalue, label="min",color='blue')

ax.axhline(y=maxvalue, label="max",color='cyan')

ax.axhline(y=meanvalue, label="mean",color='red')

ax.axhline(y=medianvalue, label="mid",color='green')

ax.legend()
#Lets see Z-score

from scipy import stats

import numpy as np

z = np.abs(stats.zscore(train['Annual_Income']))

print(z)
threshold = 3

print(np.where(z > 3))
z[98]
train['Annual_Income_log'] = np.log(train['Annual_Income']) 

train['Annual_Income_log'].hist(bins=20) 

test['Annual_Income_log'] = np.log(test['Annual_Income'])
train[['Annual_Income','Annual_Income_log']].sort_values('Annual_Income')
#Lets see scatter plot

cont_col.remove('Months_Since_Deliquency')

for col in cont_col:

    minvalue=min(train[col])

    maxvalue=max(train[col])

    meanvalue=np.mean(train[col])

    medianvalue=np.median(train[col])



    #points=[minvalue, maxvalue, meanvalue, medianvalue]

    #names=["min","max","mean","mid"]

    fig, ax = plt.subplots(figsize=(16,4))

    ax.scatter(train.index,train[col],color = 'deeppink',label = col)

    ax.axhline(y=minvalue, label="min",color='blue')

    ax.axhline(y=maxvalue, label="max",color='cyan')

    ax.axhline(y=meanvalue, label="mean",color='red')

    ax.axhline(y=medianvalue, label="mid",color='green')

    ax.legend()
threshold = 3

for col in cont_col:

    z = np.abs(stats.zscore(train[col]))

    outlier = np.where(z > threshold)

    print("No. of outliers in ",col,":",len(outlier[0]))

    print(outlier[0])

    print()



#type(outlier[0])
train['Annual_Income'][177]
#Annual Income outliered values

train[(np.abs(stats.zscore(train['Annual_Income'])) > 3)]['Annual_Income'].sort_values()
#Annual Income outliered values

train[(np.abs(stats.zscore(train['Number_Open_Accounts'])) > 3)]['Number_Open_Accounts'].sort_values()
#As we have 0's in the sample, if we apply log those will be -inf, to overcome this we apply sqrt instead of log

train['Number_Open_Accounts_log'] = np.sqrt(train['Number_Open_Accounts'])

plt.figure(figsize = (12,4))

plt.subplot(1,2,1)

train['Number_Open_Accounts_log'].hist() 

plt.subplot(1,2,2)

sns.distplot(train['Number_Open_Accounts_log'])

test['Number_Open_Accounts_log'] = np.sqrt(test['Number_Open_Accounts'])
train[['Number_Open_Accounts_log','Number_Open_Accounts']].sort_values('Number_Open_Accounts_log')
train[(np.abs(stats.zscore(train['Total_Accounts'])) > 3)]['Total_Accounts'].sort_values()
#As we have 0's in the sample, if we apply log those will be -inf, to overcome this we apply sqrt instead of log

train['Total_Accounts_log'] = np.sqrt(train['Total_Accounts'])

plt.figure(figsize = (12,4))

plt.subplot(1,2,1)

train['Total_Accounts_log'].hist() 

plt.subplot(1,2,2)

sns.distplot(train['Total_Accounts_log'])

test['Total_Accounts_log'] = np.sqrt(test['Number_Open_Accounts'])
train[['Total_Accounts_log','Total_Accounts']].sort_values('Total_Accounts_log')
train.columns
train=train.drop('Loan_ID',axis=1) 

test=test.drop('Loan_ID',axis=1)
test
X = train.drop(['Interest_Rate','Total_Accounts','Number_Open_Accounts','Annual_Income','Annual_Income_bin','LoanAmount_bin'],1) 

y = train['Interest_Rate']

test = test.drop(['Total_Accounts','Number_Open_Accounts','Annual_Income'],1)
X.columns
X=pd.get_dummies(X) 

train=pd.get_dummies(train) 

test=pd.get_dummies(test)
from sklearn.model_selection import train_test_split

x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3)
from sklearn.metrics import confusion_matrix 

from sklearn.model_selection import train_test_split 

from sklearn.svm import SVC 

from sklearn.utils import shuffle

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn import preprocessing

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier 

knn = KNeighborsClassifier(n_neighbors = 7).fit(x_train, y_train) 

pred_cv = knn.predict(x_cv)

accuracy_score(y_cv,pred_cv)
# accuracy on X_test 



accuracy = knn.score(x_cv, y_cv) 

print(accuracy)

  

# creating a confusion matrix 

knn_predictions = knn.predict(x_cv)  

cm = confusion_matrix(y_cv, knn_predictions)
sns.heatmap(cm,annot=True,fmt = '')
pred_test = knn.predict(test)
submission=pd.read_csv("../input/janata-hack-machine-learning-for-banking/sample_submission_HSqiq1Q.csv")

submission['Interest_Rate']=pred_test 

submission['Loan_ID']=test_original['Loan_ID']

submission.to_csv('knn.csv',index = False)

submission.head()
submission.to_csv('knn.csv',index = False)
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

i=1 

kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True) 

for train_index,test_index in kf.split(X,y):     

    print('\n{} of kfold {}'.format(i,kf.n_splits))     

    xtr,xvl = X.loc[train_index],X.loc[test_index]     

    ytr,yvl = y[train_index],y[test_index]         

    model = RandomForestClassifier(random_state=1, max_depth=10)     

    model.fit(xtr, ytr)     

    pred_test = model.predict(xvl)     

    score = accuracy_score(yvl,pred_test)     

    print('accuracy_score',score)     

    i+=1 



    rf_test = model.predict(test)
submission['Interest_Rate']=rf_test 

submission['Loan_ID']=test_original['Loan_ID']

submission.to_csv('rf1.csv',index = False)
rf = RandomForestClassifier(max_depth=10).fit(x_train, y_train) 

pred_cv = rf.predict(x_cv)

accuracy_score(y_cv,pred_cv)
pred_test = rf.predict(test)
submission['Interest_Rate']=pred_test 

submission['Loan_ID']=test_original['Loan_ID']

submission.to_csv('rf1.csv',index = False)
from sklearn.model_selection import GridSearchCV

# Provide range for max_depth from 1 to 20 with an interval of 2 and from 1 to 200 with an interval of 20 for n_estimators 

paramgrid = {'max_depth': list(range(1, 20, 2)), 'n_estimators': list(range(1, 200, 20))}

grid_search=GridSearchCV(RandomForestClassifier(random_state=1),paramgrid)
from sklearn.model_selection import train_test_split 

x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3, random_state=1)

# Fit the grid search model 

grid_search.fit(x_train,y_train)
grid_search.best_estimator_
i=1 

kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True) 

for train_index,test_index in kf.split(X,y):     

    print('\n{} of kfold {}'.format(i,kf.n_splits))     

    xtr,xvl = X.loc[train_index],X.loc[test_index]     

    ytr,yvl = y[train_index],y[test_index]         

    model = RandomForestClassifier(random_state=1, max_depth=15, n_estimators=181)    

    model.fit(xtr, ytr)     

    pred_test = model.predict(xvl)     

    score = accuracy_score(yvl,pred_test)     

    print('accuracy_score',score)     

    i+=1 

    pred_test = model.predict(test) 

    #pred2=model.predict_proba(test)[:,1]

    



