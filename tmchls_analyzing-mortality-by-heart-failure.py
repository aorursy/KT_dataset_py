# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#loading the dataset

h=pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

h.head()
h.shape
h.info()
h.describe()
h.isnull().sum()
#renaming the target column

h.rename(columns={'DEATH_EVENT':'death_event'},inplace=True)
h.columns
#converting age into int type

h['age']=h['age'].astype('int64')
h.info()
import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly

#plotly.offline.init_notebook_mode(connected = True)
plt.figure(figsize=(10,10))

sns.distplot(h['age'],bins=30)
plt.figure(figsize=(10,10))

sns.distplot(h['creatinine_phosphokinase'])

plt.xticks(range(0,10000,500))
plt.figure(figsize=(10,10))

sns.distplot(h['ejection_fraction'],bins=25)
plt.figure(figsize=(10,10))

sns.distplot(h['platelets'])
plt.figure(figsize=(10,10))

sns.distplot(h['serum_sodium'])
plt.figure(figsize=(10,10))

sns.distplot(h['serum_creatinine'])
plt.figure(figsize=(10,10))

sns.distplot(h['time'],bins=30)
h['anaemia'].value_counts(normalize=True)*100
(h['anaemia'].value_counts(normalize=True)*100).plot(kind='bar',color=['b','r'],rot=0)
h['diabetes'].value_counts(normalize=True)*100
(h['diabetes'].value_counts(normalize=True)*100).plot(kind='bar',color=['b','r'],rot=0)
h['high_blood_pressure'].value_counts(normalize=True)*100
(h['high_blood_pressure'].value_counts(normalize=True)*100).plot(kind='bar',color=['b','r'],rot=0)
h['sex'].value_counts(normalize=True)*100
h['smoking'].value_counts(normalize=True)*100
h['death_event'].value_counts(normalize=True)*100
x=h.drop(columns='death_event')

y=h['death_event']



from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()

model.fit(x,y)

print(model.feature_importances_) 

feat_importances = pd.Series(model.feature_importances_, index=x.columns)

feat_importances.nlargest(12).plot(kind='barh')

plt.show()
#determining correlation

plt.figure(figsize=(15,15))

sns.heatmap(h.corr(),annot=True)
px.box(h,'time')
px.box(h,'ejection_fraction')
#detecting the outliers

h[h['ejection_fraction']>=70]
#removing the outliers

h=h[h['ejection_fraction']<70]

h.head()
px.box(h,'serum_creatinine')
#converting discrete values into categorical for analysis

h['anaemia'].replace({1:'Yes',0:'No'},inplace=True)

h['diabetes'].replace({1:'Yes',0:'No'},inplace=True)

h['high_blood_pressure'].replace({1:'Yes',0:'No'},inplace=True)

h['smoking'].replace({1:'Yes',0:'No'},inplace=True)

h['death_event'].replace({1:'Yes',0:'No'},inplace=True)

h['sex'].replace({1:"Men",0:"women"},inplace=True)
e=px.scatter(h,'age',color='death_event',

             title="Distribution of death_event on basis of age",size='age')

e.show()
e=px.scatter(h,'creatinine_phosphokinase',color='death_event',

             title="Distribution of death_event on basis of Creatinine Phosphokinase (in mcg/L)",size='creatinine_phosphokinase')

e.show()
e=px.scatter(h,'serum_creatinine',color='death_event',

             title="Distribution of death_event on basis of Serum Creatinine(in mg/dL)",size='serum_creatinine')

e.show()
e=px.scatter(h,'serum_sodium',color='death_event',

             title="Distribution of death_event on basis of serum sodium(mEq/L)",size='serum_sodium')

e.show()
e=px.scatter(h,'ejection_fraction',color='death_event',

             title="Distribution of death_event on basis of Ejection fraction (in %)",size='ejection_fraction')

e.show()
e=px.scatter(h,'platelets',color='death_event',

             title="Distribution of death_event on basis of Platelets (in kiloplates/ML)",size='platelets')

e.show()
e=px.scatter(h,'time',color='death_event',

             title="Distribution of death_event on basis of Follow-up period (in days)",size='time')

e.show()
sns.countplot("death_event",data=h,hue='smoking')
sns.countplot("death_event",data=h,hue='anaemia')
sns.countplot("death_event",data=h,hue='sex')
sns.countplot("death_event",data=h,hue='diabetes')
sns.countplot("death_event",data=h,hue='high_blood_pressure')
sns.barplot(y='age',x='diabetes',data=h,ci=None)
sns.barplot(y='age',x='high_blood_pressure',data=h,ci=None)
#continuous variable analysis with respect to target variable

d=px.scatter(h,'serum_sodium',color='death_event',hover_data=['creatinine_phosphokinase',

                                                                             'serum_creatinine','ejection_fraction',

                                                                             'time','platelets'])

d.show()
#discrete variable analysis with respect to target variable

w=px.bar(h,'death_event',color='death_event',hover_data=['sex','diabetes','anaemia','high_blood_pressure','smoking'])

w.show()
#encoding the strings into numbers of needed features

h['death_event'].replace({'Yes':1,'No':0},inplace=True)

h['anaemia'].replace({'Yes':1,'No':0},inplace=True)

h['diabetes'].replace({'Yes':1,'No':0},inplace=True)

h['high_blood_pressure'].replace({'Yes':1,'No':0},inplace=True)

h['smoking'].replace({'Yes':1,'No':0},inplace=True)

h['sex'].replace({"Men":1,"women":0},inplace=True)
#model fitting without feature selection

x1=h.drop(columns=['death_event'])

y1=h['death_event']
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,roc_auc_score,classification_report

from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier as KNN

from sklearn.ensemble import VotingClassifier 
train_x,test_x,train_y,test_y=train_test_split(x1,y1,test_size=0.2,random_state=0,shuffle=True)

sc=StandardScaler()

train_x=sc.fit_transform(train_x)

test_x=sc.transform(test_x)
lr=LogisticRegression(random_state=0)

dt=DecisionTreeClassifier(random_state=0)

knn=KNN()

classifiers = [('Logistic Regression', lr),

('K Nearest Neighbours', knn),

('Decision Tree', dt)]
vc=VotingClassifier(estimators=classifiers)

vc.fit(train_x,train_y)
y_pred=vc.predict(test_x)

roc_auc_score(test_y,y_pred)
#model fitting with feature selection

x2=h.iloc[:,[4,7,11]]

x2.head()
train_x,test_x,train_y,test_y=train_test_split(x2,y1,test_size=0.2,random_state=0,shuffle=True)

sc=StandardScaler()

train_x=sc.fit_transform(train_x)

test_x=sc.transform(test_x)

vc=VotingClassifier(estimators=classifiers)

vc.fit(train_x,train_y)

y_pred=vc.predict(test_x)

roc_auc_score(test_y,y_pred)
cm=confusion_matrix(test_y,y_pred)

plt.figure(figsize=(10,10))

sns.heatmap(cm, cmap=plt.cm.Blues,annot=True)

plt.title("Ensemble Model - Confusion Matrix")

plt.yticks(range(2), ["Actual Heart Not Failed","Actual Heart Fail"], fontsize=16)

plt.xticks(range(2), ["Predicted Heart Not Failed"," Predicted Heart Fail"], fontsize=16)

plt.show()
print(classification_report(test_y,y_pred))