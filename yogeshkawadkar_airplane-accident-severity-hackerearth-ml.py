# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

sample_submission = pd.read_csv("../input/airplane-accidents-severity-dataset/sample_submission.csv")

test = pd.read_csv("../input/airplane-accidents-severity-dataset/test.csv",index_col='Accident_ID')

train = pd.read_csv("../input/airplane-accidents-severity-dataset/train.csv",index_col='Accident_ID')
import seaborn as sns

import matplotlib.pyplot as plt

sns.set

def ecdf(data):

    """Compute ECDF for a one-dimensional array of measurements."""

    # Number of data points: n

    n = len(data)



    # x-data for the ECDF: x

    x = np.sort(data)



    # y-data for the ECDF: y

    y = np.arange(1, n+1) / n



    return x, y

train.head()
train['Severity'].replace('Minor_Damage_And_Injuries',0,inplace=True)

train['Severity'].replace('Significant_Damage_And_Fatalities',1,inplace=True)

train['Severity'].replace('Significant_Damage_And_Serious_Injuries',2,inplace=True)

train['Severity'].replace('Highly_Fatal_And_Damaging',3,inplace=True)
train['Severity'].value_counts()
'''from sklearn.utils import resample

df0 = train[train['Severity']==0]

df1 = train[train['Severity']==1]

df2 = train[train['Severity']==2]

df3 = train[train['Severity']==3]



df_1 = resample(df1,replace=True,n_samples=3000,random_state=123)

df_0 = resample(df0,replace=True,n_samples=3000,random_state=123) 

 

train = pd.concat([df_1, df2, df3, df_0])

 



train['Severity'].value_counts()'''
x=train.drop('Severity',axis=1)

y=train['Severity']
l_quan=['Safety_Score','Control_Metric','Turbulence_In_gforces','Cabin_Temperature','Max_Elevation','Adverse_Weather_Metric','Total_Safety_Complaints']

l_cato=['Days_Since_Inspection','Accident_Type_Code','Violations']
for a in l_quan:

    sns.distplot(x[a])

    plt.show()
for a in l_cato:

    sns.countplot(x[a])

    plt.show()
for a in l_quan:

    x1,y1=ecdf(x[a])

    x2,y2=ecdf(np.random.normal(np.mean(x[a]),np.std(x[a]),size=10000))

    plt.plot(x1,y1,marker='+',linestyle=None)

    plt.xlabel(a)

    plt.plot(x2,y2)

    plt.legend(['Real', 'Theory'])

    plt.show()

    
x['Total_Safety_Complaints'].corr(train['Severity'])
for a in l_quan:

    print(x[a].corr(train['Severity']))

    print(a)

    print()

    
    x['Adverse_Weather_Metric']=np.log(x['Adverse_Weather_Metric'])

    x1,y1=ecdf(x['Adverse_Weather_Metric'])

    x2,y2=ecdf(np.random.normal(np.mean(x['Adverse_Weather_Metric']),np.std(x['Adverse_Weather_Metric']),size=10000))

    plt.plot(x1,y1,marker='+',linestyle=None)

    plt.xlabel('Adverse_Weather_Metric')

    plt.plot(x2,y2)

    plt.legend(['Real', 'Theory'])

    plt.show()

    sns.distplot(x['Adverse_Weather_Metric'])
x['Total_Safety_Complaints']=np.log(x['Total_Safety_Complaints']+1)

x['Total_Safety_Complaints'].unique()


x1,y1=ecdf(x['Total_Safety_Complaints'])

x2,y2=ecdf(np.random.normal(np.mean(x['Total_Safety_Complaints']),np.std(x['Total_Safety_Complaints']),size=10000))

plt.plot(x1,y1,marker='+',linestyle=None)

plt.xlabel('Total_Safety_Complaints')

plt.plot(x2,y2)

plt.legend(['Real', 'Theory'])

plt.show()

sns.distplot(x['Total_Safety_Complaints'])
test['Total_Safety_Complaints']=np.log(test['Total_Safety_Complaints']+1)

test['Adverse_Weather_Metric']=np.log(test['Adverse_Weather_Metric'])
x.head()
test.head()
from sklearn.preprocessing import StandardScaler

ss=StandardScaler().fit(x[l_quan])
x_quan=pd.DataFrame(ss.transform(x[l_quan]),columns=l_quan,index=x.index)

x[l_quan]=x_quan

x.head()
test_quan=pd.DataFrame(ss.transform(test[l_quan]),columns=l_quan,index=test.index)

test[l_quan]=test_quan

test.head()
sns.countplot(y)
from sklearn.model_selection import train_test_split,cross_val_score

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=343,stratify=y)
from sklearn.ensemble import GradientBoostingClassifier



gbc=GradientBoostingClassifier(n_estimators=200,max_depth=4,verbose=True)

cross_val_score(gbc,X_train,y_train,cv=3)
gbc.fit(X_train,y_train)
gbc.score(X_train,y_train)
gbc.score(X_test,y_test)
ans=gbc.predict(test)
sample_submission.head()
test.columns
test['Severity']=ans
test['Severity'].replace(0,'Minor_Damage_And_Injuries',inplace=True)

test['Severity'].replace(1,'Significant_Damage_And_Fatalities',inplace=True)

test['Severity'].replace(2,'Significant_Damage_And_Serious_Injuries',inplace=True)

test['Severity'].replace(3,'Highly_Fatal_And_Damaging',inplace=True)
test.drop(['Safety_Score', 'Days_Since_Inspection', 'Total_Safety_Complaints',

       'Control_Metric', 'Turbulence_In_gforces', 'Cabin_Temperature',

       'Accident_Type_Code', 'Max_Elevation', 'Violations',

       'Adverse_Weather_Metric'],axis=1,inplace=True)
test.head()
test.to_csv('ans.csv')