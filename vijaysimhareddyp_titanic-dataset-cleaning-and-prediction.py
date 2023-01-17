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
eda=pd.read_csv('/kaggle/input/titanic-dataset-from-kaggle/train.csv')
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

eda
eda.head()
eda.tail()
eda.info()
eda.describe()
eda.isnull().sum()
eda['Age']=eda['Age'].fillna(value=eda['Age'].mean())
eda.isnull().sum()
eda.boxplot()
IQR_Fare=eda['Fare'].quantile(0.75)-eda['Fare'].quantile(0.25)

IQR_Fare
Upper_OutlierLimit=eda['Fare'].quantile(0.75)+1.5*IQR_Fare

Upper_OutlierLimit
OutlierValues=eda[(eda['Fare']>Upper_OutlierLimit)]

OutlierValues
eda['Fare']=np.where(eda['Fare']>65.6,eda['Fare'].quantile(0.85),eda['Fare'])

eda
IQR_Age=eda['Age'].quantile(0.75)-eda['Age'].quantile(0.25)

IQR_Age
Upper_OutlierLimit2=eda['Age'].quantile(0.75)+1.5*IQR_Age

Upper_OutlierLimit2
OutlierValues2=eda[(eda['Age']>Upper_OutlierLimit2)]

OutlierValues2
eda['Age']=np.where(eda['Age']>54.5,eda['Age'].quantile(0.95),eda['Age'])

eda
eda.boxplot(column=['Fare'])
obj=eda.dtypes==np.object

print(obj)
eda.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)

eda
eda=pd.get_dummies(eda,drop_first=True)

eda
cols=eda.columns

cols=['PassengerId','Pclass','Age','SibSp','Parch','Fare','Sex_male','Embarked_Q','Embarked_S','Survived']
eda=eda[cols]

eda
x=eda.iloc[:,:-1].values

x.shape
y=eda.iloc[:,-1].values

y.shape
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25 , random_state=5)
x_train.shape
x_test.shape
y_train.shape
y_test.shape
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
y_test
y_pred
from sklearn.metrics import confusion_matrix

confusion = confusion_matrix(y_test, y_pred)

print(confusion)
TN = confusion [0,0]

FP = confusion [0,1]

FN = confusion [1,0]

TP = confusion [1,1]
print(confusion)

print ("TN: ", TN)

print ("FP: ", FP)

print ("FN: ", FN)

print ("TP: ", TP)
from sklearn import metrics

accuracy = metrics.accuracy_score(y_test, y_pred)

accuracy1 = (TN+TP)/(TN+TP+FN+FP)

print ("Accuracy from metrics: ", accuracy)

print ("Accuracy Calculated: ", accuracy1)
print((FP+FN)/float(TP+TN+FP+FN))

print(round(1-metrics.accuracy_score(y_test, y_pred),4))
print("RECALL:", metrics.recall_score(y_test,y_pred))

print("CALCULATED RECALL:", (TP)/(TP+FN))
print ("SPECIFICITY/TRUE NEGATIVE RATE:", (TN)/(TN+FP))
print("FALSE POSITIVE RATE: ",(FN)/(FN+TP))
print("FALSE NEGATIVE RATE: ",(FP)/(TN+FP))
print ("Precision: ", round(metrics.precision_score(y_test,y_pred),2))

print ("PRECISION CALCULATED: ", round(TP/float(TP+FP),2))
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,roc_auc_score

accuracy = accuracy_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

roc_auc = roc_auc_score(y_test, y_pred)



print('Accuracy is  :' ,round(accuracy,2)*100)

print('F1 score is :' ,round(f1,2)*100)

print('Precision is  :',round(precision,2)*100)

print('Recall is  :',round(recall,4)*100)

print('Roc Auc is  :',round(roc_auc,2)*100)