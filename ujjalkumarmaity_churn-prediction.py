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
df=pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head()
#describe data set
df.describe()
#check data set contain null value or not
df.isnull().sum()
#shape of data set
print('shape of data set',df.shape)
# number of feature
print('Number of feature',len(df.columns)-1)
# feature type
print(df.dtypes)
import matplotlib.pyplot as plt
import seaborn as sns
# check dataset balance or umbalance
sns.countplot(x='Churn',data=df)
from sklearn.preprocessing import LabelEncoder
col=df.columns
categorical=[i for i in col if df[i].dtype==object]
print('categorical feature are',categorical)

le=LabelEncoder()
for i in categorical:
    df[i]=le.fit_transform(df[i])
categorical=[i for i in col if df[i].dtype==object]
print('categorical feature are',categorical)
from sklearn.model_selection import train_test_split
x=df.iloc[:,0:-1]
y=df['Churn']
x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y)
def auc_roc(y_predict):
    from sklearn.metrics import roc_auc_score,roc_curve
    plt.figure(figsize=(10,5))
    fpr,tpr,_=roc_curve(y_test,y_predict)
    auc_score=roc_auc_score(y_test,y_predict)
    plt.plot(fpr,tpr,label='auc score= '+str(auc_score))
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.plot([0,1],[0,1],'--')
    plt.title('AUC ROC Curve')
    plt.show()
def confusion_(y_predict):
    from sklearn.metrics import confusion_matrix
    plt.figure(figsize=(10,5))
    confusion=confusion_matrix(y_test,y_predict)
    norm_cm = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]
    sns.heatmap(norm_cm,annot=confusion,center =2.2,fmt='2g', xticklabels=['Predicted: No','Predicted: Yes'], yticklabels=['Actual: No','Actual: Yes'])
    plt.title('confusion matrix')
    plt.show()
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(x_train,y_train)
print('train accuracy',model.score(x_train,y_train))
y_predict=model.predict(x_test)
print('test accuracy',accuracy_score(y_test,y_predict))
auc_roc(y_predict)
confusion_(y_predict)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
print('train accuracy',model.score(x_train,y_train))
y_predict=model.predict(x_test)
print('test accuracy',accuracy_score(y_test,y_predict))
auc_roc(y_predict)
confusion_(y_predict)
from sklearn.neighbors import KNeighborsClassifier
n_neighbors=[1,3,5,7,11,15,50]
test_acc=[]
train_acc=[]
for i in n_neighbors:    
    model=KNeighborsClassifier(n_neighbors=i)
    model.fit(x_train,y_train)
    train_acc_score=model.score(x_train,y_train)
    #print('train accuracy',train_acc_score)
    train_acc.append(train_acc_score)
    y_predict=model.predict(x_test)
    test_acc_score=accuracy_score(y_test,y_predict)
    #print('test accuracy',test_acc_score)
    test_acc.append(test_acc_score)
plt.plot(n_neighbors,train_acc,label='train accuracy')
plt.plot(n_neighbors,test_acc,label='test accuracy')
plt.xlabel('K')
plt.ylabel('accuracy')
plt.show()
print(train_acc)
print(test_acc)
k=13
model=KNeighborsClassifier(n_neighbors=k)
model.fit(x_train,y_train)
train_acc_score=model.score(x_train,y_train)
print('train accuracy',train_acc_score)
y_predict=model.predict(x_test)
test_acc_score=accuracy_score(y_test,y_predict)
print('test accuracy',test_acc_score)
auc_roc(y_predict)
confusion_(y_predict)

