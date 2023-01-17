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
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix
df=pd.read_csv('../input/drug-classification/drug200.csv')
df.info()
sns.distplot(df.Age,bins=10)
df.Sex.unique()
df.BP.unique()
df.Cholesterol.unique()
df.Drug.unique()
sns.countplot(df.Sex)
sns.countplot(df.Cholesterol)
sns.countplot(df.BP)
plt.scatter(df.Na_to_K,range(0,len(df.BP)))
plt.scatter(df.Age,range(0,len(df.BP)))
df.info()
plt.figure(figsize=(8,8))
sns.countplot(df.Sex,hue='BP',data=df)
plt.figure(figsize=(8,8))
sns.countplot(df.Sex,hue='Cholesterol',data=df)
plt.figure(figsize=(8,8))
sns.countplot(df.Sex,hue='Drug',data=df)
sns.scatterplot(df.Na_to_K,df.Age,hue='Sex',data=df)
sns.scatterplot(df.Age,df.Na_to_K,hue='BP',data=df)
sns.scatterplot(df.Age,df.Na_to_K,hue='Cholesterol',data=df)
sns.scatterplot(df.Age,df.Na_to_K,hue='Drug',data=df)
sns.boxplot(df.Age)
sns.boxplot(df.Na_to_K)
def outl(x):
    sns.boxplot(df[str(x)])
    iqr=df[str(x)].quantile(0.75)-df[str(x)].quantile(0.25)
    uiqr=df[str(x)].quantile(0.75) + (1.5*iqr)
    liqr=df[str(x)].quantile(0.25) - (1.5*iqr)
    return iqr,uiqr,liqr
outl('Na_to_K')
a=df[df['Na_to_K'] > 32.7817].index
for i in range(0,len(a)):
    df.loc[a[i],['Na_to_K']] = 32.7817
sns.boxplot(df.Na_to_K)
df=pd.get_dummies(df,drop_first=True)
df.columns
x=df.iloc[:,0:6].values
y=df.iloc[:,6:9].values
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2,random_state=0)
train_x.shape
test_x.shape
train_y.shape
test_y.shape
sc=StandardScaler()
train_x=sc.fit_transform(train_x)
test_x=sc.transform(test_x)
model=DecisionTreeClassifier()
model.fit(train_x,train_y)
pred_y1=model.predict(test_x)
model=RandomForestClassifier()
model.fit(train_x,train_y)
pred_y2=model.predict(test_x)
error = []
accuracy = []
# Calculating error for K values between 1 and 40
for i in range(1,40,2):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x, train_y)
    pred_i = knn.predict(test_x)
    error.append(np.mean(pred_i != test_y))
    accuracy.append(accuracy_score(test_y, pred_i))
plt.figure(figsize=(12, 6))
plt.plot(range(1,40,2), error, color='red', linestyle='dashed', marker=
'o',
markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
model=KNeighborsClassifier(n_neighbors=3)
model.fit(train_x,train_y)
pred_y3=model.predict(test_x)
accuracy_score(pred_y3,test_y)
accuracy_score(pred_y2,test_y)
accuracy_score(pred_y1,test_y)


