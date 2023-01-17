# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
df.shape
df.head()
df.columns
df.isna().sum()
df['status'].value_counts().plot(kind='bar',rot=1)
df['salary']=np.where(df['salary']==np.NaN,0,df['salary'])
df.head()
df['status']=np.where(df['status']=="Placed",1,0)
df.groupby(['degree_t'])['status'].sum().plot(kind="bar",rot=0)
plt.xlabel("Degree")
plt.ylabel("Placed Count")
sns.pairplot(df,hue="status")
sns.countplot(x=df['gender'],hue=df['status'])
df['ssc_b'].unique()
df['hsc_b'].unique()
df['ssc_b']=np.where(df['ssc_b']=="Central",1,0)
df['hsc_b']=np.where(df['hsc_b']=="Central",1,0)
dummies=pd.get_dummies(df['hsc_s'])
dummies1=pd.get_dummies(df['specialisation'])
df['workex']=np.where(df['workex']=="Yes",1,0)
df["gender"]=np.where(df['gender']=="M",1,0)
dummies2=pd.get_dummies(df['degree_t'])
df1=pd.concat([df,dummies,dummies1,dummies2],axis=1)
df1.head()

X=df.drop(['hsc_s','degree_t','specialisation','salary'],axis=1)
y=df['status']
X.head()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=0)
model1=RandomForestClassifier(n_estimators=4)
model1.fit(X_train,y_train)
y_pred1=model1.predict(X_test)
cf1=confusion_matrix(y_test,y_pred1)
cf1
cvs=cross_val_score(model1,X_train,y_train,cv=5)
cvs
cvs.mean()
accuracy_score(y_test,y_pred1)
print(classification_report(y_test,y_pred1))
plt.matshow(cf1)
plt.title("Confusion Matrix Plot")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
from sklearn.linear_model import LogisticRegression
model2=LogisticRegression()
model2.fit(X_train,y_train)
y_pred2=model2.predict(X_test)
cf2=confusion_matrix(y_test,y_pred2)
cf2
cvs=cross_val_score(model1,X_train,y_train,cv=5)
cvs
cvs.mean()
print(classification_report(y_test,y_pred2))
plt.matshow(cf2)
plt.title("Confusion Matrix Plot")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
accuracy_score(y_test,y_pred2)
