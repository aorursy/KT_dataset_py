# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df1=pd.read_csv("../input/diabetes.csv")
df1.head()
df1.describe()

sns.heatmap(df1.isnull(),yticklabels=False,cmap='viridis')
sns.heatmap(df1,yticklabels=False,cmap='viridis')
sns.set_style('whitegrid')
sns.countplot(x='Outcome',hue='Outcome',data=df1,palette='cubehelix')
plt.scatter(x='Outcome',y='Age',data=df1)
plt.ylabel('Age')
plt.xlabel('Outcome')
sns.distplot(df1['Age'],kde=False,color='darkblue',bins=30)
sns.distplot(df1['BloodPressure'],kde=False,color='royalblue',bins=20)
sns.jointplot(x='Age',y='BloodPressure',data=df1)
import seaborn as sns
sns.set(style="whitegrid")
#tips=sns.load_dataset("diabetes.csv")
plt.figure(figsize=(15,8))

ax=sns.barplot(x="Age", y="BloodPressure", data=df1,)
import cufflinks as cf
cf.go_offline()
df1['BMI'].iplot(kind='hist',bins=40,color='red')
from sklearn.model_selection import train_test_split
#looking one last time at df
df1.head()
x=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
y=['Output']
df2=pd.DataFrame(data=df1)
df2.head()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df1.drop('Outcome',axis=1),df1['Outcome'],test_size=0.20,random_state=101)
X_test.head()
from sklearn.linear_model import LogisticRegression
LRModel=LogisticRegression()
LRModel.fit(X_train,y_train)
predictions_diabetes=LRModel.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,predictions_diabetes))

# paitentid_54=pd.DataFrame([1,123,126,60,0,30.1,0.349,47],columns=x)
#Defining a sample data to test the model
x=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
data=[1,130,126,60,30,30.1,0.349,47]
paitentid_54=pd.DataFrame([data],columns=x)
paitentid_54.head()
df1.head()
predictions_diabetes=LRModel.predict(paitentid_54)

"""output is zero means this paitentid_54 is not going to have diabetes anytime soon"""
print(predictions_diabetes)
#this is it till now, i Will update it soon
