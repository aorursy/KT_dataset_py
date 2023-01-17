df.head()
import numpy as pd

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split
pwd
df=pd.read_csv('../input/heart.csv')
df.info()
df.describe()
plt.figure(figsize=(10,10))

sns.heatmap(df.corr(),annot=True,fmt='.1f')

plt.show()
#age analysis

df.age.value_counts()[:10]
sns.barplot(x=df.age.value_counts()[:10].index,y=df.age.value_counts()[:10].values)

plt.xlabel('Age')

plt.ylabel('Age Counter')

plt.title('Age Analysis')

plt.show()
df.target.value_counts()
countNoDisease=len(df[df.target==0])

countHaveDisease=len(df[df.target==1])

print("Percentage of People don't have heart disease: {:.2f}%".format((countNoDisease/len(df.target))*100))

print("Percentage of People have heart disease: {:.2f}%".format((countHaveDisease/len(df.target))*100))
countFemale=len(df[df.sex==0])

countMale=len(df[df.sex==1])

print("% of Female Patients: {:.2f}%".format((countFemale/len(df.sex))*100))

print("% of Male Pateints: {:.2f}%".format((countMale/len(df.sex))*100))
young_ages=df[(df.age>=29)&(df.age<40)]

middle_ages=df[(df.age>=40)&(df.age<55)]

elderly_ages=df[df.age>=55]

print("young ages",len(young_ages))

print("middle ages",len(middle_ages))

print("elderly ages",len(elderly_ages))
plt.figure(figsize=(8,8))

colors=['blue','green','red']

plt.pie([len(young_ages),len(middle_ages),len(elderly_ages)],labels=['young ages','middle ages','elderly ages'])

plt.show()
#chest pain analysis

df.cp.value_counts()
df.target.unique()
sns.countplot(df.target)

plt.title('Target 1 & 0')

plt.xlabel('Target')

plt.ylabel('Count')

plt.show()
df.corr()
from sklearn.linear_model import LogisticRegression

X=df.drop(['target'],axis=1)

y=df.target.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#Logistic Regression
lr=LogisticRegression()

lr.fit(X_train,y_train)

print("Logistic Regression Test Accuracy: {:.2f}%".format(lr.score(X_test,y_test)*100))
#KNN Model
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train,y_train)

print("KNN Test Accuracy: {:.2f}%".format(knn.score(X_test,y_test)*100))
#Support Vector
from sklearn.svm import SVC

svm=SVC(random_state=1)

svm.fit(X_train,y_train)

print("SVC Test Accuracy: {:.2f}%".format(svm.score(X_test,y_test)*100))
#Naives Bayes
from sklearn.naive_bayes import GaussianNB

nb=GaussianNB()

nb.fit(X_train,y_train)

print("Naives Bayes Test Accuracy: {:.2f}%".format(nb.score(X_test,y_test)*100))
#Random Forest
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=1000,random_state=1)

rf.fit(X_train,y_train)

print("Random Forest Test Accuracy: {:.2f}%".format(rf.score(X_test,y_test)*100))