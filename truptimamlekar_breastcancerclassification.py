# import python libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df1=pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')

df1.head()
df1.columns = [*df1.columns[:-1], 'Target']

df1=df1.drop(['Target'],axis=1)

df1.head()
sns.pairplot(df1, hue = 'diagnosis', vars = ['radius_mean', 'texture_mean', 'area_mean', 'perimeter_mean', 'smoothness_mean'] )
hist_mean=df1.hist(bins=10, figsize=(15, 10),grid=False,)
plt.figure (figsize=(6,6))

p = sns.countplot(data=df1,x = 'diagnosis',)
sns.scatterplot(x = 'area_mean', y = 'smoothness_mean', hue = 'diagnosis', data = df1)
plt.figure(figsize=(20,10)) 

sns.heatmap(df1.corr(), annot=True)
f,ax=plt.subplots(2,3,figsize=(25,20))

box1=sns.boxplot(x="radius_mean", y="diagnosis", data=df1,ax=ax[0][0], palette="muted",sym='k.')

ax[0][0].set_xlabel('radius_mean')

box1=sns.boxplot(x="texture_mean", y="diagnosis", data=df1,ax=ax[0][1], palette="muted",sym='k.')

ax[0][1].set_xlabel('texture_mean')

box1=sns.boxplot(x="perimeter_mean", y="diagnosis", data=df1,ax=ax[0][2], palette="muted",sym='k.')

ax[0][2].set_xlabel('perimeter_mean')

box1=sns.boxplot(x="area_mean", y="diagnosis", data=df1,ax=ax[1][0], palette="muted",sym='k.')

ax[1][0].set_xlabel('area_mean')

box1=sns.boxplot(x="smoothness_mean", y="diagnosis", data=df1,ax=ax[1][1], palette="muted",sym='k.')

ax[1][1].set_xlabel('smoothness_mean')

box1=sns.boxplot(x="compactness_mean", y="diagnosis", data=df1,ax=ax[1][2], palette="muted",sym='k.')

ax[1][2].set_xlabel('compactness_mean')

plt.show ()
f,ax=plt.subplots(2,2,figsize=(25,15))

sns.violinplot(x="diagnosis", y="radius_mean",ax=ax[0][0],data=df1, palette="muted")

sns.violinplot(x="diagnosis", y="texture_mean",data=df1,ax=ax[0][1], palette="muted")

sns.violinplot(x="diagnosis", y="perimeter_mean",ax=ax[1][0],data=df1, palette="muted")

sns.violinplot(x="diagnosis", y="area_mean",ax=ax[1][1],data=df1, palette="muted")

f,ax=plt.subplots(1,1,figsize=(25,6))

sns.kdeplot(df1.loc[(df1['diagnosis']=='M'), 'perimeter_mean'], color='b', shade=True, Label='M')

sns.kdeplot(df1.loc[(df1['diagnosis']=='B'), 'perimeter_mean'], color='g', shade=True, Label='B')

plt.xlabel('Perimeter Mean')
f,ax=plt.subplots(1,1,figsize=(25,6))

sns.kdeplot(df1.loc[(df1['diagnosis']=='M'), 'area_worst'], color='b', shade=True, Label='M')

sns.kdeplot(df1.loc[(df1['diagnosis']=='B'), 'area_worst'], color='g', shade=True, Label='B') 

plt.xlabel('Area Worst') 
f,ax=plt.subplots(1,1,figsize=(25,6))

sns.kdeplot(df1.loc[(df1['diagnosis']=='M'), 'perimeter_worst'], color='b', shade=True, Label='M')

sns.kdeplot(df1.loc[(df1['diagnosis']=='B'), 'perimeter_worst'], color='g', shade=True, Label='B')

plt.xlabel('Perimeter Worst') 
f,ax=plt.subplots(1,2,figsize=(25,5))

plot1=sns.scatterplot(x="area_mean", y="diagnosis",color = "red",data=df1,ax=ax[0])

ax[0].set_xlabel('Area Mean')

plot2=sns.scatterplot(x="concavity_mean", y="diagnosis",color = "green",data=df1,ax=ax[1])

ax[1].set_xlabel('Concavity Mean')

plt.show ()
x=df1['radius_mean']

y=df1['texture_mean']

N = 569

colors = np.random.rand(N)

area = (25 * np.random.rand(N))**2

df2= pd.DataFrame({'X': x,'Y': y,'Colors': colors,"bubble_size":area})
plt.scatter('X', 'Y', s='bubble_size', c='Colors', alpha=0.5, data=df2)

plt.xlabel("X", size=16)

plt.ylabel("y", size=16)

plt.title("Bubble Plot with Matplotlib", size=18)
plt.style.use('seaborn')

x  = [(i+1) for i in range(10)]

y1 = df1['radius_mean'][1:11]

y2 = df1['texture_mean'][1:11]

plt.plot(x, y1, label="radius_mean", color = 'B')

plt.plot(x, y2, label="texture_mean", color = 'R')

plt.plot()

plt.xlabel("x axis")

plt.ylabel("y axis")

plt.title("Line Graph Example")

plt.legend()

plt.show()
import matplotlib.pyplot as plt

idxes = [(i+1) for i in range(10)]

arr1  = df1['compactness_mean'][:10]

arr2  = df1['concavity_mean'][:10]

arr3  = df1['concave points_mean'][:10]

# Adding legend for stack plots is tricky.

plt.plot([], [], color='#FF4500', label = 'D 1')

plt.plot([], [], color='#00EB42', label = 'D 2')

plt.plot([], [], color='#00D3FF', label = 'D 3')

plt.stackplot(idxes, arr1, arr2, arr3, colors= ['#FF4500', '#00EB42', '#00D3FF'])

plt.title('Stack Plot Example')

plt.legend()

plt.show()
df_copy=df1[['radius_mean','texture_mean','perimeter_mean','area_mean']].copy()
df_copy.plot.area()
from pandas.plotting import parallel_coordinates

df3=df1.head(20)

fig, ax = plt.subplots(figsize=(30,8)) 

parallel_coordinates(df3,'diagnosis', colormap=plt.get_cmap("Set3"))

plt.show()
sns.jointplot(x=df1["texture_worst"], y=df1["perimeter_worst"], kind='hex')

sns.jointplot(x=df1["texture_worst"], y=df1["area_worst"], kind='kde')
X = df1.drop(['diagnosis'], axis=1)

y = df1['diagnosis']
# split data into training and testing with a ratio of 80:20 using sklearn

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=5)
from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression 

from sklearn.ensemble import RandomForestClassifier 

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier
svc_model = SVC(gamma='auto')

svc_model.fit(X_train, y_train)

y_predict = svc_model.predict(X_test)

cm = confusion_matrix(y_test, y_predict)

sns.heatmap(cm, annot= True)

accuracy1=svc_model.score(X_test,y_test)

print (accuracy1*100,'%')

print(classification_report(y_test, y_predict))
logistic = LogisticRegression()

logistic.fit(X_train,y_train)

y_pred=logistic.predict(X_test)

cm=confusion_matrix(y_test,y_pred)

sns.heatmap(cm,annot=True)

accuracy1=logistic.score(X_test,y_test)

print (accuracy1*100,'%')

print(classification_report(y_test,y_pred))
des_class=DecisionTreeClassifier()

des_class.fit(X_train,y_train)

des_predict=des_class.predict(X_test)

cm=confusion_matrix(y_test,des_predict)

print(classification_report(y_test,des_predict))

accuracy3=des_class.score(X_test,y_test)

print(accuracy3*100,'%')

sns.heatmap(cm,annot=True)
model= KNeighborsClassifier()

model.fit(X_train,y_train)

model_predict=model.predict(X_test)

print(classification_report(y_test,model_predict))

accuracy4=model.score(X_test,y_test)

print(accuracy4*100,'%')

cm=confusion_matrix(y_test,model_predict)

sns.heatmap(cm,annot=True)