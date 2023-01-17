# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# We are reading our data

df = pd.read_csv("../input/processed-cleveland-data/heart_disease.csv")
# First 5 rows of our data

df.head()
df.isnull().sum()
df.target.value_counts()
sns.countplot(x="target", data=df, palette="bwr")

plt.show()
countNoDisease = len(df[df.target == 0])

countHaveDisease = len(df[df.target== 1])

print("Percentage of Patients Haven't Heart Disease: {:.2f}%".format((countNoDisease / (len(df.target))*100)))

print("Percentage of Patients Have Heart Disease: {:.2f}%".format((countHaveDisease / (len(df.target))*100)))
sns.countplot(x='sex', data=df, palette="mako_r")

plt.xlabel("Sex (0 = female, 1= male)")

plt.show()
countFemale = len(df[df.sex == 0])

countMale = len(df[df.sex == 1])

print("Percentage of Female Patients: {:.2f}%".format((countFemale / (len(df.sex))*100)))

print("Percentage of Male Patients: {:.2f}%".format((countMale / (len(df.sex))*100)))
df.groupby('target').mean()
pd.crosstab(df.age,df.target).plot(kind="bar",figsize=(20,6))

plt.title('Heart Disease Frequency for Ages')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.savefig('heartDiseaseAndAges.png')

plt.show()

df.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',

       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'condition']
# Numeric data vs each other and condition:



plt.figure(figsize=(16, 10))

sns.pairplot(df[['resting_blood_pressure','cholesterol','max_heart_rate_achieved','st_depression','age', 'condition']], hue='condition',

           plot_kws=dict(s=25, alpha=0.75, ci=None))



plt.show()
def ctn_freq(df, cols, xaxi, hue=None,rows=4, columns=1):

    

    

    fig, axes = plt.subplots(rows, columns, figsize=(16, 12), sharex=True)

    axes = axes.flatten()



    for i, j in zip(df[cols].columns, axes):

        sns.pointplot(x=xaxi,

                      y=i,

                    data=df,

                    hue=hue,

                    ax=j,ci=False)      

        j.set_title(f'{str(i).capitalize()} vs. Age')



        

        plt.tight_layout()

ctn_freq(df, ['st_depression','max_heart_rate_achieved','resting_blood_pressure','cholesterol'], 'age', hue='condition',rows=4, columns=1)
df = pd.read_csv("../input/processed-cleveland-data/heart_disease.csv")
chest_pain=pd.get_dummies(df['cp'],prefix='cp',drop_first=True)

df=pd.concat([df,chest_pain],axis=1)

df.drop(['cp'],axis=1,inplace=True)

sp=pd.get_dummies(df['slope'],prefix='slope')

th=pd.get_dummies(df['thal'],prefix='thal')

frames=[df,sp,th]

df=pd.concat(frames,axis=1)

df.drop(['slope','thal'],axis=1,inplace=True)
df.head()

X = df.drop(['target'], axis = 1)

y = df.target.values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)
y = df.target.values

x_data = df.drop(['target'], axis = 1)

# Normalize

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=0)

#transpose matrices

x_train = x_train.T

y_train = y_train.T

x_test = x_test.T

y_test = y_test.T


from sklearn.preprocessing import StandardScaler

sc_X=StandardScaler()

X_train=sc_X.fit_transform(X_train)

X_test=sc_X.transform(X_test)
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import average_precision_score

from sklearn.model_selection import cross_val_score

from sklearn.metrics import precision_recall_curve

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import roc_curve

from sklearn.metrics import f1_score

from sklearn.metrics import auc

from sklearn.svm import SVC




#SVM classifier

svc_c=SVC(kernel='linear',random_state=0)

svc_c.fit(X_train,y_train)

svc_pred=svc_c.predict(X_test)

sv_cm=confusion_matrix(y_test,svc_pred)

sv_ac=accuracy_score(y_test, svc_pred)



#Bayes

gaussian=GaussianNB()

gaussian.fit(X_train,y_train)

bayes_pred=gaussian.predict(X_test)

bayes_cm=confusion_matrix(y_test,bayes_pred)

bayes_ac=accuracy_score(bayes_pred,y_test)



#SVM regressor

svc_r=SVC(kernel='rbf')

svc_r.fit(X_train,y_train)

svr_pred=svc_r.predict(X_test)

svr_cm=confusion_matrix(y_test,svr_pred)

svr_ac=accuracy_score(y_test, svr_pred)



#RandomForest

rdf_c=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)

rdf_c.fit(X_train,y_train)

rdf_pred=rdf_c.predict(X_test)

rdf_cm=confusion_matrix(y_test,rdf_pred)

rdf_ac=accuracy_score(rdf_pred,y_test)



# DecisionTree Classifier

dtree_c=DecisionTreeClassifier(criterion='entropy',random_state=0)

dtree_c.fit(X_train,y_train)

dtree_pred=dtree_c.predict(X_test)

dtree_cm=confusion_matrix(y_test,dtree_pred)

dtree_ac=accuracy_score(dtree_pred,y_test)



#KNN

knn=KNeighborsClassifier(n_neighbors=2)

knn.fit(X_train,y_train)

knn_pred=knn.predict(X_test)

knn_cm=confusion_matrix(y_test,knn_pred)

knn_ac=accuracy_score(knn_pred,y_test)
'''lr_c=LogisticRegression(random_state=0)

lr_c.fit(X_train,y_train)

lr_pred=lr_c.predict(X_test)

lr_cm=confusion_matrix(y_test,lr_pred)

lr_ac=accuracy_score(y_test, lr_pred)

plt.figure(figsize=(25,12))

plt.subplot(2,4,1)

plt.title("LogisticRegression_cm")

sns.heatmap(lr_cm,annot=True,cmap="Blues",fmt="d",cbar=False)'''

plt.subplot(2,4,2)

plt.title("SVM_regressor_cm")

sns.heatmap(sv_cm,annot=True,cmap="Blues",fmt="d",cbar=False)

plt.subplot(2,4,3)

plt.title("bayes_cm")

sns.heatmap(bayes_cm,annot=True,cmap="Oranges",fmt="d",cbar=False)

plt.subplot(2,4,4)

plt.title("RandomForest")

sns.heatmap(rdf_cm,annot=True,cmap="Blues",fmt="d",cbar=False)

plt.subplot(2,4,5)

plt.title("SVM_classifier_cm")

sns.heatmap(svr_cm,annot=True,cmap="Reds",fmt="d",cbar=False)

plt.subplot(2,4,6)

plt.title("DecisionTree_cm")

sns.heatmap(dtree_cm,annot=True,cmap="Blues",fmt="d",cbar=False)

plt.subplot(2,4,7)

plt.title("kNN_cm")

sns.heatmap(knn_cm,annot=True,cmap="Blues",fmt="d",cbar=False)

plt.show()

print('LogisticRegression_accuracy:\t',lr_ac)

print('SVM_regressor_accuracy:\t\t',svr_ac)

print('RandomForest_accuracy:\t\t',rdf_ac)

print('DecisionTree_accuracy:\t\t',dtree_ac)

print('KNN_accuracy:\t\t\t',knn_ac)

print('SVM_classifier_accuracy:\t',sv_ac)

print('Bayes_accuracy:\t\t\t',bayes_ac)
model_accuracy = pd.Series(data=[lr_ac,sv_ac,bayes_ac,svr_ac,rdf_ac,dtree_ac,knn_ac], 

                index=['LogisticRegression','SVM_classifier','Bayes','SVM_regressor',

                                      'RandomForest','DecisionTree_Classifier','KNN'])



fig= plt.figure(figsize=(10,7))

model_accuracy.sort_values().plot.barh()

plt.title('Model Accracy')