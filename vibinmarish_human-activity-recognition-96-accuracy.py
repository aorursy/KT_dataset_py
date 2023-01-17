import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import os

print(os.listdir("../input"))

df=pd.read_csv("../input/human-activity-recognition-with-smartphones/train.csv")

df_test=pd.read_csv("../input/human-activity-recognition-with-smartphones/test.csv")

df
print("Missing values:",df.isnull().values.any())
df['Activity'].value_counts()
df2=df.drop(['subject'],axis=1)

a=set(df['Activity'])

df2_test=df_test.drop(['subject'],axis=1)

print(a)
temp=df['Activity'].value_counts()



dta = pd.DataFrame({'Type': temp.index,

                   'Occurrence': temp.values

                  })



plt.bar(dta['Type'],dta['Occurrence'])

plt.xticks(rotation=75)

plt.xlabel('Type Of Activity')

plt.ylabel('No of Occurrence')

plt.show()
len(df2.columns)

X=pd.DataFrame(df2.drop('Activity',axis=1))

Y=df2.Activity.values.astype(object)

X_test=pd.DataFrame(df2_test.drop('Activity',axis=1))

Y_test=df2_test.Activity.values.astype(object)



from sklearn.preprocessing import StandardScaler



x_scaled=StandardScaler().fit_transform(X)

x_testscaled=StandardScaler().fit_transform(X_test)



from sklearn.preprocessing import LabelEncoder



y=LabelEncoder().fit_transform(Y)

y_test=LabelEncoder().fit_transform(Y_test)
from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



clf=[SVC(),GaussianNB(),DecisionTreeClassifier(),RandomForestClassifier()]

name=['Support Vector','Naive Bayes','Decision Tree','Random Forest']

print('Accuracy')

for model,names in zip(clf,name):

    print(names)

    print(cross_val_score(model,x_scaled,y,cv=5))

params_grid = [{'kernel': ['rbf'], 'gamma': [0.001,0.01,0.1,1],

                     'C': [1, 10, 100, 1000]},

                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]



from sklearn.model_selection import GridSearchCV





svm_model=GridSearchCV(SVC(),params_grid,cv=5,n_jobs=-1)

svm_model.fit(x_scaled,y)

print('Grid best parameter (max accuracy): ',svm_model.best_params_)

print('Grid best score: (accuracy)',svm_model.best_score_)

svm_final=svm_model.best_estimator_

print("Training score: ",svm_final.score(x_scaled,y))

print("Testing score",svm_final.score(x_testscaled,y_test))
import seaborn as sb

from sklearn.metrics import confusion_matrix



svm_predicted=svm_final.predict(x_testscaled)

svm_confuse=confusion_matrix(y_test,svm_predicted)

df_cm=pd.DataFrame(svm_confuse)



plt.figure(figsize=(5.5,4))

sb.heatmap(df_cm,annot=True,fmt='g')

plt.title("Confusion Matrix Heatmap")

plt.xlabel("True Label")

plt.ylabel("Predicted Label")

plt.show()

from sklearn.metrics import classification_report

print("Classification Report")

print(classification_report(y_test,svm_predicted))

print("Training score: ",svm_final.score(x_scaled,y))

print("Testing score",svm_final.score(x_testscaled,y_test))