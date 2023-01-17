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
import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



sns.set_style('darkgrid')

import warnings

warnings.filterwarnings('ignore')



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
heart = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
heart.head(3)
# GENDER



# female = 0

# male = 1



female = len(heart[heart['sex'] ==0])

male = len(heart[heart['sex'] ==1])



print('Percentage of female: {:.2f} %' .format(female/len(heart['sex'])*100))

print('Percentage of male: {:.2f} %' .format(male/len(heart['sex'])*100))
plt.figure(figsize=(15,8))

cbar_kws = { 'ticks' : [-1, -0.5, 0, 0.5, 1], 'orientation': 'horizontal'}

sns.heatmap(heart.corr(), cmap='PuBu', linewidths=0.1, annot=True, vmax=1, vmin=-1, cbar_kws=cbar_kws)
plt.figure(figsize=(15,8))

sns.distplot(heart['age'], hist=True, bins=30, color='grey')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.title('Distribution of age', fontsize=15)
heart['sex'].value_counts
plt.figure(figsize=(15,8))

sns.countplot(heart['sex'], palette='PuBu')

plt.xlabel('Gender')

plt.ylabel('Count')

plt.title('Gender', fontsize=15)
plt.figure(figsize=(15,8))

sns.countplot(heart['age'], hue=heart['sex'], palette='PuBu', saturation=0.8)

plt.xlabel('Age')

plt.ylabel('Count')

plt.title('Gender count', fontsize=15)

plt.legend(loc='upper right', fontsize=15, labels=['Female', 'Male'])
plt.figure(figsize=(15,8))

sns.countplot(heart['target'], palette='PuBu')

plt.xlabel('Target')

plt.ylabel('Count')

plt.title('Target count', fontsize=15)
plt.figure(figsize=(15,8))

sns.countplot(heart['age'], hue=heart['target'], palette='PuBu', saturation=0.8)

plt.xlabel('Age')

plt.ylabel('Count')

plt.title('Target count', fontsize=15)

plt.legend(loc='upper right', fontsize=15, labels=['No disease', 'Disease'])
countNoDisease = len(heart[heart.target == 0])

countHaveDisease = len(heart[heart.target == 1])

print("Percentage of Patients without Heart Disease: {:.2f}%".format((countNoDisease / (len(heart.target))*100)))

print("Percentage of Patients with Heart Disease: {:.2f}%".format((countHaveDisease / (len(heart.target))*100)))
pd.crosstab(heart.sex,heart.target).plot(kind="bar",figsize=(15,6),color=['#1CA53B','#AA1111' ])

plt.title('Heart Disease Frequency for Sex')

plt.xlabel('Sex (0 = Female, 1 = Male)')

plt.xticks(rotation=0)

plt.legend(["No Disease", "Disease"])

plt.ylabel('Frequency')

plt.show()
pd.crosstab(heart.fbs,heart.target).plot(kind="bar",figsize=(15,6),color=['#1CA53B','#AA1111' ])

plt.title('Heart Disease Frequency Fasting Blood Sugar')

plt.xlabel('FBS - (Fasting Blood Sugar > 120 mg/dl) (1 = true; 0 = false)')

plt.xticks(rotation=0)

plt.legend(["No Disease", "Disease"])

plt.ylabel('Frequency')

plt.show()
pd.crosstab(heart.cp,heart.target).plot(kind="bar",figsize=(15,6),color=['g','m'])

plt.title('Heart Disease Frequency Chest Pain')

plt.xlabel('Chest Pain Type')

plt.xticks(rotation=0)

plt.legend(["No Disease", "Disease"])

plt.ylabel('Frequency')

plt.show()
plt.scatter(x=heart.age[heart.target==1], y=heart.thalach[(heart.target==1)], c="violet")

plt.scatter(x=heart.age[heart.target==0], y=heart.thalach[(heart.target==0)])

plt.legend(["Disease", "Not Disease"])

plt.xlabel("Age")

plt.ylabel("Maximum Heart Rate")

plt.show()
pd.crosstab(heart.slope,heart.target).plot(kind="bar",figsize=(15,6),color=['r','b'])

plt.title('Heart Disease Frequency for Slope')

plt.xlabel('The Slope of The Peak Exercise ST Segment ')

plt.xticks(rotation = 0)

plt.ylabel('Frequency')

plt.show()
heart.head(2)
a = pd.get_dummies(heart['cp'], prefix = "cp")

b = pd.get_dummies(heart['thal'], prefix = "thal")

c = pd.get_dummies(heart['slope'], prefix = "slope")
frames = [heart, a, b, c]

heart = pd.concat(frames, axis = 1)

heart.head()
heart = heart.drop(columns = ['cp', 'thal', 'slope'])

heart.head()
y = heart.target.values

x_data = heart.drop(['target'], axis = 1)
# Normalize

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
x
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state=0 )
accuracies = {}



lr = LogisticRegression()

lr.fit(x_train,y_train)

acc = lr.score(x_test,y_test)*100



accuracies['Logistic Regression'] = acc

print("Test Accuracy {:.2f}%".format(acc))
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train, y_train)



acc = nb.score(x_test,y_test)*100

accuracies['Naive Bayes'] = acc

print("Accuracy of Naive Bayes: {:.2f}%".format(acc))
svm = SVC(random_state = 1)

svm.fit(x_train, y_train)



acc = svm.score(x_test,y_test)*100

accuracies['SVM'] = acc

print("Test Accuracy of SVM Algorithm: {:.2f}%".format(acc))
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

dtc.fit(x_train, y_train)



acc = dtc.score(x_test, y_test)*100

accuracies['Decision Tree'] = acc

print("Decision Tree Test Accuracy {:.2f}%".format(acc))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)

rf.fit(x_train, y_train)



acc = rf.score(x_test,y_test)*100

accuracies['Random Forest'] = acc

print("Random Forest Algorithm Accuracy Score : {:.2f}%".format(acc))
colors = ["purple", "green", "orange", "magenta","#CFC60E","#0FBBAE"]



sns.set_style("whitegrid")

plt.figure(figsize=(16,5))

plt.yticks(np.arange(0,100,10))

plt.ylabel("Accuracy %")

plt.xlabel("Algorithms")

sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette=colors)

plt.show()
rf = RandomForestClassifier()
from sklearn.model_selection import GridSearchCV
#Using grid search to get best params for Randomforest

params = {

    'n_estimators':[10,50,100,150,200,250],

    'random_state': [10,5,15,20,50]

         }

gs = GridSearchCV(rf, param_grid=params, cv=5, n_jobs=-1)

gs.fit(x_train,y_train)
# Grid Search Score with test Data

print("Grid search score with random forest classifier = ",gs.score(x_test,y_test)*100)
#Best Params

gs.best_params_
# Creating the Confusion matrix

pred = gs.predict(x_test)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_pred=pred, y_true=y_test)
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
ab = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=1000)

ab.fit(x_train,y_train)

print('AdaBoost Accuracy with Decision Tree = ',(ab.score(x_test,y_test)*100))
ab = AdaBoostClassifier(base_estimator=LogisticRegression(max_iter=1000,solver = 'lbfgs'),n_estimators=1000)

ab.fit(x_train,y_train)

print('AdaBoost Accuracy with Logistic Reg = ',(ab.score(x_test,y_test)*100))
ab = AdaBoostClassifier(algorithm='SAMME',base_estimator=SVC(kernel='linear',C = 1000, gamma=1),n_estimators=1000)

ab.fit(x_train,y_train)

print('AdaBoost Accuracy with SVC = ',(ab.score(x_test,y_test)*100))