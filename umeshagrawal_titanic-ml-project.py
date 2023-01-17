# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Data Analysis

import numpy as np

import pandas as pd



#Visualisation

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



#Machine Learning

import sklearn

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

sns.set_style('whitegrid')
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
train.info()
train.describe()
train.drop(['PassengerId','Name','Ticket','Fare'],axis =1 , inplace = True)
train.head()
corr = train.corr()

sns.heatmap(corr)
sns.heatmap(train.isnull())
train.drop('Cabin', axis=1, inplace = True)
train.head()
train['Age'].mean()
sns.boxplot(x ='Pclass', y ='Age', data =train)
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        if Pclass == 1:

            return 37

        elif Pclass == 2:

            return 29

        else:

            return 24

    else:

        return Age
train['Age']= train[['Age','Pclass']].apply(impute_age,axis = 1)
sns.heatmap(train.isnull())
train.head()
sex = pd.get_dummies(train['Sex'], drop_first = True)
sex.head()
embark = pd.get_dummies(train['Embarked'], drop_first = True)
embark.head()
train = pd.concat([train,sex,embark], axis = 1)
train.head()
train.drop(['Sex', 'Embarked'],axis=1,inplace = True)
train.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), train['Survived'], 

                                                    test_size = 0.30, random_state = 101) 
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
acc_Logistic_Regression = round(logmodel.score(X_train, y_train) * 100, 2)

acc_Logistic_Regression
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
models = []

acc = []

precision = []

recall = []

f1 = []
models.append('Logistic Regression')

from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 

                             recall_score, f1_score ,make_scorer)

print('Confusion Matrix for LR: \n',confusion_matrix(y_test, logmodel.predict(X_test)))

print('Accuracy for LR: \n',accuracy_score(y_test, logmodel.predict(X_test)))

acc.append(accuracy_score(y_test, logmodel.predict(X_test)))

print('Precision for LR: \n',precision_score(y_test, logmodel.predict(X_test)))

precision.append(precision_score(y_test, logmodel.predict(X_test)))

print('Recall for LR: \n',recall_score(y_test, logmodel.predict(X_test)))

recall.append(recall_score(y_test, logmodel.predict(X_test)))

print('f1_score for LR: \n',f1_score(y_test, logmodel.predict(X_test)))

f1.append(f1_score(y_test, logmodel.predict(X_test)))
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()

classifier.fit(X_train,y_train)
predictions = classifier.predict(X_test)
acc_Naive_Bayes = round(classifier.score(X_train, y_train) * 100, 2)

acc_Naive_Bayes
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
models.append('Naive Bayes')
print('Confusion Matrix for RF: \n',confusion_matrix(y_test, classifier.predict(X_test)))

print('Accuracy for RF: \n',accuracy_score(y_test, classifier.predict(X_test)))

acc.append(accuracy_score(y_test, classifier.predict(X_test)))

print('Precision for RF: \n',precision_score(y_test, classifier.predict(X_test)))

precision.append(precision_score(y_test, classifier.predict(X_test)))

print('Recall for RF: \n',recall_score(y_test, classifier.predict(X_test)))

recall.append(recall_score(y_test, classifier.predict(X_test)))

print('f1_score for RF: \n',f1_score(y_test, classifier.predict(X_test)))

f1.append(f1_score(y_test, classifier.predict(X_test)))
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)
predictions = knn.predict(X_test)
acc_KNN = round(knn.score(X_train, y_train) * 100, 2)

acc_KNN
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
models.append('K Nearest Neighbour')
print('Confusion Matrix for RF: \n',confusion_matrix(y_test, knn.predict(X_test)))

print('Accuracy for RF: \n',accuracy_score(y_test, knn.predict(X_test)))

acc.append(accuracy_score(y_test, knn.predict(X_test)))

print('Precision for RF: \n',precision_score(y_test, knn.predict(X_test)))

precision.append(precision_score(y_test, knn.predict(X_test)))

print('Recall for RF: \n',recall_score(y_test, knn.predict(X_test)))

recall.append(recall_score(y_test, knn.predict(X_test)))

print('f1_score for RF: \n',f1_score(y_test, knn.predict(X_test)))

f1.append(f1_score(y_test, knn.predict(X_test)))
from sklearn.svm import SVC
model = SVC()

model.fit(X_train,y_train)
predictions = model.predict(X_test)
acc_SVM = round(model.score(X_train, y_train) * 100, 2)

acc_SVM
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
models.append('Support Vector Machine')
print('Confusion Matrix for RF: \n',confusion_matrix(y_test, model.predict(X_test)))

print('Accuracy for RF: \n',accuracy_score(y_test, model.predict(X_test)))

acc.append(accuracy_score(y_test, model.predict(X_test)))

print('Precision for RF: \n',precision_score(y_test, model.predict(X_test)))

precision.append(precision_score(y_test, model.predict(X_test)))

print('Recall for RF: \n',recall_score(y_test, model.predict(X_test)))

recall.append(recall_score(y_test, model.predict(X_test)))

print('f1_score for RF: \n',f1_score(y_test, model.predict(X_test)))

f1.append(f1_score(y_test, model.predict(X_test)))
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
acc_decision_tree = round(clf.score(X_train, y_train) * 100, 2)

acc_decision_tree
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test,y_pred))
#Importing the accuracy metric from sklearn.metrics library



from sklearn.metrics import accuracy_score

print('Accuracy Score on train data: ', accuracy_score(y_true=y_train, y_pred=clf.predict(X_train)))

print('Accuracy Score on test data: ', accuracy_score(y_true=y_test, y_pred=y_pred))

clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=50)

clf.fit(X_train, y_train)

print('Accuracy Score on train data: ', accuracy_score(y_true=y_train, y_pred=clf.predict(X_train)))

print('Accuracy Score on the test data: ', accuracy_score(y_true=y_test, y_pred=clf.predict(X_test)))

models.append('Decision Tree')
print('Confusion Matrix for RF: \n',confusion_matrix(y_test, clf.predict(X_test)))

print('Accuracy for RF: \n',accuracy_score(y_test, clf.predict(X_test)))

acc.append(accuracy_score(y_test, clf.predict(X_test)))

print('Precision for RF: \n',precision_score(y_test, clf.predict(X_test)))

precision.append(precision_score(y_test, clf.predict(X_test)))

print('Recall for RF: \n',recall_score(y_test, clf.predict(X_test)))

recall.append(recall_score(y_test, clf.predict(X_test)))

print('f1_score for RF: \n',f1_score(y_test, clf.predict(X_test)))

f1.append(f1_score(y_test, clf.predict(X_test)))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()

rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
acc_random_forest = round(rfc.score(X_train, y_train) * 100, 2)

acc_random_forest
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
models.append('Random Forest')
print('Confusion Matrix for RF: \n',confusion_matrix(y_test, rfc.predict(X_test)))

print('Accuracy for RF: \n',accuracy_score(y_test, rfc.predict(X_test)))

acc.append(accuracy_score(y_test, rfc.predict(X_test)))

print('Precision for RF: \n',precision_score(y_test, rfc.predict(X_test)))

precision.append(precision_score(y_test, rfc.predict(X_test)))

print('Recall for RF: \n',recall_score(y_test, rfc.predict(X_test)))

recall.append(recall_score(y_test, rfc.predict(X_test)))

print('f1_score for RF: \n',f1_score(y_test, rfc.predict(X_test)))

f1.append(f1_score(y_test, rfc.predict(X_test)))
model_dict = {'Models': models,

             'Accuracies': acc,

             'Precision': precision,

             'Recall': recall,

             'f1-score': f1}
model_df = pd.DataFrame(model_dict)
model_df = model_df.sort_values(['Accuracies', 'f1-score', 'Recall', 'Precision'],

                               ascending=False)
model_df