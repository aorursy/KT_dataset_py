import numpy as np
import xlrd
import pandas as pd
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process,preprocessing
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix
from sklearn import metrics
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
clean_train = train[train['Age'] > 0]
print(clean_train.shape)
print(train.shape)
print(test.shape)
test['Age'].fillna(test['Age'].mean(), inplace=True)
train['Age'].fillna(train['Age'].mean(), inplace=True)
label = preprocessing.LabelEncoder()
clean_train['sex_code']=label.fit_transform(clean_train['Sex'])
test['sex_code'] = label.fit_transform(test['Sex'])
train['sex_code'] = label.fit_transform(train['Sex'])
columns = ['Age','sex_code','Pclass']
train_x = train[columns]
train_y = train['Survived']

clean_train_x = clean_train[columns]
clean_train_y = clean_train['Survived']

test_x = test[columns]
from sklearn.cross_validation import train_test_split
x3,x4,y3,y4 = train_test_split(train_x, train_y)

x1,x2,y1,y2 = train_test_split(clean_train_x,clean_train_y)
y4.shape
alg00 = naive_bayes.MultinomialNB()
alg01 = naive_bayes.MultinomialNB()

alg10 = naive_bayes.GaussianNB()
alg11 = naive_bayes.GaussianNB()

alg02 = naive_bayes.BernoulliNB()
alg20 = naive_bayes.BernoulliNB()

#Fit the algorithms for data with and without missing Age values respectively
alg00.fit(x3,y3)
alg01.fit(x1,y1)

alg10.fit(x3,y3)
alg11.fit(x1,y1)

alg02.fit(x3,y3)
alg20.fit(x1,y1)
y_pred00 = alg00.predict(x4)
y_pred01 = alg01.predict(x2)

y_pred10 = alg10.predict(x4)
y_pred11 = alg11.predict(x2)

y_pred02 = alg02.predict(x4)
y_pred20 = alg20.predict(x2)
from sklearn.metrics import accuracy_score,confusion_matrix
print(alg00.__class__.__name__)
print()

print("Missing Age Values are filled with Mean values")
print('Accuracy_score: ',accuracy_score(y4,y_pred00))
print(confusion_matrix(y4,y_pred00))

print()
print("Without Missing Age Values")
print('Accuracy_score: ',accuracy_score(y2,y_pred01))
print(confusion_matrix(y2,y_pred01))
print(alg10.__class__.__name__)
print()
print("Missing Age Values are filled with Mean values")
print('Accuracy_score: ',accuracy_score(y4,y_pred10))
print(confusion_matrix(y4,y_pred10))
print()

print("Without Missing Age values")
print('Accuracy_score: ',accuracy_score(y2,y_pred11))
print(confusion_matrix(y2,y_pred11))
print(alg02.__class__.__name__)
print()

print("Missing Age Values are filled with Mean values")
print("Accuracy_score: ",accuracy_score(y4,y_pred02))
print(confusion_matrix(y4,y_pred02))
print()

print("With Missing Age Values")
print('Accuracy_score: ',accuracy_score(y2,y_pred20))
print(confusion_matrix(y2,y_pred20))
