import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import KFold,cross_val_score

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier
# the iris data dont have the column names so we are adding the column names externaly

names=['sepal-length','sepal-width','petal-length','petal-width','class']

data=pd.read_csv('../input/iris.data',names=names)
data.head()
data.info()
data.describe()
data.isnull().sum()
data.plot.box(figsize=(10,5));
data.plot(kind='Box',subplots=True,layout=(2,2),sharex=False,sharey=False,figsize=(10,5))

plt.show()
data.columns
data.plot.scatter(x='petal-length',y='petal-width',label='Group 1');
data.plot.scatter(x='sepal-length',y='sepal-width',label='Group 2',color='red');
data.hist();
from pandas .plotting import scatter_matrix
scatter_matrix(data,diagonal='kde',figsize=(10,6));
sns.pairplot(data,hue='class');
data.shape
from sklearn.model_selection import train_test_split
x=data.values[:,0:4]

y=data.values[:,4]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=7)
x_train.shape,y_train.shape,x_test.shape,y_test.shape
models = []

models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

# evaluate each model in turn

results = []

Names = []

for name, model in models:

#   print("\r")

#    print(name,model)

    kfold = KFold(n_splits=10, random_state=7)

    cv_results =cross_val_score(model, x_train, y_train, cv=kfold)

    results.append(cv_results)

    Names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())



    print(msg)
# Compare Algorithms

fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(Names)

plt.show()

KNN = KNeighborsClassifier()

KNN.fit(x_train,y_train)

y_pred=KNN.predict(x_test)

print("Accuracy Score",accuracy_score(y_test,y_pred)*100,"\n")

print("Confusion Matrix \n",confusion_matrix(y_test,y_pred),"\n")

print("Classification Report \n\n",classification_report(y_test,y_pred))