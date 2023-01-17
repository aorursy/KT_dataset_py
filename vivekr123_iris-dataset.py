import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
# import dataset

data = pd.read_csv('../input/Iris.csv').drop(labels="Id", axis=1)

data.columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Class']

X = data.iloc[:,0:4]

Y = data.iloc[:, 4:5]

data
# divide into sepals and petals

sepal = data.iloc[:, [0, 1, 4]]

petal = data.iloc[:, [2, 3, 4]]

# divide sepals in classes

sepal_setosa = sepal[sepal['Class'] == 'Iris-setosa']

sepal_virginica = sepal[sepal['Class'] == 'Iris-virginica']

sepal_versicolor = sepal[sepal['Class'] == 'Iris-versicolor']

# scatter plot and identify clusters

plt.figure(1)

plt.scatter(sepal_setosa.iloc[:, 0], sepal_setosa.iloc[:, 1], color="r")

plt.scatter(sepal_virginica.iloc[:, 0], sepal_virginica.iloc[:, 1], color="g")

plt.scatter(sepal_versicolor.iloc[:, 0], sepal_versicolor.iloc[:, 1], color="b")

plt.yticks([])

plt.xticks([])

plt.xlabel("Sepal Length")

plt.ylabel("Sepal Width")



# divide petals in classes

plt.figure(2)

petal_setosa = petal[petal['Class'] == 'Iris-setosa']

petal_virginica = petal[petal['Class'] == 'Iris-virginica']

petal_versicolor = petal[petal['Class'] == 'Iris-versicolor']

# scatter plot and identify clusters

plt.scatter(petal_setosa.iloc[:, 0], petal_setosa.iloc[:, 1], color="r")

plt.scatter(petal_virginica.iloc[:, 0], petal_virginica.iloc[:, 1], color="g")

plt.scatter(petal_versicolor.iloc[:, 0], petal_versicolor.iloc[:, 1], color="b")

plt.yticks([])

plt.xticks([])

plt.xlabel("Petal Length")

plt.ylabel("Petal Width")
# split data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, shuffle=True)
# compare different models

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm, tree

import xgboost

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.linear_model import LogisticRegression



classifiers = []



model1 = xgboost.XGBClassifier()

classifiers.append(model1)

model2 = svm.SVC()

classifiers.append(model2)

model3 = tree.DecisionTreeClassifier()

classifiers.append(model3)

model4 = RandomForestClassifier()

classifiers.append(model4)

model5 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')

classifiers.append(model5)



print("All 4 variables used")

for clf in classifiers: 

  clf.fit(X_train, y_train)

  y_pred = clf.predict(X_test)

  acc = accuracy_score(y_test, y_pred)

  print(acc)

  cm = confusion_matrix(y_test, y_pred)

  print(cm)

 

print("Only Petal data used")

X_train = X_train[["PetalWidth", "PetalLength"]]

X_test = X_test[["PetalWidth", "PetalLength"]]

for clf in classifiers: 

  clf.fit(X_train, y_train)

  y_pred = clf.predict(X_test)

  acc = accuracy_score(y_test, y_pred)

  print(acc)

  cm = confusion_matrix(y_test, y_pred)

  print(cm)