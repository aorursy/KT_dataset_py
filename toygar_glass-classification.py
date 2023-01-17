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
from matplotlib import pyplot

from pandas.plotting import scatter_matrix

from sklearn.model_selection import train_test_split

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report
url = "/kaggle/input/glass/glass.csv"

dataset = pd.read_csv(url)
#univariate plots 

dataset.plot(kind="box", subplots=True, sharex=False, sharey=False)

pyplot.show()



dataset.hist()

pyplot.show()



#multivariate plot

scatter_matrix(dataset)

pyplot.show()
X = dataset.values[:, 0:9]

y = dataset.values[:,9]



X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
models = []

models.append(("LR", LogisticRegression(solver="liblinear", multi_class="ovr")))

models.append(("LDA", LinearDiscriminantAnalysis()))

models.append(("CART", DecisionTreeClassifier()))

models.append(("KNN", KNeighborsClassifier()))

models.append(("NB", GaussianNB()))

models.append(("SVM", SVC(gamma="auto")))
results = []

names = []



for name, model in models:

    kfold = StratifiedKFold(n_splits=8, random_state=1, shuffle=True)

    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring="accuracy")

    results.append(cv_results)

    names.append(name)

    print("%s %f (%f)" %(name, cv_results.mean(), cv_results.std()))
pyplot.boxplot(results, labels=names)

pyplot.title("Algo Comparison")

pyplot.show()
model = DecisionTreeClassifier()

model.fit(X_train, y_train)

predictions = model.predict(X_validation)



print(accuracy_score(y_validation, predictions))

print(confusion_matrix(y_validation, predictions))

print(classification_report(y_validation, predictions))