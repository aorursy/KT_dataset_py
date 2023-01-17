from IPython.display import Image

Image("../input/species-predictor/main_app.png")
import pandas as pd

import numpy as np

from sklearn import model_selection

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("../input/iris-dataset/iris.csv")
df.head(2)
df.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)

plt.show()
sns.pairplot(df); #relationship b/w attr
array = df.values

X = array[:,0:4]    #1st 4 cols are training attributes

Y = array[:,4]      #5th col is the class (species name in our case)
validation_size = 0.20

seed = 41

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
from sklearn.linear_model import LogisticRegression

logit = LogisticRegression()
logit.fit(X_train,Y_train)
logit.predict(X_test)
mysample = np.array([4.5,3.2,1.2,0.5])

ex1 = mysample.reshape(1,-1)

logit.predict(ex1)

ex2 = np.array([6.2,3.4,5.4,2.3]).reshape(1,-1)

logit.predict(ex2)
from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC
knn = KNeighborsClassifier()

dtree = DecisionTreeClassifier()

svm = SVC()
knn.fit(X_train, Y_train)

print("accuracy :" , knn.score(X_test,Y_test))
dtree.fit(X_train, Y_train)

print("accuracy :" , dtree.score(X_test,Y_test))
svm.fit(X_train, Y_train)

print("accuracy :" , svm.score(X_test,Y_test))