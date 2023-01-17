import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
path = '/kaggle/input/jds101/'

df = pd.read_csv(path +'fashion-mnist_train.csv')

df_test = pd.read_csv(path + 'new_test.csv')
X = df.iloc[:,1:]

y = df.iloc[:,0]

X.shape, y.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
def save_answers(answers):

    answers = answers.reshape(10000, 1)

    b = np.arange(1, 10001).reshape(10000, 1)

    result = np.concatenate((b, answers), axis=1)

    df = pd.DataFrame(result)

    df.columns = ['id', 'label']

    df.to_csv("answers.csv", index=False)
from sklearn.linear_model import LogisticRegression
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
logistic_regression.score(X_test, y_test)
answers = logistic_regression.predict(df_test)
save_answers(answers)
from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA
pca = PCA()
pca_train = pca.fit(X)
X_pca_train = pca_train.transform(X_train)

X_pca_test = pca_train.transform(X_test)

X_pca_competition = pca_train.transform(df_test)
logistic_regression = LogisticRegression()

logistic_regression.fit(X_pca_train, y_train)
logistic_regression.score(X_pca_test, y_test)
answers = logistic_regression.predict(X_pca_competition)

save_answers(answers)
from sklearn.svm import LinearSVC
svc = LinearSVC()
svc.fit(X_train, y_train)
svc.score(X_test, y_test)
from sklearn.svm import LinearSVC

from sklearn.decomposition import PCA
pca = PCA(n_components=200)

pca_train = pca.fit(X)
X_pca_train = pca_train.transform(X_train)

X_pca_test = pca_train.transform(X_test)

X_pca_competition = pca_train.transform(df_test)
svc = LinearSVC()

svc.fit(X_pca_train, y_train)
svc.score(X_pca_test, y_test)
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
decision_tree.score(X_test, y_test)
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
nb.score(X_test, y_test)