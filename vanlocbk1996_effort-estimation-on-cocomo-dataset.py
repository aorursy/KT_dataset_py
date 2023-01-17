from scipy.io import arff

import numpy as np

import pandas as pd
data = arff.loadarff('../input/cocomo81/cocomo81.arff')

df = pd.DataFrame(data[0])

df.head()
X = df.iloc[:, :-1]

y = df.iloc[:, -1]
y = pd.cut(y, bins=[0, 97, 430, np.inf], labels=[0, 1, 2])
from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold

from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier



kf = KFold(n_splits=5, random_state=42, shuffle=True)



aucs = []

# clf = LogisticRegression(solver='lbfgs',max_iter=1000)

# clf = RandomForestClassifier()

clf = MultinomialNB()



for train_index, test_index in kf.split(X):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    clf.fit(X_train, y_train)

    predict = clf.predict(X_test)

    fpr, tpr, thresholds = metrics.roc_curve(y_test, predict, pos_label=2)

    aucs.append(metrics.auc(fpr, tpr))

print("Auc: ", np.mean(aucs))