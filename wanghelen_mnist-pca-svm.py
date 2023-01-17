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
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
train_data.head()
test_data.head()
import numpy as np

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn.externals import joblib

from sklearn.model_selection import train_test_split
X = train_data.drop(["label"], axis=1)

Y = train_data["label"]

Y.value_counts()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=None)
print(X_train.shape, X_test.shape)
print(X_train[0:1].values.reshape(28,28))
from matplotlib import pyplot as plt

plt.imshow(X_train[0:1].values.reshape(28,28))
from sklearn.decomposition import PCA

pca = PCA(n_components = 0.9, whiten = True)

X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)
print(X_train.shape, X_test.shape)
print(pca.explained_variance_ratio_)

print(pca.explained_variance_)

print(pca.n_components_)
PKL = "svm.pkl"

def search_best_param():

    params = {"C": [1, 5, 10, 100, 1000]}

    svr = SVC(kernel="rbf")

    clf = GridSearchCV(svr, cv=5, param_grid=params, n_jobs=-1, verbose=3)

    clf.fit(X_train, y_train)

    best_model = clf.best_estimator_

    joblib.dump(best_model, PKL)  # save the best model

    print('train_data score:{:.4f}'.format(best_model.score(X_train, y_train)))

    print('test_data score:{:.4f}'.format(best_model.score(X_test, y_test)))

    print(best_model)

search_best_param()
# 训练模型

def train():

#     clf = SVC(kernel='poly', C=1, probability=True)

    clf = SVC(kernel = 'rbf',C = 10)

    clf.fit(X_train, y_train)

    joblib.dump(clf, PKL)

    print('train_data score:{:.4f}'.format(best_model.score(X_train, y_train)))

    print('test_data score:{:.4f}'.format(best_model.score(X_test, y_test)))
# train()
X_test_data = pca.transform(test_data)

clf = joblib.load(PKL)

Y_test_data = clf.predict(X_test_data)
Y_test_data[0:5]
plt.imshow(test_data[0:1].values.reshape(28,28))
Y_test_data.shape
submit_data = pd.DataFrame()

submit_data["ImageId"] = pd.Series(range(1, Y_test_data.shape[0]+1))

submit_data["Label"] = pd.Series(Y_test_data)
submit_data.head()
submit_data.to_csv("submit_data_svm.csv", index=False)