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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from matplotlib import pyplot as plt
df = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

X_test = df_test.iloc[:, 0:]
pca = PCA(X.shape[1])
pca.fit(X)
pca.explained_variance_ratio_

plt.plot([i for i in range(X.shape[1])], 
         [np.sum(pca.explained_variance_ratio_[:i+1]) for i in range(X.shape[1])])
plt.show()



pca = PCA(0.95)
pca.fit(X)

pca.n_components_

X_reduction = pca.transform(X)
def PolynomialLinearSVC():
    return Pipeline([
        ("std_scaler", StandardScaler()),
        ("LinearSVC", LinearSVC())
    ])

polynomialLinearSVC = PolynomialLinearSVC()
polynomialLinearSVC.fit(X_reduction, y)


X_predict = polynomialLinearSVC.predict(X_reduction)
polynomialLinearSVC.score(X_reduction, y)
polynomialLinearSVC.score(X_reduction, y)
X_test_reduction = pca.transform(X_test)

X_test_predict = polynomialLinearSVC.predict(X_test_reduction)
X_test_predict.shape
X_test_predict[: 20]
predictions = pd.Series(X_test_predict,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),predictions],axis = 1)
submission.to_csv("svc_submission.csv",index=False)
