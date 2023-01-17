# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/iris-dataset/iris.data.csv')
from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline
from sklearn import linear_model, datasets

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
iris = datasets.load_iris()
iris.feature_names
iris.target_names
print(iris.data.shape)
iris.target
X = iris.data
Y = labels = iris.target

feature_names = iris.feature_names
Y_names = iris.target_names

n_labels = len(Y_names)

n_samples, n_features = X.shape

print("n_labels=%d \t n_samples=%d \t n_features=%d" % (n_labels, n_samples, n_features))
iris_types = set(Y)
iris_types
X_train, X_test, Y_train, Y_test = \
    train_test_split(X, Y, train_size=0.8, random_state=0)
logreg_1 = linear_model.LogisticRegression(C=1e5)
logreg_1.fit(X_train[:,:2], Y_train)
fig, ax = plt.subplots(1, 1, figsize=(10,7))
for label, color in zip(iris_types, "brg"):
    plt.scatter(X[Y==label, 0], 
                X[Y==label, 1], color=color, label=Y_names[label])

plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
ax.legend(loc="upper right")

plt.show()
