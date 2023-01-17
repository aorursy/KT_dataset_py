import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

df_test = pd.read_csv("../input/leaf-classification/test.csv")

df_train = pd.read_csv("../input/leaf-classification/train.csv")

df_train.columns.values
df_train.head()
df_train.info()
df_train.describe()
df_train.shape
df_train.isnull().sum()
df_train.drop(['id'], axis=1, inplace=True)
df_train.head()
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

X=df_train.iloc[:,1:].values

y=df_train.iloc[:, 0].values

le=LabelEncoder()

y=le.fit_transform(y)

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=1)
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

pipe_lr=Pipeline([('scl', StandardScaler()), ('pca', PCA(n_components=10)), ('clf', LogisticRegression(random_state=1))])

pipe_lr.fit(X_train, y_train)

print(pipe_lr.score(X_test, y_test))
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores=learning_curve(estimator=pipe_lr, X=X_train, y=y_train, train_sizes=np.linspace(0.1,1.0,10), cv=10, n_jobs=-1)

train_mean=np.mean(train_scores, axis=1)

train_std=np.std(train_scores, axis=1)

test_mean=np.mean(test_scores, axis=1)

test_std=np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, color='blue', marker='o', label='training accuracy')

# plt.fill_between(train_sizes, train_mean+train_std, train_mean-train_std, color='blue')

plt.plot(train_sizes, test_mean, color='red', marker='s', label='test accuracy')

plt.grid()

# plt.fill_between(train_sizes, test_mean+test_std, test_mean-test_std, color='red')

plt.xlabel('Number of training samples')

plt.ylabel('Accuracy')

plt.ylim([0, 1])

plt.legend(loc='lower right')

plt.show()