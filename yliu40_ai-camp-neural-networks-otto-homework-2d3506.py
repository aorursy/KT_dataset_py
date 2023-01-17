import numpy as np

import pandas as pd

from patsy import dmatrices

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn import metrics

import matplotlib.pyplot as plt
data = pd.read_csv("../input/train.csv")
data.head()
columns = data.columns[1:-1]
X = data[columns]
X.head()
y = np.ravel(data['target'])
data.groupby('target').size().plot.bar()
for id in range(9):

    plt.subplot(3, 3, id + 1) # 2行4列

    # plt.axis('off') # 不显示坐标轴

    data[data.target == 'Class_' + str(id + 1)].feat_20.hist()

plt.show() 
plt.scatter(data['feat_19'], data['feat_20'])
import seaborn as sns

sns.set()

sns.heatmap(data.corr())
num_fea = X.shape[1]
model = MLPClassifier(hidden_layer_sizes=(30, 10))
model.fit(X,y)
model.intercepts_
print(model.coefs_[0].shape)

print(model.coefs_[1].shape)

print(model.coefs_[2].shape)
pred = model.predict(X)

pred
cross_val_score(model, X, y)
sum(pred == y) / len(y)
Xtest = pd.read_csv("../input/test.csv")