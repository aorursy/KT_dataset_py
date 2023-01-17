import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
train = pd.read_csv('../input/train.csv')
train.head()
sns.heatmap(train.corr());
sns.pairplot(train);
train.min()
train.max()
test = pd.read_csv('../input/test.csv')
sns.pairplot(test);
test.head()
test.min()
test.max()
def rmsle(y_true, y_pred):
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))
X = train.drop(['id', 'target'], axis=1)
y = train.target
X['bias'] = 1
X_test = test.drop('id', axis=1)
X_test['bias'] = 1
y_pred = []
k = 10
batch = X.shape[0] // k
scores = []
for i in range(k):
    X_train, y_train = pd.concat((X[:i*batch], X[(i+1)*batch:])), pd.concat((y[:i*batch], y[(i+1)*batch:]))
    X_val, y_val = X[i*batch: (i+1)*batch], y[i*batch: (i+1)*batch]
    coef = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
    scores.append(rmsle(y_val, X_val.dot(coef)))
    y_pred.append(X_test.dot(coef))
y_pred = np.array(y_pred)
y_pred = y_pred.T.mean(axis=1)
print('RMSLE:', np.mean(scores))
pd.DataFrame({'id': test.id, 'target': y_pred}).to_csv('submission.csv', index=None)
