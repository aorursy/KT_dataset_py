import numpy as np
import pandas as pd 
import os
import gc
print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')
train.shape
train.head()
train = np.array(train)
X = np.array(train[:,3:-1])
Y = np.array(train[:,-1])
X.shape, Y.shape
import gc
del train
gc.collect()
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
%matplotlib inline
X = np.concatenate((np.ones((X.shape[0], 1)), X, X**2, X**3, X**4, X**5, X**6), axis=1)
X.shape
gc.collect()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10)
del X, Y
gc.collect()
model = LinearRegression(normalize=True, n_jobs=8)
result = model.fit(X_train, Y_train)
gc.collect()
print("Train Score:",result.score(X_train, Y_train))
print("Test Score:",result.score(X_test, Y_test))
test = pd.read_csv('../input/test.csv')
test = np.array(test)
X_t = np.array(test[:,3:])
X_t.shape
del test
gc.collect()
X_t = np.concatenate((np.ones((X_t.shape[0], 1)), X_t, X_t**2, X_t**3, X_t**4, X_t**5, X_t**6), axis=1)
X_t.shape
gc.collect()
pred = result.predict(X_t)
pred.shape
submit = pd.read_csv('../input/sample_submission.csv')
submit.shape, submit.columns
submit.winPlacePerc = pd.Series(pred)
submit.to_csv('submit.csv', index=False)
