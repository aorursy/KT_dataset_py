import pandas as pd
import numpy as np
dataset = pd.read_csv('../input/breast-cancer-csv/breastCancer.csv')
dataset = dataset.replace('?', np.nan)
dataset.isnull().sum()
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.nan, strategy = 'mean')
imputer.fit(X)
X = imputer.transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)
from sklearn.model_selection import cross_val_score
acc = cross_val_score(estimator = lr, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f}%".format(acc.mean()*100))
print("Standard Deviation: {:.2f}%".format(acc.std()*100))