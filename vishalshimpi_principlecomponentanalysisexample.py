import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

cancer.keys()
df = pd.DataFrame(data=cancer['data'], columns=cancer['feature_names'])
df.head(2)
#apply feature scalling so that all the feature values come in the equal range

from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
scalled_Feature = scalar.fit_transform(df)
#Apply PCA

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(scalled_Feature)
X_PCA = pca.transform(scalled_Feature)
plt.scatter(X_PCA[:,0], X_PCA[:,1], c=cancer['target'], cmap='rainbow')

plt.xlabel = 'First Principle Component'

plt.ylabel = 'Second Principle Component'
#Apply logistic regression on it

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()



#Formulate X and y

X = X_PCA

y = cancer['target']

#split in data in train and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
#print confussion matrix and classification report

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_pred=predictions, y_true=y_test))

print('\n')

print(classification_report(y_pred=predictions, y_true=y_test))