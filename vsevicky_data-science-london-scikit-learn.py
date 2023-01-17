import numpy as np
import pandas as pd

X = pd.read_csv("../input/data-science-london-scikit-learn/train.csv")
y = pd.read_csv("../input/data-science-london-scikit-learn/trainLabels.csv")
X.head()
X.shape, y.shape
X.isnull().mean()
X.info()
X.describe()
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
df = X.corr()
fig, ax = plt.subplots(figsize=(20,20))         # Sample figsize in inches
sns.heatmap(df, linewidths=.5, ax=ax)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)
y_pred_train = classifier.predict(X_train)
from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(y_train, y_pred_train)
accuracy_score(y_train, y_pred_train)
y_pred_test = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred_test)
cm
from sklearn.svm import SVC
classifier1 = SVC(kernel='rbf')
classifier1.fit(X_train, y_train)
confusion_matrix(y_train, classifier1.predict(X_train))
confusion_matrix(y_test, classifier1.predict(X_test))
accuracy_score(y_train,classifier1.predict(X_train))
