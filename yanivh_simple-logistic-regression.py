import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import linear_model

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import roc_auc_score, confusion_matrix

%matplotlib inline



import matplotlib

import matplotlib.pyplot as plt
cc = pd.read_csv('../input/creditcard.csv')
cc.Class.value_counts()
X = cc.ix[:,'V1':'Amount'].as_matrix()

y = cc.Class.as_matrix()
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=42)
logistic = linear_model.LogisticRegression(class_weight='balanced')

logistic.fit(X_train, y_train)
predictions = logistic.predict(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions, pos_label=True)

print("ROCAUC:", roc_auc_score(y_test, predictions))

print(confusion_matrix(y_test, predictions))
plt.plot(fpr, tpr, color='darkorange',

         lw=1, label='ROC curve')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="best")

plt.show()