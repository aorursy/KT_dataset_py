# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import random



random.seed(50)



# Importing the dataset

dataset = pd.read_csv('../input/PS_20174392719_1491204439457_log.csv')

dataset.drop('nameOrig', axis=1, inplace=True)

dataset.drop('nameDest', axis=1, inplace=True)

dataset.drop('isFlaggedFraud', axis=1, inplace=True)
sample_dataframe = dataset.sample(n=100000)

X = sample_dataframe.iloc[:, :-1].values

y = sample_dataframe.iloc[:, 7].values



print(sample_dataframe.isFraud.value_counts())
# Encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder = LabelEncoder()

X[:, 1] = labelencoder.fit_transform(X[:, 1])

onehotencoder = OneHotEncoder(categorical_features = [1])

X = onehotencoder.fit_transform(X).toarray()



# Avoiding the Dummy Variable Trap

X = X[:, 1:]
# Splitting the dataset into the Training set and Test set



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1)



counts = np.unique(y_train, return_counts=True)

# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_val = sc.transform(X_val)

X_test = sc.transform(X_test)

print(counts)
# Fitting Kernel SVM to the Training set

from sklearn.svm import LinearSVC

from imblearn.under_sampling import NearMiss

from imblearn import over_sampling as os

from imblearn.pipeline import make_pipeline

from imblearn.combine import SMOTETomek

from imblearn.over_sampling import ADASYN

from imblearn.under_sampling import ClusterCentroids

from imblearn.over_sampling import RandomOverSampler
# Apply the sampling

ada = ADASYN()

X_resampled, y_resampled = ada.fit_sample(X_train, y_train)

count = np.unique(y_resampled, return_counts=True)



# Create a pipeline



pipeline4 = make_pipeline(ADASYN(),LinearSVC(random_state=1))

pipeline4.fit(X_train, y_train)

print(count)
# Classify and report the results

from imblearn.metrics import classification_report_imbalanced

print(classification_report_imbalanced(y_test, pipeline4.predict(X_test)))
# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix, precision_score,auc,roc_auc_score,roc_curve,recall_score

cm = confusion_matrix(y_val, pipeline4.predict(X_val))

roc = roc_auc_score(y_val, pipeline4.predict(X_val))

fpr, tpr, thresholds = roc_curve(y_val, pipeline4.predict(X_val))

roc_auc = auc(fpr,tpr)
# Plot ROC

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.0])

plt.ylim([-0.1,1.01])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()