import numpy as np

import pandas as pd

import sklearn

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
from imblearn.under_sampling import ClusterCentroids #undersampling

from imblearn.over_sampling import SMOTE  #oversampling

from imblearn.combine import SMOTEENN
%matplotlib inline

df = pd.read_csv("../input/creditcard.csv")
df.describe()
print(df['Class'].value_counts())
from sklearn.cross_validation import train_test_split
train, test = train_test_split(df, train_size = 0.8)
train.head()
train_label = train["Class"]

train_label

train_feature = train.drop("Class", axis = 1)
train_feature.head()
CC = ClusterCentroids()

ccx, ccy = CC.fit_sample(train_feature, train_label)
unique, counts = np.unique(ccy, return_counts=True)

print (np.asarray((unique, counts)).T)
test_label = test["Class"]

test_label

test_feature = test.drop("Class", axis = 1)
rfc = RandomForestClassifier()

rfc.fit(ccx,ccy)

y_cc_pred = rfc.predict(test_feature) 



print(classification_report(y_cc_pred,test_label))
from sklearn.metrics import confusion_matrix

confusion_matrix(test_label, y_cc_pred)
smote = SMOTE(ratio='auto', kind='regular')

smox, smoy = smote.fit_sample(train_feature, train_label)
unique_smote, counts_smote = np.unique(smoy, return_counts=True)

print (np.asarray((unique_smote, counts_smote)).T)
rfc.fit(smox,smoy)

y_smote_pred = rfc.predict(test_feature) 

print(classification_report(y_smote_pred,test_label))
confusion_matrix(test_label, y_smote_pred)
SENN = SMOTEENN(ratio = 'auto')

ennx, enny = SENN.fit_sample(train_feature, train_label)

unique_enny, counts_enny = np.unique(enny, return_counts=True)

print (np.asarray((unique_enny, counts_enny)).T)
rfc.fit(ennx, enny)

y_senn_pred = rfc.predict(test_feature) 

print(classification_report(y_senn_pred,test_label))
confusion_matrix(test_label, y_senn_pred)