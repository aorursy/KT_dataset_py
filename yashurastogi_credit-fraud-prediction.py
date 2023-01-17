# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, accuracy_score, auc, roc_curve, confusion_matrix,precision_recall_curve

import matplotlib.pyplot as plt

from itertools import cycle



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/creditcard.csv")

data.head()
data['Class'].value_counts()
data['Class'].value_counts()*100/data['Class'].count()
sns.distplot((data['Time']/(60*60))%24, kde=False, color="b")

data['Time_of_day']=(data['Time']/(60*60))%24


data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))

data['normTime_of_day'] = StandardScaler().fit_transform(data['Time_of_day'].values.reshape(-1, 1))

data = data.drop(['Time','Amount','Time_of_day'],axis=1)

data.head()
X = np.array(data.loc[:, data.columns != 'Class'])

y = np.array(data.loc[:, data.columns == 'Class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)



print("X_train: ", X_train.shape)

print("y_train: ", y_train.shape)

print("X_test: ", X_test.shape)

print("y_test: ", y_test.shape)
print("Fraud Count Before: ",(sum(y_train==1)))

print("Not Fraud Count Before: ",(sum(y_train==0)))



smo = SMOTE(random_state=1)

X_train_resample, y_train_resample = smo.fit_sample(X_train, y_train.ravel())



print("X_train: ", X_train_resample.shape)

print("y_train: ", y_train_resample.shape)



print("Fraud Count After: ",(sum(y_train_resample==1)))

print("Not Fraud Count After: ",(sum(y_train_resample==0)))

X_dump, X_RF, y_dump, y_RF = train_test_split(X_train_resample, y_train_resample, test_size=0.25, random_state=0)
sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))

sel.fit(X_RF, y_RF)
sel.get_support()
(data.loc[:, data.columns != 'Class']).columns.values[(sel.get_support())]
selected_feat= X_RF[:,(sel.get_support())]

len(selected_feat)
X_train_resample=X_train_resample[:,(sel.get_support())]

X_test=X_test[:,(sel.get_support())]

clf = LogisticRegression(solver="lbfgs").fit(X_train_resample, y_train_resample)

pred_y = clf.predict(X_test)

y_pred_proba = clf.predict_proba(X_test)



print ("")

print ("Classification Report: ")

print (classification_report(y_test, pred_y))



print ("")

print ("Accuracy Score: ", accuracy_score(y_test, pred_y))

print ("")

#fpr, tpr, thresholds = roc_curve(y_test, pred_y, pos_label=2)

#print ("AUC: ", auc(fpr, tpr))

pd.DataFrame(confusion_matrix(y_test, pred_y)).rename(columns={0:'Negative-Not Fraud',1:'Positive-Fraud'}, index={0:'Negative-Not Fraud',1:'Positive-Fraud'})
fpr, tpr, thresholds = roc_curve(y_test.ravel(),pred_y)

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