# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # for intractve graphs

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
CreditTxData = pd.read_csv('../input/creditcard.csv')
CreditTxData.shape
CreditTxData.describe()
CreditTxData.head()
CreditTxData.head().T
CreditTxData.info()
CreditTxData.isnull().sum()
CreditTxData.hist(bins=10,figsize=(9,7),grid=False);
len(CreditTxData[CreditTxData["Class"]==0])
CreditTxData['Class'].value_counts()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(CreditTxData.drop('Class',axis=1), CreditTxData['Class'], test_size=0.30, random_state=100)
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import classification_report,accuracy_score, confusion_matrix, auc ,roc_curve

clf = GaussianNB()

clf.fit(X_train,y_train)

GaussianNB(priors=None)

y_prediction = clf.predict(X_test)

print('accuracy %s' % accuracy_score(y_test, y_prediction))

cm = confusion_matrix(y_test, y_prediction)

print('confusion matrix\n %s' % cm)

clf_pf = GaussianNB()

clf_pf.partial_fit(X_train,y_train, np.unique(y_train))

GaussianNB(priors=None)

y_prediction = clf.predict(X_test)

print('accuracy %s' % accuracy_score(y_test, y_prediction))

cm = confusion_matrix(y_test, y_prediction)

print('confusion matrix\n %s' % cm)