# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from sklearn.ensemble import RandomForestClassifier as RF

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix as cf

# from keras.utils.np_utils import to_categorical

from sklearn.decomposition import PCA

from sklearn.preprocessing import label_binarize

from imblearn.over_sampling import SMOTE

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import average_precision_score

from sklearn.model_selection import cross_val_score as cv_score

import time

from sklearn.metrics import roc_curve as roc

from sklearn.metrics import auc as auc
uci_cc_df = pd.read_csv('../input/UCI_Credit_Card.csv')

uci_cc_df.head(2)
uci_cc_df.columns
X_dum = pd.concat([uci_cc_df[['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE']]] + \

          [uci_cc_df[['BILL_AMT%s' %i, 'PAY_AMT%s'%i]] for i in range(1,7)] + \

          [pd.get_dummies(uci_cc_df[['PAY_%s' %i]] for i in [0, 2, 3, 4, 5, 6])], axis = 1)

X_dum
pd.get_dummies(uci_cc_df['PAY_%s' %2])
uci_cc_df['default.payment.next.month'].unique()
X = uci_cc_df.drop(['ID', 'default.payment.next.month'], axis = 1)
y = uci_cc_df['default.payment.next.month']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)
sm = SMOTE(random_state = 42)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
X_train.shape, X_test.shape
time_rf = time.time()

rf = RF(n_estimators = 20)

rf.fit(X_train_res, y_train_res)

rf_pred = rf.predict(X_test)

fpr, tpr, _ = roc(y_test, rf_pred)

print('runtime is', time.time() - time_rf)
pred_proba_res_rf = rf.predict_proba(X_test)

fpr, tpr, _ = roc(y_test, pred_proba_res_rf[:,1])

plt.plot(fpr, tpr)

plt.plot(np.linspace(0, 1, fpr.shape[0]), np.linspace(0, 1, fpr.shape[0]), '--')

plt.title('AUC is %f' % auc(fpr, tpr))

plt.show()
forest = RF(n_estimators=20)



forest.fit(X_train_res, y_train_res)

importances = forest.feature_importances_

std = np.std([tree.feature_importances_ for tree in forest.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")



for f in range(X_train_res.shape[1]):

    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))



# Plot the feature importances of the forest

plt.figure()

plt.title("Feature importances")

plt.bar(range(X_train_res.shape[1]), importances[indices],

       color="r", yerr=std[indices], align="center")

plt.xticks(range(X_train_res.shape[1]), indices)

plt.xlim([-1, X_train_res.shape[1]])

plt.show()
top_10_feat = np.array(X.columns)[indices[:10]]

plt.bar(np.arange(10),np.sort(importances)[::-1][:10])

plt.title('Top 10 Features')

plt.xticks(np.arange(10), top_10_feat, rotation = 45)

plt.ylabel('Importance')

plt.show()