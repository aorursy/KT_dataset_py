# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import time



# Classifier Libraries

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import RobustScaler



# Other Libraries

from imblearn.under_sampling import RandomUnderSampler, NearMiss



from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.metrics import roc_auc_score, accuracy_score, classification_report



import warnings

warnings.filterwarnings("ignore")



fraud_df = pd.read_csv('../input/creditcard.csv')
fraud_df.head(2)
rob_scaler = RobustScaler()

scaled_amount = rob_scaler.fit_transform(fraud_df['Amount'].values.reshape(-1, 1))

scaled_time = rob_scaler.fit_transform(fraud_df['Time'].values.reshape(-1, 1))
fraud_df.insert(0, 'scaled_amount', scaled_amount)

fraud_df.insert(1, 'scaled_time', scaled_time)
fraud_df.columns
fraud_df.drop(['Time', 'Amount'], axis=1, inplace=True)
X = fraud_df.drop(['Class'], axis=1)

y = fraud_df['Class']
ori_X_train, ori_X_test, ori_y_train, ori_y_test = train_test_split(X, y, 

                                                                    test_size=0.3, 

                                                                random_state=89)
nm2 = NearMiss(version=2)

X_over_nm2, y_over_nm2 = nm2.fit_resample(ori_X_train, ori_y_train)

X_train, X_test, y_train, y_test = train_test_split(X_over_nm2, y_over_nm2, 

                                                    test_size=0.3, 

                                                    random_state=89)
clf_nb = GaussianNB()

clf_nb.fit(X_train, y_train)

clf_nb.score(ori_X_test, ori_y_test)
rus = RandomUnderSampler(sampling_strategy=1., random_state=0)

X_under_rus, y_under_rus = rus.fit_resample(ori_X_train, ori_y_train)

X_train, X_test, y_train, y_test = train_test_split(X_under_rus, y_under_rus, 

                                                    test_size=0.3, 

                                                    random_state=89)
clf_nb = GaussianNB()

clf_nb.fit(X_train, y_train)

clf_nb.score(ori_X_test, ori_y_test)
ra_score = roc_auc_score(ori_y_test, clf_nb.predict(ori_X_test))
print('Roc_auc_score = {:4.2f}'.format(ra_score))