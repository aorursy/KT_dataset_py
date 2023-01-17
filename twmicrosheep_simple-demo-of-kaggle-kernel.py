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
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, roc_auc_score



from sklearn.tree import DecisionTreeClassifier
X_train = pd.read_csv("../input/data_train.csv", index_col="index")

Y_train = pd.read_csv("../input/answer_train.csv", index_col="index")



X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2)
model = DecisionTreeClassifier()

model.fit(X_train, Y_train)
train_pred = model.predict(X_train)

print(classification_report(Y_train, train_pred))

print(roc_auc_score(Y_train, train_pred))
test_pred = model.predict(X_test)

print(classification_report(Y_test, test_pred))

print(roc_auc_score(Y_test, test_pred))
X_submit = pd.read_csv("../input/data_test.csv", index_col="index")

pred = model.predict(X_submit)

pred_df = pd.DataFrame(data={'default.payment.next.month': pred}).reset_index()

pred_df.to_csv("./pred.csv", index=False)