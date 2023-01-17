# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

from sklearn import svm
# load data

df_train_X = pd.read_csv("/kaggle/input/data-science-london-scikit-learn/train.csv", header=None)

df_train_y = pd.read_csv("/kaggle/input/data-science-london-scikit-learn/trainLabels.csv", header=None)

df_test_X = pd.read_csv("/kaggle/input/data-science-london-scikit-learn/test.csv", header=None)

print(df_train_X.head())
# EDA and check whether exist the unbalanced-problems

print(df_train_X.describe())

df_train_y.apply(pd.value_counts)
import os

os.getcwd()
# Train Model with svm

clf_svc = svm.SVC()

clf_svc.fit(df_train_X, df_train_y)

clf_svc.score(df_test_X, clf_svc.predict(df_test_X))
# submission the result

output = pd.DataFrame({'Id': range(1, len(df_test_X)+1),

                      'Solution': clf_svc.predict(df_test_X)})

output.to_csv('submission.csv', index=False)