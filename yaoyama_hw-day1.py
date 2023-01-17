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

#ks_projects_201612 = pd.read_csv("../input/kickstarter-projects/ks-projects-201612.csv")

data = pd.read_csv("../input/kickstarter-projects/ks-projects-201801.csv")
data.head(5)
data.columns
data['state'].unique()
data=data[data['state'].isin(['failed', 'successful'])]

data['state'].unique()
#欠損値の確認

data.isnull().sum(axis = 0)
data.describe()
data.corr()
#目的変数と説明変数

y = data['state'] =='successful'

X = data[["usd_pledged_real", "backers"]].values



#学習

from sklearn.linear_model import SGDClassifier

clf = SGDClassifier(loss='log', penalty='none', max_iter=10000, fit_intercept=True, random_state=1234, tol=1e-3)

clf.fit(X, y)



# ラベルを予測

y_est = clf.predict(X)
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, confusion_matrix



# 対数尤度を表示

display('対数尤度 = {:.3f}'.format(- log_loss(y, y_est)))



# 正答率accuracy, 適合率precision, 再現率recallを表示

display('正答率 = {:.3f}%'.format(100 * accuracy_score(y, y_est)))

display('適合率 = {:.3f}%'.format(100 * precision_score(y, y_est)))

display('再現率 = {:.3f}%'.format(100 * recall_score(y, y_est)))



# 予測値と正解のクロス集計

conf_mat = pd.DataFrame(confusion_matrix(y, y_est), 

                        index=['actual = others', 'actual = successful'], 

                        columns=['predict = others', 'predict = successful'])

display(conf_mat)