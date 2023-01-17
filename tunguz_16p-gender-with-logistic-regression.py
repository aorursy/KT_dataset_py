# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas_profiling import ProfileReport



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score, accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/cattells-16-personality-factors/16PF/data.csv", sep="\t")
data.head()
data.shape
data[data['country'] == 'US'].shape
data[data['gender'] == 1].shape
#%%time

#profile = ProfileReport(data, title='Pandas Profiling Report', html={'style':{'full_width':True}})
#profile
np.unique(data['gender'].values)
gendered_data = data[(data['gender'] == 1) | (data['gender'] == 2)]
gendered_data.shape
gendered_data['gender'] = gendered_data['gender'].values -1
gendered_data['gender'].head(20)
gendered_data.columns
gendered_data.columns[:-6]
features = gendered_data.columns[:-6]
gendered_data[features].values.max()
gendered_data[features].values.min()
gendered_data[features] = gendered_data[features].values/5.
gendered_data[features].head()
gendered_data[features].std(axis=1)
gendered_data['std'] = gendered_data[features].std(axis=1)
gendered_data = gendered_data[gendered_data['std'] > 0.0]
gendered_data.shape
X = gendered_data[features].values

Y = gendered_data['gender'].values
np.mean(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
lr = LogisticRegression(C=20)

lr.fit(X_train, y_train)

preds_0 = lr.predict(X_test)

preds_1 = lr.predict_proba(X_test)[:,1]
accuracy_score(y_test, preds_0)
0.7967605488496854
roc_auc_score(y_test, preds_1)
0.870735955752324