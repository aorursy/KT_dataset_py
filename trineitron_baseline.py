# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import linear_model # basic algorithms and evaluation



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

from pathlib import Path

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_dir = Path('/kaggle/input/aicommunity-1')
train = pd.read_csv(data_dir / 'train.csv', index_col='Unnamed: 0')

test = pd.read_csv(data_dir / 'test.csv', index_col='Unnamed: 0')
train.head()
train.info()
y_train = train.TARGET_deathRate

X_train = train.drop(columns=['TARGET_deathRate'])
X_train = X_train.drop(columns=['Geography', 'binnedInc'])

X_train = X_train.dropna(axis=1)

X_test = test[X_train.columns]
X_train
reg = linear_model.RidgeCV()

reg.fit(X_train, y_train)
submission = pd.Series(reg.predict(X_test), index=X_test.index, name='TARGET_deathRate')

submission.index.name= 'Id'

submission.to_csv('linreg_baseline.csv')