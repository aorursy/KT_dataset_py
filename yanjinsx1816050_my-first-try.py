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
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
%matplotlib inline
train = pd.read_csv('../input/train.csv')
target='Cover_Type' # Disbursed的值就是二元分类的输出
IDcol = 'Id'
x_columns = [x for x in train.columns if x not in [target, IDcol]]
X = train[x_columns]
y = train['Cover_Type']

rf1 = RandomForestClassifier(n_estimators= 263, max_depth=21, min_samples_split=41,
                                max_features=25
                             ,oob_score=True, random_state=10)
rf1.fit(X,y)
X_test = pd.read_csv('../input/test.csv')
x1_columns = [x1 for x1 in X_test.columns if x1 not in [IDcol]]
X1 = X_test[x1_columns]
print ('start predicting:')
pre_Y = rf1.predict(X1)
print ('predict over')
a = [x for x in range(15121,581013)]
predDf=pd.DataFrame(
    {'Id':a,
    'Cover_Type':pre_Y})
predDf.to_csv('mypred.csv',index=False)
print ('start over:')
