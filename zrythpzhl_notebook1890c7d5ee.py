# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn import svm, linear_model

from sklearn.ensemble import GradientBoostingClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

types = {'row_id': np.dtype(int),

         'x': np.dtype(float),

         'y' : np.dtype(float),

         'accuracy': np.dtype(int),

         'place_id': np.dtype(int) }

df_train = pd.read_csv('../input/train.csv', dtype = types, index_col = 0)

df_train.info()

#df_test = pd.read_csv('../input/test.csv', dtype = types, index_col = 0)

print('finished load data')

X = df_train[['x','y','accuracy','time']]

y = df_train['place_id']

del df_train

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

del X, y

# Any results you write to the current directory are saved as output.
#clf = linear_model.SGDClassifier()

#clf = KNeighborsClassifier(n_neighbors = 5)

#clf = GradientBoostingClassifier()

#clf.fit(X_train, y_train) 

#print('finished fit')

#accuracy = clf.score(X_test, y_test)

#print(accuracy)