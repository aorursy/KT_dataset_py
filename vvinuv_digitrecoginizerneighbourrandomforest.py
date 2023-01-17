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

# visualization

import seaborn as sns

import matplotlib.pyplot as pl

%matplotlib inline



# machine learning

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.decomposition import PCA



#Learning curve

from sklearn.model_selection import learning_curve

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import validation_curve

from sklearn.metrics import accuracy_score
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
train_data_shape = train_data.shape[0]
train_data_shape, train_data.shape, test_data.shape
#train_data.head(5)
train_labels = train_data['label'].values

train_data = train_data.drop('label', axis=1).values

test_data = test_data.values
print (train_data.shape, test_data.shape)
trans = PCA(0.95) #PCA(n_components=100)

data_new = trans.fit_transform(np.row_stack([train_data, test_data]))
train_data_new = data_new[:train_data_shape]

test_data_new = data_new[train_data_shape:]
train_data.shape, train_labels.shape, train_data_new.shape, test_data_new.shape
X_train_new, X_test_new, Y_train,  Y_test = train_test_split(train_data_new, train_labels, 

                                                             test_size=0.2, 

                                                             random_state=0, shuffle=True)
pl.scatter(X_train_new[:,0], X_train_new[:,1], c=Y_train, cmap=pl.get_cmap('nipy_spectral', 10))
clf = KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='auto', p=2, 

                           metric='minkowski', metric_params=None)
clf.fit(X_train_new, Y_train)
Y_pred = clf.predict(X_test_new)
print('Accuracy score ', accuracy_score(Y_test, Y_pred))
Y_kaggle = clf.predict(test_data_new)
print(check_output(["ls", "../working"]).decode("utf8"))
np.savetxt('../working/kneigh.txt', Y_kaggle)
X_train, X_test, Y_train,  Y_test = train_test_split(train_data, train_labels, 

                                                     test_size=0.2, 

                                                      random_state=0, shuffle=True)
random_forest = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, 

                                       min_samples_split=2, min_samples_leaf=2,

                                       random_state =0, max_features='auto')
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
print('Accuracy score ', accuracy_score(Y_test, Y_pred))
Y_kaggle_random = random_forest.predict(test_data)
np.savetxt('../working/randomforest.txt', Y_kaggle_random)