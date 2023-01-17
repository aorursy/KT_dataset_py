# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
xy_data = pd.read_csv('../input/logistic-classification-diabetes-knn/train.csv')

x_test = pd.read_csv('../input/logistic-classification-diabetes-knn/test_data.csv')

submit = pd.read_csv('../input/logistic-classification-diabetes-knn/submission_form.csv')
xy_data
print(xy_data.shape)
xy_data = np.array(xy_data)

X_train = xy_data[:, 1:9]

y_train = xy_data[:, 9]

X_train = pd.DataFrame(X_train)

y_train = pd.DataFrame(y_train)

print(X_train.head())

print(X_train.shape)

print(y_train.head())

print(y_train.shape)
print(x_test.head())

print(x_test.shape)
x_test = np.array(x_test)

X_test = x_test[:, 1:9]

X_test = pd.DataFrame(X_test)

print(X_test.head())

print(X_test.shape)
from sklearn.preprocessing import LabelEncoder

classle = LabelEncoder()

y_train = classle.fit_transform(y_train.values)

print('Labels : ', np.unique(y_train))
yo = classle.inverse_transform(y_train)

print('original Labels : ', np.unique(yo))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, p=2)

knn.fit(X_train, y_train)
y_train_pred = knn.predict(X_train)

y_test_pred = knn.predict(X_test)

print('Misclassified training samples : %d' %(y_train!=y_train_pred).sum())
y_test_pred
print(submit.head())
for i in range(len(y_test_pred)) :

  submit['Label'][i] = y_test_pred[i]
submit = submit.astype(np.int32)

submit.to_csv('submit.csv', mode='w', header=True, index=False)
submit