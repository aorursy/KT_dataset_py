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
import numpy as np # linear algebra

import pandas as pd

train_df = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

test_df = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')
train_labels = train_df['label']

train_df.drop(columns='label', axis=1, inplace=True)

test_id = test_df['id']

test_df.drop(columns='id', axis=1, inplace=True)

train_df
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_df, train_labels, random_state=42)
from sklearn.decomposition import PCA

pca = PCA(n_components=0.75, whiten=True)

X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)
from sklearn.svm import SVC

svc = SVC(kernel='rbf', C=10, gamma='auto')

svc.fit(X_train, y_train)

pred = svc.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(pred, y_test))
out_csv = test_id.to_frame()

out_csv['label'] = svc.predict(pca.transform(test_df))

out_csv
out_csv.to_csv('submission.csv', index=False)