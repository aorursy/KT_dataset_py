# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('/kaggle/input/leaf-classification/train.csv.zip')

test_data = pd.read_csv('/kaggle/input/leaf-classification/test.csv.zip')



#y = train_data['species']



#Dropping the id and species columns

X = train_data[train_data.columns.difference(['id', 'species'])] # this is pandas dataframe

#X_1 = train_data.drop(['id', 'species'], axis=1).values  # this is numpy ndarray



#X_test = test_data
train_data.head()
le = LabelEncoder().fit(train_data['species'])
y = le.transform(train_data['species']) # Assigning the species column to y
scaler = StandardScaler().fit(X)

X = scaler.transform(X)
logReg = LogisticRegression(solver='lbfgs', multi_class='multinomial', dual=False)
logReg.fit(X,y)
X_test_columns = test_data.pop('id')

X_test = test_data.values
X_test = scaler.transform(X_test)

y_test = logReg.predict_proba(X_test)
submission_csv = pd.DataFrame(y_test, index=X_test_columns, columns=le.classes_)
submission_csv.to_csv('submission.csv', index=False)