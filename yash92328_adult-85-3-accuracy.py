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
data = '/kaggle/input/adult-census-income/adult.csv'

train = pd.read_csv(data)
train.head()
train.describe()
train.isnull().sum()
train.workclass.value_counts()
from sklearn import preprocessing



for x in train.columns:

    if train[x].dtype == 'object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(train[x].values))

        train[x] = lbl.transform(list(train[x].values))
train.head()
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score



y = train['income']

del train['income']



X = train

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)



#train the RF classifier

clf = RandomForestClassifier(n_estimators = 500, max_depth = 6)

clf.fit(X_train,y_train)



prediction = clf.predict(X_test)

acc =  accuracy_score(np.array(y_test),prediction)

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



print("Accuracy: %s%%" % (100*acc))

print(confusion_matrix(y_test, prediction))

print(classification_report(y_test, prediction))