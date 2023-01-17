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
train = pd.read_csv('../input/airplane-accidents-severity-dataset/train.csv')

test = pd.read_csv('../input/airplane-accidents-severity-dataset/test.csv')

sample = pd.read_csv('../input/airplane-accidents-severity-dataset/sample_submission.csv')
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()

df2=enc.fit(train)
train.isnull().sum()
df =train.drop('Accident_ID',axis=1)

df.head()
dtrain = df.drop('Severity',axis=1)

dtarget = df['Severity']
import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.utils import to_categorical
dtarget.replace('Minor_Damage_And_Injuries',0)

dtarget.replace('Significant_Damage_And_Fatalities',1)

dtarget.replace('Significant_Damage_And_Serious_Injuries',2)

dtarget.replace('Highly_Fatal_And_Damaging',3)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dtrain, dtarget, test_size=0.20)
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
test.head()

test.head()

dtest = test.drop('Accident_ID',axis=1)
sub = classifier.predict(dtest)
sample.head()
submission = pd.DataFrame({

    'Accident_ID': test['Accident_ID'],

    'Severity': sub

})



submission.to_csv("submission.csv", index=False)
submission.head()
from IPython.display import FileLink

FileLink('submission.csv')