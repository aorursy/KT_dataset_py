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
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print (train.shape)

print (test.shape)

test_ids = test.id 



levels=train.species

train.drop(['species', 'id'], axis=1,inplace=True) 

test.drop(['id'],axis=1,inplace=True)

print ("after ")

print (levels.shape)

print (train.shape)

print (test.shape)

print ("number of classes =",levels.unique().shape)
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder().fit(levels)

levels=le.transform(levels)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

Model=LinearDiscriminantAnalysis()

Model.fit(train, levels)
predictions = Model.predict_proba(test)

print (predictions.shape)

sub = pd.DataFrame(predictions, columns=list(le.classes_))

sub.insert(0, 'id', test_ids)

sub.reset_index()

sub.to_csv('submit.csv', index = False)

sub.head()  