# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import tree



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output



#print(check_output(["ls", "../input"]).decode("utf8"))

#names = ['PassengerID', 'Survived']

df_train = pd.read_csv("../input/train.csv", usecols = ['Pclass', 'Sex', 'Survived'])



df_train['Sex'] = df_train['Sex'].replace(['male'], 0)

df_train['Sex'] = df_train['Sex'].replace(['female'], 1)

#df_train.fillna(-1)

array_train = df_train.as_matrix()

clf = tree.DecisionTreeClassifier()

#print (array_train[:,:1])

#print (array_train[:,(1,2,3)])

clf = clf.fit(array_train[:,(1,2)], array_train[:,:1])



# Any results you write to the current directory are saved as output.

df_test = pd.read_csv("../input/test.csv", usecols = ['Pclass', 'Sex'])

df_test.fillna(-1)

df_test['Sex'] = df_test['Sex'].replace(['male'], 0)

df_test['Sex'] = df_test['Sex'].replace(['female'], 1)



array_test = df_test.as_matrix()

print(clf.predict(array_test[:,(0,1)]) )