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
import pandas as pd

import numpy as np

train = pd.read_csv("/kaggle/input/1056lab-student-performance-prediction/train.csv",index_col=0)

test = pd.read_csv("/kaggle/input/1056lab-student-performance-prediction/test.csv",index_col=0)
train = train.replace({'GP':1,'MS':2,

                       'por':1,'mat':2,

                       'F':1,'M':2,

                       'U':1,'R':2,

                       'GT3':1,'LE3':2,

                       'T':1,'A':2,

                       'other':1,'services':2,'teacher':3,'health':4,'at_home':5,

                       'course':1,'reputation':2,'home':3,'other':4,

                       'mother':1,'father':2,'other':3,

                       True:1,False:2})

test = test.replace({'GP':1,'MS':2,

                       'por':1,'mat':2,

                       'F':1,'M':2,

                       'U':1,'R':2,

                       'GT3':1,'LE3':2,

                       'T':1,'A':2,

                       'other':1,'services':2,'teacher':3,'health':4,'at_home':5,

                       'course':1,'reputation':2,'home':3,'other':4,

                       'mother':1,'father':2,'other':3,

                       True:1,False:2})

import seaborn as sns

from matplotlib import pyplot



sns.set_style("darkgrid")

pyplot.figure(figsize=(17, 17))  # 図の大きさを大き目に設定

sns.heatmap(train.corr(), square=True, annot=True) 
train = train[['Medu','Fedu','studytime','failures','G3']]

test = test[['Medu','Fedu','studytime','failures']]
train
X = train.drop('G3',axis=1)

#y = train['G3'].values

y = train[['Medu','Fedu','studytime','failures']]
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth=2)

clf.fit(X, y)
predict = clf.predict(test)
submit = pd.read_csv('/kaggle/input/1056lab-student-performance-prediction/sampleSubmission.csv',index_col=0)

submit['G3'] = predict

submit.to_csv('submission.csv')