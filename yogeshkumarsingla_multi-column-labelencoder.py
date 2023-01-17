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

sample_submission = pd.read_csv("../input/cat-in-the-dat/sample_submission.csv")

test = pd.read_csv("../input/cat-in-the-dat/test.csv")

train = pd.read_csv("../input/cat-in-the-dat/train.csv")
from sklearn.preprocessing import LabelEncoder

class MultiColumnLabelEncoder:

    def __init__(self,columns=None):

        self.columns = columns

        

    def transform(self,X):

        train=X.copy()

        if self.columns is not None:

            for col in self.columns:

                train[col]=LabelEncoder().fit_transform(train[col])

        else:

            for col in X.columns:

                train[col]=LabelEncoder().fit_transform(train[col])

        return train      
le = MultiColumnLabelEncoder()

X_train_le = le.transform(train)

X_train_le.head()