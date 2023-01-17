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
pixels_train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

pixels_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
Y_pixels_train = pd.DataFrame(pixels_train['label'])

X_pixels_train = pd.DataFrame(pixels_train.iloc[:,1:])
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()

dtree.fit(X_pixels_train, Y_pixels_train)
predict = dtree.predict(pixels_test)
output = pd.DataFrame(predict, columns=['Label'])

#Adjusting the indices

output.index+=1

#Renaming

output.index.name = 'ImageId'
#Reporting Solution

output.to_csv('/kaggle/working/Solution.csv')