# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from sklearn.model_selection import train_test_split





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
df = df.drop(columns = ['cp', 'thal', 'slope'])

df.head()
y = df.target.values

x_data = df.drop(['target'], axis = 1)
# Normalize

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
x_train = x_train.T

y_train = y_train.T

x_test = x_test.T

y_test = y_test.T
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 2)  # n_neighbors means k

knn.fit(x_train.T, y_train.T)

prediction = knn.predict(x_test.T)



print("{} NN Score: {:.2f}%".format(2, knn.score(x_test.T, y_test.T)*100))