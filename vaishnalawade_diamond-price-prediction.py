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
df = pd.read_csv("/kaggle/input/diamonds/diamonds.csv", index_col=0)
df.head()
df['cut'].unique()
df['color'].unique()
df['clarity'].unique()
cut_class_dict = {"Fair":1, "Good":2, "Very Good":3, "Premium":4, "Ideal":5}
clarity_dict = {"I1":1, "SI2":2, "SI1":3, "VS2":4, "VS1":5, "VVS1":6, "VVS2":7, "IF":8}
color_dict = {"J":1,"I":2,"H":3,"G":4,"F":5,"E":6,"D":7}
df['cut'] = df['cut'].map(cut_class_dict)
df['clarity'] = df['clarity'].map(clarity_dict)
df['color'] = df['color'].map(color_dict)
import sklearn
from sklearn import svm, preprocessing
df = sklearn.utils.shuffle(df)
X = df.drop('price', axis=1).values

X = preprocessing.scale(X)
y = df['price'].values
test_size = 1000
X_train = X[:-test_size]

y_train = y[:-test_size]
X_test = X[-test_size:]

y_test = y[-test_size:]
clf = svm.SVR(kernel = 'linear')
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
for X, y in zip(X_test, y_test):

    print(f"Model Price: {clf.predict([X])[0]}, Actual Price:{y}")