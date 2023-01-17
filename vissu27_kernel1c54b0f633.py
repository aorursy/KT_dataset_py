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
digit_train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

digit_test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
digit_train_data
y_train = digit_train_data['label']
x_train = digit_train_data.iloc[:,1:]
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(x_train,y_train,test_size=0.30,random_state=42)
import matplotlib.pyplot as plt

%matplotlib inline
from sklearn import svm

from sklearn.svm import LinearSVC

clf = svm.LinearSVC()

clf.fit(X_train, Y_train)

y_pred = clf.predict(X_test)

error =0

for i in range(len(y_pred)):

    if (y_pred[i] !=  Y_test.iloc[i]):

        error = error +1

error_percentage = error/len(y_pred)*100

print("Error %:{}".format(error_percentage))
predicitions = clf.predict(digit_test_data)
df_predictions = pd.DataFrame({'Label':predicitions.tolist()})

df_predictions.index += 1 

df_predictions.index.names = ['ImageId']

df_predictions
#df_predictions.to_csv('/kaggle/input/digit-recognizer/predictions.csv')