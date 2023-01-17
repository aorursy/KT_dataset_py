# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# import the Iris dataset with pandas

mydataset = pd.read_csv('../input/Iris.csv')
x = mydataset.iloc[:,[1,2,3,4]].values

x
y = mydataset['Species']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)



print('There is {} samples in the Training set and {} samples in the Test set.'.format(X_train.shape[0],X_test.shape[0]))
from sklearn.svm import SVC

model = SVC(kernel ='rbf',random_state=0,gamma=.10,C=1.0)

model.fit(X_train,y_train)

y_predicted = model.predict(X_test) # predicted the test dataset

y_predicted_train = model.predict(X_train) 
y_predicted
y_test
model.score(X_train,y_train)

#print the accuracy of the svm classifier on training data is ...