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
#sklearn's train_test_split is a powerful package

#which can randomly split dataset into training and testing parts.

#And it is extremely easy to apply.

from sklearn.model_selection import train_test_split



#First, let's look at the iris dataset

iris = pd.read_csv('../input/Iris.csv')

iris.head()
iris.pop('Id')  #Id column will not to be used, so remove it.

target_values = iris.pop('Species') #or you can call it 'labels'

target_values.replace(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], [0, 1, 2], inplace = True)

#Split iris dataset

train_data, test_data, train_target, test_target = train_test_split(iris, target_values, test_size=0.2)



#Let's check the content of train_data

train_data.head()
#Logistic Regression model

from sklearn import linear_model



logis = linear_model.LogisticRegression()

logis.fit(train_data, train_target)

#Below is parameters of logistic regression model.
#score of logistic regression model

#The score is so-called 'R-square value' (the higher, the better)

#You can google 'R-square' to learn more about it.

logis.score(train_data, train_target)
#Check the correct rate on training data

train_pred = logis.predict(train_data) 

train_pred_boo = train_pred == train_target

print("Training correct rate: ", np.sum(train_pred_boo)/len(train_pred_boo))
#Check the correct rate on testing data

test_pred = logis.predict(test_data)

test_pred_boo = test_pred == test_target

print("Testing correct rate: ", np.sum(test_pred_boo)/len(test_pred_boo))
#k-Nearest Neighbor (k-NN) and radius-Nearest Neighbor

from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier()

knn.fit(train_data, train_target)

#Below are the parameters of knn model, check sklearn website for more information
from sklearn.neighbors import RadiusNeighborsClassifier

rnn = RadiusNeighborsClassifier()

rnn.fit(train_data, train_target)

#Below are the parameters of rnn model, check sklearn website for more information
#R-Square value -- knn

knn.score(train_data,train_target)
#R-Square value -- rnn

rnn.score(train_data, train_target)
"T"#Training correct rate -- knn

knn_pred = knn.predict(train_data)

knn_pred_boo = knn_pred == train_target



print("Training correct rate -- knn: ",np.sum(knn_pred_boo)/len(knn_pred_boo))
#Training correct rate -- rnn

rnn_pred = rnn.predict(train_data)

rnn_pred_boo = rnn_pred == train_target



print("Training correct rate -- rnn: ",np.sum(rnn_pred_boo)/len(knn_pred_boo))
#Testing correct rate -- knn

knn_test = knn.predict(test_data)

knn_test_boo = knn_test == test_target



print("Testing correct rate -- knn: ", np.sum(knn_test_boo)/len(knn_test_boo))
#Testing correct rate -- rnn

rnn_test = rnn.predict(test_data)

rnn_test_boo = rnn_test == test_target



print("Testing correct rate -- rnn: ", np.sum(rnn_test_boo)/len(rnn_test_boo))