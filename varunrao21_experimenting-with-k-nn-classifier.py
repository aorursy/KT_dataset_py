# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.neighbors import KNeighborsClassifier

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Reading the data into the dataframe



dataFrame=pd.read_csv('../input/voice.csv')
# Splitting the dataset for training and testing data.

# Only 6 features have been taken into consideration here.



# Inbuilt function used to form the training and test datasets

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(dataFrame[['meanfreq','sd','centroid','meanfun','IQR','median']],dataFrame['label'],random_state=0)



print("X_train shape: {}".format(X_train.shape))

print("y_train shape: {}".format(y_train.shape))

print("X_test shape: {}".format(X_test.shape))

print("y_test shape: {}".format(y_test.shape))
#Now the k-NN classifier is used, taking k=1



from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)



# Testing this model on the training set

print("Test score on training data:{:.2f}".format((knn.score(X_train,y_train))*100))
# Now trying it on the test data:

print("Test score on test data:{:.2f}".format((knn.score(X_test,y_test))*100))
# Trying to find where the k-NN will give maximum accuracy:

from sklearn.neighbors import KNeighborsClassifier

accuracy=[0]*30

for x in range(1,len(accuracy)):

    knn=KNeighborsClassifier(n_neighbors=x)

    knn.fit(X_train,y_train)

    accuracy[x]=(knn.score(X_test,y_test))*100



import matplotlib.pyplot as plt

plt.plot(range(0,30),accuracy)

plt.ylabel('Accuracy with varying k')

plt.show()



# It is observed that after k=1, most of the values of k show similar accuracy.