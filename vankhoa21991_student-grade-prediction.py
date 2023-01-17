# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from pandas import DataFrame,Series

import sys

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data_mat = pd.read_csv('../input/student-mat.csv')

data_por = pd.read_csv('../input/student-por.csv')

data_mat.describe()
print(type(data_mat))

data_mat=DataFrame(data_mat)

data_mat.head()[:2]
sns.factorplot('age','G3',data = data_mat)

sns.factorplot('studytime','G3',data = data_mat)

sns.factorplot('goout','G3',data = data_mat)
str_list = [] # empty list to contain columns with strings (words)



for colname, colvalue in data_mat.iteritems():

    if type(colvalue[1]) == str:

         str_list.append(colname)

# Get to the numeric columns by inversion            

num_list = data_mat.columns.difference(str_list) 

print(str_list)

print(num_list)

data_matnum = data_mat[num_list]
# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(12, 10))

plt.title('Pearson Correlation of Movie Features')

# Draw the heatmap using seaborn

sns.heatmap(data_matnum.astype(float).corr(),linewidths=0.25,vmax=1.0, square=True, cmap="YlGnBu", linecolor='black', annot=True)
np.sum(data_mat.isnull())
y_train = data_matnum['G3']

x_train = data_matnum

x_train .drop(['G3'], axis=1,inplace=True)
number_of_samples = len(y_train)

print(len(y_train))

np.random.seed(0)

random_indices = np.random.permutation(number_of_samples)

num_training_samples = int(number_of_samples*0.75)

X_train = x_train.iloc[random_indices[:num_training_samples]]

Y_train= y_train.iloc[random_indices[:num_training_samples]]

X_test= x_train.iloc[random_indices[num_training_samples:]]

Y_test=y_train.iloc[random_indices[num_training_samples:]]

Y_Train=list(Y_train)
from sklearn import neighbors



n_neighbors=5

knn=neighbors.KNeighborsRegressor(n_neighbors,weights='uniform')

knn.fit(X_train,Y_train)

y1_knn=knn.predict(X_train)

y1_knn=list(y1_knn)



train_error_knn = np.mean(abs(y1_knn-Y_train))

print(train_error_knn)



y_test=knn.predict(X_test)

y_Predict=list(y_test)



test_error_knn = np.mean(abs(y_Predict-Y_test))

print(test_error_knn)
from sklearn import svm

svm_reg=svm.SVR()

svm_reg.fit(X_train,Y_train)

y1_svm=svm_reg.predict(X_train)

y1_svm=list(y1_svm)

y2_svm=svm_reg.predict(X_test)

y2_svm=list(y2_svm)



train_error_svm = np.mean(abs(y1_svm-Y_train))

train_error_svm_std = np.std(abs(y1_svm-Y_train))

print(train_error_svm)



test_error_svm = np.mean(abs(y2_svm-Y_test))

print(test_error_svm)
train_error=[train_error_knn,train_error_svm]

test_error=[test_error_knn,test_error_svm]



col={'Train Error':train_error,'Test Error':test_error}

models=['Knn','SVM']

df=DataFrame(data=col,index=models)

df
df.plot(kind='bar')