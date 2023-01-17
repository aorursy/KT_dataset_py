# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import linear_model

from sklearn.model_selection import train_test_split

import gc

import cv2
# read the dataset

digits = pd.read_csv("../input/digit-recognizer/train.csv")

digits.info()
# head

digits.head()
four = digits.iloc[3, 1:]

four.shape
four = four.values.reshape(28, 28)

plt.imshow(four, cmap='gray')
# visualise the array

print(four[5:-5, 5:-5])
# Summarise the counts of 'label' to see how many labels of each digit are present

digits.label.astype('category').value_counts()
# Summarise count in terms of percentage 

100*(round(digits.label.astype('category').value_counts()/len(digits.index), 4))
# missing values - there are none

digits.isnull().sum()
# average values/distributions of features

description = digits.describe()

description
# Creating training and test sets

# Splitting the data into train and test

X = digits.iloc[:, 1:]

Y = digits.iloc[:, 0]



# Rescaling the features

from sklearn.preprocessing import scale

X = scale(X)



# train test split with train_size=10% and test size=90%

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.10, random_state=101)

print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)

from sklearn import svm

from sklearn import metrics



# an initial SVM model with linear kernel   

svm_linear = svm.SVC(kernel='linear')



# fit

svm_linear.fit(x_train, y_train)
# predict

predictions = svm_linear.predict(x_test)

predictions[:10]
# evaluation: accuracy

# C(i, j) represents the number of points known to be in class i 

# but predicted to be in class j

confusion = metrics.confusion_matrix(y_true = y_test, y_pred = predictions)

confusion
# measure accuracy

metrics.accuracy_score(y_true=y_test, y_pred=predictions)
# class-wise accuracy

class_wise = metrics.classification_report(y_true=y_test, y_pred=predictions)

print(class_wise)
# run gc.collect() (garbage collect) to free up memory

# else, since the dataset is large and SVM is computationally heavy,

# it'll throw a memory error while training

gc.collect()
# rbf kernel with other hyperparameters kept to default 

svm_rbf = svm.SVC(kernel='rbf')

svm_rbf.fit(x_train, y_train)
# predict

predictions = svm_rbf.predict(x_test)



# accuracy 

print(metrics.accuracy_score(y_true=y_test, y_pred=predictions))
# conduct (grid search) cross-validation to find the optimal values 

# of cost C and the choice of kernel



from sklearn.model_selection import GridSearchCV



parameters = {'C':[1, 10, 100], 

             'gamma': [1e-2, 1e-3, 1e-4]}



# instantiate a model 

svc_grid_search = svm.SVC(kernel="rbf")



# create a classifier to perform grid search

clf = GridSearchCV(svc_grid_search, param_grid=parameters, scoring='accuracy', return_train_score=True)



# fit

clf.fit(x_train, y_train)
# results

cv_results = pd.DataFrame(clf.cv_results_)

cv_results
# converting C to numeric type for plotting on x-axis

cv_results['param_C'] = cv_results['param_C'].astype('int')



# # plotting

plt.figure(figsize=(16,6))



# subplot 1/3

plt.subplot(131)

gamma_01 = cv_results[cv_results['param_gamma']==0.01]



plt.plot(gamma_01["param_C"], gamma_01["mean_test_score"])

plt.plot(gamma_01["param_C"], gamma_01["mean_train_score"])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title("Gamma=0.01")

plt.ylim([0.60, 1])

plt.legend(['test accuracy', 'train accuracy'], loc='lower right')

plt.xscale('log')



# subplot 2/3

plt.subplot(132)

gamma_001 = cv_results[cv_results['param_gamma']==0.001]



plt.plot(gamma_001["param_C"], gamma_001["mean_test_score"])

plt.plot(gamma_001["param_C"], gamma_001["mean_train_score"])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title("Gamma=0.001")

plt.ylim([0.60, 1])

plt.legend(['test accuracy', 'train accuracy'], loc='lower right')

plt.xscale('log')





# subplot 3/3

plt.subplot(133)

gamma_0001 = cv_results[cv_results['param_gamma']==0.0001]



plt.plot(gamma_0001["param_C"], gamma_0001["mean_test_score"])

plt.plot(gamma_0001["param_C"], gamma_0001["mean_train_score"])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title("Gamma=0.0001")

plt.ylim([0.60, 1])

plt.legend(['test accuracy', 'train accuracy'], loc='lower right')

plt.xscale('log')



plt.show()
# optimal hyperparameters

best_C = 1

best_gamma = 0.001



# model

svm_final = svm.SVC(kernel='rbf', C=best_C, gamma=best_gamma)



# fit

svm_final.fit(x_train, y_train)
# predict

predictions = svm_final.predict(x_test)
# evaluation: CM 

confusion = metrics.confusion_matrix(y_true = y_test, y_pred = predictions)



# measure accuracy

test_accuracy = metrics.accuracy_score(y_true=y_test, y_pred=predictions)



print(test_accuracy, "\n")

print(confusion)
