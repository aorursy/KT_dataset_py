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
from sklearn import svm

#First model -- Support Vector Classification(SVC)

svm_svc = svm.SVC()

svm_svc.fit(train_data, train_target)

#Below are the parameters of SVC, check sklearn website for details
#Training Score (Correct rate)

svm_svc.score(train_data, train_target)
#Testing Score (Correct rate)

svm_svc.score(test_data, test_target)
#Second SVM model -- NuSVC

nu_svc = svm.NuSVC()

nu_svc.fit(train_data, train_target)

#Below are the parameters of SVC, check sklearn website for details
#Training Score (Correct rate)

nu_svc.score(train_data, train_target)
#Testing Score (Correct rate)

nu_svc.score(test_data, test_target)
#Third model -- Linear SVC

l_svc = svm.LinearSVC()

l_svc.fit(train_data, train_target)

#Below are the parameters of SVC, check sklearn website for details
#Training Score (Correct rate)

l_svc.score(train_data, train_target)
l_svc.score(test_data, test_target)