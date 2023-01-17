# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#importing required packages
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import SVC
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
df1 = pd.read_csv('../input/Absenteeism_at_work.csv', delimiter=',')#dataset
features = df1.iloc[:, 1:14]#features used for training and testing
target = df1.iloc[:, 14]#values that has to be predicted
print("Missing Values: ",np.count_nonzero(features.isnull()))#checking for missing values
#converting to a numpy array
features = features.values
target = target.values
#dividing the dataset into training and testing in ratio 70%:30%
f_train, f_test, t_train, t_test = train_test_split(features, target, test_size = 0.3, random_state = 80)
#SVM
svclassifier = SVC(kernel='linear',C=1.0,gamma=0.1)  
svclassifier.fit(f_train, t_train)
y_pred = svclassifier.predict(f_test) 
print(accuracy_score(t_test, y_pred) * 100)
#Artificial Neural Network(ANN)
scaler = StandardScaler()
scaler.fit(f_train)
#transforming the data
train_data = scaler.transform(f_train)
test_data = scaler.transform(f_test)

mlp = MLPClassifier(hidden_layer_sizes=(4,4,4),activation='tanh',random_state=120, max_iter=5000)
mlp.fit(train_data,t_train)
predictions = mlp.predict(test_data)
print(accuracy_score(t_test, predictions) * 100)
#dividing the dataset into training and testing in ratio 80%:20%
f_train, f_test, t_train, t_test = train_test_split(features, target, test_size = 0.2, random_state = 100)
#Decision Trees
crt_entropy = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=5,
                                     max_features=None, max_leaf_nodes=None, min_samples_leaf=0.02,
                                     min_samples_split=0.05, min_weight_fraction_leaf=0.0,
                                     presort=False, random_state=130, splitter='random')
crt_entropy.fit(f_train, t_train)
pred_entropy = crt_entropy.predict(f_test)
print("Accuracy = ", accuracy_score(t_test, pred_entropy) * 100)
#Random Forest
rf = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, 
                            min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
                            max_features=0.39, max_leaf_nodes=None, min_impurity_decrease=0.0, 
                            min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, 
                            random_state=600, verbose=0, warm_start=False, class_weight=None)
rf.fit(f_train, t_train)
pred_rf = rf.predict(f_test)
print("Accuracy = ", accuracy_score(t_test, pred_rf) * 100)
