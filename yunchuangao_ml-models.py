import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# data visualization
import seaborn as sns
%matplotlib inline
from matplotlib import pyplot as plt
from matplotlib import style

# ML algorithms;
# Algorithms
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
# Display the folders and files in current directory;
import os
for dirname, _, filenames in os.walk('/kaggle/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Load train data already pre-processed;
titanic_train = pd.read_csv('/kaggle/input/ml-course/train_features.csv', index_col=0)
titanic_test = pd.read_csv('/kaggle/input/ml-course/test_features.csv', index_col=0)
titanic_train.head()
titanic_test.head()
# Re-organize the data; keep the columns with useful features;
input_cols = ['Pclass',"Sex","Age","Cabin","Family"]
output_cols = ["Survived"]
X_train = titanic_train[input_cols].values
y_train = titanic_train[output_cols].values
X_test = titanic_test.values
# Logistic regression;

# Construct model; the paramters are set as default values;
model = LogisticRegression(penalty='l2',tol=0.0001,random_state=None,solver='lbfgs')
# Fit the model to the data;
model.fit(X_train,y_train)

# Use the model to predict the labels of test data;
y_pred_lr=model.predict(X_test)

# Check the performance of model by using training data;
model.score(X_train,y_train)
# KNN
model = KNeighborsClassifier(n_neighbors = 3) 
model.fit(X_train, y_train)  
y_pred_knn = model .predict(X_test)  
model.score(X_train,y_train)
# Gaussian naive bayesian
from sklearn.naive_bayes import GaussianNB
model= GaussianNB()
model.fit(X_train,y_train)
y_pred_gnb=model.predict(X_test) 
model.score(X_train,y_train)
# Linear SVM
model  = LinearSVC()
model.fit(X_train, y_train)

y_pred_svc = model.predict(X_test)
model.score(X_train,y_train)
# Random forest
model  = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

y_pred_rf = model.predict(X_test)
model.score(X_train,y_train)
# Decision tree
model = DecisionTreeClassifier() 
model.fit(X_train, y_train)
y_pred_dt = model.predict(X_test) 
model.score(X_train,y_train)