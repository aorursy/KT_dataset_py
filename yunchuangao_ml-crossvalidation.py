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
# Load train data already pre-processed;
titanic_train = pd.read_csv('/kaggle/input/ml-course/train_features.csv', index_col=0)
titanic_test = pd.read_csv('/kaggle/input/ml-course/test_features.csv', index_col=0)
titanic_train.head()
# Re-organize the data; keep the columns with useful features;
input_cols = ['Pclass',"Sex","Age","Cabin","Family"]
output_cols = ["Survived"]
X_train = titanic_train[input_cols].values
y_train = titanic_train[output_cols].values
X_test = titanic_test.values

model = LogisticRegression()
model.fit(X_train,y_train)
y_pred_lr=model.predict(X_test)

# Call cross validation method;
from sklearn.model_selection import cross_val_score
score = -cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')

print(score)