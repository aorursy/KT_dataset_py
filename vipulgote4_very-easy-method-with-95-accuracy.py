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
# here we will import the libraries used for machine learning

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL

import matplotlib.pyplot as plt # this is used for the plot the graph 

import seaborn as sns # used for plot interactive graph. I like it most for plot

%matplotlib inline

from sklearn.linear_model import LogisticRegression # to apply the Logistic regression

from sklearn.model_selection import train_test_split # to split the data into two parts

from sklearn.model_selection import KFold # use for cross validation

from sklearn.model_selection import GridSearchCV# for tuning parameter

from sklearn.ensemble import RandomForestClassifier # for random forest classifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn import svm # for Support Vector Machine

from sklearn import metrics # for the check the error and accuracy of the model

# Any results you write to the current directory are saved as output.

# dont worry about the error if its not working then insteda of model_selection we can use cross_validation
import seaborn as sns

dataset=pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')

dataset
dataset.info()
dataset.drop(['Unnamed: 32','id'],axis=1,inplace=True)

dataset.columns
### As per the above output of cell. all data can be divided into three parts.lets divied the features according to their category.

'''features_mean= list(dataset.columns[1:11])

features_se= list(dataset.columns[11:20])

features_worst=list(dataset.columns[21:31])

print(features_mean)

print("----"*35)

print(features_se)

print("----"*35)

print(features_worst)

print("----"*35)'''
dataset['diagnosis']=dataset['diagnosis'].map({'M':1,'B':0})
dataset.describe()
sns.countplot(dataset['diagnosis'],label='count')
y=dataset.pop('diagnosis')

train_x,test_x,train_y,test_y = train_test_split(dataset, y,test_size = 0.2,random_state=42)
model=RandomForestClassifier(n_estimators=100)

model.fit(train_x,train_y)
prediction=model.predict(test_x)
metrics.accuracy_score(prediction,test_y)
from sklearn.metrics import accuracy_score, confusion_matrix



plt.subplots(figsize=(14,12))

sns.heatmap(confusion_matrix(test_y, prediction),annot=True,fmt="1.0f",cbar=False,annot_kws={"size": 20})

plt.title(f"Random forest model Accuracy: {accuracy_score(test_y, prediction)}",fontsize=40)

plt.xlabel("Target",fontsize=30)

plt.show()