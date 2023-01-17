# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from statsmodels.formula.api import ols

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/fishdata/fish.csv')
print(data)
for each in (set(data['Species'])):
    print("%s:\t%d" % (each, len(data.loc[data['Species']==each])))
sns.heatmap(data.corr(),annot=True)
data = data.drop(['Length2','Length3'],axis=1)
sns.pairplot(data,hue='Species')
sns.scatterplot(x='Length1',y='Width',data=data,hue='Species')
sns.scatterplot(x='Width',y='Weight',data=data,hue='Species')
results = ols('Weight ~ Width',data=data).fit()
print(results.summary())
sns.scatterplot(x='Width',y='Weight',data=data,hue='Species')
results = ols('Width ~ Length1',data=data).fit()
print(results.summary())
sns.scatterplot(x='Length1',y='Width',hue='Species',data=data)
results = ols('Weight ~ Length1',data=data).fit()
print(results.summary())
sns.scatterplot(x='Length1',y='Weight',hue='Species',data=data)
#Separating Variables
X = data.drop(['Species'],axis=1)
Y = data['Species']

from sklearn.model_selection import train_test_split
#Splitting Data Set into Training and Testing Sets
x_train, x_test, y_train, y_test = train_test_split(X,Y)
from statistics import mean
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
kfold = StratifiedKFold(5)
finalAcc = []
for train,test in kfold.split(X=X,y=Y):
#     print("Train: \n%s\nTest: \n%s" % (X.iloc[train] , X.iloc[test]))
#     Train DecisionTree with Training Set
    model = DecisionTreeClassifier().fit(X.iloc[train],Y.iloc[train])
#     Test_Set Accuracy Score
    acc = model.score(X.iloc[test],Y.iloc[test]) * 100
    finalAcc.append(acc)
    print('Tree Accuracy with Test-Set: %0.2f%s' % (acc,'%'))
print("\nThe mean accuracy of the model is %0.2f%s" % (mean(finalAcc),'%'))
#Train DecisionTree with Training Set
model = DecisionTreeClassifier().fit(x_train,y_train)
#Test_Set Accuracy Score
print('Accuracy with testing: %0.2f%s' % (model.score(x_test,y_test)*100,'%'))