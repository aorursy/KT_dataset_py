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

import sklearn

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.datasets import load_breast_cancer

from sklearn.datasets import load_digits

from sklearn import metrics

%matplotlib inline

import os

import warnings

warnings.filterwarnings('ignore')
cancer=load_breast_cancer()

digits=load_digits()
data=cancer
data
df=pd.DataFrame(data=np.c_[data['data'],data['target']],  columns=list(data['feature_names'])+['target'])

df['target']=df['target'].astype('uint16')
df
x=df.drop('target',axis=1)

y=df[['target']]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=99)
print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
print(y_train.mean())

print(y_test.mean())
shallow_tree=DecisionTreeClassifier(max_depth=2,random_state=99)
shallow_tree.fit(x_train,y_train)

y_pred=shallow_tree.predict(x_test)

score=metrics.accuracy_score(y_test,y_pred)

score
estimators=list(range(1,275,15))

abc_scores=[]

for n in estimators:

    ABC=AdaBoostClassifier(base_estimator=shallow_tree, n_estimators=n)

    

    ABC.fit(x_train,y_train)

    y_pred=ABC.predict(x_test)

    score=metrics.accuracy_score(y_test,y_pred)

    abc_scores.append(score)
abc_scores
max(abc_scores)
plt.plot(estimators, abc_scores)

plt.xlabel('n_estimators')

plt.ylabel('accuracy')

plt.show()