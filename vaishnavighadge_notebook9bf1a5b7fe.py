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
pd.pandas.set_option('display.max_columns',None)
df=pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')

df.head()
df.isnull().sum()
df.dtypes
catagarical=[feature for feature in df.columns if df[feature].dtypes=='O']

df[catagarical].head()
len(catagarical)
for feature in catagarical:

    print('the feature {} has {} unique values'.format(feature,len(df[feature].unique())))
df.size
df=pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')

pd.get_dummies(df,drop_first=True).shape
df.dtypes
df.columns
df1=pd.get_dummies(df,drop_first=True)

df1.head()
df1.columns
df1.dtypes
X=df1.drop(['class_p'],axis=1)

X
y=df1['class_p']

y
from sklearn.feature_selection import SelectKBest,chi2

select=SelectKBest(score_func=chi2,k='all')

fit=select.fit(X,y)

dfscore=pd.DataFrame(fit.scores_)

dfcolumns=pd.DataFrame(X.columns)

features=pd.concat([dfscore,dfcolumns],axis=1)

features
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn import tree



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)
parameters = {'criterion':('gini', 'entropy'), 

              'min_samples_split':[2,3,4,5], 

              'max_depth':[9,10,11,12],

              'class_weight':('balanced', None),

              

             }
tr = tree.DecisionTreeClassifier()

gsearch = GridSearchCV(tr, parameters)

gsearch.fit(X_train, y_train)

model = gsearch.best_estimator_

model
model.fit(X_train,y_train)
print(X_test)
print(y_test)
y_pred=model.predict(X_test)

y_pred
score = model.score(X_test, y_test)

score
from sklearn import metrics

cm = metrics.confusion_matrix(y_test, y_pred)

print(cm)


plt.figure(figsize=(9,9))

sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');

plt.ylabel('Actual label');

plt.xlabel('Predicted label');

all_sample_title = 'Accuracy Score: {0}'.format(score)

plt.title(all_sample_title, size = 15);