# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/bank-full.csv', sep = ';')



df
for i in df.columns:

    print (i, list(df[i]).count('unknown'))
strcols = []



for i in df.columns:

    if df[i].dtype == 'object':

        strcols.append(i)

        

strcols

    
from sklearn import preprocessing



df1 = df.copy()



for i in df.columns:

    if df[i].dtype == 'object':

        df[i] = preprocessing.LabelEncoder().fit_transform(df[i])
from sklearn.model_selection import train_test_split as tts

from sklearn.tree import DecisionTreeClassifier as dtc



x = df.drop(columns = ['y', 'poutcome'])

y = df['y']



x_train, x_test, y_train, y_test = tts(x,y)



regressor = dtc(criterion = 'entropy')



regressor.fit(x_train, y_train)



regressor.score(x_test, y_test)
from sklearn.model_selection import GridSearchCV

clf = dtc()

grid_values = {'max_depth': [2,3,4],'min_samples_split':[2,3,4]}

grid_gbm = GridSearchCV(estimator=clf, param_grid = grid_values, cv= 3,n_jobs=-1)



grid_gbm.fit(x_train, y_train)



print('Accuracy Score : ' + str(grid_gbm.best_score_))
df2 = pd.get_dummies(df1, columns = strcols)



cols = df1.columns
print (df2.columns)

    
cols
x = df2.drop(columns = ['y_yes', 'y_no'])

y = df1['y']



x_train, x_test, y_train, y_test = tts(x,y)



regressor = dtc()



regressor.fit(x_train, y_train)



regressor.score(x_test, y_test)