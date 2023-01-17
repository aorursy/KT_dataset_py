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
df = pd.read_csv('../input/train.csv', header=0)

df.head()
df_summ = df.groupby(['Sex', pd.cut(df_clean["Age"], np.arange(0, 100, 10))])['Survived']

df_summ.agg(['count', 'mean'])
df_summ = df.groupby(['Sex', pd.cut(df_clean["Fare"], np.arange(0, 200, 50))])['Survived']

df_summ.agg(['count', 'mean'])
from sklearn import preprocessing, tree

from sklearn.datasets import load_iris

from sklearn.cross_validation import train_test_split



le = preprocessing.LabelEncoder()



df_clean = df.drop(['Ticket', 'Cabin', 'Embarked'], axis=1).copy()

df_clean['Sex'] = le.fit_transform(df_clean['Sex'])



lastnames = le.fit_transform([name[:name.index(', ')] for name in df_clean['Name']])

df_clean['LastName'] = lastnames



df_clean = df_clean.drop('Name', axis=1).set_index('PassengerId')

x = df_clean.drop('Survived', axis=1)

y = df_clean['Survived']





#df.iloc[np.where(pd.isnull(df_clean))[:1]]

#df#[np.isnan(df)]



#xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.50)

    

#clf = tree.DecisionTreeClassifier()

#clf = clf.fit(xtrain, ytrain)