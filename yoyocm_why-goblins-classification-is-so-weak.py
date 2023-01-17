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
df = pd.read_csv("../input/train.csv")

print(df.describe())
df_test = pd.read_csv("../input/test.csv")

print(df_test.describe())
import seaborn as sns

sns.set()



g = sns.FacetGrid(pd.melt(df[['bone_length','rotting_flesh','hair_length','has_soul','type']], id_vars='type'), col='type')

g.map(sns.boxplot, 'value', 'variable')
print(df['type'].value_counts())
y = df['type']

df = df.drop(["type","id"],axis=1)



df = pd.get_dummies(df)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=0)
from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report



rfc = RandomForestClassifier(n_estimators=1000,max_depth=7)

rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test) 



print(classification_report(y_pred,y_test))
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression(penalty='l2',C=1000000,class_weight="balanced")

lr.fit(X_train,y_train)

y_pred= lr.predict(X_test) 



print(classification_report(y_pred,y_test))
importances = rfc.feature_importances_

std = np.std([tree.feature_importances_ for tree in rfc.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")



for f in range(df.shape[1]):

    print("%d. feature %s (%f)" % (f + 1, df.columns[indices[f]], importances[indices[f]]))
import re 



pattern = re.compile("^color_.*")

cols_to_drop = [ x for x in df.columns if re.match(pattern,x)]



df = df.drop(cols_to_drop,axis=1)

#df_test = df_test.drop(cols_to_drop,axis=1)
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=0)



rfc = RandomForestClassifier(n_estimators=1000,max_depth=7)

rfc.fit(X_train,y_train)

y_pred= rfc.predict(X_test) 



print(classification_report(y_pred,y_test))
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression(penalty='l2',C=1000000,class_weight="balanced")

lr.fit(X_train,y_train)

y_pred= lr.predict(X_test) 



print(classification_report(y_pred,y_test))