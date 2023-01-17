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
df = pd.read_csv("../input/diabetes.csv")

print(df.describe())
import seaborn as sns

sns.set()



sns.pairplot(df, hue="Outcome")
y = df["Outcome"]

df = df.drop("Outcome",axis=1)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.1, random_state=0)



rfc = RandomForestClassifier(n_estimators=1000,max_depth=7,n_jobs=-1)

rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test) 



print(classification_report(y_pred,y_test))
importances = rfc.feature_importances_

std = np.std([tree.feature_importances_ for tree in rfc.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")



for f in range(df.shape[1]):

    print("%d. %s (%f)" % (f + 1, df.columns.values[indices[f]], importances[indices[f]]))