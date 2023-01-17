# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

random_seed = 5



# Any results you write to the current directory are saved as output.
reuters = pd.read_csv("../input/reuters-newswire-2017-edited.csv")
reuters = reuters[reuters.category!=7]
reuters.head()
vec = TfidfVectorizer(ngram_range=(1,3))
vec.fit(reuters.headline_text.values)
feature =  vec.transform(reuters.headline_text.values).toarray()
reuters.category.value_counts()
X_train,X_test,Y_train,Y_test = train_test_split(feature, reuters.category,random_state=random_seed,stratify=reuters.category)
Y_test.shape
len(vec.get_feature_names())
Rm = RandomForestClassifier(n_estimators=200)
Rm.fit(X_train,Y_train)
Rm.score(X_test,Y_test)
Rm.score(X_train,Y_train)
Y_pred = Rm.predict(X_test)
accuracy_score(Y_test,Y_pred)
Rm.feature_importances_
featureimportance = pd.DataFrame({"feature":vec.get_feature_names(),"importance":Rm.feature_importances_})
featureimportance.head()
pltdata = featureimportance.sort_values("importance",ascending=False)[featureimportance.importance>0.001]
sns.set(font_scale=1.2)

plt.figure(figsize = (16,6))

sns.barplot(data = pltdata,x='feature',y='importance')

plt.xticks(rotation=90);