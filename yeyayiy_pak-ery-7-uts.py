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
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
%matplotlib inline
data = pd.read_csv("../input/cervical-cancer-behavior/sobar-72.csv")
data.head(5)
data.drop("behavior_sexualRisk", axis=1 ,inplace=True)
data
sex = pd.get_dummies(data["behavior_eating"], drop_first=True)
sex
data =pd.concat([data,sex], axis=1)
data
data.drop("behavior_eating",axis=1 ,inplace=True)
data
x=data.drop("ca_cervix", axis=1)
y=data["ca_cervix"]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
y_pred

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

from sklearn.metrics import accuracy_score
predic = knn.predict(x_test)
accuracy_score(y_test, predic)
#print(#classifier.predict(sc.transform([[0,30,87000]])))



