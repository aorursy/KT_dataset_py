# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

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
df=pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
df.head()
from imblearn.over_sampling import SMOTE
count=pd.value_counts(df['Class'])
count.plot(kind='bar',rot=0)
X=df.drop('Class',axis=1)
y=df['Class']
X_res,y_res=SMOTE().fit_resample(X,y)
X_res.shape
count=pd.value_counts(y_res)
count.plot(kind='bar',rot=0)
plt.scatter(df['Amount'],df['Class'])
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
model=RandomForestClassifier(n_estimators=5)
X_train,X_test,y_train,y_test=train_test_split(X_res,y_res,test_size=0.20,random_state=0)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
from sklearn.metrics  import confusion_matrix,accuracy_score
confusion_matrix(y_pred,y_test)
accuracy_score(y_test,y_pred)


