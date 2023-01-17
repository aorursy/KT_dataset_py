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
data=pd.read_csv("/kaggle/input/predicting-a-pulsar-star/pulsar_stars.csv")
data.head()
x_data=data.iloc[:,[0,7]].values
y=data["target_class"].values.reshape(-1,1)
x_data
#normalization
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=420)
#Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)

print("score:",dt.score(x_test,y_test))
#Random forest Classification
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=100,random_state=1)
rf.fit(x_train,y_train)

print("random forest algo result,",rf.score(x_test,y_test)) #score:accuracy


#Confusion Matrix.
y_pred=rf.predict(x_test)
y_true=y_test
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_true, y_pred)

#Visualition with Confusion Matrix
import seaborn as sns
import matplotlib.pyplot as plt

f,ax=plt.subplots(figsize=(5,5)) 
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax) 
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()

