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
data = pd.read_csv('../input/creditcardfraud/creditcard.csv')
data.head()
data.info()
classes = data['Class'].value_counts()
classes[0],classes[1]
m = classes[0]/data.shape[0]
n = classes[1]/data.shape[0]


print("Percentage of valid cases: ",m*100)
print("Percentage of fraud cases: ",n*100)
import matplotlib.pyplot as plt 
classes.plot(kind = "bar")
plt.xlabel("Class")
plt.ylabel("Number of observartions")
plt.title("Counts of different classes")
plt.show()
# Comparison between fraud and non-fraud cases
plt.scatter(data.loc[data['Class'] == 0]['V11'], data.loc[data['Class'] == 0]['V12'],label='Class #0', alpha=0.5, linewidth=0.15,c='g')
plt.scatter(data.loc[data['Class'] == 1]['V11'], data.loc[data['Class'] == 1]['V12'],label='Class #1', alpha=0.5, linewidth=0.15,c='r')
plt.show()

X = data.drop(['Class'],axis=1)
Y = data['Class']
print(X.shape)
print(Y.shape)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
X_train,x_test,Y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=100)
print(X_train.shape)
print(Y_train.shape)
print(x_test.shape)
print(y_test.shape)

ss = StandardScaler()
X_ = ss.fit_transform(X_train-X_train.mean()/X_train.std())
x_ = ss.transform(x_test-x_test.mean()/x_test.std())

log_reg = LogisticRegression()
log_reg.fit(X_,Y_train)
log_reg.coef_
log_reg.intercept_
#Making predictions
predictions = log_reg.predict(x_test)
predictions[:20]
score = accuracy_score(predictions,y_test)
print("The accuracy score is: ",score*100)
