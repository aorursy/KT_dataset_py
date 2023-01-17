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
#read dataset
df=pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
RANDOM_SEED = 20
df.head(10)

#check null values
df.isnull().sum()
#determine dependant and independant variables
X=df.drop(['Class'],axis=1)
X
y=df['Class']
y
X.shape
y.shape
#check imbalalencing
LABELS=['Normal','Fraud']
import matplotlib.pyplot as plt
count_classes = pd.value_counts(df['Class'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.title("Transaction Class Distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency")
#from above graph we see that data imbalencing is much more so covert them into balenced dataset
#to check it in values use following piece of code

fraud = df[df['Class']==1]
normal = df[df['Class']==0]
print(fraud.shape,normal.shape)

from imblearn.combine import SMOTETomek
from imblearn.under_sampling import NearMiss
# Implementing Oversampling for Handling Imbalanced 
smk = SMOTETomek(random_state=42)
X_res,y_res=smk.fit_sample(X,y)

X_res.shape,y_res.shape
from collections import Counter
print('Original dataset shape {}'.format(Counter(y)))
print('Resampled dataset shape {}'.format(Counter(y_res)))
count=pd.value_counts(y_res)
count.plot(kind='bar',rot=0)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_res,y_res,random_state=0,test_size=0.20)
from sklearn.linear_model import LogisticRegression
reg=LogisticRegression(max_iter=120000)
reg
reg.fit(X_train,y_train)
print(X_test)
print(y_test)
y_pred=reg.predict(X_test)
print(y_pred)
from sklearn.metrics import classification_report,accuracy_score
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
