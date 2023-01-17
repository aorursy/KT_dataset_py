# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
emp=pd.read_csv("../input/HR-Employee-Attrition.csv")
emp.shape
emp.info()
emp.describe().T
emp.Attrition.value_counts()
emp.isna().sum()
emp.columns
emp.duplicated().sum()
emp.head(5)
emp['Attrition'] = emp['Attrition'].map(lambda x: 1 if x== 'Yes' else 0)
emp.head(5)
cat_col = emp.select_dtypes(exclude=np.number)    ### to select all category types

cat_col

num_col = emp.select_dtypes(include=np.number)

num_col
for i in cat_col:

    print(emp[i].value_counts())
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics
one_hot = pd.get_dummies(cat_col)

one_hot.head(5)
emp = pd.concat([num_col,one_hot],sort=False,axis=1)

emp.head()
x = emp.drop(columns='Attrition')
y = emp['Attrition']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=0)

logreg = LogisticRegression()

logreg.fit(x_train,y_train)

train_pred = logreg.predict(x_train)
metrics.confusion_matrix(y_train,train_pred)
metrics.accuracy_score(y_train,train_pred)
test_Pred = logreg.predict(x_test)
metrics.confusion_matrix(y_test,test_Pred)
metrics.accuracy_score(y_test,test_Pred)
from sklearn.metrics import classification_report

print(classification_report(y_test, test_Pred))
import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, logreg.predict(x_test))

fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(x_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from math import sqrt
x_train, x_test, y_train, y_test = train_test_split(

x, y, test_size = 0.3, random_state = 100)

y_train=np.ravel(y_train)

y_test=np.ravel(y_test)

#y_train = y_train.ravel()

#y_test = y_test.ravel()
accuracy_train_dict={}

accuracy_test_dict={}

df_len=round(sqrt(len(emp)))

for k in range(3,df_len):

    K_value = k+1

    neigh = KNeighborsClassifier(n_neighbors = K_value, weights='uniform', algorithm='auto')

    neigh.fit(x_train, y_train) 

    y_pred_train = neigh.predict(x_train)

    y_pred_test = neigh.predict(x_test)    

    train_accuracy=accuracy_score(y_train,y_pred_train)*100

    test_accuracy=accuracy_score(y_test,y_pred_test)*100

    accuracy_train_dict.update(({k:train_accuracy}))

    accuracy_test_dict.update(({k:test_accuracy}))

    print ("Accuracy for train :",train_accuracy ," and test :",test_accuracy,"% for K-Value:",K_value)
elbow_curve_train = pd.Series(accuracy_train_dict,index=accuracy_train_dict.keys())

elbow_curve_test = pd.Series(accuracy_test_dict,index=accuracy_test_dict.keys())

elbow_curve_train.head(10)
ax=elbow_curve_train.plot(title="Accuracy of train VS Value of K ")

ax.set_xlabel("K")

ax.set_ylabel("Accuracy of train")
ax=elbow_curve_test.plot(title="Accuracy of test VS Value of K ")

ax.set_xlabel("K")

ax.set_ylabel("Accuracy of test")
from sklearn.naive_bayes import GaussianNB
NB=GaussianNB()

NB.fit(x_train, y_train)
GaussianNB(priors=None,var_smoothing=1e-09)
train_pred=NB.predict(x_train)

accuracy_score(train_pred,y_train)
test_pred=NB.predict(x_test)

accuracy_score(test_pred,y_test)