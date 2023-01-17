# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")

df.head()
df.shape


df.info()
plt.figure(figsize=(10,12))

sns.heatmap(df.corr())
z=df['Class'].value_counts(sort=True).sort_index()

z.plot(kind='bar')
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

z=scaler.fit_transform(df['Amount'].values.reshape(1,-1))

df['normAmount']=z.reshape(-1,1)
Y=df['Class']

df=df.drop(['Amount'],axis=1)

X_TEST=df.drop(['Class'],axis=1)

Y_TEST=df["Class"]
df.head()
df['Class'].value_counts()
Y=df['Class']

fraud_indices=np.array(Y[Y==1].index)

normal_indices=np.array(Y[Y==0].index)

number_fraud=Y[Y==1].count()
random_normal_indices=np.random.choice(normal_indices,number_fraud,replace=True)

print((random_normal_indices).reshape(1,-1))

under_sample_indices=np.concatenate([random_normal_indices,fraud_indices])

import random

random.shuffle(under_sample_indices)
print(under_sample_indices)
df.head()
X_under_sample=df.iloc[under_sample_indices]

X_under_sample=X_under_sample.drop(["Class"],axis=1)

Y_under_sample=df['Class'].iloc[under_sample_indices]

print(Y_under_sample[:15])

#print(X_under_sample.shape,Y_under_sample.shape)
Y_under_sample.value_counts()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X_under_sample,Y_under_sample,test_size=0.33)

print(len(x_test)/(len(x_test)+len(x_train)))
from sklearn.model_selection import cross_val_score

def kfold():

    c_param_range = [0.001,0.01,0.1,1,10,100,1000]

    values=[]    

    for var in c_param_range:

        model=LogisticRegression(C=var,penalty="l1")

        scores=cross_val_score(model,

                               X_under_sample,Y_under_sample,cv=5,scoring='recall')

        print("C=",var)

        print("scores",scores)

        values.append(scores.mean())

        print("Mean is ",scores.mean())

    return c_param_range[values.index(max(values))]
best_c=kfold()

print("best_c",best_c)
model=LogisticRegression(C=best_c,penalty='l1')

model.fit(x_train,y_train)

y_pred_undersample = model.predict(x_test)

from sklearn.metrics import confusion_matrix

# Compute confusion matrix

cnf_matrix = confusion_matrix(y_test,y_pred_undersample)

np.set_printoptions(precision=2)



print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

model=LogisticRegression(C=best_c,penalty='l1')

model.fit(x_train,y_train)



y_pred_undersample = model.predict(X_TEST)

from sklearn.metrics import confusion_matrix

# Compute confusion matrix

cnf_matrix = confusion_matrix(Y_TEST,y_pred_undersample)

np.set_printoptions(precision=2)



print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report 

lr = LogisticRegression(C = best_c, penalty = 'l1')

y_pred_undersample_score = lr.fit(x_train,y_train).decision_function(x_test)



fpr, tpr, thresholds = roc_curve(y_test,y_pred_undersample_score)

roc_auc = auc(fpr,tpr)



# Plot ROC

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.0])

plt.ylim([-0.1,1.01])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()