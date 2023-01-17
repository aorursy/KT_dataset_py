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

import seaborn as sns

%matplotlib inline

df = pd.read_csv('../input/health-care-data-set-on-heart-attack-possibility/heart.csv')
print(df.nunique())

category_var = [i for i in df.columns if (df[i].nunique()<10)&(i!='target')]

continuous_var = [i for i in df.columns if df[i].nunique()>10]

print('category_var',category_var)

print('continuous_var',continuous_var)
print(df.head())

print('-'*10)

print(df.target.value_counts()) #data is balance

#df.isna().sum() # no NaN

sns.pairplot(df[continuous_var])
fig = plt.figure(figsize=(16,8))

for index,item in enumerate(category_var):

    ax = plt.subplot(2,4,index+1)

    sns.barplot(x=df[item].unique(),y=df[item].value_counts(),ax=ax)

plt.show()
from sklearn.preprocessing import StandardScaler

y=df.target

scaler = StandardScaler()

scaler.fit(df[continuous_var])

df[continuous_var]=scaler.transform(df[continuous_var])

df = df.drop('target',axis=1)

print(df.head())
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(df,y,test_size=.2)

print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

svc = SVC(probability =True)

svc.fit(X_train,y_train)

cv_score = cross_val_score(svc,X_test,y_test,cv=7,scoring='accuracy')

y_pred = svc.predict(X_test)

print('accuracy:{:.4}'.format(accuracy_score(y_test,y_pred)))

print(f'mean_of_accuracy:{cv_score.mean()}')
from sklearn.metrics import auc,roc_curve

y_proba = svc.predict_proba(X_test)

proba_answer = {'proba':y_proba[:,1],

        'answer':y_test}

proba_answer = pd.DataFrame(proba_answer).sort_values(by='proba',ascending=False).reset_index(drop=True)

#print(proba_answer)

y_scale = 1/len(proba_answer[proba_answer['answer']==1])

x_scale = 1/len(proba_answer[proba_answer['answer']==0])

position_x,position_y = 0,0

x_list = [0]

y_list = [0]

for i in range(len(proba_answer)):

    if (proba_answer['answer'][i]==1):       

        position_y += y_scale

    else:

        position_x +=x_scale

    x_list.append(position_x)

    y_list.append(position_y)

    

#print(x_list)

#print(y_list)

fpr,tpr,thresholds = roc_curve(proba_answer['answer'],proba_answer['proba'])

area_of_curve = auc(fpr,tpr)

#print(fpr,tpr,thresholds)

#print(area_of_curve)

plt.title('ROC_AUC_curve')

plt.ylabel('True positive rate')

plt.xlabel('False positive rate')

plt.plot(x_list,y_list,color='y',label='%.2f'%area_of_curve)

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.plot([0,1],[0,1],'r--',label=0.5)

plt.legend()

plt.show()