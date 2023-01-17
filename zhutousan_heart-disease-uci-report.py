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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_predict,cross_val_score

from sklearn.metrics import accuracy_score,recall_score,confusion_matrix

from sklearn.metrics import roc_curve,precision_recall_curve

from sklearn.metrics import auc

from sklearn.ensemble import ExtraTreesClassifier
df = pd.read_csv('../input/heart.csv')
df.head()
df.info()
df.isnull().sum()
sns.countplot('target',hue='target',data=df,dodge=False)

plt.legend(loc='upper left')
sns.pairplot(df,hue='target')
print('Sex Feature unique values is \n{0}'.format(df['sex'].value_counts()))

print('Fbs Feature unique values is \n{0}'.format(df['fbs'].value_counts()))

print('Restecg Feature unique values is \n{0}'.format(df['restecg'].value_counts()))

print('Exang Feature unique values is \n{0}'.format(df['exang'].value_counts()))

print('Slope Feature unique values is \n{0}'.format(df['slope'].value_counts()))

print('Ca Feature unique values is \n{0}'.format(df['ca'].value_counts()))

print('Thal Feature unique values is \n{0}'.format(df['thal'].value_counts()))
plt.figure(figsize=(10,10))

corr = df.corr()

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)]=True

sns.heatmap(corr,mask=mask,annot=True)
x_train,x_test,y_train,y_test = train_test_split(df.drop('target',1).values,df['target'].values,test_size=0.2,random_state=42)
ex_tree = ExtraTreesClassifier(max_depth=2,n_estimators=300)

ex_tree.fit(x_train,y_train)

importance =[]

for name,score in zip(df.columns,ex_tree.feature_importances_):

    importance.append([name,score])

importance.sort(key=lambda x:x[1],reverse=True)

for i in importance:

    print(i)
voting_clf = VotingClassifier([('lr',LogisticRegression()),

                               ('svm',SVC(C=1,probability=True)),

                               ('random',RandomForestClassifier(max_depth=2)),

                               ('knn',KNeighborsClassifier(n_neighbors=20))],voting='soft')
score = cross_val_score(voting_clf,x_train,y_train,cv=3,verbose=1,scoring='accuracy')
score
voting_clf.fit(x_train,y_train)
y_pre = voting_clf.predict(x_test)
accuracy_score(y_test,y_pre)
recall_score(y_test,y_pre)
sns.heatmap(confusion_matrix(y_test,y_pre),annot=True)
y_pre_pro = voting_clf.predict_proba(x_test)[:,1]
fpr,tpr,th = roc_curve(y_test,y_pre_pro)
plt.plot(fpr,tpr)

plt.plot([0,1],[0,1],'k--')

plt.xlim((0,1))

plt.ylim((0,1))

plt.grid(True)
auc(fpr,tpr)
pre,recall,th = precision_recall_curve(y_test,y_pre_pro)

plt.plot(th,pre[:-1],label='precision')

plt.plot(th,recall[:-1],label='recall')

plt.legend()