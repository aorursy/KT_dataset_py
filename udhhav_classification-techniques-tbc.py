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

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,f1_score, r2_score



sns.set_style(style='darkgrid')
df = pd.read_csv('../input/social-network-ads/Social_Network_Ads.csv')

df.head()
df['Purchased'].value_counts()
df.drop('Gender',axis=1, inplace=True)

df.drop('User ID',axis=1, inplace=True)
df.head()
sns.countplot(df['Purchased'])
X= df.iloc[:,:-1].values

y=df.iloc[:,-1].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
lr = LogisticRegression()

lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
print(accuracy_score(y_test,y_pred))

print(classification_report(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))
print(lr.predict(sc.transform([[46,41000]])))
clf = SVC(kernel='linear', random_state=0)

clf.fit(X_train,y_train)
pred =clf.predict(X_test)
print('Accuracy:',accuracy_score(y_test,pred)*100)

print('F1 Score: ',f1_score(y_test,pred))

print('R2_Score: ',r2_score(y_test,pred))
X = df.iloc[:,:-1].values

y = df.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
classifier = SVC(kernel = 'rbf', random_state = 0)

classifier.fit(X_train, y_train)
print(classifier.predict(sc.transform([[30,87000]])))
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

print(cm)

accuracy_score(y_test, y_pred)
X = df.iloc[:,:-1].values

y= df.iloc[:,-1].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
sc = StandardScaler()

X_train= sc.fit_transform(X_train)

X_test = sc.transform(X_test)
clf = KNeighborsClassifier(n_neighbors= 5, metric= 'minkowski', p=2)

clf.fit(X_train,y_train)
print(clf.predict(sc.transform([[30,87000]])))
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

print('Confusion Matrix:\n',cm)

print('Accuracy: {:.2f}%'.format(accuracy_score(y_test, y_pred)*100))