# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.model_selection import GridSearchCV



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

df.head()
df.tail()
df.info()
df.shape
df.hist(edgecolor='black', linewidth=1.2)

fig=plt.gcf()

fig.set_size_inches(12,6)

plt.show()
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.violinplot(x='target',y='age',data=df)

plt.subplot(2,2,2)

sns.violinplot(x='target',y='cp',data=df)

plt.subplot(2,2,3)

sns.violinplot(x='target',y='trestbps',data=df)

plt.subplot(2,2,4)

sns.violinplot(x='target',y='chol',data=df)

plt.scatter(x=df.age[df.target==1], y=df.thalach[(df.target==1)], c="green")

plt.scatter(x=df.age[df.target==0], y=df.thalach[(df.target==0)])

plt.title("Umur VS heart rate")

plt.legend(["Sakit", "Tidak Sakit"])

plt.xlabel("Umur")

plt.ylabel("Makasimal heart rate")

plt.show()
pd.crosstab(df.slope,df.target).plot(kind="bar",figsize=(15,6),color=['Red','Blue' ])

plt.title('Penyakit hati terhadap slope')

plt.xlabel('Slope ')

plt.xticks(rotation = 0)

plt.ylabel('Frekuensi')

plt.show()
x = df.drop(columns=['target'])

x.head()
y = df['target'].values

y[0:5]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y)
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train,y_train)
knn.predict(X_test)[0:5]
hasil = knn.score(X_test, y_test)

print('Accuracy: ',hasil)
daftar = []

for i in range(1,20):

    knn2 = KNeighborsClassifier(n_neighbors = i)  

    knn2.fit(X_train, y_train)

    daftar.append(knn2.score(X_test, y_test))

    

acc = max(daftar)*100

print("Maximum KNN Score is ",acc)
logistic_regression= LogisticRegression()

logistic_regression.fit(X_train,y_train)

y_pred=logistic_regression.predict(X_test)

print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))