# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sn

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
data = pd.read_csv('../input/heart-attack-prediction/data.csv')

data.head()
#data.drop(['slope', 'ca','thal'], axis=1, inplace=True)
data.head()
data.mean()
data.median()
data.head()
data.replace('?', np.nan, inplace=True)
median=data['chol'].median()

data['chol'].fillna(median,inplace=True)

median=data['slope'].median()

data['slope'].fillna(median,inplace=True)

median=data['ca'].median()

data['ca'].fillna(median,inplace=True)

median=data['thal'].median()

data['thal'].fillna(median,inplace=True)

median=data['fbs'].median()

data['fbs'].fillna(median,inplace=True)

median=data['restecg'].median()

data['restecg'].fillna(median,inplace=True)
cols=data.columns

for col in cols:

    median=data[col].median()

    data[col].fillna(median,inplace=True)

    
data.head()
data.corr()
sn.heatmap(data.corr())
sn.jointplot(x="oldpeak",y="cp",data=data,kind="reg")
data.hist(figsize=(10,10))
corr = data.corr()

f,ax=plt.subplots(figsize=(20,1))

sn.heatmap(corr.sort_values(by=['cp'],ascending=False).head(1), cmap='Blues')

plt.title("features correlation with the Research", weight='bold', fontsize=18)

plt.xticks(weight='bold')

plt.yticks(weight='bold', color='dodgerblue', rotation=0)

plt.show()
corr = data.corr()

f,ax=plt.subplots(figsize=(20,1))

sn.heatmap(corr.sort_values(by=['oldpeak'],ascending=False).head(1), cmap='Blues')

plt.title("features correlation with the Research", weight='bold', fontsize=18)

plt.xticks(weight='bold')

plt.yticks(weight='bold', color='dodgerblue', rotation=0)

plt.show()
real_x=data.iloc[:,:-1].values

real_y=data.iloc[:,-1].values
train_x,test_x,train_y,test_y=train_test_split(real_x,real_y,test_size=0.25,random_state=0)
sc=StandardScaler()

train_x=sc.fit_transform(train_x)

test_x=sc.fit_transform(test_x)
cls=KNeighborsClassifier(n_neighbors=5)

cls.fit(train_x,train_y)

pred_y=cls.predict(test_x)
pred_y
test_y
from sklearn.metrics import confusion_matrix

cs=confusion_matrix(test_y,pred_y)

cs
print(cls.score(train_x,train_y)*100)

print(cls.score(test_x,test_y)*100)
from sklearn.metrics import accuracy_score
ac=accuracy_score(test_y,pred_y)

ac*100
plt.figure(figsize=(10,10))

plt.subplot(2,1,1)

plt.plot(test_y,color="green")

plt.grid()

plt.subplot(2,1,2)

plt.plot(pred_y,color="blue")



plt.grid()
error = []



# Calculating error for K values between 1 and 40

for i in range(1, 40):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(train_x, train_y)

    pred_i = knn.predict(test_x)

    error.append(np.mean(pred_i != test_y))
plt.figure(figsize=(12, 6))

plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',

         markerfacecolor='blue', markersize=10)

plt.title('Error Rate K Value')

plt.xlabel('K Value')

plt.ylabel('Mean Error')

plt.grid()
x_min, x_max = real_x[:, 0].min() - 1, real_x[:, 0].max() + 1

y_min, y_max = real_x[:, 1].min() - 1, real_x[:, 1].max() + 1

h=0.2

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Put the result into a color plot

plt.figure()

plt.scatter(real_x[:, 0], real_x[:, 1])

plt.xlim(xx.min(), xx.max())

plt.ylim(yy.min(), yy.max())

plt.title("Data points")

plt.show()
no_neighbors = np.arange(1, 9)

train_accuracy = np.empty(len(no_neighbors))

test_accuracy = np.empty(len(no_neighbors))



for i, k in enumerate(no_neighbors):

    # We instantiate the classifier

    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data

    knn.fit(train_x,train_y)

    

    # Compute accuracy on the training set

    train_accuracy[i] = knn.score(train_x, train_y)



    # Compute accuracy on the testing set

    test_accuracy[i] = knn.score(test_x, test_y)



# Visualization of k values vs accuracy



plt.title('k-NN: Varying Number of Neighbors')

plt.plot(no_neighbors, test_accuracy, label = 'Testing Accuracy')

plt.plot(no_neighbors, train_accuracy, label = 'Training Accuracy')

plt.legend()

plt.xlabel('Number of Neighbors')
x=[[40,1,3,120,150,0,0,150,1,1.5,2,0,7]] 
pred_y1=cls.predict(x)

pred_y1