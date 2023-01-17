import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder,StandardScaler

from sklearn.metrics import accuracy_score,precision_recall_curve,precision_score,recall_score,confusion_matrix

from sklearn.model_selection import KFold

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import NearMiss
dataset = pd.read_csv('../input/heart.csv')
columns = dataset.columns[0:-1]



print(columns)
sns.countplot(dataset.target)

plt.show()
smote = SMOTE()

dataset_x,dataset_y = smote.fit_sample(dataset.drop('target',axis=1),dataset[['target']])
dataset_x = pd.DataFrame(dataset_x)

dataset_x.columns = columns
dataset_y = pd.DataFrame(dataset_y)

dataset_y.columns = ['target']

dataset_y.head()
sns.countplot(dataset_y.target)
dataset = dataset_x.join(dataset_y,how='right')

dataset.head()
normal = dataset[dataset.target == 0]

disease = dataset[dataset.target == 1]
for i in columns:

    sns.distplot(normal[i],color='g')

    sns.distplot(disease[i],color='r')

    plt.show()
sns.heatmap(dataset.corr())
dataset = dataset[['cp','thalach','oldpeak','target']]

dataset.head()
train_x, test_x, train_y, test_y = train_test_split(dataset.drop('target',axis=1),dataset[['target']],test_size = 0.20,random_state = 42)
scaler = StandardScaler()

train_x = scaler.fit_transform(train_x)

test_x = scaler.transform(test_x)
cl = LogisticRegression()

cl.fit(train_x,train_y)

pred = cl.predict(test_x)

cm = confusion_matrix(test_y,pred)

print("Confusion Matrix")

print(cm)

print("Accuracy Score: ",accuracy_score(test_y,pred))

print("Precision Score: ",precision_score(test_y,pred))

print("Recall Score: ",recall_score(test_y,pred))
cl = KNeighborsClassifier(n_neighbors=5)

cl.fit(train_x,train_y)

pred = cl.predict(test_x)

cm = confusion_matrix(test_y,pred)

print("Confusion Matrix")

print(cm)

print("Accuracy Score: ",accuracy_score(test_y,pred))

print("Precision Score: ",precision_score(test_y,pred))

print("Recall Score: ",recall_score(test_y,pred))
cl = RandomForestClassifier()

cl.fit(train_x,train_y)

pred = cl.predict(test_x)

cm = confusion_matrix(test_y,pred)

print("Confusion Matrix")

print(cm)

print("Accuracy Score: ",accuracy_score(test_y,pred))

print("Precision Score: ",precision_score(test_y,pred))

print("Recall Score: ",recall_score(test_y,pred))
cl = GaussianNB()

cl.fit(train_x,train_y)

pred = cl.predict(test_x)

cm = confusion_matrix(test_y,pred)

print("Confusion Matrix")

print(cm)

print("Accuracy Score: ",accuracy_score(test_y,pred))

print("Precision Score: ",precision_score(test_y,pred))

print("Recall Score: ",recall_score(test_y,pred))
cl = SVC()

cl.fit(train_x,train_y)

pred = cl.predict(test_x)

cm = confusion_matrix(test_y,pred)

print("Confusion Matrix")

print(cm)

print("Accuracy Score: ",accuracy_score(test_y,pred))

print("Precision Score: ",precision_score(test_y,pred))

print("Recall Score: ",recall_score(test_y,pred))
cl = MLPClassifier(hidden_layer_sizes = 3,activation='relu',solver='adam',warm_start = False,max_iter=500)

cl.fit(train_x,train_y)

pred = cl.predict(test_x)

cm = confusion_matrix(test_y,pred)

print("Confusion Matrix")

print(cm)

print("Accuracy Score: ",accuracy_score(test_y,pred))

print("Precision Score: ",precision_score(test_y,pred))

print("Recall Score: ",recall_score(test_y,pred))