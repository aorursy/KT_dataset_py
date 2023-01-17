# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics

from sklearn.svm import SVC

from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster import DBSCAN





import sklearn.utils

import itertools

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/heart.csv",na_values="?")



train, validate, test = np.split(data.sample(frac=1), [int(.7*len(data)), int(.8*len(data))])
data.head()
print("train data")

train
print("validate")

validate
print("test")

test
d =data[data.target==1]

not_d=data[data.target==0]
y=data.target.values

x_data=data.drop(["target"],axis=1)

x=(x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))
#train test split



x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.3,random_state=1)
#find k value

print("kNN")

score_list=[]

for each in range(1,150):

    knn2=KNeighborsClassifier(n_neighbors=each)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))

plt.plot(range(1,150),score_list)

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()
KNNfind = KNeighborsClassifier(n_neighbors = 40) #n_neighbors = K value

KNNfind.fit(x_train,y_train) #learning model

prediction = KNNfind.predict(x_test)

print("{}-NN Score: {}".format(40,KNNfind.score(x_test,y_test)))

KNNscore = KNNfind.score(x_test,y_test)

yprediciton2= KNNfind.predict(x_test)

ytrue = y_test





CM = confusion_matrix(ytrue,yprediciton2)



f, ax = plt.subplots(figsize=(5,5))

sns.heatmap(CM,annot = True, linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("Prediction(Ypred)")

plt.ylabel("Ytrue")

plt.show()
Accuracy = (30+43) / (30+11+7+43)

Precision = (30) / (30+7)

Recall = (30) / (30+11)

F1_score = 2*(Recall * Precision) / (Recall + Precision)

print("Accuracy : ", Accuracy)

print("Precision : ", Precision)

print("Recall  : ", Recall)

print("F1 score : ", F1_score)
print("SVM")

svc = SVC(probability = True)

svc.fit(x_train, y_train)

print("Accuracy on training set: {:.2f}".format(svc.score(x_train, y_train)))

print("Accuracy on test set: {:.2f}".format(svc.score(x_test, y_test)))



print ("SVM Accuracy:", svc.score(x_test,y_test))
print("SVM re-scale")

scaler = MinMaxScaler()

x_train_scaled = scaler.fit_transform(x_train)

x_test_scaled = scaler.fit_transform(x_test)

svc = SVC()

svc.fit(x_train_scaled, y_train)

print("****Results after scaling****")

print("Accuracy on training set: {:.2f}".format(svc.score(x_train_scaled, y_train)))

print("Accuracy on test set: {:.2f}".format(svc.score(x_test_scaled, y_test)))



print ("SVM Accuracy:", svc.score(x_test,y_test))



SVMScore = svc.score(x_test,y_test)
yprediciton2= svc.predict(x_test)

ytrue = y_test





CM = confusion_matrix(ytrue,yprediciton2)



f, ax = plt.subplots(figsize=(5,5))

sns.heatmap(CM,annot = True, linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("Prediction(Ypred)")

plt.ylabel("Ytrue")

plt.show()
Accuracy = (30+41) / (30+11+9+41)

Precision = (30) / (30+9)

Recall = (30) / (30+11)

F1_score = 2*(Recall * Precision) / (Recall + Precision)

print("Accuracy : ", Accuracy)

print("Precision : ", Precision)

print("Recall  : ", Recall)

print("F1 score : ", F1_score)
print("DT")

tree = DecisionTreeClassifier(random_state=0)

tree.fit(x_train, y_train)

print("Accuracy on training set: {:.3f}".format(tree.score(x_train, y_train)))

print("Accuracy on test set: {:.3f}".format(tree.score(x_test, y_test)))



print ("DT Accuracy:", tree.score(x_test,y_test))



print("DT with depth 4")

tree = DecisionTreeClassifier(max_depth=4, random_state=0)

tree.fit(x_train, y_train)

print("Accuracy on training set: {:.3f}".format(tree.score(x_train, y_train)))

print("Accuracy on test set: {:.3f}".format(tree.score(x_test, y_test)))



print ("DT Accuracy:", tree.score(x_test,y_test))



DTScore = tree.score(x_test,y_test)
yprediciton2= tree.predict(x_test)

ytrue = y_test





CM = confusion_matrix(ytrue,yprediciton2)



f, ax = plt.subplots(figsize=(5,5))

sns.heatmap(CM,annot = True, linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("Prediction(Ypred)")

plt.ylabel("Ytrue")

plt.show()
Accuracy = (29+39) / (29+12+11+39)

Precision = (29) / (29+11)

Recall = (29) / (29+12)

F1_score = 2*(Recall * Precision) / (Recall + Precision)

print("Accuracy : ", Accuracy)

print("Precision : ", Precision)

print("Recall  : ", Recall)

print("F1 score : ", F1_score)
print("k-Means")

sns.set_style('whitegrid')

sns.lmplot('thalach', 'chol',data=data, hue='target',

           palette='coolwarm',size=6,aspect=1,fit_reg=False)
data_target = data[data['target'] == 0]
print("DBSCAN")

db = DBSCAN(eps=11, min_samples=6).fit(data)



data['target'] = db.labels_

plt.figure(figsize=(12, 8))

sns.scatterplot(data['thalach'], data['chol'], hue=data['target'], 

                palette=sns.color_palette('hls', np.unique(db.labels_).shape[0]))

plt.title('DBSCAN with epsilon 11, min samples 6')

plt.show()


