# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

%matplotlib inline
import numpy as np
import pandas as pd
from sklearn import svm
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import random
from matplotlib.colors import ListedColormap
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#import data
data = pd.read_csv('../input/data.csv')
data.head()
data.drop('Unnamed: 32', axis=1,inplace = True)
data.drop('id', axis=1,inplace = True)
#plot the PCC figure
corr = data.corr(method = 'pearson')
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(10, 275, as_cmap=True)

sns.heatmap(corr, cmap=cmap, square=True,
            linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=ax)
#drop those related features
drop_features = ['perimeter_mean','radius_mean','compactness_mean','concave points_mean','radius_se',
              'perimeter_se','radius_worst','perimeter_worst','compactness_worst','concave points_worst',
              'compactness_se','concave points_se','texture_worst','area_worst']
data1 = data.drop(drop_features, axis = 1)
#replace M and B in 'diagnosis' with 1 and 0 respectively for later classification problem
data1["class"] = data['diagnosis'].map({'M':1, 'B':0})
data1 = data1.drop('diagnosis', axis=1, inplace=True)
x = data1.copy(deep = True)
x = x.drop('class', axis=1, inplace=True)
x = x
y = data1['class'] 
#calculate the scores for each features in order to find out which features are more important.
feature_ranking = SelectKBest(chi2, k=5)
fit = feature_ranking.fit(x, y)

fmt = '%-8s%-20s%s'

print(fmt % ('', 'Scores', 'Features'))
for i, (score, feature) in enumerate(zip(feature_ranking.scores_, x.columns)):
    print(fmt % (i, score, feature))
data_norm = (data1 - data1.min()) / (data1.max() - data1.min())
data_norm.head()
X_norm = data_norm
y_norm = data_norm['class']
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,6))
#PCA model, after normalization
pca = PCA(n_components=2)
X_r = pca.fit(X_norm).transform(X_norm)

#get the first class
data_0 = []
for i, label in enumerate(y_norm):
    if label == 0:
        data_0.append(X_r[i].tolist())
        
data_0_array = np.asarray(data_0)
 #get the second class
data_1 = []
for i, label in enumerate(y_norm):
    if label == 1:
        data_1.append(X_r[i].tolist())
        
data_1_array = np.asarray(data_1)
 #plot these two classes in one single plot
ax1.scatter(x=data_0_array[:,0], y=data_0_array[:,1], c='purple', label='Benign')
ax1.legend()
ax1.scatter(x=data_1_array[:,0], y=data_1_array[:,1], c='yellow', label='Malignant')
ax1.legend()
ax1.set_title('Principal Component Analysis after normalization (PCA)')
ax1.set_xlabel('1st principal component')
ax1.set_ylabel('2nd principal component')

#PCA model, before normalization

X = data1
y = data1['class']

pca = PCA(n_components=2)
X_r1 = pca.fit(X).transform(X)

data1_0 = []
for i, label in enumerate(y):
    if label == 0:
        data1_0.append(X_r1[i].tolist())
        
data1_0_array = np.asarray(data1_0)

data1_1 = []
for i, label in enumerate(y):
    if label == 1:
        data1_1.append(X_r1[i].tolist())
        
data1_1_array = np.asarray(data1_1)

ax2.scatter(x=data1_0_array[:,0], y=data1_0_array[:,1], c='purple', label='Benign')
ax2.legend()
ax2.scatter(x=data1_1_array[:,0], y=data1_1_array[:,1], c='yellow', label='Malignant')
ax2.legend()
ax2.set_title('Principal Component Analysis before normalization (PCA)')
ax2.set_xlabel('1st principal component')
ax2.set_ylabel('2nd principal component')
X_train, X_test, y_train, y_test = train_test_split(X_r, y, test_size=0.33, random_state=42)
print(X_train.shape)
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
print(accuracy_score(y_test, pred))
confusion_matrix = confusion_matrix(y_test, pred)
print(confusion_matrix)
plt.scatter(X_test[:, 0], X_test[:, 1], c=pred, label=pred)
plt.title('Classification using KNN', fontsize=12)
plt.xlabel('1st principal component')
plt.ylabel('2nd principal component')
plt.legend(labels=pred)
clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print((accuracy_score(y_test, pred)))

plt.scatter(X_test[:, 0], X_test[:, 1], c=pred, label=pred)
plt.title('Classification using SVM', fontsize=12)
plt.xlabel('1st principal component')
plt.ylabel('2nd principal component')
plt.legend(labels=pred)
X_train, X_test, y_train, y_test = train_test_split(X_r, y, test_size=0.33, random_state=42)
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred= gnb.predict(X_test)

score = accuracy_score(y_test, y_pred, normalize = True)
print(score)
X = data_norm
y = data_norm['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
print(accuracy_score(y_test, pred))
clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print((accuracy_score(y_test, pred)))
sns.countplot(data1['class'],label="Count")
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
score = accuracy_score(y_test, y_pred, normalize = True)
print(score)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
log_score = logreg.score(X_test, y_test)
print(log_score)
