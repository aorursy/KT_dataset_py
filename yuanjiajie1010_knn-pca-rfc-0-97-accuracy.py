import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier as RFC

from sklearn.neighbors import KNeighborsClassifier as KNN

from sklearn.model_selection import cross_val_score
train=pd.read_csv("../input/digit-recognizer/train.csv")

test=pd.read_csv("../input/digit-recognizer/test.csv")
print(train.shape)

print(test.shape)
train.head()
#check labels

train.label.unique()
y_train=train["label"]

train.drop("label",axis=1,inplace=True)

sns.countplot(y_train)
#check missing values

train.isnull().values.any()
#plot first 20 digits

fig,axes=plt.subplots(4,5,figsize=(6,4),subplot_kw={"xticks":[],"yticks":[]})

for i,ax in enumerate(axes.flat):

    ax.imshow(train.values[i,:].reshape(28,28))

plt.show()
%%time

X=train.values

y=y_train.values

pca_line=PCA().fit(X)

plt.figure(figsize=[20,5])

plt.plot(np.cumsum(pca_line.explained_variance_ratio_))

plt.xlabel("number of components after dimension reduction")

plt.ylabel("cumulative explained variance ratio")

plt.show()
score=[]

for i in range(1,101,10):

    X_dr=PCA(i).fit_transform(X)

    once=cross_val_score(RFC(n_estimators=20,random_state=0),X_dr,y,cv=5).mean()

    score.append(once)

plt.figure(figsize=(10,5))

plt.plot(range(1,101,10),score)

plt.show()
score=[]

for i in range(20,30):

    X_dr=PCA(i).fit_transform(X)

    once=cross_val_score(RFC(n_estimators=20,random_state=0),X_dr,y,cv=5).mean()

    score.append(once)

plt.figure(figsize=(10,5))

plt.plot(range(20,30),score)

plt.show()
#PCA+KNN

score=[]

for i in range(10):

    X_dr=PCA(30).fit_transform(X)

    once=cross_val_score(KNN(i+1),X_dr,y,cv=5).mean()

    score.append(once)

plt.figure(figsize=(10,5))

plt.plot(range(10),score)

plt.show()
pca=PCA(n_components=30)

pca.fit(X)

train_dr=pca.transform(X)

test_dr=pca.transform(test)

clf=KNN(3)

clf.fit(train_dr,y)

results=clf.predict(test_dr)

results=pd.Series(results,name="Label")
#accuracy 0.97

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("knn_digit_recognizer.csv",index=False)