import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

%matplotlib inline

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
df=pd.read_csv('../input/Wine.csv')
df.head()
df.describe()
df.info()
correlation=df.corr()
plt.figure(figsize=(25,25))
sns.heatmap(correlation,annot=True,cmap='coolwarm')
X=df.drop('Customer_Segment',axis=1)
y=df['Customer_Segment']
sc=StandardScaler()
X=sc.fit_transform(X)
(X_train,X_test,Y_train,Y_test)=train_test_split(X,y,test_size=0.30)
pca=PCA(0.95)
pca.fit(X_train)
pca.explained_variance_ratio_
pca=PCA(3)
pca.fit(X_train)
pca.explained_variance_ratio_
pca_train=pca.transform(X_train)
pca_test=pca.transform(X_test)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(pca_train[Y_train==1,0],pca_train[Y_train==1,1],pca_train[Y_train==1,2], c='red', marker='x')
ax.scatter(pca_train[Y_train==2,0],pca_train[Y_train==2,1],pca_train[Y_train==2,2], c='blue', marker='o')
ax.scatter(pca_train[Y_train==3,0],pca_train[Y_train==3,1],pca_train[Y_train==3,2], c='green', marker='^')

ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')

plt.show()
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(pca_test[Y_test==1,0],pca_test[Y_test==1,1],pca_test[Y_test==1,2], c='red', marker='x')
ax.scatter(pca_test[Y_test==2,0],pca_test[Y_test==2,1],pca_test[Y_test==2,2], c='blue', marker='o')
ax.scatter(pca_test[Y_test==3,0],pca_test[Y_test==3,1],pca_test[Y_test==3,2], c='green', marker='^')

ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')

plt.show()
gnb = GaussianNB()
gnb.fit(pca_train,Y_train)
Ypreds=gnb.predict(pca_test)
gnb.score(pca_test,Y_test)
scores = cross_val_score(gnb, pca_train, Y_train, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
cm = confusion_matrix(Y_test,Ypreds)
xy=np.array([1,2,3])
plt.figure(figsize=(10,10))
sns.heatmap(cm,annot=True,square=True,cmap='coolwarm',xticklabels=xy,yticklabels=xy)
