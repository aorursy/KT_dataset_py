import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import confusion_matrix
%matplotlib inline
df=pd.read_csv('../input/mushrooms.csv')
df.head()
def Encoder(val):
    if val in category:
        return category[val]
    else:
        category[val]=len(category)
    return category[val]
df.info()
df.shape
for i in range(df.shape[1]):
    category={}
    df.iloc[:,i]=df.iloc[:,i].apply(Encoder)
df.head()
sns.countplot(x='class',data=df)
correlation=df.corr()
plt.figure(figsize=(15,15))
sns.heatmap(correlation,annot=True,cmap='coolwarm')
df['veil-type'].value_counts()
df.describe()
X=df.drop(['class','veil-type'],axis=1)
y=df['class']
X.head()
(X_train,X_test,Y_train,Y_test)=train_test_split(X,y,test_size=0.30)
svc=SVC()
param_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 1e-5],'C': [1, 10, 100]},
              {'kernel': ['linear'], 'C': [1, 10, 100]}]

grid=GridSearchCV(svc,param_grid,cv=10,scoring='accuracy')
print("Tuning hyper-parameters")
grid.fit(X_train,Y_train)
print(grid.best_params_)
print(np.round(grid.best_score_,3))
svc=SVC(C=100,gamma=0.001,kernel='rbf')
svc.fit(X_train,Y_train)
svc.score(X_test,Y_test)
Ypreds=svc.predict(X_test)
cm = confusion_matrix(Y_test,Ypreds)
xy=np.array([0,1])
plt.figure(figsize=(10,10))
sns.heatmap(cm,annot=True,square=True,cmap='coolwarm',xticklabels=xy,yticklabels=xy)
