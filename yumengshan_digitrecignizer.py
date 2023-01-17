import pandas as pd
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA,KernelPCA
from sklearn.model_selection import cross_val_score,GridSearchCV,KFold
from sklearn.svm import SVC,LinearSVR
train_df=pd.read_csv('../input/train.csv')
test_df=pd.read_csv('../input/test.csv')
train_df.info()
train_df.isnull().sum().sort_values(ascending=False)[:5]
x_train=train_df.values[:,1:]
y_train=train_df.values[:,0]
import matplotlib.pyplot as plt
%matplotlib inline
p1=x_train[3333].reshape(28,28)
plt.imshow(p1,cmap=plt.cm.gray_r)
plt.hist(y_train)
x_train_c=x_train.copy()
x_train_s=x_train/255
x_sample=x_train_s[0:5000]
y_sample=y_train[0:5000]
class grid():
    def __init__(self,model):
        self.model=model
    def grid_get(self,X,y,param_grid):
        grid_search=GridSearchCV(self.model,param_grid,cv=5,scoring='accuracy')
        grid_search.fit(X,y)
        print(grid_search.best_params_,grid_search.best_score_)
        return grid_search.best_params_
grid=grid(SVC()).grid_get(x_sample,y_sample,{'C':[9,10,12],'kernel':['rbf'],'gamma':[0.01,0.013,0.05,0.1]})
def cross_validation(k,x,y):
    svc=SVC(C=9, gamma=0.02, kernel='rbf')
    score=cross_val_score(svc,x,y,cv=3)
    return score.mean()
svc=SVC(C=9, gamma=0.02, kernel='rbf')
def decomposition(n,matrice):
    pca=PCA(n_components=n)
    matrice_pca=pca.fit_transform(matrice)
    return matrice_pca
n_array=[30,50,70,80,90,100,200,300,500,700]
n_score=[]
for n in n_array:
    train_x=decomposition(n,x_train_s)
    n_score.append(cross_val_score(svc,train_x,y_train).mean())
plt.plot(n_array,n_score,'r')
pca=PCA(70)
x_train_scaled=pca.fit_transform(x_train_s)
x_test_s=test_df/255
x_test_scaled=pca.transform(x_test_s)
svc.fit(x_train_scaled,y_train)
result=svc.predict(x_test_scaled)
result
result=np.c_[range(1,len(result)+1),result]
result=pd.DataFrame(result,columns=['ImageId','Label'])
result.to_csv('submission.csv',index=False)