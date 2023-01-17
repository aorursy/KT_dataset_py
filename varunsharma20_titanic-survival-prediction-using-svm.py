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
data = pd.read_csv('../input/titanic/train.csv')
data.head()
data.shape
data.columns
data.info()
to_drop = ["Name","Ticket","Cabin","Embarked"]

data_clean = data.drop(to_drop,axis = 1)

data_clean.head()
data_clean.info()
value = {'Age':data_clean['Age'].mean()}
data_clean = data_clean.fillna(value)
data_clean.head()
survived_counts = data_clean['Survived'].value_counts()
survived_counts
from matplotlib import pyplot as plt
survived_counts.plot(kind='bar')
survived_classification = data_clean['Sex'].groupby(data_clean['Survived']).value_counts()
survived_classification
survived_classification.unstack(level=0).plot(kind = 'bar',title = 'Survived and death counts plot')
plt.ylabel('Counts')
plt.show()
data_clean.Sex.replace({'male':1,'female':0},inplace=True)
data_clean.head()
x_col = data_clean.drop('Survived',axis=1)
y_col = data_clean['Survived']

print(type(x_col))
x = x_col.values
y = y_col.values
x_ = (x - x.mean())/x.std()
y[y==0]=-1
print(y[:10])


class MySVM():
    
    def __init__(self,C=0.01):
        self.C=C
        self.W = 0.0
        self.bias = 0.0
        
        
    def hingeloss(self,X,Y,W,bias):
        m = X.shape[0]
        loss = 0.0
        loss += 0.5*np.dot(W,W.T)
        for i in range(m):
            ti = Y[i]*(np.dot(W,X[i].T)+bias)
            loss += self.C*max((0,1-ti))
        return loss[0][0]
    
    def train(self,X,Y,max_iter=300,learning_rate = 0.01,batch_size=100):
        m,n = np.shape((X))
        c = self.C
        W = np.zeros((1,n))
        bias = 0
        gradw = 0
        gradb = 0
        l = []
    
        for i in range(max_iter):
            losses = self.hingeloss(X,Y,W,bias)
            l.append(losses)
            ids = np.arange(m)
            np.random.shuffle(ids)
            
            for batch_start in range(0,m,batch_size):
                gradw = 0
                gradb = 0
                for j in range(batch_start,batch_start+batch_size):
                    if j<m:
                        i = ids[j]
                        ti = Y[i]*(np.dot(W,X[i].T)+bias)     
            
                        if ti>1:
                            gradw+=0 
                            gradb+=0
                        
                        else:
                            gradw += c*Y[i]*X[i]
                            gradb += c*Y[i]
                    
                W = W - learning_rate*W + learning_rate*gradw
                bias = bias + learning_rate*gradb

                
                self.W = W   
                self.bias = bias
        
        return self.W,self.bias,l
    
    
    
    
svm = MySVM()
w,b,losses = svm.train(x_,y)
print(w)
print(b)
w.shape
plt.plot(losses)
def predict(X,W,bias):
        y = (np.dot(X,W.T)+bias)
        y_pred = np.sign(y)
        return y_pred
    
y = predict(x_,w,b)
y = y.astype('int')
y.shape
test_data = pd.read_csv('../input/titanic/test.csv')
test_data.head()
to_drop = ["Name","Ticket","Cabin","Embarked"]

test_data = test_data.drop(to_drop,axis = 1)
value = {'Age':test_data['Age'].mean()}
test_data = test_data.fillna(value)
test_data.head()


test_data.Sex.replace({'male':1,'female':0},inplace=True)
test_data.head()
test_data.info()
value = {'Fare':test_data['Fare'].median()}
test_data = test_data.fillna(value)


x_test = (test_data-test_data.mean())/test_data.std()

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

clf = SVC(kernel='linear')
clf.fit(x_,y_col)

clf.class_weight_
clf.intercept_
y_train_predict = clf.predict(x_)
score = accuracy_score(y_train_predict,y_col)
score
x_test.shape
y_pred_ = clf.predict(x_test)
y_pred_[:10]

df = pd.DataFrame({'PassengerId': test_data.PassengerId,'Survived': y_pred_})
df.to_csv('predictions.csv',index=False)
df = pd.read_csv('./predictions.csv')
df
