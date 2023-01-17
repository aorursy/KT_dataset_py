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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder,MinMaxScaler
import seaborn as sns
from sklearn.decomposition import PCA
df=pd.read_csv('/kaggle/input/logistic-regression/Social_Network_Ads.csv')
df.head()
# Drop User id
len(df['User ID'].unique())
df.drop(columns=['User ID'],inplace=True)
df.describe()
df.isnull().sum()
df.dtypes
#conert categorical feature to numarical feature
le=LabelEncoder()
df['Gender']=le.fit_transform(df['Gender'])
#Normalize the data
sc=MinMaxScaler()
df_n=sc.fit_transform(df.iloc[:,:-1])
#train test split
x_train,x_test,y_train,y_test=train_test_split(df_n,df['Purchased'])
y_train.reset_index(drop=True,inplace=True)
y_test.reset_index(drop=True,inplace=True)
x=x_train
y=y_train
#pairplot
sns.pairplot(df,hue='Purchased')
sns.boxplot(x='Purchased',y='EstimatedSalary',data=df)
sns.boxplot(x='Purchased',y='Age',data=df)
#pie plot
df_gender=df[['Gender','Purchased']].groupby('Purchased').sum()
df_gender.index=['Male','Female']
df_gender['Gender'].plot(kind='pie',autopct='%1.1f%%')
plt.show()
def sigmoid(x,w,b):
    return 1/(1+np.exp(-(np.dot(x,w)+b)))
def loss(x,w,y,b):
    s=sigmoid(x,w,b)
    return np.mean(-(y*np.log(s))- ((1-y)*np.log(1-s)))
def grad(x,y,w,b):
    s=sigmoid(x,w,b)    
    return np.dot(x.T,(s-y))/x.shape[0]
def accuracy(y_pred,y_test):
    return np.mean(y_pred==y_test)
# initilize w and b
def gradientdescent(x,y):
    w=np.zeros((x.shape[1]))
    b=np.zeros(1)
    ite=1000 #number of iteration
    eta=0.7 #learning rate
    loss_v=[]
    for i in range(ite):
        probability=sigmoid(x,w,b)
        l=loss(x,w,y,b)
        gradient=grad(x,y,w,b)
        w=w- (eta*gradient)
        b=b-(eta*np.sum(probability-y)/x.shape[0])
        loss_v.append(l)
        if i%100==0:
            print(l)
    return w,b,loss_v
w,b,loss_v=gradientdescent(x,y)
y_pred=sigmoid(x_test,w,b)
for j,i in enumerate(y_pred):
    if i<0.5:
        y_pred[j]=0
    else:
        y_pred[j]=1

print('test accuracy',accuracy(y_pred,y_test))
plt.plot(range(len(loss_v)),loss_v)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.show()
batch_size=8
def sgd(x,y,batch_size):
    # initilize w and b
    w=np.zeros((x_train.shape[1]))
    b=np.zeros(1)
    ite=1000 #number of iteration
    eta=0.7 #learning rate
    loss_v=[]
    for i in range(1000):
        ind=np.random.choice(len(y_train),batch_size)
        x_b=x[ind]
        y_b=y[ind]
        p=sigmoid(x_b,w,b)
        l=loss(x_b,w,y_b,b)
        gradient=grad(x_b,y_b,w,b)
        w=w- (0.1*gradient)
        b=b-(eta*np.sum(p-y_b)/x.shape[0])
        if i%10==0:
            loss_v.append(l)
        if i%100==0:
            print('loss',l)
    return w,b,loss_v
w,b,loss_v=sgd(x,y,32)
y_pred=sigmoid(x_test,w,b)
for j,i in enumerate(y_pred):
    if i<0.5:
        y_pred[j]=0
    else:
        y_pred[j]=1

print('test accuracy',accuracy(y_pred,y_test))
plt.plot(range(len(loss_v)),loss_v)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.show()
batch_size=8
def sgdmomentum(x,y,batch_size):
    # initilize w and b
    w=np.zeros((x_train.shape[1]))
    b=np.zeros(1)
    ite=1000 #number of iteration
    eta=0.7 #learning rate
    alpha=0.9
    loss_v=[]
    v_t=np.zeros((x_train.shape[1])) 
    v_b=np.zeros(1)
    for i in range(1000):
        ind=np.random.choice(len(y_train),batch_size)
        x_b=x[ind]
        y_b=y[ind]
        p=sigmoid(x_b,w,b)
        l=loss(x_b,w,y_b,b)
        gradient=grad(x_b,y_b,w,b)
        v_t =(alpha*v_t) + (eta*gradient)
        w=w-v_t
        v_b=(alpha*v_b) + (eta*np.sum(p-y_b)/x.shape[0])
        b=b-v_b
        if i%10==0:
            loss_v.append(l)
        if i%100==0:
            print('loss',l)
    return w,b,loss_v
w,b,loss_v=sgdmomentum(x,y,32)
plt.plot(range(len(loss_v)),loss_v)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.show()
#Predction
y_pred=sigmoid(x_test,w,b)
for j,i in enumerate(y_pred):
    if i<0.5:
        y_pred[j]=0
    else:
        y_pred[j]=1

print('test accuracy',accuracy(y_pred,y_test))
from sklearn.linear_model import LogisticRegression
model= LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print('test accuracy',accuracy(y_pred,y_test))
