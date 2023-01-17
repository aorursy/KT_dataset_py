# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
X_train = np.array(pd.read_csv("/kaggle/input/digit-recognizer/train.csv"))
Y_train = np.array([X_train[:,0]])
X_train = X_train[:,1:]
X_test = np.array(pd.read_csv("/kaggle/input/digit-recognizer/test.csv"))
X_train = X_train.T/255
X_test = X_test.T/255
shape = (Y_train.size, Y_train.max()+1)
one_hot = np.zeros(shape)
rows = np.arange(Y_train.size)
one_hot[rows, Y_train] = 1
Y_train = one_hot.T
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
n = X_train.shape[0]
m = X_train.shape[1]
def softmax(x) :
  x = x - np.max(x)
  return np.exp(x) / np.sum(np.exp(x), axis=0,keepdims=True)
def relu(X) :
  X[X<0]=0
  return X
def initialize_wts(struct) :
  W = {}
  b = {}
  for i in range(1,len(struct)) :
    W["W" + str(i)] = np.random.randn(struct[i],struct[i-1])*np.sqrt(2/struct[i-1])
    #W["W" + str(i)] = np.random.randn(struct[i],struct[i-1])*0.01
    #W["W" + str(i)] = np.random.rand(struct[i],struct[i-1])
    b["b" + str(i)] = np.zeros((struct[i],1))
  return W,b
def forward_prop(X_train,W,b) :
  return relu(np.dot(W,X_train) + b)
def model(struct,X_train,Y_train,lr,W,b,num_itr) :
  n = X_train.shape[0]
  m = X_train.shape[1]
  for k in range(0,num_itr) :
    A = {}
    A["A" + str(0)] = np.copy(X_train)
    for i in range(1,len(struct)-1) :
      A["A" + str(i)] = forward_prop(A["A" + str(i-1)],W["W" + str(i)],b["b" + str(i)])
    A["A" + str(len(struct) - 1)] = softmax(np.dot(W["W" + str(len(struct) - 1)] , A["A" + str(len(struct) - 2)]) + b["b" + str(len(struct) - 1)])
    dW = {}
    db = {}
    dZ = {}
    dZ["dZ" + str(len(struct) - 1)] = A["A" + str(len(struct) - 1)] - Y_train
    dW["dW" + str(len(struct) - 1)] = (1/m)*(np.dot(dZ["dZ" + str(len(struct) - 1)],A["A" + str(len(struct) - 2)].T))
    db["db" + str(len(struct) - 1)] = (1/m)*(np.sum(dZ["dZ" + str(len(struct) - 1)],axis = 1, keepdims = True))
    for i in range(0,len(W) - 1) :
      dZ["dZ" + str(len(W) - 1 - i)] = np.dot(W["W" + str(len(W) - i)].T,dZ["dZ" + str(len(W) - i)])*np.int64(A["A" + str(len(W) - 1 - i)] > 0)
      dW["dW" + str(len(W) - 1 - i)] = (1/m)*np.dot(dZ["dZ" + str(len(W) - 1 - i)] , A["A" + str(len(W) - 1 - i - 1)].T)
      db["db" + str(len(W) - i - 1)] = (1/m)*np.sum(dZ["dZ" + str(len(W) - 1 - i)],axis = 1,keepdims = True)
    for i in range(1,len(W) + 1) :
      W["W" + str(i)] = W["W" + str(i)] - lr*dW["dW" + str(i)]
      b["b" + str(i)] = b["b" + str(i)] - lr*db["db" + str(i)]
    cost1 = -np.mean(Y_train * np.log(A["A" + str(len(struct) - 1)] + 1e-8))
  return W,b,A["A" + str(len(struct) - 1)],float(cost1)
def train(struct,X_train,Y_train,lr,epoks,batch_size):
  costu = []
  W,b = initialize_wts(struct)
  for j in range(0,epoks) :
    for i in range(0,int(X_train.shape[1]/batch_size)+1) :
      if i != int(X_train.shape[1]/batch_size) :
        X_batch = X_train[:,i*batch_size:(i+1)*batch_size]
        Y_batch = Y_train[:,i*batch_size:(i+1)*batch_size]
        lim = (i+1)*batch_size
      elif i*batch_size == X_train.shape[1] :
        continue
      else :
        X_batch = X_train[:,i*batch_size:-1]
        Y_batch = Y_train[:,i*batch_size:-1]
        lim = X_train.shape[1]
      W,b,A,cost1 = model(struct , X_batch , Y_batch , lr ,W,b,num_itr = 1)
      costu.append(cost1)
      print("Epok : {}  Data_Passed : {}/{}  Train_Loss : {}".format(j+1,lim,X_train.shape[1],cost1))
  A = {}
  A["A" + str(0)] = np.copy(X_train)
  for i in range(1,len(struct)-1) :
    A["A" + str(i)] = forward_prop(A["A" + str(i-1)],W["W" + str(i)],b["b" + str(i)])
  A = softmax(np.dot(W["W" + str(len(struct) - 1)] , A["A" + str(len(struct) - 2)]) + b["b" + str(len(struct) - 1)])
  plt.plot(list(range(len(costu))),costu)
  return W,b,A
W,b,A = train([n,600,400,300,200,100,50,10],X_train,Y_train,0.1,10,64)
def get_output(X,W,b) :
  struct = len(W) + 1
  A = {}
  A["A" + str(0)] = np.copy(X)
  for i in range(1,struct-1) :
    A["A" + str(i)] = forward_prop(A["A" + str(i-1)],W["W" + str(i)],b["b" + str(i)])
  A["A" + str(struct - 1)] = softmax(np.dot(W["W" + str(struct - 1)] , A["A" + str(struct - 2)]) + b["b" + str(struct - 1)])
  return A["A" + str(struct - 1)]
def predict(A,Y) :
  for i in range(0,A.shape[1]) :
    a = np.max(A[:,i])
    A[:,i] = (A[:,i] == a)
  fai = np.absolute(A - Y)
  count = 0
  for i in range(0,fai.shape[1]) :
    if np.sum(fai[:,i]) == 0 :
      count += 1
  return (count/A.shape[1])*100
A_train = get_output(X_train,W,b)
print("Train accuracy is {}".format(predict(A_train,Y_train)))
A_test  = get_output(X_test,W,b)
Y_pred = np.argmax(A_test , axis = 0)
print(Y_pred)
f = np.arange(A_test.shape[1]) + 1
final = np.zeros((A_test.shape[1] , 2))
final[:,0] = np.copy(f)
final[:,1] = np.copy(Y_pred)
# final = np.vstack([np.array(["ImageId","Label"]),final])
np.savetxt("finalresults09.csv",final,delimiter = ",",fmt="%d")





