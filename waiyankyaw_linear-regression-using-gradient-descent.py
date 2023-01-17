import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
datasets = pd.read_csv("../input/population-vs-profit/ex1data1.csv")

print(datasets.head())

print("number of observations:",datasets.shape[0])
x = datasets.iloc[:,0]

y = datasets.iloc[:,1]

plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

plt.scatter(x,y)

plt.xlabel('Populations in 10,000s')

plt.ylabel('Profit in $10,000s')

plt.show()

m = x.shape[0]

# creating ones column to the size of m

temp = (np.array(x)).reshape(m,1) 

X = np.c_[np.ones(m),x]# np.concatenate to join two array along an axis

y = np.array(y).reshape(m,1)

print(X.shape)

print(y.shape)

#First we need to initialize weights

def initialize_parameter(n):

  '''

  input: n - the number of features 

  output: the randomly printed weights in (n,1) matrix

  '''

  return (np.random.randn(n)).reshape(2,1)



def compute_cost(X,y,W):

  '''

  input: X - training sets

         y - target value

  output: cost - computed cost using simple linear equation 

  '''

  m = X.shape[0]

  y_hat = np.dot(X,W)

  error = np.square(y-y_hat)

  cost = 1/(2*m)*np.sum(error)

  return cost



def grad(X,y,W):

  h = np.dot(X,W)

  grad = np.dot(X.T,(h-y))

  return grad

 
def batch_gradient(X,y,W,epochs,learning_rate):

  '''

  input: X - the number of training sets

         W - pre-initialize weights 

         epochs -the number of steps to iterate through the entire data to update W

  output: J_history - the computed cost for each epoch in a list format

          W         -  the updated weights 

  '''

  J_history = np.zeros((epochs))

  Weights = []

  m = X.shape[0]

  for i in range(epochs):

    h = np.dot(X,W)

    error = h-y

    J_history[i] = compute_cost(X,y,W)

    W = W -(learning_rate/m)*grad(X,y,W)    

    Weights.append(W)

  return J_history,W,Weights

def Stochastic(X,y,W,epochs,learning_rate):

  '''

  input: X - the number of training sets

         W - pre-initialize weights 

         epochs -the number of steps to iterate through the entire data to update W

  output: J_history - the computed cost for each epoch in a list format

          W         -  the updated weights 

  '''

  m = len(y)

  Weights = []

  J_history = np.zeros((epochs))

  for i in range(epochs):

    tempcost = 0.0

    for j in range(m):

      rand_indx = np.random.randint(0,m)

      X_i= X[rand_indx,:].reshape(1,X.shape[1])

      y_i = y[rand_indx].reshape(1,1)

      h = np.dot(X_i,W)

      error = h-y_i

      W = W -(1/m)*learning_rate*grad(X,y,W)  

      tempcost += compute_cost(X,y,W)

      Weights.append(W)

    J_history[i] = tempcost/m    

  return J_history,W,Weights
def create_mini_batches(X, y, batch_size): # correct

    mini_batches = [] 

    data = np.hstack((X, y)) 

    np.random.shuffle(data) 

    n_minibatches = data.shape[0] // batch_size 

    i = 0  

    for i in range(n_minibatches + 1): 

        mini_batch = data[i * batch_size:(i + 1)*batch_size, :] 

        X_mini = mini_batch[:, :-1] 

        Y_mini = mini_batch[:, -1].reshape((-1, 1)) 

        mini_batches.append((X_mini, Y_mini)) 

    if data.shape[0] % batch_size != 0: 

        mini_batch = data[i * batch_size:data.shape[0]] 

        X_mini = mini_batch[:, :-1] 

        Y_mini = mini_batch[:, -1].reshape((-1, 1)) 

        mini_batches.append((X_mini, Y_mini)) 

    return mini_batches 



def Mini_Batch(X,y,W,epochs,learning_rate,batch_size):

  '''

  input: X - the X training sets (m,n)

         y - the true value(n,1)

         W - the pre-initialized weights

         batch_size - number of batch size to be separated from a complete datsets

  output: J_history - the computed cost over each epoch

              W     - the updated weight over each epoch 

  '''

  J_history = []

  Weights =[]

  for i in range(epochs):

    mini_batches = create_mini_batches(X,y,batch_size)

    for mini_batch in mini_batches:

      X_mini,y_mini = mini_batch

      W = W -(1/batch_size)*learning_rate*grad(X_mini,y_mini,W)      

      J_history.append(compute_cost(X_mini,y_mini,W))

  return J_history,W,Weights
def train_model(X,y,learning_rate=1e-4,epochs=1500,type='Batch',batch_size = 5):



    W = initialize_parameter(n=2)*0.08

    if type == 'Batch':

        J,W,Weight_history =batch_gradient(X,y,W,epochs,learning_rate)

        return J,W,Weight_history

    if type == 'Stochastic':    

        J,W,Weight_history =Stochastic(X,y,W,epochs,learning_rate)

        return J,W,Weight_history    

    if type == 'Mini_Batch':

        J,W,Weight_history = Mini_Batch(X,y,W,epochs,learning_rate,batch_size) 

        return J,W,Weight_history

    else:

        print(" Re-enter the valid type")

    return None    
fig = plt.figure(figsize=(30,5))

fig.suptitle('Different types of Gradient Descent')



J_Batch,W_batch,Weight_history_batch= train_model(X,y,learning_rate=1e-4,epochs=1600,type='Batch',batch_size =_)

sub1 = plt.subplot(1,3,1)

sub1.plot(J_Batch)

sub1.set(xlabel='Iterations',ylabel='Cost',title = 'Batch_Gradient')



J_Sto,W_sto,Weight_history_sto= train_model(X,y,learning_rate=1e-4,epochs=40,type='Stochastic',batch_size =_)

sub2 = plt.subplot(1,3,2)

sub2.plot(J_Sto)

sub2.set(xlabel='Iterations',ylabel='Cost',title = 'Stochastic_Gradient')



J_Mini,W_mini,Weight_history_mini= train_model(X,y,learning_rate=1e-4,epochs=20,type='Mini_Batch',batch_size = 5)

sub3 = plt.subplot(1,3,3)

sub3.plot(J_Mini)

sub3.set(xlabel='Iterations',ylabel='Cost',title = 'Mini_Batch_Gradient')



print("Minimum cost from Batch_Gradient:", J_Batch[-1])

print("Minimum cost from Stochastic_Gradient:", J_Sto[-1])

print("Minimum cost from Mini_Batch_Gradient:", J_Mini[-1])



print(W_batch)

print(W_sto)

print(W_mini)
plt.figure(figsize= (30,10))

plt.subplot(131)

plt.scatter(x,y)

plt.plot(x,X.dot(W_batch),color ='red')

plt.xlabel('Populations in 10,000s')

plt.ylabel('Profit in $10,000s')

plt.title("Batch_Gradient_Descent")



plt.subplot(132)

plt.scatter(x,y)

plt.plot(x,X.dot(W_sto),color ='red')

plt.xlabel('Populations in 10,000s')

plt.ylabel('Profit in $10,000s')

plt.title("Stochastic_Gradient_Descent")



plt.subplot(133)

plt.scatter(x,y)

plt.plot(x,X.dot(W_mini),color ='r')

plt.xlabel('Populations in 10,000s')

plt.ylabel('Profit in $10,000s')

plt.title("Mini_Batch_Gradient_Descent")