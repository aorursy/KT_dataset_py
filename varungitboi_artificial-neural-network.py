import numpy as np 

import matplotlib

import matplotlib.pyplot as plt

import pandas as pd

import os

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder,MinMaxScaler

print(os.listdir("../input"))

np.random.seed(0)

# Any results you write to the current directory are saved as output.
types = ['f8', 'f8']

train_set = np.genfromtxt("../input/overfit/data_overfit.csv", dtype=types, delimiter=',',names=True)

#enc=LabelEncoder()

#sca=MinMaxScaler()

#y=enc.fit_transform(train_set[:,2])

#train_set[:,2]=y

#trainY=np.eye(2)[train_set[:,2].astype(int)]

data=[]

for tup in train_set:

    data.append(list(tup))

data=np.array(data)

np.random.shuffle(data)

'''sca.fit(trainX)

trainX=sca.transform(trainX)'''
trainX,trainY=data[:4,0],data[:4,1]

testX,testY=data[4:,0],data[4:,1]
print(trainX.shape,trainY.shape)

#print(valX.shape,valY.shape)

print(testX.shape,testY.shape)
plt.scatter(data[:,0], data[:,1], marker='.')
#utility sigmoid activation function

def sigmoid(x):

    return 1/(1+np.exp(-x))
#the actual class that gives the model

class Model:

    def __init__(self):

        self.layer_count=0

        self.weights={}

        self.biases={}

        self.l_dims=[]

        self.grad_W={}

        self.grad_b={}

        self.layer_ops={} #stores the output of each layer.

        self.summary={} #stores the name and shape of each layer

        self.history={"train_loss":[],"val_loss":[],"train_acc":[],"val_acc":[]}

        self.best=None

    

    #returns the layer parameters

    def Dense(self,nodes=1,input_shape=None):

        self.layer_count+=1

        if input_shape:

            self.l_dims.append(input_shape)

        self.l_dims.append(nodes)

        return np.random.random_sample((nodes,self.l_dims[self.layer_count-1])),np.random.random_sample((nodes,1))

        

    #adds the weights passed as layers

    def add(self,weights):

        self.weights[f"w_{self.layer_count}"]=weights[0]

        self.biases[f"b_{self.layer_count}"]=weights[1]

        self.summary[f"Dense_{self.layer_count}"]=weights[0].shape

        

    def predict(self,x):

        self.layer_ops["v_0"]=x

        for i in range(self.layer_count-1):

            x=self.layer_ops[f"v_{i}"]

            x=np.dot(x,self.weights[f'w_{i+1}'].T)

            x=np.add(x,self.biases[f"b_{i+1}"].T)

            x=sigmoid(x)

            self.layer_ops[f'v_{i+1}']=x #saving the layer output

        x=np.dot(x,self.weights[f'w_{self.layer_count}'].T)

        self.layer_ops[f'v_{self.layer_count}']=x

        return x

    

    def fit(self,X,Y,validation_data=None,epochs=100000,batch_size=1,lr=0.005):

        batches=X.shape[0]//batch_size

        best_acc=0

        if validation_data:

            valX,valY=validation_data

        for e in range(epochs):

            loss=0

            for i in range(batches):

                x=X[i*batch_size:i+1*batch_size]

                y=Y[i*batch_size:i+1*batch_size]

                v=self.predict(x)

                grad_base=((v-y)).T #the centerpiece of the gardient for every layer

                for j in reversed(range(1,self.layer_count+1)):

                    self.grad_W[f'{"w_"}{j}']=grad_base

                    #calculation of gradient for each layer

                    for k in reversed(range(j,self.layer_count)):

                        self.grad_W[f'{"w_"}{j}']=np.dot(self.weights[f'{"w_"}{k+1}'].T,self.grad_W[f'{"w_"}{j}'])

                        self.grad_W[f'{"w_"}{j}']=self.grad_W[f'{"w_"}{j}']*(self.layer_ops[f'{"v_"}{k}']*

                                                                             (1-self.layer_ops[f'{"v_"}{k}'])).T

                    self.grad_b[f"b_{j}"]=np.sum(self.grad_W[f"w_{j}"],axis=1).reshape(self.biases[f"b_{j}"].shape) #bias gradient

                    self.grad_W[f'{"w_"}{j}']=np.dot(self.grad_W[f'{"w_"}{j}'],self.layer_ops[f'{"v_"}{j-1}']) #weight gradient

                    #bias and weights adjustment

                    self.weights[f'{"w_"}{j}']-=lr*self.grad_W[f'{"w_"}{j}']

                    self.biases[f'b_{j}']-=lr*self.grad_b[f"b_{j}"]

                    #self.weights[f'{"w_"}{j}']/=trainX.shape[0]

                    #self.biases[f'{"b_"}{j}']/=trainX.shape[0]

                loss+=np.sum((v-y)*(v-y))/(2*batch_size) #batch loss aggregation

            val_loss=0

            train_acc=accuracy_score(np.argmax(trainY,axis=1),np.argmax(self.predict(trainX),axis=1))

            #saving performance on train set

            self.history["train_acc"].append(train_acc)

            self.history["train_loss"].append(loss)

            label=self.predict(trainX)

            #validation

            if validation_data:

                val_v=self.predict(valX)

                #print(val_v.shape,valY.shape)

                val_loss+=np.sum((val_v-valY)*(val_v-valY))/(2*valX.shape[0])

                val_acc=accuracy_score(np.argmax(valY,axis=1),np.argmax(self.predict(valX),axis=1))

                self.history["val_acc"].append(val_acc)

                self.history["val_loss"].append(val_loss)

                if best_acc<val_acc:

                    self.best=Model()

                    for key,value in self.weights.items():

                        self.best.weights[key]=np.copy(value)

                    for key,value in self.biases.items():

                        self.best.biases[key]=np.copy(value)

                    best_acc=val_acc

                print(f"epoch {e}  train loss: {loss}  validation loss:{val_loss}")

            else:

                print(f"epoch {e}  train loss: {loss} ")
model=Model()

model.add(model.Dense(nodes=6,input_shape=trainX.shape[1]))

model.add(model.Dense(nodes=7))

model.add(model.Dense(nodes=7))

model.add(model.Dense(trainY.shape[1]))

model.fit(trainX[:4],trainY[:4],validation_data=[testX,testY],batch_size=1,epochs=200000,lr=0.01)
# Plot training & validation loss values/

plt.plot(model.history['train_loss'])

plt.plot(model.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()
yout=model.predict(data[:,0].reshape((-1,1)))

plt.scatter(data[:,0],yout,marker='.')
plt.figure()

plt.scatter(data[:,0], data[:,1], marker='.',)

plt.scatter(data[:,0],yout,marker='.',color=["orange"])

plt.show()