import pandas as pd

import numpy as np

data = pd.read_csv("../input/xclara.csv")
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

data = scaler.fit_transform(data)
from sklearn.model_selection import train_test_split

x_train,x_test = train_test_split(data)
from matplotlib import pyplot as plt

plt.scatter(x_train[:,0],x_train[:,1])

plt.show()
rand_init_1 = data[np.random.randint(0,x_train.shape[0],3)]

x_train = np.append(x_train,np.zeros((x_train.shape[0],1)),axis=1)

rand_init_1
def index(a):

    a = list(a)

    return a.index(min(a))

index([2,5,0,-4])
def distance(test,data):

    return (np.sum((test - data)**2,axis = 1,keepdims=True))**(0.5)

#data[data[:,2]==2,:2]
rand_init_1= np.ndarray((10,2)) #2(n) is the dimension of each example and 3000(m) is the number of examples 

rand_init_2= np.ndarray((10,2))

rand_init_3= np.ndarray((10,2))

check = []

for k in range(10):

    rand_init_1[k,:] = data[np.random.randint(0,data.shape[0],1),:2]

    rand_init_2[k,:] = data[np.random.randint(0,data.shape[0],1),:2]

    rand_init_3[k,:] = data[np.random.randint(0,data.shape[0],1),:2]

    for i in range(10):

        dist_11 = distance(rand_init_1[k,:],data[:,0:2])

        dist_12 = distance(rand_init_2[k,:],data[:,0:2])

        dist_13 = distance(rand_init_3[k,:],data[:,0:2])

        if data.shape[1] !=3:

            data = np.c_[data,np.zeros((3000,1))]

        for j in range(data.shape[0]):

            data[j,2] = index(np.array([dist_11[j],dist_12[j],dist_13[j]]))

        rand_init_1[k,:] = np.sum(data[data[:,2]==0,:2],axis=0)/(data[data[:,2]==0].shape[0])

        rand_init_2[k,:] = np.sum(data[data[:,2]==1,:2],axis=0)/(data[data[:,2]==1].shape[0])

        rand_init_3[k,:] = np.sum(data[data[:,2]==2,:2],axis=0)/(data[data[:,2]==2].shape[0])

    temp1 = sum((distance(rand_init_1[k,:],data[data[:,2]==0,:2])))+sum((distance(rand_init_2[k,:],data[data[:,2]==1,:2])))+sum((distance(rand_init_3[k,:],data[data[:,2]==2,:2])))

    check.append(temp1/3000)

       
pt1 = rand_init_1[check.index(min(check)),:]

pt2 = rand_init_2[check.index(min(check)),:]

pt3 = rand_init_3[check.index(min(check)),:]
plt.scatter(data[:,0],data[:,1],c = "cyan")

plt.scatter(pt1[0],pt1[1],c = "red")

plt.scatter(pt2[0],pt2[1],c = "red")

plt.scatter(pt3[0],pt3[1],c = "red")



plt.show
