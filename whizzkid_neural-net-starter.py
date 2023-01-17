import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import torch

import torch.nn.functional as F

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import DataLoader



is_cuda=torch.cuda.is_available()
df=pd.read_csv('../input/Admission_Predict.csv')

df.head()
x=df.iloc[:380,-1].value_counts().index.values

y=df.iloc[:380,-1].value_counts().values

plt.bar(x,y,width=0.1)
plt.figure(figsize=(10,10))

plt.subplot(2,2,1)

plt.scatter(df.iloc[:,1],df.iloc[:,-1])

plt.xlabel("GRE Score")

plt.subplot(2,2,2)

plt.scatter(df.iloc[:,2],df.iloc[:,-1])

plt.xlabel("TOEFL Score")

plt.subplot(2,2,3)

plt.scatter(df.CGPA,df.iloc[:,-1])

plt.xlabel("CGPA")
df.columns
columns_to_take=['GRE Score','TOEFL Score','CGPA','Chance of Admit ']

inp=len(columns_to_take)-1

rows_to_take=380

train_loader=DataLoader(df.loc[:rows_to_take,columns_to_take].values,batch_size=64,shuffle=True) 

test_loader=DataLoader(df.loc[rows_to_take:,columns_to_take].values) 
class Model(torch.nn.Module):

    

    def __init__(self):

        super(Model,self).__init__()

        

        self.l1=torch.nn.Linear(inp,100)

        self.l2=torch.nn.Linear(100,200)

        self.l3=torch.nn.Linear(200,1)

        

        

    def forward(self,x):

        

        x=torch.relu(self.l1(x))

        x=torch.relu(self.l2(x))

        x=self.l3(x)

        

        return x
model=Model()

if(torch.cuda.is_available()):

    model=model.cuda()

criterion=torch.nn.MSELoss()

optim=torch.optim.Adam(model.parameters(),lr=0.01)
earr=[]

larr=[]

for epoch in range(500):



    for data in train_loader:

        xd,yd=data[:,:-1],data[:,-1:]

        xd,yd=xd.to(torch.float32),yd.to(torch.float32)

        if(torch.cuda.is_available()):

            xd,yd=xd.cuda(),yd.cuda()

        

        output=model(xd)

        loss=criterion(output,yd)



        optim.zero_grad()

        loss.backward()

        optim.step()

    

    

    earr.append(epoch)

    larr.append(loss)

    if(epoch%100==0):

        print("Epoch {} Loss {}".format(epoch,loss))

plt.plot(earr,larr)
outarr=[]

exparr=[]

for data in test_loader:

    

    xd,yd=data[:,:-1],data[:,-1:]

    xd,yd=xd.to(torch.float32),yd.to(torch.float32)

    if(torch.cuda.is_available()):

        xd,yd=xd.cuda(),yd.cuda()

        

    output=model(xd)

    outarr.append(output.detach().cpu().numpy())

    exparr.append(yd.cpu().numpy())

    

#     print(output.detach().cpu().numpy(),yd.cpu().numpy())

plt.plot(np.ravel(exparr),np.ravel(exparr))

plt.scatter(np.ravel(outarr),np.ravel(exparr))