# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
from torch import optim
import torch.nn.functional as F
from tqdm import notebook
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import copy
from sklearn.ensemble import RandomForestClassifier
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd

import warnings
warnings.filterwarnings('ignore')
#loading dataframe
df = pd.read_csv(dirname + "/train.csv")
df.head()

def get_data(data):
    # take only this specific column
    data = data[['Age', 'Sex', 'Fare', 'Pclass', 'SibSp', 'Parch']]
    
    # replace male by 1, female by 0
    data.replace({ 'male' : 1, 'female' : 0 }, inplace=True)
    
    # replace null/nan data by the mean (age and fare columns)
    data['Fare'].fillna(int(data['Fare'].mean()), inplace=True)
    data['Age'].fillna(int(data['Age'].mean()), inplace=True)
    
    
    # transform into a numpy array
    data = data.to_numpy()
    
    # normalize (make sure the data is between -1 and 1)
    for i in range(data.shape[1]):
        data[:,i] = (data[:,i] - data[:,i].mean()) / data[:,i].std()
    
    return data
# lets take out first the label
y=df['Survived'].values
#print(y.shape)-->(891,)
#y_binarized is used just stratifying the data in train_test_split
#df['Survived'].value_counts(normalize=True)#ensures all 0,1's are in equal ratio
#y_binarized=df['Survived'].values
#print(y_binarized)

#loading data into x
x= get_data(df)
#print(x.shape)--->(891,6)

#splitting data for evaluation and training  x_train,y_train --->training purpose y_test,x_test--->valuation purpose
#x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,stratify=y_binarized)
#print(x_train.shape)-->(668,6)
#print(x_test.shape)-->(223,6)
#print(y_train.shape)-->(668,)
#print(y_test.shape)-->(223,)

#converting numpy arrays into tensors using pytorch
x_torch_train=torch.from_numpy(x)
y_torch_trian=torch.from_numpy(y)
#x_torch_test=torch.from_numpy(x_test)
#y_torch_test=torch.from_numpy(y_test)
#This class is used to to create above network
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.net=nn.Sequential(
                                nn.Linear(6,512),
                                nn.Sigmoid(),
                                nn.Linear(512,256),
                                nn.Sigmoid(),
                                nn.Linear(256,1),
                                nn.Sigmoid()
                                )
    def forward(self,X):
        return self.net(X)[:,0]
    #Fit function is used to iterate over train data and update parameters    
    def fit(self,X,Y,optimizer=0,loss_fn=nn.BCELoss(),epochs=1000):
        loss_arr=[]
        min_loss=8
        if optimizer == 0:
            optimizer=optim.Adam(self.net.parameters(),lr=0.00125)
        for i in notebook.tqdm(range(epochs),total=epochs,unit='epoch'):
            loss=loss_fn(self(X),Y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_arr.append(loss.item())
            if min_loss > loss.item():
                min_loss = loss.item()
                best_model = copy.deepcopy(self.state_dict())
        print('========================')
        print('     Min loss %0.2f' % min_loss)
        print('========================')
        self.load_state_dict(best_model)
        return loss_arr
    #calaculate accuracy after training the data
    def accuracy(self,X,Y):
        correct=0
        total=0
        bin_train=self(X)
        bin_train=(bin_train>0.5).int()
        Y=Y.int()
        for i,j in zip(bin_train,Y):
            total+=1
            if i==j:
                correct+=1
        accuracy=100*(correct/total)
        print('total =',total)
        print('correct predictions =',correct)
        print('accuracy =',round(accuracy,2))
#intializing model and moving all data to gpu
model=Network()
device=torch.device("cuda")
print(device)
x_torch_train=x_torch_train.to(device)
y_torch_trian=y_torch_trian.to(device)
#x_torch_test=x_torch_test.to(device)
#y_torch_test=y_torch_test.to(device)
model.to(device)
#loss_fn=nn.BCELoss()
optimizer=optim.SGD(model.parameters(),lr=1)#---->#specifiy learning rate if you prefer using lr=learning_rate_value
losslist=model.fit(x_torch_train.float(),y_torch_trian.float(),epochs=10000)
print('----------Train Accuracy---------- ')
model.accuracy(x_torch_train.float(),y_torch_trian.float())
print('----------Test  Accuracy---------- ')
model.accuracy(x_torch_test.float(),y_torch_test.float())
plt.plot(losslist)
plt.show()
test_df = pd.read_csv(dirname + "/test.csv")
test_x = get_data(test_df)
torch_test_x=torch.from_numpy(test_x)
torch_test_x=torch_test_x.to(device)
# Get test data predictions
test_preds =model(torch_test_x.float())
test_preds=(test_preds>0.5).int()
test_preds=test_preds.tolist()

# Add passengers ids to the test predictions
passenger_ids = test_df['PassengerId'].to_numpy()
# combine passenger ids with the predictions
final_result = np.array(list(map(list, zip(passenger_ids,test_preds))))
# arraay final_result to dataframe
df_final = pd.DataFrame(data=final_result, columns=["PassengerId", "Survived"])
# save the result
#df_final.tail()
df_final.to_csv('submission_final13.csv', index=False)
