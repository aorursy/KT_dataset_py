import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
train_data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test_data = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

train_labels = train_data.pop("label").values
train_data = train_data.values

IMG_SIZE = (28, 28)
IMG_X, IMG_Y = IMG_SIZE
IMG_LEN = IMG_X * IMG_Y

train_labels
def show_example(i):
    temp = train_data[i].reshape(IMG_SIZE)
    plt.imshow(temp,cmap='gray')
    plt.show()

def preprocess():
    X = []
#     for i in len(train_data):
#         pass
        
    return X
show_example(2)
IN_FEATURES = 28 * 28

class DeepNetwork(nn.Module):
    def __init__(self):
        super(DeepNetwork, self).__init__()
        
        self._input_ = nn.Identity(IMG_LEN)
        
        self._hidden_1 = nn.Linear(in_features=IMG_LEN * 1, out_features=IMG_LEN * 2)
        self._hidden_2 = nn.Linear(in_features=IMG_LEN * 2, out_features=IMG_LEN * 4)
        self._hidden_3 = nn.Linear(in_features=IMG_LEN * 4, out_features=IMG_LEN * 2)
        
        self._output_ = nn.Linear(in_features=IMG_LEN * 2, out_features=10)
        
    def forward(self, x):
        x = self._input_(x)
        
        x = self._hidden_1(x)
        x = F.relu(x)
        
        x = self._hidden_2(x)
        x = F.relu(x)
        
        x = self._hidden_3(x)
        x = F.relu(x)
        
        x = self._output_(x)
        
        return x
model = DeepNetwork()

lossfn = nn.CrossEntropyLoss()

opt = optim.SGD(model.parameters(), lr=0.01)
for i in range(len(train_data)):
    
    X = torch.tensor( np.array([ float(px/256) for px in train_data[i] ]) ).float()
    y = torch.tensor([ train_labels[i] ])
    
    _y = model(X).unsqueeze(dim=0)
    
    loss = lossfn(_y, y)
    loss.backward()
    
    opt.step()
    
    if i % 100 == 0:
        print(f'Loss: {loss}')
