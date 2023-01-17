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
import random
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, label_ranking_average_precision_score, label_ranking_loss, coverage_error 
from sklearn.utils import shuffle
from scipy.signal import resample
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torchvision import transforms
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torch import autograd
from torch.autograd import Variable

%matplotlib inline
train_df = pd.read_csv("../input/heartbeat/mitbih_train.csv", header = None)
test_df = pd.read_csv("../input/heartbeat/mitbih_test.csv", header = None)

df = pd.concat([train_df, test_df], axis=0)
df.head()


df.info()

df[187].value_counts()
data_array = df.values
inputs = data_array[:, :-1]
labels = data_array[:, -1].astype(int)

print(data_array.shape)

del train_df
del test_df
del data_array
c0 = np.argwhere(labels == 0).flatten()
c1 = np.argwhere(labels == 1).flatten()
c2 = np.argwhere(labels == 2).flatten()
c3 = np.argwhere(labels == 3).flatten()
c4 = np.argwhere(labels == 4).flatten()
x = np.arange(0, 187)*8/1000
plt.figure(figsize=(20,12))
plt.plot(x, inputs[c0, :][0], label="Cat. N")
plt.plot(x, inputs[c1, :][0], label="Cat. S")
plt.plot(x, inputs[c2, :][0], label="Cat. V")
plt.plot(x, inputs[c3, :][0], label="Cat. F")
plt.plot(x, inputs[c4, :][0], label="Cat. Q")
plt.legend()
plt.title("1-beat ECG for every category", fontsize=20)
plt.ylabel("Amplitude", fontsize=15)
plt.xlabel("Time (ms)", fontsize=15)
plt.show()
def stretch(x):
    l = int(187 * (1 + (random.random()-0.5)/3))
    y = resample(x, l)
    if l < 187:
        y_ = np.zeros(shape=(187, ))
        y_[:l] = y
    else:
        y_ = y[:187]
    return y_

def amplify(x):
    alpha = (random.random()-0.5)
    factor = -alpha*x + (1+alpha)
    return x*factor

def augment(x):
    result = np.zeros(shape= (4, 187))
    for i in range(3):
        if random.random() < 0.33:
            new_y = stretch(x)
        elif random.random() < 0.66:
            new_y = amplify(x)
        else:
            new_y = stretch(x)
            new_y = amplify(new_y)
        result[i, :] = new_y
    return result

plt.plot(inputs[10000,:])

result = np.apply_along_axis(augment, axis=1, arr=inputs[c3]).reshape(-1, 187)
c = np.ones(shape=(result.shape[0],), dtype=int)*3
inputs = np.vstack([inputs, result])
labels = np.hstack([labels, c])
print(inputs.shape)
subC0 = np.random.choice(c0, 900)
subC1 = np.random.choice(c1, 900)
subC2 = np.random.choice(c2, 900)
subC3 = np.random.choice(c3, 900)
subC4 = np.random.choice(c4, 900)




x_test = np.vstack([inputs[subC0], inputs[subC1], inputs[subC2], inputs[subC3], inputs[subC4]]).reshape(-1,1,187)
y_test = np.hstack([labels[subC0], labels[subC1], labels[subC2], labels[subC3], labels[subC4]])

x_train = np.delete(inputs, [subC0, subC1, subC2, subC3, subC4], axis=0).reshape(-1,1,187)
y_train = np.delete(labels, [subC0, subC1, subC2, subC3, subC4], axis=0)

x_train, y_train = shuffle(x_train, y_train, random_state=0)
x_test, y_test = shuffle(x_test, y_test, random_state=0)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

del inputs
del labels




n = x_train.shape[0]

#train val split
def train_val_split(x,y,n,split_ratio,seed):
    np.random.seed(100)
    n_val = int(split_ratio*n)

    idxs = np.random.permutation(n)
    train_idxs, val_idxs = idxs[n_val:], idxs[:n_val]
    
    return x[train_idxs],y[train_idxs],x[val_idxs],y[val_idxs]
x_train,y_train,x_val,y_val = train_val_split(x_train,y_train,n,0.2,343543)
print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)

class MyDataset(IterableDataset):
    def __init__(self,data):
        self.data = data
    def __iter__(self):
        return iter(self.data)

train_inputs = torch.from_numpy(x_train)
train_targets = torch.from_numpy(y_train)
train_ds = TensorDataset(train_inputs,train_targets)
# train_iter = MyDataset(train_ds)
train_loader = DataLoader(train_ds,batch_size=64,shuffle=True)

val_inputs = torch.from_numpy(x_val)
val_targets = torch.from_numpy(y_val)
val_ds = TensorDataset(val_inputs,val_targets)
# train_iter = MyDataset(train_ds)
val_loader = DataLoader(val_ds,batch_size=64,shuffle=True)

test_inputs = torch.from_numpy(x_test)
test_targets = torch.from_numpy(y_test)
test_ds = TensorDataset(test_inputs,test_targets)
# test_iter = MyDataset(x_test)
test_loader = DataLoader(test_ds,batch_size=64,shuffle=True)

for xbatch,ybatch in train_loader:
    print(xbatch)
    print(ybatch)
    break




#Resnet Block
class ResNet(nn.Module):
    def __init__(self, module_1, module_2):
        super(ResNet, self).__init__()
        self.module_1 = module_1
        self.module_2 = module_2
        self.shortcut = nn.Identity() 
    def forward(self, inputs):
        return self.module_2(self.shortcut(self.module_1(inputs)) + self.shortcut(inputs))
          
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.model = nn.Sequential(
          nn.Conv1d(1,32,kernel_size=5,stride=1,padding=2),
          # 5 Residual blocks
          ResNet(
              nn.Sequential(
                  nn.Conv1d(32,32,kernel_size=5,stride=1,padding=2),
                  nn.ReLU(),
                  nn.Conv1d(32,32,kernel_size=5,stride=1,padding=2),
              ),
              nn.Sequential(
                  nn.ReLU(),
                  nn.MaxPool1d(5,stride=2)
              )
          ),
          ResNet(
              nn.Sequential(
                  nn.Conv1d(32,32,kernel_size=5,stride=1,padding=2),
                  nn.ReLU(),
                  nn.Conv1d(32,32,kernel_size=5,stride=1,padding=2),
              ),
              nn.Sequential(
                  nn.ReLU(),
                  nn.MaxPool1d(5,stride=2)
              )
          ),
          ResNet(
              nn.Sequential(
                  nn.Conv1d(32,32,kernel_size=5,stride=1,padding=2),
                  nn.ReLU(),
                  nn.Conv1d(32,32,kernel_size=5,stride=1,padding=2),
              ),
              nn.Sequential(
                  nn.ReLU(),
                  nn.MaxPool1d(5,stride=2)
              )
          ),
          ResNet(
              nn.Sequential(
                  nn.Conv1d(32,32,kernel_size=5,stride=1,padding=2),
                  nn.ReLU(),
                  nn.Conv1d(32,32,kernel_size=5,stride=1,padding=2),
              ),
              nn.Sequential(
                  nn.ReLU(),
                  nn.MaxPool1d(5,stride=2)
              )
          ),
          ResNet(
              nn.Sequential(
                  nn.Conv1d(32,32,kernel_size=5,stride=1,padding=2),
                  nn.ReLU(),
                  nn.Conv1d(32,32,kernel_size=5,stride=1,padding=2),
              ),
              nn.Sequential(
                  nn.ReLU(),
                  nn.MaxPool1d(5,stride=2)
              )
          ),
          nn.Flatten(),
            
          nn.Linear(in_features=64,out_features=32),
          nn.ReLU(),
          nn.Linear(in_features=32,out_features=5)
      )
    
    def forward(self, x):
#           x = x.view(x.size(0), 784)
#           c = self.label_emb(labels)
#           x = torch.cat([x, c], 1)
          out = self.model(x)
            
          return out
model = nn.Sequential(
          nn.Conv1d(1,32,kernel_size=5,stride=1,padding=2),
#           nn.ReLU(),
          # 5 Residual blocks
          ResNet(
              nn.Sequential(
                  nn.Conv1d(32,32,kernel_size=5,stride=1,padding=2),
                  nn.ReLU(),
                  nn.Conv1d(32,32,kernel_size=5,stride=1,padding=2),
              ),
              nn.Sequential(
                  nn.ReLU(),
                  nn.MaxPool1d(5,stride=2)
              )
          ),
          ResNet(
              nn.Sequential(
                  nn.Conv1d(32,32,kernel_size=5,stride=1,padding=2),
                  nn.ReLU(),
                  nn.Conv1d(32,32,kernel_size=5,stride=1,padding=2),
              ),
              nn.Sequential(
                  nn.ReLU(),
                  nn.MaxPool1d(5,stride=2)
              )
          ),
          ResNet(
              nn.Sequential(
                  nn.Conv1d(32,32,kernel_size=5,stride=1,padding=2),
                  nn.ReLU(),
                  nn.Conv1d(32,32,kernel_size=5,stride=1,padding=2),
              ),
              nn.Sequential(
                  nn.ReLU(),
                  nn.MaxPool1d(5,stride=2)
              )
          ),
          ResNet(
              nn.Sequential(
                  nn.Conv1d(32,32,kernel_size=5,stride=1,padding=2),
                  nn.ReLU(),
                  nn.Conv1d(32,32,kernel_size=5,stride=1,padding=2),
              ),
              nn.Sequential(
                  nn.ReLU(),
                  nn.MaxPool1d(5,stride=2)
              )
          ),
          ResNet(
              nn.Sequential(
                  nn.Conv1d(32,32,kernel_size=5,stride=1,padding=2),
                  nn.ReLU(),
                  nn.Conv1d(32,32,kernel_size=5,stride=1,padding=2),
              ),
              nn.Sequential(
                  nn.ReLU(),
                  nn.MaxPool1d(5,stride=2)
              )
          ),
          nn.Flatten(),
          nn.Linear(in_features=64,out_features=32),
          nn.ReLU(),
          nn.Linear(in_features=32,out_features=32),
          nn.Linear(in_features=32,out_features=5)
      ).to('cuda')
# model[1].weight
# network = Network().to('cuda')
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001,betas=(0.9,0.999))

train_epoch_loss = []
val_accuracy = []


def train(num_epochs, model, loss_fn, optimizer):
    for epoch in range(num_epochs):
        train_step = 0
        avg_loss = 0.0
        correct = 0.0
        total = 0
        #training
        model.train()
        for xb,yb in train_loader:  
            xb,yb = xb.to('cuda').float(), yb.to('cuda')
            preds = model(xb)
            preds_act = torch.nn.functional.softmax(preds,dim=1)
            _,y_pred = torch.max(preds_act,1)
            loss = loss_fn(preds,yb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_step += 1
            avg_loss += loss.item()
            
            total += yb.size(0)
            correct += (y_pred == yb).sum().item()
            
        train_epoch_loss.append(avg_loss / train_step)
        print('Epoch [{}/{}]'.format(epoch+1,num_epochs))
        print('Avg Train Loss : {}'.format(avg_loss / train_step))
        print('Train Accuracy : {}'.format(100 *correct / total))
        
        #Evaluation
        correct = 0.0
        val_step = 0
        total = 0
        model.eval()
        for xb,yb in val_loader:
            xb,yb = xb.to('cuda').float(), yb.to('cuda')
            preds = model(xb) 
            _,y_pred = torch.max(preds,1)
            correct += (y_pred == yb).sum().item()
            val_step += 1
            total += yb.size(0)
        val_accuracy.append(correct / total)
        print('Validation Accuracy : {}'.format(correct / total))
            
            
        
train(5,model,loss_fn,optimizer)
plt.plot(train_epoch_loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(val_accuracy)
correct = 0.0
total = 0
original_output = []
predicted_output = []
model.eval()
last_conv_output = []
last_conv_pool_output = []
eachlayer_output = []
for xb,yb in test_loader:
    xb,yb = xb.to('cuda').float(), yb.to('cuda')
#     preds = model(xb)
    n = 5
    for i,l in enumerate(model.children()):
        xb = l(xb)
        eachlayer_output.append(xb)
        if(i == n):
            last_conv_pool_output.append(xb.view(xb.shape[0],-1))
            n = n + 11
    preds = eachlayer_output[-1]
    preds = torch.nn.functional.softmax(preds,dim=1)
    print(preds.shape)
    _,y_pred = torch.max(preds,1)
    print(y_pred.shape)
    correct += (y_pred == yb).sum().item()
    total += yb.size(0)
    y_pred = y_pred.cpu()
    predicted_output.append(y_pred.numpy())
    yb = yb.cpu()
    original_output.append(yb.numpy())
val_accuracy.append(correct / total)
print('Test Accuracy : {}'.format(correct / total))
for i in range(len(last_conv_pool_output)):
    print(last_conv_pool_output[i])
tsne_input = torch.cat(last_conv_pool_output,dim=0)
print(tsne_input.shape)

from tsnecuda import TSNE
X_embedded = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(tsne_input)
x = [11,12,13]

for i,ele in enumerate(x):
    print(i)
    print(ele)
for l in model.modules():
    print(l)
import numpy as np
predicted_output = np.concatenate(predicted_output,axis=0)
original_output = np.concatenate(original_output,axis=0)




print(classification_report(original_output,predicted_output))


import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greys):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(original_output,predicted_output)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize=(10, 10))
plot_confusion_matrix(cnf_matrix, classes=['N', 'S', 'V', 'F', 'Q'],normalize=True,
                      title='Confusion matrix, without normalization')
plt.show()
df_ptbdb_normal = pd.read_csv('../input/heartbeat/ptbdb_normal.csv', header=None)
df_ptbdb_abnormal = pd.read_csv('../input/heartbeat/ptbdb_abnormal.csv', header=None)

print(df_ptbdb_normal.info())
print(df_ptbdb_abnormal.info())
df_ptbdb = pd.concat([df_ptbdb_normal, df_ptbdb_abnormal], axis=0)
df_ptbdb.info()

inputs = df_ptbdb.values[:, :-1]
labels = df_ptbdb.values[:, -1].astype(int)

print(inputs.shape)
print(labels.shape)


c0 = np.argwhere(labels == 0).flatten()
c1 = np.argwhere(labels == 1).flatten()
x = np.arange(0, 187)*8/1000
plt.figure(figsize=(20,12))
plt.plot(x, inputs[c0, :][0], label="Cat. Normal")
plt.plot(x, inputs[c1, :][0], label="Cat. Abnormal")
plt.legend()
plt.title("1-beat ECG for every category", fontsize=20)
plt.ylabel("Amplitude", fontsize=15)
plt.xlabel("Time (ms)", fontsize=15)
plt.show()
x_train,y_train,x_test,y_test = train_val_split(inputs,labels,inputs.shape[0],0.1,100)

x_train,y_train,x_val,y_val = train_val_split(x_train,y_train,x_train.shape[0],0.2,100)


print(x_train.shape)
print(y_train.shape)

print(x_val.shape)
print(y_val.shape)

print(x_test.shape)
print(y_test.shape)
