import torch
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as cm
import sys
import time
import numpy as np
import math
import pandas as pd
from PIL import Image, ImageOps, ImageFilter
from datetime import datetime
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
import random
import datetime
import os

from sklearn import preprocessing 
from sklearn.model_selection import KFold

print(torch.__version__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
from mlxtend.data import loadlocal_mnist

train_x, train_y = loadlocal_mnist(
        images_path='../input/coic/train-images-idx3-ubyte', 
        labels_path='../input/coic/train-labels-idx1-ubyte')

test_x, test_y = loadlocal_mnist(
        images_path='../input/coic/test-images-idx3-ubyte', 
        labels_path='../input/coic/test-labels-idx1-ubyte')

final_x, final_y = loadlocal_mnist(
        images_path='../input/coic/final-test-images-idx3-ubyte', 
        labels_path='../input/coic/test-labels-idx1-ubyte')

print("Train images shape : ",train_x.shape)
print("Train labels shape : ",train_y.shape)

print("Test images shape : ",test_x.shape)
print("Test labels shape : ",test_y.shape)

print("Final images shape : ",final_x.shape)
def creatData(img_data,lable):
    train_data = []
    for i in range(len(img_data)):
        img =  img_data[i].reshape(28,28)
        lab = lable[i]
        train_data.append([img,lab])  
    return np.array(train_data)
train_data = creatData(train_x, train_y)
test_data = creatData(test_x, test_y)

final_data = creatData(final_x, final_y)
#train_data = train_data[0:10000]
def onehot(length, num):
    d = np.zeros(length)
    d[num] = 1
    return d
class creatDataset(Dataset):
    def __init__(self, data, transform):
        self.files = data[:,0]
        self.labels = data[:,1]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_array = self.files[idx].reshape(28,28)
        img = Image.fromarray(np.uint8(img_array))
        image = self.transform(img)
        lab = onehot(10,self.labels[idx])
        lab_num = self.labels[idx]
        return image,lab,lab_num
data_transformer = transforms.Compose([
                transforms.Resize((28,28)),
                transforms.ToTensor()
                ])
display_dataloader = DataLoader(creatDataset(train_data,data_transformer),
                               batch_size=32, 
                               shuffle=True,
                               num_workers=2, 
                               pin_memory=True)
plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.imshow(transforms.ToPILImage()(display_dataloader.dataset[15][0]).convert('RGB'))
plt.subplot(1,2,2)
plt.imshow(transforms.ToPILImage()(display_dataloader.dataset[78][0]).convert('RGB'))
print(np.argmax(display_dataloader.dataset[15][1]), np.argmax(display_dataloader.dataset[78][1]))
model = models.resnet101(pretrained=False, progress=True)

model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

model.fc = nn.Sequential(
          nn.Dropout(p=0.1, inplace=False),
          nn.Linear(in_features=2048, out_features=256, bias=True),
          #nn.Dropout(p=0.2, inplace=False),
          nn.Linear(in_features=256, out_features=10))

model.to(device)

def train(epoch,fnum,dloader):
    for step, (x, y, y2) in enumerate(dloader):
        data = Variable(x).cuda()   # batch x
        #y = np.array([np.argmax(y)])
        target = Variable(y2).cuda()   # batch y
        output = model(data)

        loss = loss_func(output, target.long())   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients
        if step==0:
            start = time.time()
            ti = 0
        elif step==100:
            ti = time.time()-start #total time = ti*(length/100)
            #print(ti)
            ti = ti*(len(dloader)/100)
        if step % 100 == 0:
            second = ti*(((len(dloader)-step)/len(dloader)))#*(5-epoch)*(4-fnum)
            print('Train Fold:{}/4  Ep: {}/15 [{}/{} ({:.0f}%)]\t Loss: {:.6f}\t Remain : {} '.
                     format(fnum+1,
                            epoch+1, 
                            step * len(data), 
                            len(dloader.dataset),
                            100.*step/len(dloader), 
                            loss.data.item(),
                            datetime.timedelta(seconds = int(second))))
        data.cpu()
        target.cpu()
        torch.cuda.empty_cache()
    print("Finish")
def val(dloader):
    los = []
    acc_num = 0
    for step, (x, y, y2) in enumerate(dloader):
        data = Variable(x).cuda()
        #y = np.array([np.argmax(y)])
        target = Variable(y2).cuda()
        with torch.no_grad():
            output = model(data) 
        loss = loss_func(output, target.long())
        los.append(loss.item())
        
        out = target.cpu().data.numpy().squeeze()
        pre = np.argmax(output.cpu().data.numpy().squeeze(),axis = 1)
        acc_num += (out == pre).sum()
             
        if step %100 == 0:
            print('[{}/{} ({:.1f}%)]'.format(step * len(data), 
                                        len(dloader.dataset),
                                        100.*step/len(dloader)))
    print("")
    print("Acc : [{}/{} ({:.1f}%)]".format(acc_num,
                                         len(dloader.dataset),
                                         100.*acc_num/len(dloader.dataset)))
    torch.cuda.empty_cache()
    los = np.array(los)
    avg_val_loss = los.sum()/len(los)
    print("Avg val loss: {:.8f}".format(avg_val_loss))
    print("")
    return 100.*acc_num/len(dloader.dataset),avg_val_loss
display_val_loss = []
display_val_acc = []

fold = KFold(n_splits = 4, random_state = 10)
for fold_num, (trn_idx, val_idx) in enumerate(fold.split(train_data)):
    train_d = train_data[trn_idx, :]
    train_v = train_data[val_idx, :]

    val_dataloader = DataLoader(creatDataset(train_v,data_transformer),batch_size=32, shuffle=False,num_workers=2, pin_memory=True)
    train_dataloader = DataLoader(creatDataset(train_d,data_transformer),batch_size=32, shuffle=True,num_workers=2, pin_memory=True)
    for epoch in range(15):
        ###########################################
        if epoch<5:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        elif (epoch>=5) and (epoch<10):
            optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.000002)
        #optimizer = torch.optim.Adam(model.parameters(), lr=0.00002/(2**epoch))
        #optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        loss_func = nn.CrossEntropyLoss() 
        #loss_func = torch.nn.MSELoss()
        ###########################################
        train(epoch,fold_num,train_dataloader) 
        ac,los = val(val_dataloader)
        display_val_acc.append(ac)
        display_val_loss.append(los)
fig = plt.figure()
plt.plot(display_val_loss)
fig = plt.figure()
plt.plot(display_val_acc)
test_dataloader = DataLoader(creatDataset(test_data,data_transformer),batch_size=32, shuffle=False,num_workers=2, pin_memory=True)
loss_func = nn.CrossEntropyLoss()
ac,los = val(test_dataloader)
final_dataloader = DataLoader(creatDataset(final_data,data_transformer),batch_size=32, shuffle=False,num_workers=2, pin_memory=True)
loss_func = nn.CrossEntropyLoss()
def final(dloader):
    ans = []
    index = 1
    for step, (x, y, y2) in enumerate(dloader):
        data = Variable(x).cuda()
        #y = np.array([np.argmax(y)])
        target = Variable(y2).cuda()
        with torch.no_grad():
            output = model(data) 
        #loss = loss_func(output, target.long())
        
        #out = target.cpu().data.numpy().squeeze()
        pre = np.argmax(output.cpu().data.numpy().squeeze(),axis = 1)
        
        for id in range(len(pre)):
            ans.append([index,pre[id]])
            index+=1
            
             
        if step %100 == 0:
            print('[{}/{} ({:.1f}%)]'.format(step * len(data), 
                                        len(dloader.dataset),
                                        100.*step/len(dloader)))
    torch.cuda.empty_cache()
    
    return ans
ans = final(final_dataloader)

sub =  pd.DataFrame(ans)
sub = sub.rename(index=str, columns={0: "id", 1: "Class"})
sub.to_csv('submission.csv', index=False)
torch.save(model.state_dict(), 'dl_course_c1_model.pkl')
!ls