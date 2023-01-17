!pip install efficientnet_pytorch



import torch

import torch.nn as nn #网络的基础包

import torch.nn.functional as F

import torch.optim as optim  #优化器包

import torch.utils.data as Data    #批量训练用的包

import torch.utils.model_zoo as model_zoo

from torch.utils.data import Dataset



from torchvision import models 

from torchvision import transforms as tfs



import numpy as np  # 科学计算



import pandas as pd  # 数据分析

from pandas import Series, DataFrame



import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline



import cv2



from PIL import Image



import seaborn as sns



from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.model_selection import train_test_split



from scipy.special import softmax



import os



import tqdm

from tqdm.notebook import tqdm



from efficientnet_pytorch import EfficientNet



from transformers import *



from albumentations import *

from albumentations.pytorch import ToTensor



import gc



import warnings



from sklearn import metrics

warnings.filterwarnings("ignore")

# coding=utf-8
class Net(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = EfficientNet.from_pretrained('efficientnet-b0')

        

        self.domain_layers = nn.Sequential(nn.Conv2d(1280, 1280//16, kernel_size=1),  nn.ReLU(True),

                                           nn.Conv2d(1280//16, 1280//16, kernel_size=1),  nn.ReLU(True), 

                                           nn.Conv2d(1280//16, 1280, kernel_size=1))

        self.class_layers=nn.Sequential(nn.Conv2d(1280, 600, kernel_size=1), nn.BatchNorm2d(600), nn.ReLU(True),

                                          nn.Conv2d(600, 345, kernel_size=1), nn.BatchNorm2d(345), nn.ReLU(True))

        

        self.se_layers = nn.Sequential(nn.Conv2d(1280, 1280//16, kernel_size=1),  nn.ReLU(True), nn.Conv2d(1280//16, 1280, kernel_size=1))

        

    def feature_normalize(self,data):

        mu = data.mean()

        std = data.std()

        return (data - mu)/std

    

    def forward(self, x, domain, a):

        feat = self.model.extract_features(x)

        feat = F.avg_pool2d(feat, feat.size()[2:]).reshape(-1, 1280)

        feat = feat.unsqueeze(2).unsqueeze(3)

     

        domain_feature = self.domain_layers(feat)

        

        with torch.no_grad():

            new_domain = self.feature_normalize(a*domain+(1-a)*domain_feature)



        new_feature = feat*self.se_layers(new_domain)



        return_feature = self.class_layers(new_feature).squeeze(2).squeeze(2)



        return return_feature,new_domain

labels=[]

for dirname, _, filenames in os.walk('/kaggle/input/test33/quickdraw/quickdraw'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))

    labels.append(dirname.split('/')[-1])

labels=labels[1:]

print(len(labels))
folder_path='/kaggle/input/test33/'

folders=['clipart/','infograph/','painting/','quickdraw/','real/','sketch/']
random_type = folders[np.random.randint(0,len(folders))]

random_label = labels[np.random.randint(0,len(labels))]



for dirname, _, filenames in os.walk(folder_path+random_type+random_type+random_label+'/'):

    pass



sample_file = filenames[np.random.randint(0,len(filenames))]

sample_path=  os.path.join(folder_path+random_type+random_type+random_label+'/',sample_file)



sample_img = cv2.imread(sample_path)



plt.imshow(sample_img)

plt.show()

random_type,random_label
AUG_TRAIN = Compose([

    VerticalFlip(p=0.5),

    HorizontalFlip(p=0.5),

#     ShiftScaleRotate(rotate_limit=25.0, p=0.7),

#     OneOf([IAAEmboss(p=1),

#     IAASharpen(p=1),

#     Blur(p=1)], p=0.5),

#     IAAPiecewiseAffine(p=0.5),

    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), always_apply=True),

    ToFloat(max_value=255),

    ToTensor()

], p=1)



AUG_TEST = Compose([

    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), always_apply=True),

    ToFloat(max_value=255),

    ToTensor()

], p=1)
folders=['clipart/','infograph/','painting/','quickdraw/','real/','sketch/']

train_txts = ['/kaggle/input/test22/'+folder[:-1]+'_train.txt' for folder in folders]

test_txts = ['/kaggle/input/test22/'+folder[:-1]+'_test.txt' for folder in folders]



train_imgs= [[],[],[],[],[]]

train_labels=[[],[],[],[],[]]



test_imgs= [[],[],[],[],[],[]]

test_labels=[[],[],[],[],[],[]]



for train_txt in train_txts:

    file_object = open(train_txt)

    try:

        file_context = file_object.read()

    finally:

        file_object.close()

    data=file_context.split('\n')[:-1]

    for index in range(len(data)):

        if data[index][0:3]=='cli':

            data[index]='clipart/'+data[index]  

            train_imgs[0].append('/kaggle/input/test33/'+data[index].split()[0])

            train_labels[0].append(int(data[index].split()[1]))

        elif data[index][0:3]=='inf':

            data[index]='infograph/'+data[index]

            train_imgs[1].append('/kaggle/input/test33/'+data[index].split()[0])

            train_labels[1].append(int(data[index].split()[1]))

        elif data[index][0:3]=='pai':

            data[index]='painting/'+data[index]  

            train_imgs[2].append('/kaggle/input/test33/'+data[index].split()[0])

            train_labels[2].append(int(data[index].split()[1]))

        elif data[index][0:3]=='qui':

            data[index]='quickdraw/'+data[index]

            train_imgs[3].append('/kaggle/input/test33/'+data[index].split()[0])

            train_labels[3].append(int(data[index].split()[1]))

        elif data[index][0:3]=='rea':

            data[index]='real/'+data[index]

            train_imgs[4].append('/kaggle/input/test33/'+data[index].split()[0])

            train_labels[4].append(int(data[index].split()[1]))

        elif data[index][0:3]=='ske':

            data[index]='sketch/'+data[index]

            test_imgs[5].append('/kaggle/input/test33/'+data[index].split()[0])

            test_labels[5].append(int(data[index].split()[1]))

            



for test_txt in test_txts:

    file_object = open(test_txt)

    try:

        file_context = file_object.read()

    finally:

        file_object.close()

    data=file_context.split('\n')[:-1]

    for index in range(len(data)):

        if data[index][0:3]=='cli':

            data[index]='clipart/'+data[index]  

            test_imgs[0].append('/kaggle/input/test33/'+data[index].split()[0])

            test_labels[0].append(int(data[index].split()[1]))

        elif data[index][0:3]=='inf':

            data[index]='infograph/'+data[index]

            test_imgs[1].append('/kaggle/input/test33/'+data[index].split()[0])

            test_labels[1].append(int(data[index].split()[1]))

        elif data[index][0:3]=='pai':

            data[index]='painting/'+data[index]  

            test_imgs[2].append('/kaggle/input/test33/'+data[index].split()[0])

            test_labels[2].append(int(data[index].split()[1]))

        elif data[index][0:3]=='qui':

            data[index]='quickdraw/'+data[index]

            test_imgs[3].append('/kaggle/input/test33/'+data[index].split()[0])

            test_labels[3].append(int(data[index].split()[1]))

        elif data[index][0:3]=='rea':

            data[index]='real/'+data[index]

            test_imgs[4].append('/kaggle/input/test33/'+data[index].split()[0])

            test_labels[4].append(int(data[index].split()[1]))

        elif data[index][0:3]=='ske':

            data[index]='sketch/'+data[index]

            test_imgs[5].append('/kaggle/input/test33/'+data[index].split()[0])

            test_labels[5].append(int(data[index].split()[1]))
for i in range(len(train_imgs)): #range(6)

    len_ = len(train_imgs[i]) 

    train_imgs[i] = train_imgs[i][:-(len_%80)]

    train_labels[i] = train_labels[i][:-(len_%80)]

    

for i in range(len(test_imgs)): 

    len_ = len(test_imgs[i])

    test_imgs[i] = test_imgs[i][:-(len_%80)]

    test_labels[i] = test_labels[i][:-(len_%80)]

    

print(list(len(train_imgs[i]) for i in range(len(train_imgs))))

print(list(len(train_labels[i]) for i in range(len(train_labels))))

print(list(len(test_imgs[i]) for i in range(len(test_imgs))))

print(list(len(test_labels[i]) for i in range(len(test_labels))))
test_size = 16000



train_imgs = [train_imgs[i][0:test_size] for i in range(len(train_imgs))]

train_labels = [train_labels[i][0:test_size] for i in range(len(train_labels))]

test_imgs = [test_imgs[i][0:test_size//2] for i in range(len(test_imgs))]

test_labels = [test_labels[i][0:test_size//2] for i in range(len(test_labels))]



print(list(len(train_imgs[i]) for i in range(len(train_imgs))))

print(list(len(train_labels[i]) for i in range(len(train_labels))))

print(list(len(test_imgs[i]) for i in range(len(test_imgs))))

print(list(len(test_labels[i]) for i in range(len(test_labels))))
class HWDataset(Dataset):



    def __init__(self, data_path,data_label, augmentations):

        self.data_path = data_path

        self.data_label = data_label

        self.augment = augmentations

        

    def __len__(self):

        return len(self.data_path)



    def __getitem__(self, idx):

        path, label = self.data_path[idx],self.data_label[idx]

        im = cv2.imread(path)

        im = cv2.resize(im,(224,224))

        im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

        im = self.augment(image=im)

        im = im['image']

        

        return im, label
batch_size = 80

num_workers = 2



train_datasets = [HWDataset(train_imgs[i],train_labels[i],AUG_TRAIN) for i in range(5)]

train_loaders = [torch.utils.data.DataLoader(train_datasets[i],batch_size=batch_size,\

                                           num_workers=num_workers,shuffle=True) for i in range(5)]



test_datasets = [HWDataset(test_imgs[i],test_labels[i],AUG_TEST) for i in range(6)]

test_loaders = [torch.utils.data.DataLoader(test_datasets[i],batch_size=batch_size,\

                                           num_workers=num_workers,shuffle=True) for i in range(6)]
for i_batch,(img,label) in enumerate(train_loaders[0]):

    print(i_batch)

    print(img.shape)

    print(label.shape)

    if i_batch==0:

        break
model=Net().cuda()

criterion = nn.CrossEntropyLoss().cuda()  #分类交叉熵Cross-Entropy 作损失函数

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7)



device = 'cuda'



domain_info = [torch.rand(80,1280,1,1).cuda() for i in range(6)]
data = open("log.txt","a")

# data.write("test\ntest\ntest")

# data.close()
train_domains=['clipart','infograph','painting','quickdraw','real']

test_domains=['clipart','infograph','painting','quickdraw','real','sketch']
# model.load_state_dict(torch.load("../input/test1/2_loss3.57e+02_acc5d_48.6_acc5wd_80.6_acc1wd_0.0.pth"))
num_epochs = 12



for epoch in range(num_epochs):

    print('Epoch {}/{}'.format(epoch+1, num_epochs))

    print('-' * 10)

    data.write('Epoch {}/{}\n'.format(epoch+1, num_epochs)+'-' * 10)

    

    data.write('train 5 with/without domain:  \n')

    train_loss, test_loss1, test_loss2,test_loss3 = [], [], [],[]

    model.train()

    running_loss = 0

    for i in range(5):

        tk0 = tqdm(train_loaders[i], total=int(len(train_loaders[i])))

        for im,labels in tk0:

            inputs = im.to(device, dtype=torch.float)

            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()

            

            if np.random.rand(1)>0.5:

                outputs,new_domain = model(inputs.cuda(),domain_info[i],1-1/((epoch+1)*1.5))

                domain_info[i] = new_domain

            else:

                outputs,new_domain = model(inputs.cuda(),domain_info[5],1-1/((epoch+1)*1.5))

                domain_info[5] = new_domain

            

            loss = criterion(outputs, labels)

            loss.backward()

    

            optimizer.step()

            running_loss += loss.item()

            tk0.set_postfix(loss=(loss.item()))



        epoch_loss = running_loss / (len(train_loaders[i])/batch_size)

        train_loss.append(epoch_loss)

        

        data.write(train_domains[i]+' '+str(epoch_loss)+'\n')

    data.write(str(np.mean(train_loss))+'\n')

        

    print('Training Loss: {:.8f}'.format(np.mean(train_loss)))

    

    



###################################### test with domain info ###########################





    acc1,acc2,acc3=[],[],0.0

    

    data.write('test 5 with domain:  \n')

    for i in range(5):

        tk1 = tqdm(test_loaders[i], total=int(len(test_loaders[i])))

        model.eval()

        running_loss = 0.0

        y, preds = [], []

        with torch.no_grad():

            for (im, labels) in tk1:

                inputs = im.to(device, dtype=torch.float)

                labels = labels.to(device, dtype=torch.long)

                

                outputs,new_domain = model(inputs.cuda(),domain_info[i],1-1/((epoch+1)*1.5))

                

                loss = criterion(outputs, labels)

                y.extend(labels.cpu().numpy().astype(int))

                preds.extend(F.softmax(outputs, 1).cpu().numpy())

                running_loss += loss.item()

                tk1.set_postfix(loss=(loss.item()))



            epoch_loss = running_loss / (len(test_loaders[i])/batch_size)

            test_loss1.append(epoch_loss)

            preds = np.array(preds)

            # convert multiclass labels to binary class

            y = np.array(y)

            labels = preds.argmax(1)

            for class_label in np.unique(y):

                idx = y == class_label

                acc = (labels[idx] == y[idx]).astype(np.float).mean()*100

    #             print('accuracy for class', labels[class_label], 'is', acc)



            acc = (labels == y).mean()*100

            acc_withd = acc

            acc1.append(acc_withd)

            new_preds = np.zeros((len(preds),))

            temp = preds[labels != 0, 1:]

            new_preds[labels != 0] = temp.sum(1)

            new_preds[labels == 0] = 1 - preds[labels == 0, 0]

            y = np.array(y)

            y[y != 0] = 1

            print(f'test domain {test_domains[i]} withd loss: {epoch_loss:.3},  Acc: {acc_withd:.3}')

            data.write(test_domains[i]+' '+str(epoch_loss)+' '+str(acc_withd)+'\n')

    data.write(str(np.mean(acc1))+'\n')

    

    

    data.write('test 5 without domain:  \n')

    for i in range(5):

        tk1 = tqdm(test_loaders[i], total=int(len(test_loaders[i])))

        model.eval()

        running_loss = 0

        y, preds = [], []

        with torch.no_grad():

            for (im, labels) in tk1:

                inputs = im.to(device, dtype=torch.float)

                labels = labels.to(device, dtype=torch.long)



                outputs,new_domain = model(inputs.cuda(),domain_info[5],1-1/((epoch+1)*1.5))



                loss = criterion(outputs, labels)

                y.extend(labels.cpu().numpy().astype(int))

                preds.extend(F.softmax(outputs, 1).cpu().numpy())

                running_loss += loss.item()

                tk1.set_postfix(loss=(loss.item()))



            epoch_loss = running_loss / (len(test_loaders[i])/batch_size)

            test_loss2.append(epoch_loss)

            preds = np.array(preds)

            # convert multiclass labels to binary class

            y = np.array(y)

            labels = preds.argmax(1)

            for class_label in np.unique(y):

                idx = y == class_label

                acc = (labels[idx] == y[idx]).astype(np.float).mean()*100

    #             print('accuracy for class', labels[class_label], 'is', acc)



            acc = (labels == y).mean()*100

            acc_withoutd = acc

            acc2.append(acc_withd)

            new_preds = np.zeros((len(preds),))

            temp = preds[labels != 0, 1:]

            new_preds[labels != 0] = temp.sum(1)

            new_preds[labels == 0] = 1 - preds[labels == 0, 0]

            y = np.array(y)

            y[y != 0] = 1

            data.write(test_domains[i]+' '+str(epoch_loss)+' '+str(acc_withoutd)+'\n')

            print(f'test domain {test_domains[i]} withoutd loss: {epoch_loss:.3},  Acc: {acc_withoutd:.3}')            

    data.write(str(np.mean(acc2))+'\n')

    

    data.write('testing3:  \n')

    

    tk1 = tqdm(test_loaders[5], total=int(len(test_loaders[5])))

    model.eval()

    running_loss = 0

    y, preds = [], []

    with torch.no_grad():

        for (im, labels) in tk1:

            inputs = im.to(device, dtype=torch.float)

            labels = labels.to(device, dtype=torch.long)

            

            outputs,new_domain = model(inputs.cuda(),domain_info[5],1-1/((epoch+1)*1.5))

            

            loss = criterion(outputs, labels)

            y.extend(labels.cpu().numpy().astype(int))

            preds.extend(F.softmax(outputs, 1).cpu().numpy())

            running_loss += loss.item()

            tk1.set_postfix(loss=(loss.item()))



        epoch_loss = running_loss / (len(test_loaders[5])/batch_size)

        test_loss3.append(epoch_loss)

        preds = np.array(preds)

        # convert multiclass labels to binary class

        y = np.array(y)

        labels = preds.argmax(1)

        for class_label in np.unique(y):

            idx = y == class_label

            acc = (labels[idx] == y[idx]).astype(np.float).mean()*100

#             print('accuracy for class', labels[class_label], 'is', acc)



        acc = (labels == y).mean()*100

        acc_3 = acc

        new_preds = np.zeros((len(preds),))

        temp = preds[labels != 0, 1:]

        new_preds[labels != 0] = temp.sum(1)

        new_preds[labels == 0] = 1 - preds[labels == 0, 0]

        y = np.array(y)

        y[y != 0] = 1

        print(f'test domain real withoutd loss: {epoch_loss:.3},  Acc: {acc_3:.3}')

        data.write(test_domains[5]+' '+str(epoch_loss)+' '+str(acc_3)+'\n')

    data.write(str(acc_3)+'\n')

    

    torch.save(model.state_dict(),f"{epoch}_loss{np.mean(train_loss):.3}_acc5d_{np.mean(acc1):.3}_acc5wd_{np.mean(acc2):.3}_acc1wd_{acc3:.3}.pth")

data.close()