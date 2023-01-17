# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory





import os

print(os.listdir("../input/aptos2019-blindness-detection/"))



# Any results you write to the current directory are saved as output.
# import sys

# package_dir = "../input/pretrainedmodels/pretrained-models/pretrained-models.pytorch-master/"

# sys.path.insert(0, package_dir)



# import pretrainedmodels
# import torch.nn as nn

# import torch

# model = pretrainedmodels.__dict__['vgg16'](pretrained=None)

# model.last_linear = nn.Sequential( nn.Linear(in_features=4096, out_features=5),

#                                     nn.Softmax()

#                                   )

# model



# model.load_state_dict(torch.load("../input/mymodel/vgg_stact_dict.pt"))

# from PIL import Image

# import matplotlib.pyplot as plt 

# train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

# c = 0

# imgs = []

# plt.figure(figsize=(15, 10/5*3))

# for i in range(10):

#     f = os.path.join('../input/aptos2019-blindness-detection/train_images/{0}.png'.format(train['id_code'][i]))

#     img = Image.open(f)

#     plt.subplot(2 ,5, i+1)

#     plt.imshow(img)  

# plt.show()

from __future__ import unicode_literals

from PIL import Image

import os

import torch

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

# %matplotlib inline

from sklearn.model_selection import train_test_split

from torchvision import transforms as tfs

from torch.utils.data import DataLoader,Dataset

class Config:

    data_dir = '../input/aptos2019-blindness-detection/'

    crop_size = 224

    train_batch_size = 64

    test_batch_size = 1

    lr = 1e-3

    momentum = 0.9

    epochs = 20

    print_every = 5

    

opt = Config()
def read_file(data_dir, split = 'train'):

    file = os.path.join(data_dir + ('test.csv' if split is 'test' else 'train.csv' ))

    dataset = pd.read_csv(file)

    if split is 'test':

        data = [os.path.join(data_dir, 'test_images/{0}.png' .format(dataset.iloc[i].values[0]) )

                for i in range(len(dataset))]

        label = 0

        return data, label

    else :

        data =[os.path.join(data_dir, 'train_images/{0}.png'.format(dataset.iloc[i].values[0])) 

               for i in range(len(dataset))]

        label = dataset.iloc[:,1].values 

        train_data, eval_data, train_label, eval_label = train_test_split(data, label)

        if split is 'eval':

            return eval_data, eval_label

        else:

            return train_data, train_label



def transforms(img, crop_size):

    img_tfs = tfs.Compose([

        tfs.RandomResizedCrop(crop_size),

        tfs.RandomHorizontalFlip(p=0.2),

#         tfs.RandomRotation(degrees = (30,90)),

        tfs.ToTensor(),

        tfs.Normalize([0.5, 0.5 ,0.5],[0.5, 0.5, 0.5])

        

    ])

    img = img_tfs(img)

    return img



class APTOSSet(Dataset):

    def __init__(self, transform , split = 'train',

        data_dir=opt.data_dir,crop_size=opt.crop_size):

        data_list, label = read_file(data_dir, split = split)

        self.transform = transform

        self.data_list = data_list

        self.label = label

        self.crop_size = crop_size

        self.split = split

    def __getitem__(self, idx):

        img = self.data_list[idx]

        img = Image.open(img) 

        img = transforms(img, self.crop_size)

        if self.split is 'test' :

            return img

        else:

            label = self.label[idx]

            return img , label



    def __len__(self):

        return len(self.data_list)



train_set = APTOSSet(split='train', transform=transforms)

eval_set  = APTOSSet(split='eval', transform= transforms)

test_set = APTOSSet(split='test',transform= transforms)

APT_train = DataLoader(train_set, opt.train_batch_size, shuffle=True,num_workers= 0)

APT_eval = DataLoader(eval_set, opt.train_batch_size, shuffle=True,num_workers= 0)

APT_test = DataLoader(test_set, opt.test_batch_size, shuffle=False,num_workers= False)

# for inputs, labels in APT_train:

#     print(inputs)



#可视化

# vis_trains = []

# for i in range(10):

#     img, _ = train_set[i]

#     vis_trains.append(img)

# plt.figure(figsize=(15,len(vis_trains)/5*3))

# for c in range(len(vis_trains)):

#     plt.subplot(len(vis_trains)/5, 5, c+1)

#     plt.imshow(vis_trains[c])

# plt.show()

    
# import  torchvision.models  as models

# import torch.nn as nn

# model = pretrainedmodels.__dict__['vgg16'](pretrained=None)

model = torch.load('../input/mymodel/vgg16_model.pt')

for params in model.parameters():

    params.requires_grad = False

model.eval()

# from torch import optim

# import torch

# criterion = nn.NLLLoss()

# optimizer = optim.SGD(model.parameters(), lr = opt.lr, momentum = opt.momentum)

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

device

model.to(device)
# model.to(device)



# step = 0

# run_loss = 0

# lst_tloss = []

# ac_train = []

# for epoch in range(opt.epochs):

#     train_correct = 0

#     for inputs, labels in APT_train:

#         step +=1

#         inputs = inputs.to(device)

#         labels = labels.to(device)

#         optimizer.zero_grad()

#         outputs = model.forward(inputs)

#         loss = criterion(outputs, labels)

#         pred = outputs.data.max(1, keepdim=True)[1]

#         loss.backward()

#         optimizer.step()

#         train_correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()

#         train_accuracy = train_correct/len(APT_train.dataset)

#         run_loss += loss.item()

#     lst_tloss.append(run_loss)

#     ac_train.append(train_accuracy)

     

#     test_loss = 0

#     accuracy = 0

#     model.eval()

#     with torch.no_grad():

#         for inputs, labels in APT_eval:

#             inputs, labels = inputs.to(device), labels.to(device)

#             outputs = model.forward(inputs)

#             pred = outputs.data.max(1, keepdim=True)[1]

#             batch_loss = criterion(outputs, labels)

#             test_loss += batch_loss

#             accuracy += pred.eq(labels.data.view_as(pred)).cpu().sum().item()

#     print(f"Epoch {epoch+1}/{opt.epochs}.. "

#           f"Train loss: {run_loss/len(APT_train):.3f}.. "

#           f"Train accuracy: {train_correct/len(APT_train.dataset):.3f}.."

#           f"Test loss: {test_loss/len(APT_eval):.3f}.. "

#           f"Test accuracy: {accuracy/len(APT_eval.dataset):.3f}")

#     run_loss = 0

#     model.train()

                    
# import matplotlib.pyplot as plt

# a=[2,6,5,4,3]

# plt.plot(np.arange(1, 6), a, color= 'b')

# plt.show()
# model.to(device)



# step = 0

# run_loss = 0

# lst_tloss = []

# ac_train = []

# test_losses = []

# test_acces = []

# for epoch in range(opt.epochs):

#     train_correct = 0

#     for inputs, labels in APT_train:

#         step +=1

#         inputs = inputs.to(device)

#         labels = labels.to(device)

#         optimizer.zero_grad()

#         outputs = model.forward(inputs)

#         pred = outputs.data.max(1, keepdim=True)[1]

#         loss = criterion(outputs, labels)

#         loss.backward()

#         optimizer.step()

#         train_correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()

#         run_loss += loss.item()

       

#         if step % opt.print_every ==0:

#             test_loss = 0

#             correct = 0

#             model.eval()

#             with torch.no_grad():

#                 for inputs, labels in APT_eval:

#                     inputs, labels = inputs.to(device), labels.to(device)

#                     outputs = model.forward(inputs)

#                     pred = outputs.data.max(1, keepdim=True)[1]

#                     batch_loss = criterion(outputs, labels)

#                     test_loss += batch_loss

#                     correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()

#             accuracy = float(correct)/ len(APT_eval.dataset)

#             test_losses.append(test_loss/ len(APT_eval))  

#             test_acces.append(accuracy)

            

#             lst_tloss.append(run_loss/opt.print_every)

#             ac_train.append(train_correct/(opt.print_every*opt.train_batch_size))

#             print(f"Epoch {epoch+1}/{opt.epochs}.. "

#                   f"Train loss: {run_loss/opt.print_every:.3f}.. "

#                   f"Train accuracy: {train_correct/(opt.print_every*opt.train_batch_size):.3f}.."

#                   f"Test loss: {test_loss/len(APT_eval):.3f}.. "

#                   f"Test accuracy: {accuracy:.3f}")

#             run_loss = 0

#             train_correct = 0

#             model.train()

#             torch.save(model.state_dict(),'vgg_stact_dict.bin')

#             torch.save(model, 'vgg16_model.bin')

            

# import matplotlib.pyplot as plt

# # % inline matplotlib

# plt.subplot(211)

# plt.title('loss')

# plt.plot(np.arange(1, len(lst_tloss)+1), lst_tloss, color= 'b')

# plt.plot(np.arange(1, len(test_losses)+1), test_losses, color='r')



# plt.subplot(212)

# plt.title('acc')

# plt.plot(np.arange(1, len(ac_train)+1), ac_train, color ='b')

# plt.plot(np.arange(1, len(test_acces)+1), test_acces, color = 'r')

# plt.show()
def test_predict(model):

    model.eval()

    prediction = []

    for data in APT_test:

        data = data.to(device)

        outputs = model(data)

        pred = outputs.data.max(1, keepdim=True)[1]

        prediction.append(int(pred))

    return prediction



sub = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

sub['diagnosis'] = test_predict(model)

sub.to_csv('submission.csv', index= False)

sub.head()