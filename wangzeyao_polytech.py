import torch

import torch.nn as nn

import torch.optim as optim

from torch.optim import lr_scheduler

from torch.autograd import Variable

import numpy as np

import torchvision

from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt

import cv2 as cv

import time

import os

import copy

from glob2 import glob

from tqdm import tqdm_notebook

from IPython.display import display

from PIL import Image

import numpy as np

import ignite



print(torch.__version__)



train_path = '/kaggle/input/polytech-ds-2019/polytech-ds-2019/training/'

validation_path = '/kaggle/input/polytech-ds-2019/polytech-ds-2019/validation'

test_path = '/kaggle/input/polytech-ds-2019/polytech-ds-2019/kaggle_evaluation'
class Food11Dataset(torch.utils.data.Dataset):

  

  def __init__(self, img_dir,model,sample_num = None):

    

    super().__init__()

    

    # store directory names

    self.sample_num = sample_num

    self.img_dir = img_dir

    if self.sample_num == None:

        self.img_names = [x.split("/")[-1] for x in glob(img_dir + "/*")]

    else:

        self.img_names = [x.split("/")[-1] for x in glob(img_dir + "/*")[0:self.sample_num]]

    

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                     std=[0.229, 0.224, 0.225])

    

    self.transform_train = transforms.Compose([transforms.RandomResizedCrop(224),

                                         transforms.RandomHorizontalFlip(),

                                         transforms.ToTensor(),normalize])

    

    self.transform_val  = transforms.Compose([transforms.Resize((224,224)),

                                              transforms.ToTensor(),normalize])

    self.transform_show  = transforms.Compose([transforms.Resize((224,224)),

                                              transforms.ToTensor()])

    self.model = model

    

  

  def __len__(self):

    return len(self.img_names)

    

  def __getitem__(self,i):

    return self._read_img_and_gt(i)

  

  def _read_img_and_gt(self, i):

    img = Image.open(self.img_dir + "/" + self.img_names[i])

    if self.model == "train":

        gt = self.img_names[i].split('_')[0]

        return self.transform_train(img),int(gt)

    elif self.model == 'validation':

        gt = self.img_names[i].split('_')[0]

        return self.transform_val(img),int(gt)

    elif self.model == 'test':

        return self.transform_val(img),self.img_names[i].split('.')[0]

    elif self.model == 'show':

        gt = self.img_names[i].split('_')[0]

        return self.transform_val(img),self.transform_show(img)
food11_train = Food11Dataset(train_path,model='train')



food11_samples = Food11Dataset(train_path,model='train',sample_num = int(len(food11_train)/10))



food11_val = Food11Dataset(validation_path,model = 'validation')

print(len(food11_train),len(food11_val),len(food11_samples))

data = food11_val[3]

img = transforms.ToPILImage()(data[0]).convert('RGB')

plt.imshow(img)
train_dl = torch.utils.data.DataLoader(food11_train,batch_size=32,num_workers=2,shuffle=True)

val_dl = torch.utils.data.DataLoader(food11_val,batch_size=32,num_workers=2)

samples_dl = torch.utils.data.DataLoader(food11_samples,batch_size=32,num_workers=2,shuffle=True)
# net = torchvision.models.vgg16_bn(pretrained=True)  

# net = torchvision.models.wide_resnet50_2(pretrained = True)

net = torchvision.models.resnext101_32x8d(pretrained=True)

print(net)
print(net.fc)

in_features = net.fc.in_features

features = list(net.fc.children())[:-1]

features.extend([nn.Linear(in_features,11),nn.LogSoftmax(dim=1)])

net.fc = nn.Sequential(*features)

print(net)
net.load_state_dict(torch.load('../input/net-weight/resnext101_zhongming40.pt'))
def unfreezClassifier(net):

    for child in net.children():

        for param in child.parameters():

            param.requires_grad = False

    for param in net.fc.parameters():

        param.requires_grad = True

    return net
N_EPOCHS = 2





criterion = nn.NLLLoss()

# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9,weight_decay = 0.0005)



large_lr_params = list(map(id, net.fc.parameters())) # return parameters‘ adress 

small_lr_params = filter(lambda p: id(p) not in large_lr_params, net.parameters()) 

optimizer = optim.SGD([

{'params': small_lr_params},

{'params': net.fc.parameters(), 'lr': 0.001}], 0.0001, momentum=0.9, weight_decay=1e-4)



# optimizer = optim.Adam(net.parameters())

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.1,patience=5)

# scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max = (10))
net.cuda()

## NUMBER OF EPOCHS TO TRAIN





epoch_loss, epoch_acc, epoch_val_loss, epoch_val_acc,lr = [], [], [], [],[]

best_val = 0

best_model_wts = copy.deepcopy(net.state_dict())

data_dl = train_dl

train_data = food11_train

for e in range(N_EPOCHS):



    print("EPOCH:", e)



    running_loss = 0

    running_accuracy = 0



    net.train()



    for i, batch in enumerate(tqdm_notebook(data_dl)):

        x = batch[0]

        labels = batch[1]



        if torch.cuda.is_available():

            x = x.cuda()

            labels = labels.cuda()



        y = net(x)



        loss = criterion(y, labels)



        optimizer.zero_grad()



        loss.backward()



        optimizer.step()



        with torch.no_grad():

            running_loss += loss.item()

            running_accuracy += (y.max(1)[1] == labels).sum().item()



    print("Training accuracy:", running_accuracy / float(len(train_data)),

          "Training loss:", running_loss / float(len(train_data)))



    epoch_loss.append(running_loss / len(train_data))

    epoch_acc.append(running_accuracy / len(train_data))





    net.eval()



    running_val_loss = 0

    running_val_accuracy = 0



    for i, batch in enumerate(tqdm_notebook(val_dl)):

        with torch.no_grad():

            x = batch[0]

            labels = batch[1]



            x = x.cuda()

            labels = labels.cuda()



            y = net(x)



            # Compute the loss

            loss = criterion(y, labels)



            running_val_loss += loss.item()

            running_val_accuracy += (y.max(1)[1] == labels).sum().item()



    print("Validation accuracy:", running_val_accuracy / float(len(food11_val)),

          "Validation loss:", running_val_loss / float(len(food11_val)))

    if running_val_accuracy > best_val:

        best_val = running_val_accuracy

        best_model_wts = copy.deepcopy(net.state_dict())



    epoch_val_loss.append(running_val_loss / len(food11_val))

    epoch_val_acc.append(running_val_accuracy / len(food11_val))

#     lr.append(scheduler.get_lr())

    scheduler.step((running_val_loss / len(food11_val)))

    
import matplotlib.pyplot as plt

import numpy as np



x = np.arange(N_EPOCHS)

plt.figure()

plt.plot(x,epoch_acc,label='train')

plt.plot(x, epoch_val_acc,label = 'val')

plt.legend()

plt.title('Accuracy')



plt.figure()

plt.plot(x, epoch_loss,label='train')

plt.plot(x, epoch_val_loss,label='val')

plt.legend()

plt.title('Loss')

food11_show = Food11Dataset(validation_path,model = 'show')

show_dl = torch.utils.data.DataLoader(food11_show,batch_size=32,num_workers=2,shuffle=True)

net.cuda()

iter_dl = iter(show_dl)

batch = next(iter_dl)

y = net(batch[0].cuda())

def display_tensor(t):

  trans = transforms.ToPILImage()

  display(trans(t))

for i in range(5):

    display_tensor(batch[1][i,:,:,:])

    print(y.max(1)[1][i].item())
def confusion_matrix():

    net.cuda()

    net.eval()

    

    right_count = np.zeros((11))

    total = np.zeros((11))

    confusion_matrix = np.zeros((11,11))



    for i, batch in enumerate(tqdm_notebook(eval_dl)):



      with torch.no_grad():

        x = batch[0]

        label = batch[1].item()



        x = x.cuda()



        y = net(x)

        out_put = y.max(1)[1].item()

        if not label == out_put:

            confusion_matrix[label][out_put] += 1

    return confusion_matrix
import seaborn as sn

import pandas as pd

eval_dl = torch.utils.data.DataLoader(food11_val,batch_size=1,num_workers=2)

matrix = confusion_matrix()



labels = ["Bread","Dairy products", "Dessert", "Egg", "Fried food", "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup", "Vegetable/Fruit"] # labels for axis

df_cm = pd.DataFrame(matrix)

sn.heatmap(df_cm, xticklabels=labels, yticklabels=labels)
food11_test = Food11Dataset(test_path,model = 'test')

print(len(food11_test))

test_dl = torch.utils.data.DataLoader(food11_test,batch_size=1,num_workers=1)
def test(test_dl,net):

    net.cuda()

    net.eval()

    headers = ['Id','Category']

    rows=[]

    for i, batch in enumerate(tqdm_notebook(test_dl)):



      with torch.no_grad():

        # Get a batch from the dataloader

        x = batch[0]

        id = batch[1][0]

        id = np.int64(id)



        # move the batch to GPU

        x = x.cuda()



        # Compute the network output

        y = net(x)



        out_put = y.max(1)[1].item()

        out_put = ' '+str(out_put)

        row = {'Id':id,'Category':out_put}

        rows.append(row)



    print(len(rows))

    return headers,rows

    

headers,rows = test(test_dl,net)
import csv

with open('submission.csv','w',newline='') as f:

    f_csv = csv.DictWriter(f, headers)

    f_csv.writeheader()

    f_csv.writerows(rows)
from IPython.display import FileLink

FileLink(r'submission.csv')