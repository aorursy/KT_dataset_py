## My first kernel on Kaggle :)

## Credits 

## https://www.kaggle.com/abhishek/very-simple-pytorch-training-0-59

import pandas as pd

import time

import torchvision

import torch.nn as nn

from tqdm import tqdm_notebook as tqdm

from PIL import Image, ImageFile

from torch.utils.data import Dataset

import torch

import torch.optim as optim

from torchvision import transforms

import torch.optim.lr_scheduler as lr_scheduler

device = torch.device("cuda:0")

ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

#!pip install pretrainedmodels

#import  pretrainedmodels

from sklearn.model_selection import train_test_split
## read the files

train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

sub = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')
## split train data into train and validation data 

X_train, X_val, y_train, y_val = train_test_split(train['id_code'], train['diagnosis'], test_size=0.33, random_state=42)

df = pd.DataFrame({'id_code': X_train,'diagnosis': y_train})

df.to_csv( 'train.csv' , index=False)
df = pd.DataFrame({'id_code': X_val,'diagnosis': y_val})

df.to_csv( 'val.csv' , index=False)
train = pd.read_csv('train.csv')

val = pd.read_csv('val.csv')


from PIL import Image

from torch.utils.data import Dataset



class AptosDataset(Dataset):

    def __init__(self, 

                 csv_file, 

                 root_dir, 

                 transform=None):

        self.data = pd.read_csv(csv_file)

        self.root_dir = root_dir

        self.transform = transform



    def __len__(self):

        return len(self.data)



    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir, 

                                self.data.loc[idx, 'id_code'] + '.png')

        image = Image.open(img_name)

        #image = image.resize((256, 256), resample=Image.BILINEAR)

        label_tensor = torch.tensor(self.data.loc[idx, 'diagnosis'])



        if self.transform:

            image = self.transform(image)



        return {'image': image,

                'labels': label_tensor

                }
import torch.nn as nn

import torchvision.models as models

# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

model = models.resnet101(pretrained=False)

model.load_state_dict(torch.load("../input/resnet101/resnet101-5d3b4d8f.pth"))

num_features = model.fc.in_features

print(num_features)

model.fc = nn.Linear(2048, 1)

model = model.to(device)
def train_model(model, data_loader, dataset_size, optimizer, scheduler, num_epochs):  

    since = time.time()

    criterion =  nn.MSELoss()

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        print('-' * 10)

        scheduler.step()

        model.train()

        running_loss = 0.0

        tk0 = tqdm(data_loader, total=int(len(data_loader)))

        counter = 0

        for bi, d in enumerate(tk0):

            inputs = d["image"]

            labels = d["labels"].view(-1, 1)

            inputs = inputs.to(device, dtype=torch.float)

            labels = labels.to(device, dtype=torch.float)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):

                outputs = model(inputs)

                loss = criterion(outputs, labels)

                loss.backward()

                optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            counter += 1

            tk0.set_postfix(loss=(running_loss / (counter * data_loader.batch_size)))

        epoch_loss = running_loss / len(data_loader)

        print('Training Loss: {:.4f}'.format(epoch_loss))

    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model 
train_transform = transforms.Compose([

        # Data augmentation is a good practice for the train set

        # Here, we randomly crop the image to 224x224 and

        # randomly flip it horizontally. 

        transforms.Resize((224,224)),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406],

                                 [0.229, 0.224, 0.225])

    ])





train_dataset = AptosDataset(csv_file='train.csv' , 

                             root_dir='../input/aptos2019-blindness-detection/train_images',

                             transform=train_transform)

train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

lr_min = 1e-4

lr_max = 1e-3



plist = [

         {'params': model.layer4.parameters(), 'lr': 1e-4, 'weight': 0.001},

         {'params': model.fc.parameters(), 'lr': 1e-3}

     ]



optimizer_ft = optim.Adam(plist, lr=0.001)

scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10)
since = time.time()

model=train_model(model,train_dataset_loader,len(train_dataset),optimizer_ft,scheduler,num_epochs=10)

#model=train_model_patience(model,train_dataset_loader,len(train_dataset),optimizer_ft,scheduler,num_epochs=1)

time_elapsed = time.time() - since

print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

torch.save(model.state_dict(), "aptos_model.bin")
print(os.listdir("."))
import torch

from torchvision import transforms



# define some re-usable stuff

IMAGE_SIZE = 224

NUM_CLASSES = 5

TEST_BATCH_SIZE = 1

device = torch.device("cuda:0")





# make some augmentations on training data

test_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406],

                                 [0.229, 0.224, 0.225])

])
# import pretrainedmodels

# import sys

# package_dir = "../input/resnet101/"

# sys.path.insert(0, package_dir)

# model_pt = pretrainedmodels.__dict__['resnet101'](pretrained=None)

# model_pt.avg_pool = nn.AdaptiveAvgPool2d(1)

# model_pt.last_linear = nn.Sequential(

#                       nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

#                       nn.Dropout(p=0.25),

#                       nn.Linear(in_features=2048, out_features=2048, bias=True),

#                       nn.ReLU(),

#                       nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

#                       nn.Dropout(p=0.5),

#                       nn.Linear(in_features=2048, out_features=1, bias=True),

#                      )

# # setting strict=False to get around the weight loading problem 

# # model_pt.load_state_dict(torch.load("../input/20epoch/aptos_model.bin"), strict=False)

# model_pt.load_state_dict(torch.load("aptos_model.bin") , strict=False)

# # model_pt = model_pt.to(device)
# for param in model_pt.parameters():

#     param.requires_grad = False

model_pt=model 

model_pt.eval()
class AptosTestDataset(Dataset):



    def __init__(self, 

                 csv_file, 

                 root_dir, 

                 transform=None):

        self.data = pd.read_csv(csv_file)

        self.root_dir = root_dir

        self.transform = transform



    def __len__(self):

        return len(self.data)



    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir, 

                                self.data.loc[idx, 'id_code'] + '.png')

        image = Image.open(img_name)

        if self.transform:

            image = self.transform(image)



        return {'image': image}
val_dataset = AptosTestDataset(csv_file='val.csv',

                                      transform=test_transform, root_dir='../input/aptos2019-blindness-detection/train_images')



val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

val_preds = np.zeros((len(val_dataset), 1))

tk0 = tqdm(val_data_loader)

for i, x_batch in enumerate(tk0):

    x_batch = x_batch["image"]

    pred = model_pt(x_batch.to(device))

    val_preds[i * 32:(i + 1) * 32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)
val = pd.read_csv('val.csv')
import numpy as np

import pandas as pd

import os

import scipy as sp

from functools import partial

from sklearn import metrics

from collections import Counter

import json



class OptimizedRounder(object):

    def __init__(self):

        self.coef_ = 0



    def _kappa_loss(self, coef, X, y):

        X_p = np.copy(X)

        for i, pred in enumerate(X_p):

            if pred < coef[0]:

                X_p[i] = 0

            elif pred >= coef[0] and pred < coef[1]:

                X_p[i] = 1

            elif pred >= coef[1] and pred < coef[2]:

                X_p[i] = 2

            elif pred >= coef[2] and pred < coef[3]:

                X_p[i] = 3

            else:

                X_p[i] = 4



        ll = metrics.cohen_kappa_score(y, X_p, weights='quadratic')

        return -ll



    def fit(self, X, y):

        loss_partial = partial(self._kappa_loss, X=X, y=y)

        initial_coef = [0.5, 1.5, 2.5, 3.5]

        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')



    def predict(self, X, coef):

        X_p = np.copy(X)

        for i, pred in enumerate(X_p):

            if pred < coef[0]:

                X_p[i] = 0

            elif pred >= coef[0] and pred < coef[1]:

                X_p[i] = 1

            elif pred >= coef[1] and pred < coef[2]:

                X_p[i] = 2

            elif pred >= coef[2] and pred < coef[3]:

                X_p[i] = 3

            else:

                X_p[i] = 4

        return X_p



    def coefficients(self):

        return self.coef_['x']

optR = OptimizedRounder()

optR.fit(val_preds.astype(int), val['diagnosis'])

print(optR.coefficients)
coefficients = optR.coefficients()

valid_predictions = optR.predict(val_preds.astype(int), coefficients)
np.unique(valid_predictions)
import sklearn

acc = sklearn.metrics.accuracy_score(val['diagnosis'], valid_predictions)

print(' accuracy on validation set : {}'.format(acc)) 
# i was never able to get this working correctly for classification 

# kagglers plz let me know where i am botching this up

# sub = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

# test_preds = np.zeros((len(test_dataset), NUM_CLASSES))

# tk0 = tqdm(test_data_loader , total=int(len(test_data_loader)))

# for i, x_batch in enumerate(tk0):

#     x_batch = x_batch["image"]

#     pred = model_pt(x_batch.to(device))

#     test_preds[i * TEST_BATCH_SIZE:(i + 1) * TEST_BATCH_SIZE, :] = pred.detach().cpu().squeeze().numpy()

    

# test_preds = torch.from_numpy(test_preds).float().to(device).sigmoid()

# test_preds = test_preds.detach().cpu().squeeze().numpy()
test_dataset = AptosTestDataset(csv_file='../input/aptos2019-blindness-detection/test.csv',

                                      transform=test_transform, root_dir='../input/aptos2019-blindness-detection/test_images')

test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

test_preds1 = np.zeros((len(test_dataset), 1))

tk0 = tqdm(test_data_loader)

for i, x_batch in enumerate(tk0):

    x_batch = x_batch["image"]

    pred = model_pt(x_batch.to(device))

    test_preds1[i * 32:(i + 1) * 32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)
test_preds1
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

test_preds2 = np.zeros((len(test_dataset), 1))

tk0 = tqdm(test_data_loader)

for i, x_batch in enumerate(tk0):

    x_batch = x_batch["image"]

    pred = model_pt(x_batch.to(device))

    test_preds2[i * 32:(i + 1) * 32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

test_preds3 = np.zeros((len(test_dataset), 1))

tk0 = tqdm(test_data_loader)

for i, x_batch in enumerate(tk0):

    x_batch = x_batch["image"]

    pred = model_pt(x_batch.to(device))

    test_preds3[i * 32:(i + 1) * 32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

test_preds4 = np.zeros((len(test_dataset), 1))

tk0 = tqdm(test_data_loader)

for i, x_batch in enumerate(tk0):

    x_batch = x_batch["image"]

    pred = model_pt(x_batch.to(device))

    test_preds4[i * 32:(i + 1) * 32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

test_preds5 = np.zeros((len(test_dataset), 1))

tk0 = tqdm(test_data_loader)

for i, x_batch in enumerate(tk0):

    x_batch = x_batch["image"]

    pred = model_pt(x_batch.to(device))

    test_preds5[i * 32:(i + 1) * 32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

test_preds6 = np.zeros((len(test_dataset), 1))

tk0 = tqdm(test_data_loader)

for i, x_batch in enumerate(tk0):

    x_batch = x_batch["image"]

    pred = model_pt(x_batch.to(device))

    test_preds6[i * 32:(i + 1) * 32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

test_preds7 = np.zeros((len(test_dataset), 1))

tk0 = tqdm(test_data_loader)

for i, x_batch in enumerate(tk0):

    x_batch = x_batch["image"]

    pred = model_pt(x_batch.to(device))

    test_preds7[i * 32:(i + 1) * 32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

test_preds8 = np.zeros((len(test_dataset), 1))

tk0 = tqdm(test_data_loader)

for i, x_batch in enumerate(tk0):

    x_batch = x_batch["image"]

    pred = model_pt(x_batch.to(device))

    test_preds8[i * 32:(i + 1) * 32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

test_preds9 = np.zeros((len(test_dataset), 1))

tk0 = tqdm(test_data_loader)

for i, x_batch in enumerate(tk0):

    x_batch = x_batch["image"]

    pred = model_pt(x_batch.to(device))

    test_preds9[i * 32:(i + 1) * 32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

test_preds10 = np.zeros((len(test_dataset), 1))

tk0 = tqdm(test_data_loader)

for i, x_batch in enumerate(tk0):

    x_batch = x_batch["image"]

    pred = model_pt(x_batch.to(device))

    test_preds10[i * 32:(i + 1) * 32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)
test_preds = (test_preds1 + test_preds2 + test_preds3 + test_preds4 + test_preds5+test_preds6 + test_preds7 + test_preds8 + test_preds9 + test_preds10)/10.0
test_preds
np.unique(test_preds)
test_predictions = optR.predict(test_preds.astype(int), coefficients)
test_predictions
sample = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")

sample.diagnosis = test_predictions

sample.to_csv("submission.csv", index=False)
sample=pd.read_csv("submission.csv").head()
np.unique(sample['diagnosis'])