# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/223333/resnet152/resnet152"))



# Any results you write to the current directory are saved as output.
# import os

import cv2

import time

import shutil

import matplotlib.pyplot as plt

from tqdm import *

from sklearn.model_selection import train_test_split



from torch.utils.data import DataLoader, Dataset

import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms

import torch.nn.functional as F

import torchvision.models as models

import torch.optim as optim

import torch.nn as nn

import torch
img_size = 224

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet_cls = models.resnet152()

resnet_weights_path = "../input/resnet152/resnet152.pth"

resnet_cls.load_state_dict(torch.load(resnet_weights_path))



class AvgPool(nn.Module):

    def forward(self, x):

        return F.avg_pool2d(x, x.shape[2:])

    

class ResNet152(nn.Module):

    def __init__(self,num_outputs):

        super(ResNet152,self).__init__()

        self.resnet = resnet_cls

        layer4 = self.resnet.layer4

        self.resnet.layer4 = nn.Sequential(

                                    nn.Dropout(0.5),

                                    layer4

                                    )

        self.resnet.avgpool = AvgPool()

        self.resnet.fc = nn.Linear(2048, num_outputs)

        for param in self.resnet.parameters():

            param.requires_grad = False



        for param in self.resnet.layer4.parameters():

            param.requires_grad = True



        for param in self.resnet.fc.parameters():

            param.requires_grad = True

            

    def forward(self,x):

        out = self.resnet(x)

        return out
# resnet_cls = models.resnet152()

# resnet_weights_path = "../input/resnet152/resnet152.pth"

# resnet_cls.load_state_dict(torch.load(resnet_weights_path))



# class AvgPool(nn.Module):

#     def forward(self, x):

#         return F.avg_pool2d(x, x.shape[2:])

    

# class ResNet152(nn.Module):

#     def __init__(self,num_outputs):

#         super(ResNet152,self).__init__()

#         self.resnet = resnet_cls

#         layer4 = self.resnet.layer4

#         self.resnet.layer4 = nn.Sequential(

#                                     nn.Dropout(0.5),

#                                     layer4

#                                     )

#         self.resnet.avgpool = AvgPool()

#         self.resnet.fc = nn.Sequential(

#                           nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

#                           nn.Dropout(p=0.25),

#                           nn.Linear(in_features=2048, out_features=2048, bias=True),

#                           nn.ReLU(),

#                           nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

#                           nn.Dropout(p=0.5),

#                           nn.Linear(in_features=2048, out_features=num_outputs, bias=True),

#                          )

#         for param in self.resnet.parameters():

#             param.requires_grad = False



#         for param in self.resnet.layer4.parameters():

#             param.requires_grad = True



#         for param in self.resnet.fc.parameters():

#             param.requires_grad = True

            

#     def forward(self,x):

#         out = self.resnet(x)

#         return out
class RetinopathyDatasetTest(Dataset):

    def __init__(self, csv_file, transform):

        self.data = pd.read_csv(csv_file)

        self.transform = transform



    def __len__(self):

        return len(self.data)



    def __getitem__(self, idx):

        img_name = os.path.join('../input/aptos2019-blindness-detection/test_images', self.data.loc[idx, 'id_code'] + '.png')

#         image = cv2.imread(img_name)

#         image = cv2.resize(image,(512,512))

        image = load_ben_color(img_name)

        image = self.transform(image)

        return {'image': image}
def crop_image1(img,tol=7):

    # img is image data

    # tol  is tolerance

        

    mask = img>tol

    return img[np.ix_(mask.any(1),mask.any(0))]



def crop_image_from_gray(img,tol=7):

    if img.ndim ==2:

        mask = img>tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    elif img.ndim==3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        mask = gray_img>tol

        

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0): # image is too dark so that we crop out everything,

            return img # return original image

        else:

            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

    #         print(img1.shape,img2.shape,img3.shape)

            img = np.stack([img1,img2,img3],axis=-1)

    #         print(img.shape)

        return img

    

def load_ben_color(path, sigmaX=10 ):

    image = cv2.imread(path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = crop_image_from_gray(image)

    image = cv2.resize(image, (img_size, img_size))

    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)

        

    return image
# checkpoint = torch.load('../input/aptos7-12-224/aptosresnet101_224/model_best.pth.tar')

# checkpoint = torch.load('../input/152-512/aptosresnet152_512/model_best.pth.tar')

# print(checkpoint['best_prec1'])

# print(checkpoint['epoch'])
submit = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

checkpoint = torch.load('../input/223333/resnet152/resnet152/model_best.pth.tar')

#checkpoint = torch.load("../input/blindmodel-resnet152up/resnet152up/model_best.pth.tar")

#creat model

NeuralNet = ResNet152(num_outputs = 5).to(device)

#NeuralNet = torch.nn.DataParallel(NeuralNet).cuda()

#cudnn.benchmark = True

NeuralNet.load_state_dict(checkpoint['state_dict'])

NeuralNet.eval()

predicted = []



test_transform = transforms.Compose([transforms.ToPILImage(),

                               #transforms.Pad(64,padding_mode='reflect'),

                               transforms.RandomHorizontalFlip(),

                               #transforms.RandomVerticalFlip(),

                               transforms.ToTensor(),

                               transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])



test_dataset = RetinopathyDatasetTest(csv_file='../input/aptos2019-blindness-detection/sample_submission.csv',

                                  transform=test_transform)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

test_preds1 = np.zeros((len(test_dataset), 1))

tk0 = tqdm(test_data_loader)

for i, x_batch in enumerate(tk0):

    x_batch = x_batch["image"].to(device)

    pred = NeuralNet(x_batch)

    pred_np = np.argmax(pred.data.cpu().numpy(),axis=1)

    test_preds1[i * 32:(i + 1) * 32] = pred_np.reshape(-1,1)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

test_preds2 = np.zeros((len(test_dataset), 1))

tk0 = tqdm(test_data_loader)

for i, x_batch in enumerate(tk0):

    x_batch = x_batch["image"].to(device)

    pred = NeuralNet(x_batch)

    pred_np = np.argmax(pred.data.cpu().numpy(),axis=1)

    test_preds2[i * 32:(i + 1) * 32] = pred_np.reshape(-1,1)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

test_preds3 = np.zeros((len(test_dataset), 1))

tk0 = tqdm(test_data_loader)

for i, x_batch in enumerate(tk0):

    x_batch = x_batch["image"].to(device)

    pred = NeuralNet(x_batch)

    pred_np = np.argmax(pred.data.cpu().numpy(),axis=1)

    test_preds3[i * 32:(i + 1) * 32] = pred_np.reshape(-1,1)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

test_preds4 = np.zeros((len(test_dataset), 1))

tk0 = tqdm(test_data_loader)

for i, x_batch in enumerate(tk0):

    x_batch = x_batch["image"].to(device)

    pred = NeuralNet(x_batch)

    pred_np = np.argmax(pred.data.cpu().numpy(),axis=1)

    test_preds4[i * 32:(i + 1) * 32] = pred_np.reshape(-1,1)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

test_preds5 = np.zeros((len(test_dataset), 1))

tk0 = tqdm(test_data_loader)

for i, x_batch in enumerate(tk0):

    x_batch = x_batch["image"].to(device)

    pred = NeuralNet(x_batch)

    pred_np = np.argmax(pred.data.cpu().numpy(),axis=1)

    test_preds5[i * 32:(i + 1) * 32] = pred_np.reshape(-1,1)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

test_preds6 = np.zeros((len(test_dataset), 1))

tk0 = tqdm(test_data_loader)

for i, x_batch in enumerate(tk0):

    x_batch = x_batch["image"].to(device)

    pred = NeuralNet(x_batch)

    pred_np = np.argmax(pred.data.cpu().numpy(),axis=1)

    test_preds6[i * 32:(i + 1) * 32] = pred_np.reshape(-1,1)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

test_preds7 = np.zeros((len(test_dataset), 1))

tk0 = tqdm(test_data_loader)

for i, x_batch in enumerate(tk0):

    x_batch = x_batch["image"].to(device)

    pred = NeuralNet(x_batch)

    pred_np = np.argmax(pred.data.cpu().numpy(),axis=1)

    test_preds7[i * 32:(i + 1) * 32] = pred_np.reshape(-1,1)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

test_preds8 = np.zeros((len(test_dataset), 1))

tk0 = tqdm(test_data_loader)

for i, x_batch in enumerate(tk0):

    x_batch = x_batch["image"].to(device)

    pred = NeuralNet(x_batch)

    pred_np = np.argmax(pred.data.cpu().numpy(),axis=1)

    test_preds8[i * 32:(i + 1) * 32] = pred_np.reshape(-1,1)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

test_preds9 = np.zeros((len(test_dataset), 1))

tk0 = tqdm(test_data_loader)

for i, x_batch in enumerate(tk0):

    x_batch = x_batch["image"].to(device)

    pred = NeuralNet(x_batch)

    pred_np = np.argmax(pred.data.cpu().numpy(),axis=1)

    test_preds9[i * 32:(i + 1) * 32] = pred_np.reshape(-1,1)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

test_preds10 = np.zeros((len(test_dataset), 1))

tk0 = tqdm(test_data_loader)

for i, x_batch in enumerate(tk0):

    x_batch = x_batch["image"].to(device)

    pred = NeuralNet(x_batch)

    pred_np = np.argmax(pred.data.cpu().numpy(),axis=1)

    test_preds10[i * 32:(i + 1) * 32] = pred_np.reshape(-1,1)
test_preds = (test_preds1 + test_preds2 + test_preds3 + test_preds4 + test_preds5

             + test_preds6 + test_preds7 + test_preds8 + test_preds9 + test_preds10) / 10.0
coef = [0.5, 1.5, 2.5, 3.5]



for i, pred in enumerate(test_preds):

    if pred < coef[0]:

        test_preds[i] = 0

    elif pred >= coef[0] and pred < coef[1]:

        test_preds[i] = 1

    elif pred >= coef[1] and pred < coef[2]:

        test_preds[i] = 2

    elif pred >= coef[2] and pred < coef[3]:

        test_preds[i] = 3

    else:

        test_preds[i] = 4
sample = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")

sample.diagnosis = test_preds.astype(int)

sample.to_csv("submission.csv", index=False)
sample