import pandas as pd
import cv2
import skimage.io as io
import os
from tqdm import tqdm
import random

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models


import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)


BASE_DIR = "/kaggle/input/siim-isic-melanoma-classification/"

meta_data = pd.read_csv(BASE_DIR + 'train.csv')
meta_data.columns = ["image_name", "patient_id", "sex", "age", "anatom", "diagnosis", "bening_m", "target"]
meta_data.sort_values(by=['image_name'], inplace=True)


images_name, targets =meta_data.image_name, meta_data.target


neg_values = sum(meta_data['target'] == 0)
pos_values = sum(meta_data['target'] == 1)

targets = targets.to_numpy()
len(targets) == (neg_values + pos_values)
print("ratios: Neg Class : ", neg_values/len(targets), " Pos Class : ", pos_values/len(targets))
targets_0 = np.where(targets == 0)
targets_1 = np.where(targets == 1)

print(len(targets_0[0]), len(targets_1[0]))
def hair_remove(image):
    # convert image to grayScale
    grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # kernel for morphologyEx
    kernel = cv2.getStructuringElement(1,(17,17))
    
    # apply MORPH_BLACKHAT to grayScale image
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    
    # apply thresholding to blackhat
    _,threshold = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
    
    # inpaint with original image and threshold image
    final_image = cv2.inpaint(image,threshold,1,cv2.INPAINT_TELEA)
    
    return final_image
class CustomLoader:
    
    def __init__(self, target_0, target_1, batch_size = 64):
        self.Path = "../input/resize-jpg-siimisic-melanoma-classification/300x300/"
        self.dataset = torchvision.datasets.ImageFolder(root = self.Path,
                        transform=transforms.Compose([ transforms.Resize((256,256)),
                        transforms.RandomAffine(degrees = random.randint(0, 360), translate=None, scale=None, shear=None, resample=False, fillcolor=0),
                        transforms.RandomCrop(size = 224, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomRotation(degrees = random.randint(0, 360), resample=False, expand=False, center=None, fill=None),
                        transforms.RandomVerticalFlip(p=0.5),
                        
                        transforms.ToTensor(), 
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]))
        self.target_0 = target_0[0]
        self.target_1 = target_1[0]
        self.batch_size = batch_size
        self.labels = [0]*batch_size
        
        for x in range(int(batch_size/2), batch_size):
            self.labels[x] = 1

    def get_minibatch(self ):
        
        
        indices_0 = random.sample(range(0, len(self.target_0)), int(self.batch_size/2))
        indices_1 = random.sample(range(0, len(self.target_1)), int(self.batch_size/2))

        comb_indices = [0]*self.batch_size
        
        test_size = len(self.dataset) - (len(self.target_0) + len(self.target_1))
        
        for x in range(self.batch_size):
            if x < int(self.batch_size/2):
                comb_indices[x] = self.target_0[indices_0[x]] + test_size
            else:
                comb_indices[x] = self.target_1[indices_1[x - int(self.batch_size)]] + test_size

        # Creating  data samplers and loaders:
        subset = torch.utils.data.Subset(self.dataset, comb_indices)
        train_loader = torch.utils.data.DataLoader(subset, batch_size=64, num_workers = 4, )
                                         
        return train_loader, self.labels
    
custom_dataset_loader = CustomLoader(targets_0, targets_1)
train_loader,labels= custom_dataset_loader.get_minibatch()
m,l = next(iter(train_loader))
print(m.shape)
img = m[0]
print(img.shape)
npimg = img.numpy()
plt.imshow(np.transpose(npimg, (1, 2, 0)))
MyNetwork = models.alexnet(pretrained=True)
#Frozen the weights of the cnn layers towards the beginning 
layers_to_freeze = [MyNetwork.features[0],MyNetwork.features[3],MyNetwork.features[6]]
for layer in layers_to_freeze:
    for params in layer.parameters():
        params.requires_grad = False
MyNetwork.classifier[0] = nn.Dropout(p = 0.2, inplace = False)
MyNetwork.classifier[3] = nn.Dropout(p = 0.2, inplace = False)
MyNetwork.classifier[6] = nn.Linear(in_features=4096, out_features=2, bias=True)

MyNetwork.classifier = nn.Sequential(*list(MyNetwork.classifier) + [nn.LogSoftmax(dim=1)])
print(MyNetwork)
def cal_acc(pred, labels):
    countLab_1, countLab_0, count_pred_1, count_pred_0 = 0, 0, 0, 0
    for i in range(len(labels)):
        if labels[i] == 1:
            countLab_1 += 1
        else:
            countLab_0 += 1
        if pred[i] == labels[i] and labels[i] == 1:
            count_pred_1 += 1
        elif pred[i] == labels[i] and labels[i] == 0:
            count_pred_0 += 1
    
    if countLab_1 == 0:
        countLab_1 = 1
    if countLab_0 == 0:
        countLab_0 = 1
    return countLab_1, countLab_0, count_pred_1, count_pred_0
network = MyNetwork
use_cuda = True
if use_cuda and torch.cuda.is_available():
    network.cuda()
    print('cuda')


optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
# weights = [pos_values/len(targets), neg_values/len(targets)]
# class_weights = torch.FloatTensor(weights).cuda()
LossFunc = nn.CrossEntropyLoss()
total_loss = 0
total_correct_0, total_correct_1 = 0, 0
x = 0
total_val_0, total_val_1= 0, 0
total_train_0, total_train_1 = 0, 0

for batches in range(300*15,6000):
        
    train_loader, labels = custom_dataset_loader.get_minibatch()
    images, _ = next(iter(train_loader))
    labels = torch.tensor(labels)
    if use_cuda and torch.cuda.is_available():
        images = images.cuda()
        labels = labels.cuda()

    pred = network(images)
    #print(torch.sigmoid(pred), labels)
    loss = LossFunc(torch.sigmoid(pred),labels)
    optimizer.zero_grad() # because each time its adds gradients into previous gradients
    loss.backward() # calculating gradient
    optimizer.step() # update weights / thetas




    total_loss += loss.item()

    tt1, tt0, tc1, tc0  = cal_acc(pred.argmax(dim = 1), labels)
    total_train_1, total_train_0, total_correct_1, total_correct_0 = total_train_1 + tt1, total_train_0 + tt0, total_correct_1 + tc1, total_correct_0 + tc0

    if batches % 300 == 0 and batches != 0:
        print("epoch : ",batches/300,"Traning Accuracy of class 1 : ",total_correct_1,'/',total_train_1,'  ',total_correct_1*1.0/total_train_1 )
        print("epoch : ",batches/300,"Traning Accuracy of class 0 : ",total_correct_0,'/',total_train_0,'  ',total_correct_0*1.0/total_train_0,"Train Loss : ",total_loss*1.0/(len(train_loader)*300) )
        total_loss = 0
        total_correct_0, total_correct_1 = 0, 0
        x = 0
        total_val_0, total_val_1= 0, 0
        total_train_0, total_train_1 = 0, 0
        
    torch.cuda.empty_cache()
meta_data = pd.read_csv(BASE_DIR + 'test.csv')
meta_data.columns = ["image_name", "patient_id", "sex", "age", "anatom"]

images_name = meta_data.image_name
print(len(images_name))
test_indices = list(range(len(images_name)))
subset = torch.utils.data.Subset(custom_dataset_loader.dataset, test_indices)
test_loader = torch.utils.data.DataLoader(subset, batch_size=64, num_workers = 4, )
import csv

x = 0
with open("sample_submission.csv", 'w') as file:
    writer = csv.writer(file)
    writer.writerow(["image_name","target"])
    for i,(data) in enumerate(test_loader):
        images,_ = data
        images = images.cuda()
        # forward + backward + optimize
        Output = network(images)
        Output = Output.argmax(dim = 1)
        for out in Output:
            writer.writerow([images_name[x],str(out.item())])
            x += 1