import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import models
from torch.nn import functional as F

import numpy as np
import pandas as pd


import cv2

import os

import albumentations as A
from albumentations.pytorch import ToTensor

import matplotlib.pyplot as plt

from PIL import Image

!pip install efficientnet_pytorch

from efficientnet_pytorch import EfficientNet


train_dir = '../input/korpus-ml-2/train/train/train'
test_dir = '../input/korpus-ml-2/test/test/test'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class DataChart(Dataset):
    def __init__(self, name_dir, transform=None, mode='train'):

        self.name_dir = name_dir
        self.file_name = os.listdir(name_dir)
        self.transform = transform
        self.mode = mode
        self.label = 0
        self.name_to_label = {'bar_chart' : 1, 'diagram' : 2, 'flow_chart' : 3,
                          'graph' : 4, 'growth_chart' : 5, 'pie_chart' : 6,
                          'table' : 7, 'just_image' : 0}

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, idx):

        if self.file_name[idx].split('.')[-1] == 'gif':
            gif = cv2.VideoCapture(os.path.join(self.name_dir, self.file_name[idx]))
            _, image = gif.read()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.imread(os.path.join(self.name_dir, self.file_name[idx]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']

        if self.mode != 'test':
            for name, label in self.name_to_label.items():
                if name in self.file_name[idx]:
                    self.label = label
                    break

            return image, self.label
        else:
            return image

img_size = 224
num_workers = 2
batch_size = 8
train_transform = A.Compose([A.Resize(img_size,img_size),
                             A.ShiftScaleRotate(rotate_limit=10, scale_limit=0.2), A.HueSaturationValue(),
                             A.CLAHE(), A.RGBShift(),
                             A.RandomBrightness(), A.Normalize(), ToTensor()])


test_transform = A.Compose([A.Resize(img_size, img_size),
                           A.Normalize(), ToTensor()])
train_set = DataChart(train_dir, train_transform, 'train')

test_set = DataChart(test_dir, test_transform, mode='test')
train_loader = DataLoader(train_set, batch_size, num_workers=num_workers, shuffle=True)

test_loader = DataLoader(test_set,batch_size, num_workers=num_workers)
plt.figure(figsize=(20,16))
image, label = next(iter(train_loader))

grid = torchvision.utils.make_grid(image[:16])
plt.imshow(np.transpose(grid.numpy(), (1,2,0)))
print(label[:16])

model1 = models.resnext101_32x8d(True, True)

for param in model1.parameters():
    param.requires_grad = False

model2 = models.wide_resnet101_2(True, True)

for param in model2.parameters():
    param.requires_grad = False


model3 = EfficientNet.from_pretrained('efficientnet-b7', num_classes=8)

model1.fc = nn.Sequential(
    nn.Linear(2048, 1024),
    nn.Dropout(0.25),
    nn.ReLU(),
    nn.Linear(1024,512),
    nn.Dropout(0.25),
    nn.ReLU(),
    nn.Linear(512,8)
)

model2.fc = nn.Sequential(
    nn.Linear(2048, 1024),
    nn.Dropout(0.4),
    nn.ReLU(),
    nn.Linear(1024,512),
    nn.Dropout(0.3),
    nn.ReLU(),
    nn.Linear(512,8)
)
# for param in model1.layer3.parameters():
#     param.requires_grad = False

class EnsembleNet(nn.Module):
    def __init__(self, model1, model2, model3):
        super(EnsembleNet, self).__init__()

        self.model1 = model1
        self.model2 = model2
        self.model3 = model3



    def forward(self, x):
        x1 = F.softmax(self.model1(x),dim=1)
        x2 = F.softmax(self.model2(x),dim=1)
        x3 = F.softmax(self.model3(x),dim=1)

    
    
        out = (x1 + x2 + x3)


        return out
lr = 0.001
n_epoch = 5

model = model1.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0, factor= 0.5)
model.train()
train_loss = []

loader = train_loader

for epoch in range(n_epoch):
    true_train_samples = 0
    for i, (data, target) in enumerate(loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()       
        optimizer.step()

        train_loss.append(loss.item())

        true_train_samples += (output.argmax(dim=1,keepdim=False) == target).sum().item()
  
    sheduler.step(np.mean(train_loss))


    print(f"эпохa: {epoch+1}, лосс: {np.mean(train_loss)}, LR: {optimizer.param_groups[0]['lr']} train_acc: {round(true_train_samples / len(loader.dataset), 4)}")
#model = EnsembleNet(model1, model2, model3).to(device)

model.eval()

result = []



with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        outputs = model(data)
        _, pred = torch.max(outputs.data,1)
        result.extend(pred.tolist())

len(result)
#model3.load_state_dict(torch.load('/content/drive/My Drive/WeightNet/ef.pth'))
#torch.save(model2.state_dict(), f'/content/drive/My Drive/WeightNet/ef2.pth')
sample = pd.read_csv('../input/korpus-ml-2/sample_submission.csv')
sample.head(3)
sample['image_name'] = os.listdir(test_dir)
sample['label'] = result
sample.to_csv('./submit.csv', index=False)