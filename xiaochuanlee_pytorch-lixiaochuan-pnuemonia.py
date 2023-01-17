# model
import torch
import torch.nn as nn
import torchvision.models as models
%matplotlib inline

class model(nn.Module):
    def __init__(self,):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        self.f = nn.Sequential(
            nn.Linear(56 * 56 * 64, 128),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.layer(x)
        x = x.view(-1, 56 * 56 * 64)
        x = self.f(x)
        return x
    def update_param(self):
        return self.parameters()


class CNN(nn.Module):
    def __init__(self,):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.2),
            nn.Conv2d(32, 20, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(20, 20, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.2),
            nn.Conv2d(20, 20, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(20, 20, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.2),
        )
        self.f = nn.Sequential(nn.Linear(14 * 14 * 20, 64), nn.Linear(64, 1))

    def forward(self, x):
        x = self.conv(x)
        x = self.f(x)
        return x
    def update_param(self):
        return self.parameters()


class Vgg_m(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.vgg16(pretrained=True)
        for parma in self.model.parameters():
            parma.requires_grad = False
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Linear(25088, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(4096, 2),
        )
    def forward(self, x):
        return self.model(x)

    def update_param(self):
        return self.classifier.parameters()

#dataset
import glob
import os
from functools import reduce
import cv2
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset

def get_img(filepath):
    img = cv2.imread(filepath).astype(np.float32)
    img = cv2.resize(img, (112, 112))
    return img

class Data(Dataset):
    def __init__(self, root="/kaggle/input/chest-xray-pneumonia/chest_xray", pos="PNEUMONIA", neg="NORMAL",
                 case="train"):
        self.case = case
        self.trans=transforms.ToTensor()
        assert case in ["train", "val", "test"], "case must in [\"train\",\"val\",\"test\"]"
        self.paths = [(os.path.join(root, self.case, j), j) for j in [neg, pos]]
        self.file_list = reduce(lambda a, b: a + b,
                                [list(map(lambda j: (j, int(x[1] == pos)), glob.glob(os.path.join(x[0], '*g')))) for x
                                 in self.paths])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        return self.trans(get_img(self.file_list[idx][0]))/255, np.array([self.file_list[idx][1]]).astype(np.float32)
# *_*coding:utf-8 *_*
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as opt
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import *
train_set=Data(case="train")
test_set=Data(case="test")
val_set=Data(case="val")
train_d=DataLoader(train_set, batch_size=1, shuffle=True)
test_d=DataLoader(test_set, batch_size=1, shuffle=True)
val_d=DataLoader(val_set, batch_size=1, shuffle=True)
def save_model(model, path):
    torch.save(model.state_dict(), path)
class robot:
    def __init__(self,model,train=train_d,test=test_d,val=test_d):
        self.model=model
        self.train=train
        self.test=test
        self.val=val

    def runtrain(self,lr=0.001,weight_decay=0.0002,no_cuda=False,epochs=20,prefix="simple"):
        loss1=[]
        loss2=[]
        Loss=nn.BCELoss()
        device = torch.device("cpu" if no_cuda else "cuda:0")
        model = self.model.to(device)

        optimizer=opt.Adam(self.model.update_param(), lr=lr, weight_decay=weight_decay)
        for epoch in range(epochs):
            train_total_loss = 0
            for data in tqdm(self.train):
                x, y = data
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                y_pred = model(x)

                loss = Loss(y_pred, y)
                train_total_loss += loss.item()
                loss.backward()
                optimizer.step()
            train_loss = train_total_loss / len(self.train.dataset)
            test_loss = test(model, device, self.test)
            save_model(model, "hed.model")
            if epoch % 10 == 0:
                if not os.path.isdir(prefix):
                    os.mkdir(prefix)
                name = "{0}-ep{1}.model".format(prefix, epoch)
                save_model(model, os.path.join(prefix, name))
            loss1.append(train_loss)
            loss2.append(test_loss)
        plt.plot(len(loss1),loss1,label="train loss")
        plt.plot(len(loss2),loss2,label="test loss")
        plt.legend()
        plt.show()



def test(model, device, test_loader):
    l=nn.BCELoss()
    total_loss = 0
    with torch.no_grad():
        for data in test_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            y_pred= model(x)
            loss = l(y_pred, y)
            total_loss += loss.item()
    loss = total_loss / len(test_loader.dataset)
    return loss


if __name__=="__main__":
    mo=[model(),CNN(),Vgg_m()]
    for m in mo:
        r=robot(m)
        r.runtrain()
