import pandas as pd

import numpy as np

import cv2 as cv

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

import seaborn as sns

import torch

from pathlib import Path

from torch.utils.data import Dataset,DataLoader

from PIL import Image

from torchvision import transforms as T

import torch.nn as nn

import torch.nn.functional as F

from torchvision.transforms import Resize, Compose, ToTensor, Grayscale

import torchvision.models as models

from fastprogress.fastprogress import master_bar, progress_bar

from torchvision.transforms.functional import to_grayscale

from sklearn.metrics import accuracy_score, roc_auc_score
path = Path('../input/chest-xray-pneumonia/chest_xray')

im_sz = 256

bs = 16
train_normal = Path('../input/chest-xray-pneumonia/chest_xray/train/NORMAL')

train_disease = Path('../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA')



train_data = [(o,0) for o in train_normal.iterdir()]

train_data_disease = [(o,1) for o in train_disease.iterdir()]

train_data.extend(train_data_disease)



train_data = pd.DataFrame(train_data, columns=["filepath","disease"])

train_data.head()
valid_normal = Path('../input/chest-xray-pneumonia/chest_xray/val/NORMAL')

valid_disease = Path('../input/chest-xray-pneumonia/chest_xray/val/PNEUMONIA')



valid_data = [(o,0) for o in valid_normal.iterdir()]

valid_data_disease = [(o,1) for o in valid_disease.iterdir()]

valid_data.extend(valid_data_disease)



valid_data = pd.DataFrame(valid_data, columns=["filepath","disease"])

valid_data.head()
test_normal = Path('../input/chest-xray-pneumonia/chest_xray/test/NORMAL')

test_disease = Path('../input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA')



test_data = [(o,0) for o in test_normal.iterdir()]

test_data_disease = [(o,1) for o in test_disease.iterdir()]

test_data.extend(test_data_disease)



test_data = pd.DataFrame(test_data, columns=["filepath","disease"])

test_data.to_csv('test.csv',index=False)

test_data.head()
def list_files(path:Path):

    """

    This function is used to list files in a directory

    """

    return [o for o in path.iterdir()]
def get_device():

    """

    This is used to get the device to run the training

    """

    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



device = get_device()
fig,a =  plt.subplots(1,5)

fig.set_figheight(30)

fig.set_figwidth(30)

for i in range(5):

    img = cv.imread(str(train_data.iloc[i]['filepath']))

    a[i].imshow(img)

    title = "Normal Xray " + str(i)

    a[i].set_title(title)
fig,a =  plt.subplots(1,5)

fig.set_figheight(30)

fig.set_figwidth(30)

for i in range(-5, 0):

    img = cv.imread(str(train_data.iloc[i]['filepath']))

    a[i].imshow(img)

    title = "Pneumonia Xray " + str(5 + i)

    a[i].set_title(title)
class PneumoniaDatset(Dataset):

    def __init__(self, df, transforms=None, is_test=False):

        self.df = df

        self.transforms = transforms

        self.is_test = is_test

    

    def __getitem__(self, idx):

        image_path = self.df.iloc[idx]['filepath']

        img = Image.open(image_path)

        

        if self.transforms:

            img = self.transforms(img)

        if self.is_test:

            return img

        else:

            disease = self.df.iloc[idx]['disease']

            return img, torch.tensor([disease], dtype=torch.float32)

        

    def __len__(self):

        return self.df.shape[0]
train_tfms = Compose([ Grayscale(), Resize((512,512)), ToTensor()])

test_tfms = Compose([Grayscale(), Resize((512,512)) , ToTensor()])
class PneumoniaModel(nn.Module):

    def __init__(self, backbone=models.resnet18(pretrained=True), n_out=1):

        super().__init__()

        backbone = backbone

        in_features = backbone.fc.in_features

        #make 3 input layer to work with 1 layer : https://discuss.pytorch.org/t/grayscale-images-for-resenet-and-deeplabv3/48693/2?u=vishnurapps

        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        self.classifier = nn.Sequential(nn.Linear(in_features, n_out))

        

    def forward(self, x):

        #Size of the input data             torch.Size([64, 1, 512, 512])

        x = self.backbone(x)               #torch.Size([64, 512, 16, 16])

        x = F.adaptive_avg_pool2d(x, 1)    #torch.Size([64, 512, 1, 1])

        x = torch.flatten(x, 1)            #torch.Size([64, 512])

        x = self.classifier(x)             #torch.Size([64, 1])

        return x
model = PneumoniaModel()

model.to(device)

opt = torch.optim.AdamW(model.parameters(), lr=1e-5,weight_decay=0.01)
train_ds = PneumoniaDatset(df=train_data,transforms=train_tfms)

train_dl = DataLoader(dataset=train_ds,batch_size=bs,shuffle=True,num_workers=4)
def training_step(xb,yb,model,loss_fn,opt,device,scheduler):

    xb,yb = xb.to(device), yb.to(device)

    out = model(xb)

    opt.zero_grad()

    loss = loss_fn(out,yb)

    loss.backward()

    opt.step()

    scheduler.step()

    return loss.item()
def validation_step(xb,yb,model,loss_fn,device):

    xb,yb = xb.to(device), yb.to(device)

    out = model(xb)

    loss = loss_fn(out,yb)

    out = torch.sigmoid(out)

    return loss.item(),out
def get_data(train_df,valid_df,train_tfms,test_tfms,bs):

    train_ds = PneumoniaDatset(df=train_data,transforms=train_tfms)

    valid_ds = PneumoniaDatset(df=valid_data,transforms=test_tfms)

    train_dl = DataLoader(dataset=train_ds,batch_size=bs,shuffle=True,num_workers=4)

    valid_dl = DataLoader(dataset=valid_ds,batch_size=bs*2,shuffle=False,num_workers=4)

    return train_dl,valid_dl
def fit(epochs,model,train_dl,valid_dl,opt,device=None,loss_fn=F.binary_cross_entropy_with_logits):

    

    device = device if device else get_device()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, len(train_dl)*epochs)

    val_rocs = [] 

    tloss = []

    vloss = []

    

    #Creating progress bar

    mb = master_bar(range(epochs))

    mb.write(['epoch','train_loss','valid_loss','val_roc'],table=True)



    for epoch in mb:    

        trn_loss,val_loss = 0.0,0.0

        val_preds = np.zeros((len(valid_dl.dataset),1))

        val_targs = np.zeros((len(valid_dl.dataset),1))

        

        #Training

        model.train()

        

        #For every batch 

        for xb,yb in progress_bar(train_dl,parent=mb):

#             print(xb.shape)

            trn_loss += training_step(xb,yb,model,loss_fn,opt,device,scheduler) 

        trn_loss /= mb.child.total

        tloss.append(trn_loss)



        #Validation

        model.eval()

        with torch.no_grad():

            for i,(xb,yb) in enumerate(progress_bar(valid_dl,parent=mb)):

                loss,out = validation_step(xb,yb,model,loss_fn,device)

                val_loss += loss

                bs = xb.shape[0]

                val_preds[i*bs:i*bs+bs] = out.cpu().numpy()

                val_targs[i*bs:i*bs+bs] = yb.cpu().numpy()



        val_loss /= mb.child.total

        vloss.append(val_loss)

        val_roc = roc_auc_score(val_targs.reshape(-1),val_preds.reshape(-1))

        val_rocs.append(val_roc)



        mb.write([epoch,f'{trn_loss:.6f}',f'{val_loss:.6f}',f'{val_roc:.6f}'],table=True)

    return model,val_rocs, tloss, vloss
train_dl,valid_dl = get_data(train_data,valid_data,train_tfms,test_tfms,bs=64)
model, val_rocs, train_loss, valid_loss = fit(10,model,train_dl,valid_dl,opt)
plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

plt.plot(train_loss, '-bx')

plt.plot(valid_loss, '-rx')

plt.xlabel('epoch')

plt.ylabel('loss')

plt.legend(['Training', 'Validation'])

plt.title('Loss vs. No. of epochs');

plt.grid()
test_ds = PneumoniaDatset(df=test_data,transforms=test_tfms,is_test=True)

test_dl = DataLoader(dataset=test_ds,batch_size=bs,shuffle=False,num_workers=4)
def get_preds(model,device=None,tta=3):

    if device is None:

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    preds = np.zeros(len(test_ds))

    for tta_id in range(tta):

        test_preds = []

        with torch.no_grad():

            for xb in test_dl:

                xb = xb.to(device)

                out = model(xb)

                out = torch.sigmoid(out)

                test_preds.extend(out.cpu().numpy())

            preds += np.array(test_preds).reshape(-1)

        #print(f'TTA {tta_id}')

    preds /= tta

    for i, x in enumerate (preds):

        if x >= 0.5:

            preds[i] = 1

        else :

            preds[i] = 0

    return preds



preds = get_preds(model)
submission = pd.read_csv('test.csv')

submission['prediction'] = preds

submission.to_csv('submission.csv',index=False)
cm = confusion_matrix(submission['disease'], submission['prediction'])

sns.heatmap(cm, annot=True, fmt="d")