import pandas as pd

import numpy as np

import cv2 as cv

import matplotlib.pyplot as plt

import torch

from pathlib import Path

from torch.utils.data import Dataset,DataLoader

from PIL import Image

import albumentations as A

from albumentations.pytorch.transforms import ToTensor

import torch.nn as nn

import torch.nn.functional as F

import torchvision.models as models

from fastprogress.fastprogress import master_bar, progress_bar

from torchvision.transforms.functional import to_grayscale

from sklearn.metrics import accuracy_score, roc_auc_score

from torchvision.utils import make_grid

!pip install efficientnet-pytorch

from efficientnet_pytorch import EfficientNet
normal = Path('../input/surface-crack-detection/Negative')

cracks = Path('../input/surface-crack-detection/Positive')



train_data = [(o,0) for o in normal.iterdir()]

train_data_cracks = [(o,1) for o in cracks.iterdir()]

train_data.extend(train_data_cracks)



train_data = pd.DataFrame(train_data, columns=["filepath","cracks"])

train_data.head()
bs = 64

lr=1e-5

wd=0.01
np.random.seed(42)

msk = np.random.rand(len(train_data)) < 0.9



train_df = train_data[msk].reset_index()

val_df = train_data[~msk].reset_index()
fig,a =  plt.subplots(1,5)

fig.set_figheight(30)

fig.set_figwidth(30)

for i in range(5):

    img = cv.imread(str(train_data.iloc[i]['filepath']))

    a[i].imshow(img)

    title = "Good Concrete " + str(i)

    a[i].set_title(title)
fig,a =  plt.subplots(1,5)

fig.set_figheight(30)

fig.set_figwidth(30)

for i in range(-5, 0):

    img = cv.imread(str(train_data.iloc[i]['filepath']))

    a[i].imshow(img)

    title = "Concrete with cracks " + str(5 + i)

    a[i].set_title(title)
class ConcreteDataset(Dataset):

    def __init__(self, df, transforms=None, is_test=False):

        self.df = df

        self.transforms = transforms

        self.is_test = is_test

    

    def __len__(self):

        return self.df.shape[0]

    

    def __getitem__(self, idx):

#         row = self.df.iloc[idx]

        img = Image.open(self.df.iloc[idx]['filepath'])

        

        if self.transforms:

            #when using albumation we have to pass data as dictionary to transforms.

            #The output after transformation is also a dictionary

            #we need to take the value from dictionary. That is why we are giving an image at the end.

            img = self.transforms(**{"image": np.array(img)})["image"]

        

        if self.is_test:

            return img

        else:

            cracks_tensor = torch.tensor([self.df.iloc[idx]['cracks']], dtype=torch.float32)

            return img, cracks_tensor

        
imagenet_stats = {'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}

train_tfms = A.Compose([

    A.Cutout(p=0.5),

    A.RandomRotate90(p=0.5),

    A.Flip(p=0.5),

    ToTensor(normalize=imagenet_stats)

        ])

    

test_tfms = A.Compose([

        ToTensor(normalize=imagenet_stats)

        ])
train_ds = ConcreteDataset(train_df, transforms=train_tfms)

val_ds = ConcreteDataset(val_df, transforms=test_tfms, is_test=True)

len(train_ds), len(val_ds)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, 

                      num_workers=3, pin_memory=True)

val_dl = DataLoader(val_ds, batch_size=64, 

                    num_workers=2, pin_memory=True)
xb, yb = next(iter(train_dl))

xb.shape, yb.shape
class ConcreteModel(nn.Module):

    def __init__(self, eff_name = 'efficientnet-b0', n_outs = 1):

        super().__init__()

        #downloading the pretrained efficientnet model

        self.backbone = EfficientNet.from_pretrained(eff_name)

        #getting the number of input layers in the classifiers

        in_features = getattr(self.backbone,'_fc').in_features

        #replacing the existing classifier with new one

        self.classifier = nn.Sequential(nn.Linear(in_features, in_features//2),

                                       nn.Dropout(p=0.2),

                                       nn.Linear(in_features//2, in_features//4),

                                       nn.Dropout(p=0.2),

                                       nn.Linear(in_features//4, n_outs))

    

    def forward(self, input_of_model):

        

        """

        here the input shape is torch.Size([64, 3, 227, 227])

        we need to extract the features. In my understanding it means taking the output before passing to classifier

        https://github.com/lukemelas/EfficientNet-PyTorch#example-feature-extraction

        """

        out_before_classifier = self.backbone.extract_features(input_of_model) #the output size is torch.Size([64, 1280, 7, 7])

        

        #to convert the 7x7 to 1x1 we use a adaptive average pool 2d

        pool_output = F.adaptive_avg_pool2d(out_before_classifier, 1) #the output is torch.Size([64, 1280, 1, 1])

        

        """

        now before passing to the classifier, we need to flatten it. Using view operation for the same

        the size parameter is the length on a particular axis. size(0) = 64 size(1) = 1280 size(2) and size(3) is 1

        """

        classifier_in = pool_output.view(pool_output.size(0),-1) #this operation will convert torch.Size([64, 1280, 1, 1]) to torch.Size([64, 1280])

        

        #this is then fed into a custom classifier which outputs the predicition

        classifier_out = self.classifier(classifier_in) #the classifier output will be of size torch.Size([64, 1])

        return classifier_out
def get_device():

    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



def get_model(model_name='efficientnet-b0',lr=1e-5,wd=0.01,freeze_backbone=False,opt_fn=torch.optim.AdamW,device=None):

    device = device if device else get_device()

    model = ConcreteModel(eff_name=model_name)

    if freeze_backbone:

        for parameter in model.backbone.parameters():

            parameter.requires_grad = False

    opt = opt_fn(model.parameters(),lr=lr,weight_decay=wd)

    model = model.to(device)

    return model, opt



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

    train_ds = ConcreteDataset(df=train_df,transforms=train_tfms)

    valid_ds = ConcreteDataset(df=valid_df,transforms=test_tfms)

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
train_dl,valid_dl = get_data(train_df,val_df,train_tfms,test_tfms,bs)

model, opt = get_model(model_name='efficientnet-b0',lr=1e-4,wd=1e-4)
model,val_rocs, train_loss, valid_loss = fit(10,model,train_dl,valid_dl,opt)
plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

plt.plot(train_loss, '-bx')

plt.plot(valid_loss, '-rx')

plt.xlabel('epoch')

plt.ylabel('loss')

plt.legend(['Training', 'Validation'])

plt.title('Loss vs. No. of epochs');

plt.grid()