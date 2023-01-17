import torch

import torchvision

from torch.utils import data

from torchvision import transforms

from PIL import Image

from skimage import io

from skimage import color

import matplotlib.pyplot as plt

import os

import numpy as np
img_rows=473

img_cols=473

img_channels=3

train_images_path='../input/adechallengedata2016/ADEChallengeData2016/images/training'

train_labels_path='../input/adechallengedata2016/ADEChallengeData2016/annotations/training'

val_images_path='../input/adechallengedata2016/ADEChallengeData2016/images/validation'

val_labels_path='../input/adechallengedata2016/ADEChallengeData2016/annotations/validation'

test_images_path='../input/release_test/release_test/testing'
transform=transforms.Compose([

    transforms.Resize((img_rows,img_cols)), #input should be PIL Image

    transforms.ToTensor() # channel first

])



class Mydataset(data.Dataset):

    def __init__(self,root1,root2,transform=None):

        imgs=sorted(os.listdir(root1))

        labs=sorted(os.listdir(root2))

        self.imgs=[os.path.join(root1,img) for img in imgs] #所有图片的路径的list

        self.labs=[os.path.join(root2,lab) for lab in labs]

        self.transform=transform

    

    def __getitem__(self,index):

        img_path=self.imgs[index]

        lab_path=self.labs[index]

        

        image=Image.open(img_path)

        image=image.resize((473,473))

        image=np.asarray(image)

        if len(image.shape)!=3:

            image=color.gray2rgb(image)

        #image=torch.from_numpy(image)

        

        label=Image.open(lab_path)

        label=label.resize((473,473))

        label=np.asarray(label)

        #label=torch.from_numpy(label)

        if self.transform is not None:

            image=self.transform(image)

            label=self.transform(label)

        label=label.squeeze(0)

        return image,label

    

    def __len__(self):

        return len(self.imgs)
class Mydataset1(data.Dataset):

    def __init__(self,root,transform):

        imgs=os.listdir(root)

        self.imgs=[os.path.join(root,img) for img in imgs]

        self.transform=transform

    def __getitem__(self,index):

        img_path=self.imgs[index]

        image=Image.open(img_path)

        image=image.resize((473,473))

        image=np.asarray(image)

        if len(image.shape)!=3:

            image=color.gray2rgb(image)

        if self.transform is not None:

            image=self.transform(image)

        return image

    def __len__(self):

        return len(self.imgs)
# train_dataset=Mydataset(train_images_path,train_labels_path,transform=transforms.ToTensor())

# print(len(train_dataset[0]))

# image,label=train_dataset[0]

# print(image.size())

# print(label.size())

# image=np.transpose(image,(1,2,0))

# # label=np.transpose(label,(1,2,0))

# plt.subplot(1,2,1)

# plt.imshow(image)

# plt.subplot(1,2,2)

# plt.imshow(label.view(473,473),cmap='gray')
train_dataset=Mydataset(train_images_path,train_labels_path,transform=transforms.ToTensor())

train_loader=data.DataLoader(train_dataset,batch_size=4,shuffle=True)

val_dataset=Mydataset(val_images_path,val_labels_path,transform=transforms.ToTensor())

val_loader=data.DataLoader(val_dataset,batch_size=1,shuffle=True)

test_dataset=Mydataset1(test_images_path,transform=transforms.ToTensor())

test_loader=data.DataLoader(test_dataset,batch_size=1)
!pip install torchsummary
import torch.nn as nn

import torch.nn.functional as F

from torch.nn import init

from torchsummary import summary
class identity_block(nn.Module):

    def __init__(self,ch_in,ch_out,pad,dilation):

        super(identity_block,self).__init__()

        self.pad=pad

        self.conv1=nn.Conv2d(ch_in,ch_out[0],kernel_size=1,stride=1,padding=0)

        self.bn1=nn.BatchNorm2d(ch_out[0])

        

        #self.pad=nn.ZeroPad2d(pad)

        self.conv2=nn.Conv2d(ch_out[0],ch_out[1],kernel_size=3,stride=1,dilation=dilation)

        self.bn2=nn.BatchNorm2d(ch_out[1])

        

        self.conv3=nn.Conv2d(ch_out[1],ch_out[2],kernel_size=1,stride=1,padding=0)

        self.bn3=nn.BatchNorm2d(ch_out[2])

        

    def forward(self,x):

        identity=x

        out=F.relu(self.bn1(self.conv1(x)))

        out=F.pad(out,(self.pad,self.pad,self.pad,self.pad))

        out=F.relu(self.bn2(self.conv2(out)))

        out=self.bn3(self.conv3(out))

        

        out=out+identity

        out=F.relu(out)

        

        return out
class conv_block(nn.Module):

    def __init__(self,ch_in,ch_out,s,pad,dilation):

        super(conv_block,self).__init__()

        self.pad=pad

        self.conv1=nn.Conv2d(ch_in,ch_out[0],kernel_size=1,stride=s,padding=0)

        self.bn1=nn.BatchNorm2d(ch_out[0])

        

        #self.pad=nn.ZeroPad2d(pad)

        self.conv2=nn.Conv2d(ch_out[0],ch_out[1],kernel_size=3,stride=1,dilation=dilation)

        self.bn2=nn.BatchNorm2d(ch_out[1])

        

        self.conv3=nn.Conv2d(ch_out[1],ch_out[2],kernel_size=1,stride=1,padding=0)

        self.bn3=nn.BatchNorm2d(ch_out[2])

        

        self.shortcut=nn.Sequential(

            nn.Conv2d(ch_in,ch_out[2],kernel_size=1,stride=s,padding=0),

            nn.BatchNorm2d(ch_out[2])

        )

        

    def forward(self,x):

        identity=x

        

        out=F.relu(self.bn1(self.conv1(x)))

        out=F.pad(out,(self.pad,self.pad,self.pad,self.pad))

        out=F.relu(self.bn2(self.conv2(out)))

        out=self.bn3(self.conv3(out))

        out=out+self.shortcut(identity)

        out=F.relu(out)

        

        return out
class resnet50(nn.Module):

    def __init__(self):

        super(resnet50,self).__init__()

        

        self.pre=nn.Sequential(

            nn.ZeroPad2d(1),

            nn.Conv2d(3,64,kernel_size=3,stride=2,padding=0),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.ZeroPad2d(1),

            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=0),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.ZeroPad2d(1),

            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=0),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

            nn.ZeroPad2d(1),

            nn.MaxPool2d(kernel_size=(3,3),stride=(2,2))

        )



        self.layer1=self._make_layer(128,[64,64,256],s=1,pad=1,dilation=1,num_blocks=3)

        self.layer2=self._make_layer(256,[128,128,512],s=2,pad=1,dilation=1,num_blocks=4)

        self.layer3=self._make_layer(512,[256,256,1024],s=1,pad=2,dilation=2,num_blocks=6)

        self.layer4=self._make_layer(1024,[512,512,2048],s=1,pad=4,dilation=4,num_blocks=3)

    

    def _make_layer(self,ch_in,ch_out,s,pad,dilation,num_blocks):

        layers=[]

        layers.append(conv_block(ch_in,ch_out,s,pad,dilation))

        

        for i in range(1,num_blocks):

            layers.append(identity_block(ch_out[2],ch_out,pad,dilation))



        return nn.Sequential(*layers)

    

    def forward(self,x):

        out=self.pre(x)

        out=self.layer1(out)

        out=self.layer2(out)

        #aux_out=self.layer3(out) #(-1,1024,60,60) aux_loss

        out=self.layer3(out)

        out=self.layer4(out)

        

        return out
# device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model=resnet50().to(device)

# summary(model,(3,473,473))
class interp_block(nn.Module):

    def __init__(self,ch_in,ch_out,kernel,stride):

        super(interp_block,self).__init__()

        self.avg=nn.AvgPool2d(kernel_size=kernel,stride=stride)

        self.conv=nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1)

        self.bn=nn.BatchNorm2d(ch_out)

        

    def forward(self,x):

        out=self.bn(self.conv(self.avg(x)))

        out=F.relu(out)

        out=F.interpolate(out,(x.size(2),x.size(3)),mode='bilinear')

        return out
class pyramid_pooling_module(nn.Module):

    def __init__(self,sizes=(1,2,3,6)):

        super(pyramid_pooling_module,self).__init__()

        

        k_s_map={1:60,2:30,3:20,6:10}

        self.layers=[]

        self.layers=nn.ModuleList([interp_block(2048,512,k_s_map[i],k_s_map[i]) for i in sizes])

            

    def forward(self,x):

        out1=self.layers[0](x)

        out2=self.layers[1](x)

        out3=self.layers[2](x)

        out4=self.layers[3](x)

        out=torch.cat((x,out1,out2,out3,out4),1)

        return out
class pspnet(nn.Module):

    def __init__(self,n_classes=150):

        super(pspnet,self).__init__()

        self.res=resnet50()

        self.psp=pyramid_pooling_module()

        self.final=nn.Sequential(

            nn.Conv2d(4096,512,kernel_size=3,stride=1,padding=1),

            nn.BatchNorm2d(512),

            nn.ReLU(inplace=True),

            nn.Dropout2d(p=0.1),

            nn.Conv2d(512,n_classes,kernel_size=1,stride=1),

        )

#         self.classifier=nn.Sequential(

#             nn.Conv2d(1024,512,kernel_size=3,stride=1,padding=1),

#             nn.BatchNorm2d(512),

#             nn.ReLU(inplace=True),

#             nn.Conv2d(512,n_classes,kernel_size=1,stride=1)

#         )

        

    def forward(self,x):

        f=self.res(x)

        out=self.psp(f)

        out=self.final(out)

        out=F.interpolate(out,(x.size(2),x.size(3)),mode='bilinear')

        #out=torch.sigmoid(out)

#         aux=self.classifier(class_f)

#         aux=F.interpolate(aux,(x.size(2),x.size(3)),mode='bilinear')

#         aux=torch.sigmoid(aux)

        return out
# device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model=pspnet().to(device)

# summary(model,(3,473,473))
from torch import optim

from torch.autograd import Variable

from torch.optim.lr_scheduler import MultiStepLR

from torch.optim.lr_scheduler import ReduceLROnPlateau

import time
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train():

    model.train()

    seg_criterion=nn.CrossEntropyLoss()

    #cls_criterion=nn.CrossEntropyLoss()

    epoch_losses=0

    start=time.time()

    

    for i,(image,label) in enumerate(train_loader):

        input=image.to(device)

        target=label.to(device).long()

        

        optimizer.zero_grad()

        out=model(input)

        seg_loss=seg_criterion(out,target)

        #cls_loss=cls_criterion(out_cls,target)

        #loss=seg_loss

        epoch_losses+=seg_loss.item()

        seg_loss.backward()

        optimizer.step()

        if i>=100:

            break

        

    print('Time for one epoch is {} sec'.format(time.time()-start))

    print('epoch {} loss {}'.format(epoch,epoch_losses))

    
def visual():

    with torch.no_grad():

        model.eval()

        gt=np.zeros((100,1,473,473))

        pred_out=np.zeros((100,1,473,473))

        #pred_outcls=np.zeros((100,1,473,473))

        for i,(image,label) in enumerate(val_loader):

            image=image.to(device)

            label=label.to(device)

            out,out_cls=model(image)

            gt[i,:,:,:]=label.cpu().numpy()

            pred_out[i,:,:,:]=np.asarray(np.argmax(out.cpu().numpy(),axis=1),dtype=np.uint8)

            #pred_outcls[i,:,:,:]=out_cls.cpu().numpy()

            if i>=10:

                break

        return gt,pred_out
# def visual():

#     with torch.no_grad():

#         model.eval()

#         gt=np.zeros((100,473,473))

#         pred_out=np.zeros((100,473,473))

#         pred_outcls=np.zeros((100,473,473))

#         #pred=np.zeros((100,150,473,473))

#         for i,(image,label) in enumerate(val_loader):

#             image=image.to(device)

#             label=label.to(device)

#             out,out_cls=model(image)

#             gt[i,:,:]=label.cpu().numpy()

#             pred_out[i,:,:]=torch.max(out,1)[1].squeeze_(0).cpu().numpy()

#             pred_outcls[i,:,:]=torch.max(out_cls,1)[1].squeeze_(0).cpu().numpy()

#             #pred[i,:,:,:]=out.squeeze_(0).cpu().numpy()

#             if i>=10:

#                 break

#         return gt,pred_out,pred_outcls
model=pspnet().to(device)

optimizer=optim.Adam(model.parameters(),lr=0.01)

scheduler=MultiStepLR(optimizer,milestones=[20,30],gamma=0.1)

for epoch in range(1,30+1):

    train()

    #test()

    #scheduler.step()

gt,pred_out=visual()

print(gt[2])

print(pred_out[2])



f,ax=plt.subplots(1,2)

ax[0].imshow(gt[2].reshape((473,473)),cmap='gray')

ax[1].imshow(pred_out[2].reshape((473,473)),cmap='gray')

#ax[2].imshow(pred_outcls[2].reshape((473,473)),cmap='gray')

#ax[3].imshow(pred[0,:,:,0],cmap='gray')
!nvidia-smi