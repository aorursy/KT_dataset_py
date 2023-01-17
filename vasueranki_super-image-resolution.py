import os

'''

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

'''

!pip install torchsummary 

#!pip install skimage

#!pip install pytorch_colours

from torchsummary import summary

#from pytorch_colours import colours

import numpy 

import pandas 

import torch 

import PIL

from PIL import Image

import matplotlib.pyplot as plt

from torch import nn,optim

import torch.nn.functional as F

from torchvision import datasets,transforms 

from torchvision.utils import save_image

import torchvision.transforms.functional as TF

import warnings 

warnings.filterwarnings("ignore")

batch_size=8
transform = transforms.Compose([transforms.Resize((256,256)),

                                transforms.RandomHorizontalFlip(p=0.5),

                                transforms.ToTensor(),

                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

output = datasets.ImageFolder("../input/caltech256/256_ObjectCategories/",transform=transform)

output_iterator = torch.utils.data.DataLoader(output,shuffle=True,batch_size=batch_size,drop_last=True)

print("Creating iterator object is complete")

'''

jj,_=next(iter(output_iterator))

jj.numpy()

jj=numpy.squeeze(jj)

plt.imshow(numpy.transpose(jj,(1,2,0)))

'''
class SISR(nn.Module):

    def __init__(self):

        super(SISR,self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=1,stride=1,padding=0,padding_mode='zeros')

        self.gn1   = nn.GroupNorm(num_groups=8,num_channels=16)

        #Output is 64x64x16

        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=2,stride=2,padding=0,padding_mode='zeros')

        self.gn2   = nn.GroupNorm(num_groups=16,num_channels=32)

        #Output is 32x32x32

        self.convt1= nn.ConvTranspose2d(in_channels=32,out_channels=64,kernel_size=4,stride=2,padding=1,padding_mode='zeros')

        self.gnt1  = nn.GroupNorm(num_groups=32,num_channels=64)

        #Output is 64x64x64

        self.convt2 = nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=4,stride=2,padding=1,padding_mode='zeros')

        self.gnt2 = nn.GroupNorm(num_groups=64,num_channels=64)

        #Output is 128x128x64

        self.convt3 = nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=2,stride=2,padding=0,padding_mode='zeros')

        self.gnt3 = nn.GroupNorm(num_groups=16,num_channels=32)

        #Output is 256x256x32

        self.conv3 = nn.Conv2d(in_channels=32,out_channels=16,kernel_size=1,stride=1,padding=0,padding_mode='zeros')

        self.gn3   = nn.GroupNorm(8,16)

        #Output is 256x256x16

        self.conv4 = nn.Conv2d(in_channels=16,out_channels=3,kernel_size=1,stride=1,padding=0,padding_mode='zeros')

        

        

        #Resblocks

        #For 64x64x16 to 64x64x64

        self.res1 = nn.Conv2d(in_channels=16,out_channels=64,kernel_size=1,stride=1,padding=0,padding_mode='zeros')

        self.gres = nn.GroupNorm(32,64)

        # For 32x32x32 to 128x128x64                      

        self.res2 = nn.ConvTranspose2d(in_channels=32,out_channels=64,kernel_size=4,stride=2,padding=1,padding_mode='zeros')

        self.gres1 =nn.GroupNorm(32,64)

        self.res3 = nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=2,stride=2,padding=0,padding_mode='zeros')

        self.gres2 = nn.GroupNorm(32,64)

        

    def forward(self,x):

            alpha=0.02

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            weight = torch.tensor([0.25]).to(device)

            a=F.leaky_relu(self.gn1(self.conv1(x)),alpha)

            b=F.leaky_relu(self.gn2(self.conv2(a)),alpha)

            x=F.leaky_relu(self.gnt1(self.convt1(b)),alpha)

            x= x+ F.prelu(self.gres(self.res1(a)),weight)

            x=F.leaky_relu(self.gnt2(self.convt2(x)),alpha)

            x = x+ F.prelu(self.gres2(self.res3(self.gres1(self.res2(b)))),weight)

            x=F.leaky_relu(self.gnt3(self.convt3(x)),alpha)

            x=F.leaky_relu(self.gn3(self.conv3(x)))

            x=F.prelu(self.conv4(x),weight)

            return x
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Network = SISR().to(device)

MSE = nn.MSELoss()

L1  = nn.L1Loss()

print(Network)

print(summary(Network,input_size=(3,64,64)))
epochs = 5

beta1= 0.5

beta2= 0.999

lr = 1e-4

Optimizer = optim.Adam(Network.parameters(),lr=lr,betas=(beta1,beta2))

scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(Optimizer,mode='min',factor=0.1,patience=int(epochs/10),verbose=True)
def DSSIM(ypred,ytrue):

    varx = torch.var(ytrue)

    vary = torch.var(ypred)

    ux = torch.mean(ytrue)

    uy = torch.mean(ypred)

    sum1 = ytrue-ux

    sum2 = ypred-uy

    length = 256**2

    cov = torch.sum(torch.mul(sum1,sum2))

    cov = cov/length

    c1 = 0.01*(4)

    c2 = 0.03*(4)

    SSIM = ((2*ux+uy)*(2*cov+c1))/((ux**2+uy**2+c1)*(varx**2+vary**2+c2))

    DSSIM = (1-SSIM)*0.5

    return DSSIM
for epoch in range(0,epochs):

    step=0

    for step,(GT,labels) in enumerate(output_iterator):

        LR = F.interpolate(GT,64)

        GT = GT.to(device)

        LR = LR.to(device)

        Pred = Network(LR)

        error_MSE = MSE(GT,Pred)

        error_L1 = 0.1*L1(GT,Pred)

        #error_DSSIM = DSSIM(GT,Pred)

        error_MSE.backward(retain_graph=True)

        error_L1.backward(retain_graph=True)

        #error_DSSIM.backward()

        Optimizer.step()

        if step%250==0:

            print('[%d/%d] [%d/%d] MSE Loss :%.4f L1 Loss %.4f '%(epoch+1,epochs,step,len(output_iterator),error_MSE.item(),error_L1.item(),error_DSSIM.item()))

        step = step+1

    scheduler1.step(error_MSE)        
if not os.path.exists("../output_images"):

    os.mkdir("../output_images")

Network.eval()

transform1 = transforms.Compose([transforms.Resize((256,256)),

                                transforms.ToTensor(),

                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

test = datasets.ImageFolder("../input/set-5-14-super-resolution-dataset/Set5/",transform=transform1)

test_iterator = torch.utils.data.DataLoader(test,shuffle=True,batch_size=1,drop_last=True)

print(test)

print(test_iterator)

print("Creating iterator object for testing is complete")

for ii,(test_image,_) in enumerate(test_iterator):

    LR = F.interpolate(test_image,64)

    LR=LR.to(device)

    test_image=test_image.to(device)

    HR = Network(LR)

    MSE_Loss = MSE(HR,test_image)

    #PSNR = 20*torch.log10(255/MSE_1)

    #print(PSNR.item())

    x=str(ii)

    save_image(HR[:,:,:]+1./2.,os.path.join('../output_images',x+'.png'))

    HR=HR.to('cpu').detach()

    HR=HR.numpy()

    HR=numpy.squeeze(HR)

    plt.imshow(numpy.transpose(HR,(1,2,0)))

        
import shutil 

shutil.make_archive('images','zip','../output_images')