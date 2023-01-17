import torch

import torchvision

from torchvision.datasets import ImageFolder

import torchvision.transforms as T

from torchvision import datasets

from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

import torchvision.utils as vutils

import torch.nn.parallel

import torch.backends.cudnn as cudnn

import torch.nn as nn

import torch.nn.functional as F

import numpy as np

import matplotlib.pyplot as plt

from torchvision.models import vgg19

%matplotlib inline

import pickle

from skimage import io, color
cuda = True if torch.cuda.is_available() else False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def display(img):

    plt.figure()

    plt.set_cmap('gray')

    plt.imshow(img)

    plt.show()



def combineLAB(l, a, b):

    shape = (l.shape[0], l.shape[1], 3)

    zeros = np.zeros(shape)

    zeros[:, :, 0] = l

    zeros[:, :, 1] = a

    zeros[:, :, 2] = b

    return zeros



def lab_normal_image(path):

    l, a ,b = load_img_for_training(path)

    return l, a ,b



def l_image(l):

    l=l.reshape(256,256)

    l=(l*50)+50

    shape = (l.shape[0], l.shape[1], 3)

    zeros = np.zeros(shape)

    zeros[:, :, 0] = l

    rgb = color.lab2rgb(zeros)

    return rgb



def a_image(a):

    a=a.reshape(256,256)

    a=a*100

    shape = (a.shape[0], a.shape[1], 3)

    zeros = np.zeros(shape)

    zeros[:, :, 1] = a

    rgb = color.lab2rgb(zeros)

    return rgb



def b_image(b):

    b=b.reshape(256,256)

    b=b*100

    shape = (b.shape[0], b.shape[1], 3)

    zeros = np.zeros(shape)

    zeros[:, :, 2] = b

    rgb = color.lab2rgb(zeros)

    return rgb





def rgb_image(l, a ,b):

    l=l.reshape(256,256)

    a=a.reshape(256,256)

    b=b.reshape(256,256)

    l=(l*50)+50

    a,b=a*100 , b*100

    lab = combineLAB(l, a ,b)

    rgb = color.lab2rgb(lab)

    return rgb



def load_img_for_training(img):

    #img = io.imread(img_path)

    #img = skimage.transform.resize(img,(256,256))

    lab = color.rgb2lab(img)

    l, a, b = (lab[:, :, 0]-50)/50, lab[:, :, 1]/100, lab[:, :, 2]/100 

    #lgray = get_l_from_gray(img)

    return l, a ,b

PATH = "../input/flowers-recognition/flowers/"

dataset_color = datasets.ImageFolder(root= PATH, transform=T.Compose([

                               T.Resize([256,256]),

                               ]))



print(len(dataset_color))

#print(len(dataset_color2))

#dataset_color=dataset_color1 +dataset_color2

#print(len(dataset_color))
class GAN_dataset(Dataset):

    def __init__(self, dataset_input,n):

        self.dataset1 = dataset_input

        self.n=n



    def __getitem__(self, index):

        x1,l1 = self.dataset1[index]

        l_dat,a_dat,b_dat=lab_normal_image(x1)

        l_dat=l_dat.reshape(1,256,256)

        a_dat=a_dat.reshape(1,256,256)

        b_dat=b_dat.reshape(1,256,256)

        l_dat = l_dat.astype('float32') 

        a_dat = a_dat.astype('float32')

        b_dat = b_dat.astype('float32')  

        l_dat=torch.from_numpy(l_dat)

        a_dat=torch.from_numpy(a_dat)

        b_dat=torch.from_numpy(b_dat)

        return l_dat,a_dat,b_dat



    def __len__(self):

        #return len(self.dataset1)

        return self.n
dataset = GAN_dataset(dataset_color,2500)
validation_split = .1

shuffle_dataset = True

random_seed= 42



# Creating data indices for training and validation splits:

dataset_size = len(dataset)

indices = list(range(dataset_size))

split = int(np.floor(validation_split * dataset_size))

if shuffle_dataset :

    np.random.seed(random_seed)

    np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]



# Creating PT data samplers and loaders:

train_sampler = SubsetRandomSampler(train_indices)

valid_sampler = SubsetRandomSampler(val_indices)



train_loader = torch.utils.data.DataLoader(dataset, batch_size=20,

                                           sampler=train_sampler,num_workers=2)

validation_loader = torch.utils.data.DataLoader(dataset, batch_size=10,

                                                sampler=valid_sampler,num_workers=2)
def double_conv(in_c, out_c):

  conv = nn.Sequential(

      nn.Conv2d(in_c, out_c, kernel_size = 3, padding = 1),

      nn.BatchNorm2d(out_c),

      nn.ReLU(inplace  = True),

      nn.Conv2d(out_c, out_c, kernel_size = 3, padding = 1),

      nn.BatchNorm2d(out_c),

      nn.ReLU(inplace  = True),

  )

  return conv



def crop_img(tensor, target_tensor):

  target_size = target_tensor.size()[2]

  tensor_size = tensor.size()[2]

  delta = tensor_size - target_size

  delta = delta//2

  return tensor[:,:, delta:tensor_size-delta, delta:tensor_size-delta]



class Unet(nn.Module):

  def __init__(self):

      

    super(Unet, self).__init__()



    self.max_pool_2x2 = nn.MaxPool2d(kernel_size = 1, stride  =2)

    self.down_conv_1 = double_conv(1, 64)

    self.down_conv_2 = double_conv(64, 128)

    self.down_conv_3 = double_conv(128, 256)

    self.down_conv_4 = double_conv(256, 512)

    self.down_conv_5 = double_conv(512, 1024)



    self.up_trans_1 = nn.ConvTranspose2d(in_channels = 1024, out_channels = 512, kernel_size = 2, stride = 2)

    self.up_conv_1 = double_conv(1024, 512)

    self.up_trans_2 = nn.ConvTranspose2d(in_channels = 512, out_channels = 256, kernel_size = 2, stride = 2)

    self.up_conv_2 = double_conv(512, 256)

    self.up_trans_3 = nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 2, stride = 2)

    self.up_conv_3 = double_conv(256, 128)

    self.up_trans_4 = nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 2, stride = 2)

    self.up_conv_4 = double_conv(128, 64)

    self.out = nn.Sequential(

        nn.Conv2d(in_channels = 64, out_channels = 1, kernel_size = 1),

        nn.BatchNorm2d(1),

        nn.Tanh()

        )





    



  def forward(self, image):

    #encoder

    x1 = self.down_conv_1(image)   #input 64x5x64x64

    #print(x1.shape)

    x2 = self.max_pool_2x2(x1)

    #print(x2.shape)

    x3 = self.down_conv_2(x2)     #  64x128x32x32

    #print(x3.shape)

    x4 = self.max_pool_2x2(x3)

    #print(x4.shape)

    x5 = self.down_conv_3(x4)     #  64x256x16x16

    #print(x5.shape)

    x6 = self.max_pool_2x2(x5)

    #print(x6.shape)

    x7 = self.down_conv_4(x6)     #  64x512x8x8

    #print(x7.shape)

    x8 = self.max_pool_2x2(x7)

    #print(x8.shape)

    x9 = self.down_conv_5(x8)    #   64x1024x4x4

    #print(x9.shape)



    #decoder

    x = self.up_trans_1(x9)

    y = crop_img(x7, x)

    x = self.up_conv_1(torch.cat([x, y], 1))



    x = self.up_trans_2(x)

    y = crop_img(x5, x)

    x = self.up_conv_2(torch.cat([x, y], 1))



    x = self.up_trans_3(x)

    y = crop_img(x3, x)

    x = self.up_conv_3(torch.cat([x, y], 1))



    x = self.up_trans_4(x)

    y = crop_img(x1, x)

    x = self.up_conv_4(torch.cat([x, y], 1))



    x = self.out(x)      #output size : 64x3x64x64





    return x

modela = Unet()

if  cuda:

  modela  = modela.cuda()



modelb = Unet()

if  cuda:

  modelb  = modelb.cuda()
criteriona = nn.MSELoss()

criterionb = nn.MSELoss()



if cuda:

  criteriona = criteriona.cuda()

if cuda:

  criterionb = criterionb.cuda()

# specify loss function

optimizera = torch.optim.Adam(modela.parameters(), lr=0.001)

optimizerb = torch.optim.Adam(modelb.parameters(), lr=0.001)
n_epochs = 135



for epoch in range(1, n_epochs+1):

    train_loss = 0.0

    validation_loss=0.0

    i=0

    for inp, outa ,outb in train_loader:

        i=i+1

        batch = inp.size(0)

        inp = inp.to(device)

        outa = outa.to(device)

        outb = outb.to(device)

        optimizera.zero_grad()       

        rta = modela(inp)   

        lossA = criteriona(rta, outa)

        lossA.backward()

        optimizera.step()

        train_loss += lossA.item()*inp.size(0)

        optimizerb.zero_grad()       

        rtb = modelb(inp)   

        lossB = criterionb(rtb, outb)

        lossB.backward()

        optimizerb.step()

        train_loss += lossB.item()*inp.size(0)

    for inp, outa, outb in validation_loader:

        batch = inp.size(0)

        inp = inp.to(device)

        outa = outa.to(device)

        outb = outb.to(device)

        rta = modela(inp)   

        lossA = criteriona(rta, outa)

        rtb = modelb(inp)   

        lossB = criterionb(rtb, outb)

        validation_loss += lossA.item()*inp.size(0) + lossB.item()*inp.size(0)      

    # print avg training statistics 

    train_loss = train_loss/len(train_loader)

    validation_loss=validation_loss/len(validation_loader)

    print('Epoch: {} \tTraining Loss A: {:.6f} \tValidation Loss A: {:.6f}'.format(epoch, train_loss,validation_loss))
filenamea = 'final_modela.sav'

pickle.dump(modela, open(filenamea, 'wb'))

filenameb = 'final_modelb.sav'

pickle.dump(modelb, open(filenameb, 'wb'))
valiter = iter(train_loader)

val_l,val_a,val_b = valiter.next()

i_l = val_l.numpy()

o_a = val_a.numpy()

o_b = val_b.numpy()
val_l=val_l.to(device)

output_A = modela(val_l)



output_A = output_A.cpu()

output_A = output_A.detach().numpy()

output_B = modelb(val_l)



output_B = output_B.cpu()

output_B = output_B.detach().numpy()
def imshow(img):

    plt.imshow(img)


fig = plt.figure(figsize=(35,8))

for idx in range(0,10):

    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])

    a=rgb_image(i_l[idx],o_a[idx],o_b[idx])

    plt.imshow(a)

fig = plt.figure(figsize=(35,8))

# display 20 images

for idx in range(0,10):

    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])

    a=rgb_image(i_l[idx],output_A[idx],output_B[idx])

    plt.imshow(a)
fig = plt.figure(figsize=(35,8))

for idx in range(10,20):

    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])

    a=rgb_image(i_l[idx],o_a[idx],o_b[idx])

    plt.imshow(a)

fig = plt.figure(figsize=(35,8))

# display 20 images

for idx in range(10,20):

    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])

    a=rgb_image(i_l[idx],output_A[idx],output_B[idx])

    plt.imshow(a)
valiter = iter(validation_loader)

val_l,val_a,val_b = valiter.next()

i_l = val_l.numpy()

o_a = val_a.numpy()

o_b = val_b.numpy()
modela = pickle.load(open(filenamea, 'rb'))

modelb = pickle.load(open(filenameb, 'rb'))
val_l=val_l.to(device)

output_A = modela(val_l)



output_A = output_A.cpu()

output_A = output_A.detach().numpy()

output_B = modelb(val_l)



output_B = output_B.cpu()

output_B = output_B.detach().numpy()


fig = plt.figure(figsize=(35,8))

for idx in range(0,10):

    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])

    a=rgb_image(i_l[idx],o_a[idx],o_b[idx])

    plt.imshow(a)

fig = plt.figure(figsize=(35,8))

# display 20 images

for idx in range(0,10):

    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])

    a=rgb_image(i_l[idx],output_A[idx],output_B[idx])

    plt.imshow(a)
fig = plt.figure(figsize=(35,8))

for idx in range(10,20):

    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])

    a=rgb_image(i_l[idx],o_a[idx],o_b[idx])

    plt.imshow(a)

fig = plt.figure(figsize=(35,8))

# display 20 images

for idx in range(10,20):

    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])

    a=rgb_image(i_l[idx],output_A[idx],output_B[idx])

    plt.imshow(a)
modela = pickle.load(open(filenamea, 'rb'))

modelb = pickle.load(open(filenameb, 'rb'))