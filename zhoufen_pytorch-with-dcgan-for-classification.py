# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# import os

import numpy as np

# import math



# import torchvision.transforms as transforms

from torchvision.utils import save_image



from torch.utils.data import DataLoader

from torchvision import datasets

from torch.autograd import Variable



import torch.nn as nn

import torch.nn.functional as F

import torch

import matplotlib.pyplot as plt



# from sklearn.model_selection import train_test_split
# Load data to DataLoader

train = pd.read_csv(r"/kaggle/input/digit-recognizer/train.csv",dtype = np.float32)



train_lables = train.label.values

train_features = train.loc[:,train.columns != "label"].values/255 # normalization



# create feature and targets tensor for train set. As you remember we need variable to accumulate gradients. Therefore first we create tensor, then we will create variable

train_features_torch = torch.from_numpy(train_features)

train_labels_torch = torch.from_numpy(train_lables).type(torch.IntTensor) # data type is int



batch_size = 128



# Pytorch train and test sets

train = torch.utils.data.TensorDataset(train_features_torch,train_labels_torch)

train_loader = DataLoader(train, batch_size = batch_size, shuffle = True)



# visualize one of the images in data set

plt.imshow(train_features[10].reshape(28,28))

plt.axis("off")

plt.title(str(train_lables[10]))

plt.savefig('graph.png')

plt.show()
class Map(dict):

    """

    To define a dictionary of parameters

    Example:

    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])

    """

    def __init__(self, *args, **kwargs):

        super(Map, self).__init__(*args, **kwargs)

        for arg in args:

            if isinstance(arg, dict):

                for k, v in arg.items():

                    self[k] = v



        if kwargs:

            for k, v in kwargs.items():

                self[k] = v



    def __getattr__(self, attr):

        return self.get(attr)



    def __setattr__(self, key, value):

        self.__setitem__(key, value)



    def __setitem__(self, key, value):

        super(Map, self).__setitem__(key, value)

        self.__dict__.update({key: value})



    def __delattr__(self, item):

        self.__delitem__(item)



    def __delitem__(self, key):

        super(Map, self).__delitem__(key)

        del self.__dict__[key]

opt = Map(n_epochs=200, batch_size=64, lr=0.0002, b1=0.5, b2=0.999, n_cpu=8, latent_dim=100, num_classes=10, img_size=28, channels=1, sample_interval=400)
def weights_init_normal(m):

    classname = m.__class__.__name__

    if classname.find("Conv") != -1:

        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find("BatchNorm") != -1:

        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)

        torch.nn.init.constant_(m.bias.data, 0.0)

        



class Generator(nn.Module):

    def __init__(self):

        super(Generator, self).__init__()



        self.label_emb = nn.Embedding(opt.num_classes, opt.latent_dim)



        self.init_size = opt.img_size // 4  # Initial size before upsampling

        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))



        self.conv_blocks = nn.Sequential(

            #input size: (100, 128, 7, 7)

            nn.BatchNorm2d(128),

            nn.Upsample(scale_factor=2),

            nn.Conv2d(128, 128, 3, stride=1, padding=1),

            #out size: (100, 128, 7, 7)

            nn.BatchNorm2d(128, 0.8),

            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),

            nn.Conv2d(128, 64, 3, stride=1, padding=1),

            #state size: (100, 64, 28, 28)

            nn.BatchNorm2d(64, 0.8),

            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),

            #state size: (100, 1, 28, 28)

            nn.Tanh(),

        )



    def forward(self, noise):

        #input size: (100,)

        out = self.l1(noise)

        #out size: (100, 128*7**2)

        out = out.view(out.shape[0], 128, self.init_size, self.init_size)

        #out size: (100, 128, 7, 7)

        img = self.conv_blocks(out)

        #img size: (100, 1, 28, 28)

        return img





class Discriminator(nn.Module):

    def __init__(self):

        super(Discriminator, self).__init__()

        ndf = 28

        nc = 1

        self.main = nn.Sequential(

            # input is (nc) x 28 x 28

            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),

            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf) x 14 x 14

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),

            nn.BatchNorm2d(ndf * 2),

            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*2) x 7 x 7

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),

            nn.BatchNorm2d(ndf * 4),

            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*4) x 3 x 3

            nn.Conv2d(ndf * 4, ndf*8, 4, 2, 1, bias=False),

            #state size. (ndf*8, 1, 1)

        )

        

        # Multi-output layers

        # Unsupervised layer

        self.adv_layer = nn.Sequential(nn.Linear(ndf*8, 1), nn.Sigmoid())

        # Supervised layer

        self.aux_layer = nn.Sequential(nn.Linear(ndf*8, opt.num_classes + 1), nn.Softmax())





    def forward(self, img):

        out = self.main(img)

        out = out.view(out.shape[0], -1)

        validity = self.adv_layer(out)

        label = self.aux_layer(out)



        return validity, label

cuda = True if torch.cuda.is_available() else False



# Loss functions

adversarial_loss = torch.nn.BCELoss()

auxiliary_loss = torch.nn.CrossEntropyLoss()



# Initialize generator and discriminator

generator = Generator()

discriminator = Discriminator()



if cuda:

    generator.cuda()

    discriminator.cuda()

    adversarial_loss.cuda()

    auxiliary_loss.cuda()



# Initialize weights

generator.apply(weights_init_normal)

discriminator.apply(weights_init_normal)



# Optimizers

optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))



FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor



# ----------

#  Training

# ----------



for epoch in range(opt.n_epochs):

    for i, (imgs, labels) in enumerate(train_loader):



        batch_size = imgs.shape[0]



        # Adversarial ground truths

        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)

        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        fake_aux_gt = Variable(LongTensor(batch_size).fill_(opt.num_classes), requires_grad=False)



        # Configure input

        real_imgs = Variable(imgs.type(FloatTensor))

        real_imgs = real_imgs.view(batch_size, 1, 28, 28)

        labels = Variable(labels.type(LongTensor))



        # -----------------

        #  Train Generator

        # -----------------



        optimizer_G.zero_grad()



        # Sample noise and labels as generator input

        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))



        # Generate a batch of images

        gen_imgs = generator(z)



        # Loss measures generator's ability to fool the discriminator

        validity, _ = discriminator(gen_imgs)

        g_loss = adversarial_loss(validity, valid)



        g_loss.backward()

        optimizer_G.step()

        



        # ---------------------

        #  Train Discriminator

        # ---------------------



        optimizer_D.zero_grad()



        # Loss for real images

        real_pred, real_aux = discriminator(real_imgs)

        d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2



        # Loss for fake images

        fake_pred, fake_aux = discriminator(gen_imgs.detach())

        d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, fake_aux_gt)) / 2



        # Total discriminator loss

        d_loss = (d_real_loss + d_fake_loss) / 2



        # Calculate discriminator accuracy

        pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)

        gt = np.concatenate([labels.data.cpu().numpy(), fake_aux_gt.data.cpu().numpy()], axis=0)

        d_acc = np.mean(np.argmax(pred, axis=1) == gt)



        d_loss.backward()

        optimizer_D.step()



        print(

            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"

            % (epoch, opt.n_epochs, i, len(train_loader), d_loss.item(), 100 * d_acc, g_loss.item())

        )
test_df = pd.read_csv(r"/kaggle/input/digit-recognizer/test.csv",dtype = np.float32)

test_features_numpy = test_df.loc[:,test_df.columns != "label"].values/255 # normalization

test_featuresTrain = torch.from_numpy(test_features_numpy)

test = torch.utils.data.TensorDataset(test_featuresTrain)

test_loader = DataLoader(test, batch_size = batch_size, shuffle = False)



pre_label = np.zeros(1)

for i, imgs in enumerate(test_loader):



    batch_size = imgs[0].shape[0]



    real_imgs = Variable(imgs[0].type(FloatTensor))

    real_imgs = real_imgs.view(batch_size, 1, 28, 28)

    real_pred, real_aux = discriminator(real_imgs)



    # Calculate discriminator accuracy

    pred = np.concatenate([real_aux.data.cpu().numpy()], axis=0)

    pred_output = [np.argmax(arr) if np.argmax(arr)<10 else arr.argsort()[-2] for arr in pred]

    pre_label = np.concatenate([pre_label, pred_output], axis=0)



result_df = pd.DataFrame({'ImageId': range(1, pre_label[1:].shape[0]+1),'Label': pre_label[1:]})

result_df = result_df.astype('int32')

result_df.to_csv('csv_to_submit.csv', header=['ImageId', 'Label'], index=False)