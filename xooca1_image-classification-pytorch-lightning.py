import os

import collections

from datetime import datetime, timedelta



'''



os.environ["XRT_TPU_CONFIG"] = "tpu_worker;0;10.0.0.2:8470"



_VersionConfig = collections.namedtuple('_VersionConfig', 'wheels,server')

VERSION = "torch_xla==nightly"

CONFIG = {

    'torch_xla==nightly': _VersionConfig('nightly', 'XRT-dev{}'.format(

        (datetime.today() - timedelta(1)).strftime('%Y%m%d')))}[VERSION]



DIST_BUCKET = 'gs://tpu-pytorch/wheels'

TORCH_WHEEL = 'torch-{}-cp36-cp36m-linux_x86_64.whl'.format(CONFIG.wheels)

TORCH_XLA_WHEEL = 'torch_xla-{}-cp36-cp36m-linux_x86_64.whl'.format(CONFIG.wheels)

TORCHVISION_WHEEL = 'torchvision-{}-cp36-cp36m-linux_x86_64.whl'.format(CONFIG.wheels)



!export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

!apt-get install libomp5 -y

!apt-get install libopenblas-dev -y



!pip uninstall -y torch torchvision

!gsutil cp "$DIST_BUCKET/$TORCH_WHEEL" .

!gsutil cp "$DIST_BUCKET/$TORCH_XLA_WHEEL" .

!gsutil cp "$DIST_BUCKET/$TORCHVISION_WHEEL" .

!pip install "$TORCH_WHEEL"

!pip install "$TORCH_XLA_WHEEL"

!pip install "$TORCHVISION_WHEEL"

import torch_xla.core.xla_model as xm

'''
!ls
!pip install pytorch_lightning
import numpy as np 

import pandas as pd

from torchvision import datasets, transforms, models

import os

import shutil

import torch

import matplotlib.pyplot as plt

import numpy as np

import torch.nn.functional as F

import torch.nn as nn

#torchvision.utils.save_image

from torchvision import datasets, transforms ,utils

print(os.listdir("../input"))

from PIL import Image

import numpy as np

import cv2

from matplotlib import pyplot as plt

from IPython.display import display, HTML 

from matplotlib.pyplot import imshow

import numpy as np

from PIL import Image

import os

import pytorch_lightning as pl

##!pip install pretrainedmodels

#import pretrainedmodels

from torch.utils.data import DataLoader

from torchvision import transforms

from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.model_selection import train_test_split

test = pd.read_csv('../input/test_ApKoW4T.csv')

sample = pd.read_csv('../input/sample_submission_ns2btKE.csv')

train = pd.read_csv('../input/train/train.csv')
convertlabeldict = {1: 'Cargo', 

2:'Military', 

3:'Carrier', 

4:'Cruise', 

5:'Tankers'}

train['category_label'] = train['category'].map(convertlabeldict)
train.head()
train, holdout = train_test_split(train, test_size=0.1, random_state=0, 

                               stratify=train['category'])
train = train.reset_index(drop=True)

train.head()
holdout = holdout.reset_index(drop=True)

holdout.head()
train['category'].value_counts()
holdout['category'].value_counts()
#trainfilenames = train['image'].tolist()

basedir = '../input/train/images/'

destinationfolder = '../train/'

for i,row in train.iterrows():

    currentfileloc = basedir + row['image']

    newdirname = destinationfolder + str(row['category'])

    if not os.path.exists(newdirname):

        os.makedirs(newdirname)

    shutil.copy(currentfileloc, newdirname)
basedir = '../input/train/images/'

destinationfolder = '../holdout/'

for i,row in holdout.iterrows():

    currentfileloc = basedir + row['image']

    newdirname = destinationfolder + str(row['category'])

    if not os.path.exists(newdirname):

        os.makedirs(newdirname)

    shutil.copy(currentfileloc, newdirname)
basedir = '../input/train/images/'

destinationfolder = '../test/'

for i,row in test.iterrows():

    currentfileloc = basedir + row['image']

    #newdirname = destinationfolder

    if not os.path.exists(destinationfolder):

        os.makedirs(destinationfolder)

    shutil.copy(currentfileloc, destinationfolder)
!ls ../
def random_brightness(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    rand = random.uniform(0.6, 1.0)

    hsv[:, :, 2] = rand*hsv[:, :, 2]

    rand = random.uniform(1.0, 1.5)

    hsv[:, :, 1] = rand*hsv[:, :, 1]

    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return new_img





def zoom(image,rows,cols):

    zoom_pix = random.randint(5, 10)

    zoom_factor = 1 + (2*zoom_pix)/rows

    image = cv2.resize(image, None, fx=zoom_factor,

                       fy=zoom_factor, interpolation=cv2.INTER_LINEAR)

    top_crop = (image.shape[0] - rows)//2

    left_crop = (image.shape[1] - cols)//2

    image = image[top_crop: top_crop+rows,

                  left_crop: left_crop+cols]

    return image
import random



def createaugimagesv2(dirname,no_of_images):

    filename = os.listdir(dirname)

    filename = random.sample(filename, no_of_images)

    for images in filename:

        if images[-8:]!='_enh.jpg' and images[-9:]!='_enh1.jpg':

            imagepath = dirname + images

            image = cv2.imread(imagepath)

            rows,cols,channel = image.shape

            image = np.fliplr(image)



            op1 = random.randint(0, 1)

            op2 = random.randint(0, 1)

            op3 = random.randint(0, 1)

            if op1:

                image = random_brightness(image)

            if op2:

                image = zoom(image,rows,cols)

            newimagepath = dirname + images.split('.')[0] + '_enh.jpg'

            try:

                image = cv2.resize(image, (224, 224))

                cv2.imwrite(newimagepath, image)

            except:

                print("file {0} is not converted".format(images))
import random

def im_convert(tensor):

    image = tensor.clone().detach().numpy()

    image = image.transpose(1, 2, 0)

    image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))

    image = image.clip(0, 1)

    return image

def createaugimages(dirname,no_of_images):

    filename = os.listdir(dirname)

    filename = random.sample(filename, no_of_images)

    for images in filename:

        if images[-9:]!='_enh1.jpg':

            imagepath = dirname + images

            pil_im = Image.open(imagepath, 'r').convert('RGB')

            op1 = random.randint(0, 1)

            if op1 ==1:

                changeimg = transforms.Compose([ 

                                        transforms.RandomRotation(5),

                                        transforms.Resize(224),

                                        transforms.ToTensor()

                                       ])

            else:

                changeimg = transforms.Compose([ 

                            transforms.RandomHorizontalFlip(),

                            transforms.RandomRotation(10),

                            transforms.Resize(224),

                            transforms.ToTensor()

                           ])



            img = changeimg(pil_im)

            newimagepath = dirname + images.split('.')[0] + '_enh1.jpg'

            utils.save_image(img,newimagepath)   
def resizeall(dirname):

    filename = os.listdir(dirname)

    non3channel = []

    for images in filename:

        imagepath = dirname + images

        image = cv2.imread(imagepath)

        if image.shape[2] !=3:

            non3channel.append(images)

    return non3channel
convertlabeldict = {1: 'Cargo', 

2:'Military', 

3:'Carrier', 

4:'Cruise', 

5:'Tankers'}
a = 1908 - 1095

b = 1908 - 1050

c = 1908 - 824

d = 1908 - 749
dirname = '../train/5/'

no_of_images = a

createaugimagesv2(dirname,no_of_images)

dirname = '../train/2/'

no_of_images = b

createaugimagesv2(dirname,no_of_images)

dirname = '../train/3/'

no_of_images = 824

createaugimagesv2(dirname,no_of_images)

dirname = '../train/4/'

no_of_images = 749

createaugimagesv2(dirname,no_of_images)
dirname = '../train/1/'

no_of_images = 500

createaugimages(dirname,no_of_images)

dirname = '../train/5/'

no_of_images = 500

createaugimages(dirname,no_of_images)

dirname = '../train/2/'

no_of_images = 500

createaugimages(dirname,no_of_images)

dirname = '../train/3/'

no_of_images = 824

createaugimages(dirname,no_of_images)

dirname = '../train/4/'

no_of_images = 749

createaugimages(dirname,no_of_images)
dirname = '../train/3/'

no_of_images = 288

createaugimagesv2(dirname,no_of_images)

dirname = '../train/4/'

no_of_images = 400

createaugimagesv2(dirname,no_of_images)
!ls ../train

!ls ../train/1 | wc -l

!ls ../train/5 | wc -l

!ls ../train/2 | wc -l

!ls ../train/4 | wc -l

!ls ../train/3 | wc -l
!ls ../holdout

!ls ../holdout/1 | wc -l

!ls ../holdout/5 | wc -l

!ls ../holdout/2 | wc -l

!ls ../holdout/4 | wc -l

!ls ../holdout/3 | wc -l
class Pl_densenet201(pl.LightningModule):



    def __init__(self,lr=0.001,classes=5):

        super(Pl_densenet201, self).__init__()

        self.lr = lr

        self.classes = classes

        self.create_densenet201_model()



        

    def create_densenet201_model(self):

        self.models = models.densenet201(pretrained=True)

        self.models.classifier = nn.Sequential(nn.Linear(1920, 720),

                                nn.ReLU(),

                                nn.Dropout(0.5),

                                nn.Linear(720, 256),

                                nn.ReLU(),

                                nn.Dropout(0.4),

                                nn.Linear(256, 64),

                                nn.ReLU(),

                                nn.Dropout(0.3),

                                nn.Linear(64, self.classes))



    def forward(self, x):

        output = self.models(x)

        return output



    def training_step(self, batch, batch_idx):

        images, target = batch

        preds = self.forward(images)

        loss = F.cross_entropy(preds, target)

        tensorboard_logs = {'train_loss': loss}

        return {'loss': loss, 'log': tensorboard_logs}

    

    def validation_step(self, batch, batch_idx):

        images, target = batch

        preds = self.forward(images)

        loss = F.cross_entropy(preds, target)

        return {'val_loss': loss}



    def validation_end(self, outputs):

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        #print(avg_loss)

        tensorboard_logs = {'val_loss': avg_loss}

        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

        

    def test_step(self, batch, batch_idx):

        images, target = batch

        preds = self.forward(images)

        return {'test_loss': F.cross_entropy(preds, target)}



    def test_end(self, outputs):

        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        tensorboard_logs = {'test_loss': avg_loss}

        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}



    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.models.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

        return [optimizer], [scheduler]



def data_transforms(transform_type ='train'):

    if transform_type=='train':

        transforms_ret = transforms.Compose([

                                    transforms.RandomResizedCrop(256,scale=(0.8, 1.0),ratio=(0.75, 1.33)),

                                    transforms.RandomRotation(degrees=15),

                                    transforms.CenterCrop(224),

                                    transforms.ToTensor(),

                                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

                                   ])

    else:

        transforms_ret = transforms.Compose([transforms.Resize(256),

                                transforms.CenterCrop(224), 

                                transforms.ToTensor(),

                                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

                              ])

    return transforms_ret



def prepare_data(train_dir,test_dir,validation_split=0.2):

    test_dataset = datasets.ImageFolder(test_dir,transform=data_transforms(transform_type='test'))

    train_dataset = datasets.ImageFolder(train_dir,transform=data_transforms(transform_type='train'))

    val_len = int(len(train_dataset) * validation_split)

    train_len = int(len(train_dataset) - val_len)

    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_len, val_len])

    print(len(test_dataset),len(train_dataset),len(val_dataset))

    return train_dataset,val_dataset,test_dataset



def create_dataloader(dataset,use_tpu=False,batch_size=64):

    sampler = None

    if use_tpu:

        sampler = torch.utils.data.distributed.DistributedSampler(

            dataset,

            num_replicas=xm.xrt_world_size(),

            rank=xm.get_ordinal(),

            shuffle=True

        )



        loader = torch.utils.data.DataLoader(

            dataset,

            sampler=sampler,

            batch_size=batch_size

        )

    else:

        loader = torch.utils.data.DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle = True

        )

    return loader





from pytorch_lightning import Trainer

from argparse import Namespace

train_dir = '/kaggle/train/'

test_dir = '/kaggle/holdout/'



train_dataset,val_dataset,test_dataset = prepare_data(train_dir,test_dir,validation_split=0.2)



train_dataloader = create_dataloader(train_dataset,use_tpu=False,batch_size=64)

val_dataloader = create_dataloader(val_dataset,use_tpu=False,batch_size=64)

test_dataloader = create_dataloader(test_dataset,use_tpu=False,batch_size=64)



model = Pl_densenet201(lr=0.001,classes=5)



# most basic trainer, uses good defaults

#trainer = Trainer(num_tpu_cores=8, precision=16,max_epochs=2)

trainer = Trainer(gpus=1, max_epochs=22)

#trainer = Trainer()

#trainer = Trainer(num_tpu_cores=8, max_epochs=2, precision=16)

trainer.fit(model,train_dataloader,val_dataloader,test_dataloader) 
def predict(model, test_image_name,transform,image_size,modeltype='other'):

    test_image = Image.open(test_image_name).convert('RGB')

    test_image_tensor = transform(test_image)

    if torch.cuda.is_available():

        test_image_tensor = test_image_tensor.view(1, 3, image_size, image_size).cuda()

    else:

        test_image_tensor = test_image_tensor.view(1, 3, image_size, image_size)

    with torch.no_grad():

        model.eval()

        if modeltype == 'inception':

            out = model(test_image_tensor)[0]

        else:

            out = model(test_image_tensor)

        ps = torch.exp(out)

    return test_image_name,ps
def extractfilename(val):

    return os.path.split(val)[1]



def maxtensorval(val):

    #ps = torch.exp(val)

    #ps = F.softmax(val,dim=1)

    top_p, top_class = val.topk(1)

    return top_class+1



image_size = 224 

test_transforms = transforms.Compose([transforms.Resize(256),

                                    transforms.CenterCrop(image_size), 

                                  transforms.ToTensor(),

#                                  transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

                              ])

basedir = '../test/'

prediction_df = pd.DataFrame(columns=['image', 'category'])

modelname = 'other'

print("Prediction of model {0} started".format(modelname))

for i,row in test.iterrows():

    pathfile = basedir + row['image']

    test_image_name,imagetype = predict(model, pathfile,test_transforms,image_size,modeltype=modelname)

    prediction_df.loc[i] = [test_image_name,imagetype]

print("Prediction for model {0} completed".format(modelname))

prediction_df['image']  = prediction_df['image'].apply(extractfilename)

prediction_df['category'] = prediction_df['category'].apply(maxtensorval)

prediction_df['category'] = prediction_df['category'].apply(int)

prediction_df.head()
prediction_df['category'].value_counts()
prediction_df.to_csv('prediction.csv',index=False)