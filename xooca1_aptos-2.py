!pip install albumentations



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

import math

import os

from albumentations.augmentations.transforms import *

from albumentations import (

    Compose, ToFloat, FromFloat, RandomRotate90, Flip, OneOf, MotionBlur, MedianBlur, Blur,

    ShiftScaleRotate, OpticalDistortion, GridDistortion, RandomBrightnessContrast,

    HueSaturationValue,

)

##!pip install pretrainedmodels

#import pretrainedmodels
test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

sample = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
train.head()
!ls /kaggle/input
from pathlib import Path

from sklearn.model_selection import train_test_split

#train_80 = train.sample(frac=0.8)

train_80,validate = train_test_split(train,stratify=train['diagnosis'],test_size=0.2)

train_80['image_loc']='/kaggle/images/train/'

train_80['basedir']='/kaggle/input/aptos2019-blindness-detection/train_images/'

validate = train.loc[~train.index.isin(train_80.index)]

validate['image_loc']='/kaggle/images/val/'

validate['basedir']='/kaggle/input/aptos2019-blindness-detection/train_images/'

train.shape,train_80.shape,validate.shape,test.shape
train = pd.concat([train_80,validate])

train = train.reset_index(drop=True)
train.tail()
train_80['diagnosis'].value_counts()

validate['diagnosis'].value_counts()
def augment(aug, image):

    return aug(image=image)['image']
def crop_image_from_gray(img,tol=7):

    if img.ndim ==2:

        mask = img>tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    elif img.ndim==3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        mask = gray_img>tol

        

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0): # image is too dark so that we crop out everything,

            return img # return original image

        else:

            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

    #         print(img1.shape,img2.shape,img3.shape)

            img = np.stack([img1,img2,img3],axis=-1)

    #         print(img.shape)

        return img
def transform_image(p=0.5):

    return Compose([

        #ToFloat(),

        OneOf([RandomRotate90(),

        Flip()]),

        #RandomRotate90(),

        #Flip(),

        #RandomSizedBBoxSafeCrop(224, 224, erosion_rate=0.0, interpolation=1, always_apply=False, p=1.0),

        #MedianBlur(blur_limit=3, p=1.0),

        #OneOf([

        #    MotionBlur(p=0.2),

        #    GaussianBlur(blur_limit=5, always_apply=True, p=0.5),

       #     MedianBlur(blur_limit=3, p=0.1),

        #    Blur(blur_limit=3, p=0.1),

        #], p=0.2),

        #CLAHE(clip_limit=(2,2), tile_grid_size=(10, 10), always_apply=True, p=0.5),

        #InvertImg(always_apply=True, p=1.0),

        #GaussianBlur(blur_limit=5, always_apply=True, p=1.0),

        #RGBShift(r_shift_limit=(20,20), g_shift_limit=(30,30), b_shift_limit=(40,40), always_apply=True, p=1.0),

        #ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=1.0),

        #RandomBrightnessContrast(brightness_limit=(-0.2,0.4), contrast_limit=(0.2,0.2), always_apply=False, p=0.5),

        #GaussianBlur(blur_limit=7, always_apply=True, p=1.0),

        #MedianBlur(blur_limit=3, p=1.0),

        #Blur(blur_limit=3, p=1.0),

        #OpticalDistortion(p=1.0),

        #OneOf([

        #    OpticalDistortion(p=0.3),

        #    GridDistortion(p=0.1),

        #], p=0.2),   

        #HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5),

        #HueSaturationValue(hue_shift_limit=(-2,2), sat_shift_limit=20, val_shift_limit=20, p=1.0),

    ], p=p)
def al_transform(imageloc):

    img = cv2.imread(imageloc)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #img = img*2

    #img = crop_image_from_gray(img,tol=7)

    #augmentation = transform_image(p=0.9)

    #final = augment(augmentation,img)

    #final = augment(augmentation,np.array(img))

    #final = final*2

    #print(final)

    #print(augmentation)

    #img = augmentation(image=img)['image']

    #final = Image.fromarray(img)

    return img
basedir = '/kaggle/input/aptos2019-blindness-detection/train_images/6194e0fff071.png'

#img = cv2.imread(basedir)

img = al_transform(basedir)

#newimagepath = newdirname + '/'+row['id_code'] +'.png'

#cv2.imwrite(newimagepath, image)

plt.figure(figsize=(6, 6))

plt.imshow(img)
#basedir = '../input/aptos2019-blindness-detection/train_images/61bbc11fe503.png'

img = cv2.imread(basedir)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#img = al_transform(basedir)

#newimagepath = newdirname + '/'+row['id_code'] +'.png'

#cv2.imwrite(newimagepath, image)

plt.figure(figsize=(6, 6))

plt.imshow(img)
#!rm -rf '../imagestrain'

#!rm -rf '../imagestest'
!ls 
#trainfilenames = train['image'].tolist()

#basedir = '../input/aptos2019-blindness-detection/train_images/'

#destinationfolder = '/kaggle/images/'

for i,row in train.iterrows():

    currentfileloc = str(row['basedir']) + row['id_code'] +'.png'

    newdirname = str(row['image_loc']) +str(row['diagnosis'])

    #print(currentfileloc)

    if not os.path.exists(newdirname):

        os.makedirs(newdirname)

    image = al_transform(str(currentfileloc))

    newimagepath = newdirname + '/'+row['id_code'] +'.png'

    cv2.imwrite(newimagepath, image)
basedir = '/kaggle/input/aptos2019-blindness-detection/test_images/'

destinationfolder = '/kaggle/images/'

for i,row in test.iterrows():

    currentfileloc = basedir + row['id_code'] +'.png'

    newdirname = destinationfolder + 'test/'

    if not os.path.exists(newdirname):

        os.makedirs(newdirname)

    image = al_transform(currentfileloc)

    newimagepath = newdirname + row['id_code'] +'.png'

    cv2.imwrite(newimagepath, image)
!ls /kaggle/images/train/0 | wc -l

!ls /kaggle/images/train/1 | wc -l

!ls /kaggle/images/train/2 | wc -l

!ls /kaggle/images/train/3 | wc -l

!ls /kaggle/images/train/4 | wc -l
def new_images(dirname,diagnosis,newdirname,no_of_images):

    files = [dirname+file+'.png' for file in train[train.diagnosis==diagnosis]['id_code'].tolist()]

    if no_of_images > 0:

        files = random.sample(files, no_of_images)

    for file in files:

        image = al_transform(file)

        newimagepath = newdirname + str(diagnosis) +'/'+file.split('/')[-1].split('.')[0] + '_enh'+'.png'

        #print(newimagepath)

        cv2.imwrite(newimagepath, image)

dirname = '../input/aptos2019-blindness-detection/train_images/'    

newdirname = '../images/train/'

imagenum=[0,0,0,0]

diagnosis = [1,2,3,4]

#for imagenum,diagnosis in zip(imagenum,diagnosis):

#    new_images(dirname,diagnosis,newdirname,imagenum)
!ls /kaggle/images/train/0 | wc -l

!ls /kaggle/images/train/1 | wc -l

!ls /kaggle/images/train/2 | wc -l

!ls /kaggle/images/train/3 | wc -l

!ls /kaggle/images/train/4 | wc -l
!ls 
!ls /kaggle/images/val/0 | wc -l

!ls /kaggle/images/val/1 | wc -l

!ls /kaggle/images/val/2 | wc -l

!ls /kaggle/images/val/3 | wc -l

!ls /kaggle/images/val/4 | wc -l
basedir = '../images/test/0005cfc8afb6.png'

img = cv2.imread(basedir)

img = al_transform(currentfileloc)

#newimagepath = newdirname + '/'+row['id_code'] +'.png'

#cv2.imwrite(newimagepath, image)

plt.figure(figsize=(6, 6))

plt.imshow(img)
!ls ../images/test
basedir = '../input/aptos2019-blindness-detection/train_images/042470a92154.png'

#img = cv2.imread(basedir)

img = al_transform(basedir)

#newimagepath = newdirname + '/'+row['id_code'] +'.png'

#cv2.imwrite(newimagepath, image)

plt.figure(figsize=(10, 10))

plt.imshow(img)
test.head()
train.head()
sample.head()
convertlabeldict = {0: 'No DR', 

1:'Mild', 

2:'Moderate', 

3:'Severe', 

4:'Proliferative DR'}

#train['diagnosis'] = train['diagnosis'].map(convertlabeldict)
train.head()
train['diagnosis'].value_counts()
!ls ../imagestrain/3
data = '../imagestrain/3/50a2aef380c8.png'

pil_im = Image.open(data, 'r')

imshow(np.asarray(pil_im))
def make_weights_for_balanced_classes(images, nclasses):                        

    count = [0] * nclasses                                                      

    for item in images:                                                         

        count[item[1]] += 1                                                     

    weight_per_class = [0.] * nclasses                                      

    N = float(sum(count))                                                   

    for i in range(nclasses):                                                   

        weight_per_class[i] = N/float(count[i])                                 

    weight = [0] * len(images)                                              

    for idx, val in enumerate(images):                                          

        weight[idx] = weight_per_class[val[1]]                                  

    return weight 



def load_split_train_test(datadir, batch_size=32,image_size=224,train_ind=True):

    train_transforms = transforms.Compose([transforms.Resize(256),

                                            transforms.RandomHorizontalFlip(),

                                            transforms.RandomRotation(10),

                                           transforms.CenterCrop(image_size),

                                          transforms.ToTensor(),

                                          transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

                                       ])

    data_set = datasets.ImageFolder(datadir,transform=train_transforms)

    

    if train_ind:

        weights = make_weights_for_balanced_classes(data_set.imgs, len(data_set.classes))                                                                

        weights = torch.DoubleTensor(weights)                                       

        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights)) 

        loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, #shuffle = True,                              

                                    sampler = sampler, 

                                     num_workers=4, pin_memory=True)

    else:

        sampler=None

        loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle = True,                              

                                            sampler = sampler, 

                                             num_workers=4, pin_memory=True)



    return loader
data_dir = '../images/train'

batch_size = 32

image_size=224

#valsize = 0.2

#train_transforms = transforms.Compose([transforms.Resize(image_size),

#                                      transforms.ToTensor(),

#                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

#                                   ])

#training_loader = load_split_train_test(data_dir,batch_size,image_size,train_ind=True)

#train_data = datasets.ImageFolder(data_dir,transform=train_transforms)

#trainloader = torch.utils.data.DataLoader(train_data,batch_size=batch_size)
#for data,target in training_loader:

#    print(data)

#    print(target)
classes = ('No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR') 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device
#checkpoint_densenet201 = torch.load("../input/pytorch-pretrained-image-models/densenet201.pth")

#checkpoint_densenet121 = torch.load("../input/pytorch-pretrained-image-models/densenet121.pth")

#checkpoint_resnet50 = torch.load("../input/pytorch-pretrained-image-models/resnet50.pth")

#checkpoint_resnet34 = torch.load("../input/pytorch-pretrained-image-models/resnet34.pth")

#checkpoint_inceptionv3 = torch.load("../input/pretrained-pytorch-models/inception_v3_google-1a9a5a14.pth")

#checkpoint_densenet161 = torch.load("../input/pretrained-pytorch-models/densenet161-17b70270.pth")

#checkpoint_densenet161 = torch.load("../input/resnet152/resnet152.pth")
def definemodel(modelnumber,pretrained=False,freezelonlylastlayer = 'yes',lr=0.0001):

    train_data_dir = '/kaggle/images/train'

    validate_data_dir = '/kaggle/images/val'

    valsize=0.2

    if modelnumber == 0:

        batch_size = 32

        image_size=224

        #training_loader, validation_loader = load_split_train_test(data_dir,batch_size,valsize,image_size)

        training_loader = load_split_train_test(train_data_dir,batch_size,image_size,train_ind=False)

        validation_loader = load_split_train_test(validate_data_dir,batch_size,image_size,train_ind=False)

        if pretrained:

            model  = models.densenet201(pretrained=True)

        else:

            model  = models.densenet201(pretrained=False)

            checkpoint = torch.load("../input/pytorch-pretrained-image-models/densenet201.pth")

            model.load_state_dict(checkpoint)

        #criterion = nn.CrossEntropyLoss()        

        if freezelonlylastlayer == 'yes':    

            for param in model.parameters():

                param.requires_grad = False

            model.classifier = nn.Sequential(                        

                        nn.BatchNorm1d(1920, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

                          nn.Dropout(p=0.4),

                          nn.Linear(in_features=1920, out_features=1920, bias=True),

                          nn.LeakyReLU(),

                          #nn.BatchNorm1d(1920, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

                          #nn.Dropout(p=0.5),

                          nn.Linear(in_features=1920, out_features=5, bias=True),)

            optimizer = torch.optim.Adam(model.classifier.parameters(), lr = lr, weight_decay=1e-5) 

        else:

            model.classifier = nn.Sequential(                        

                        nn.BatchNorm1d(1920, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

                          nn.Dropout(p=0.4),

                          nn.Linear(in_features=1920, out_features=1920, bias=True),

                          nn.LeakyReLU(),

                          #nn.BatchNorm1d(1920, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

                          #nn.Dropout(p=0.5),

                          nn.Linear(in_features=1920, out_features=5, bias=True),)

            optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=1e-5) 

        criterion = nn.CrossEntropyLoss()

        modelname = 'densenet201'

    elif modelnumber == 1:

        batch_size = 64

        image_size=224

        training_loader = load_split_train_test(train_data_dir,batch_size,image_size,train_ind=False)

        validation_loader = load_split_train_test(validate_data_dir,batch_size,image_size,train_ind=False)

        #training_loader, validation_loader = load_split_train_test(data_dir,batch_size,valsize,image_size)

        #model = models.resnet50(pretrained=True)

        if pretrained:

            model  = models.resnet50(pretrained=True)

        else:

            model  = models.resnet50(pretrained=False)

            checkpoint = torch.load("../input/pytorch-pretrained-image-models/resnet50.pth")

            model.load_state_dict(checkpoint)

        #for param in model2.parameters():

        #    param.requires_grad = False



        #criterion = nn.CrossEntropyLoss()

        

        if freezelonlylastlayer == 'yes':    

            for param in model.parameters():

                param.requires_grad = False

            model.fc = nn.Sequential(                        

                        nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

                          nn.Dropout(p=0.4),

                          nn.Linear(in_features=2048, out_features=2048, bias=True),

                          nn.LeakyReLU(),

                          #nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

                          #nn.Dropout(p=0.5),

                          nn.Linear(in_features=2048, out_features=5, bias=True),)

            optimizer = torch.optim.Adam(model.fc.parameters(), lr = lr, weight_decay=1e-5) 

        else:

            model.fc = nn.Sequential(                        

                        nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

                          nn.Dropout(p=0.4),

                          nn.Linear(in_features=2048, out_features=2048, bias=True),

                          nn.LeakyReLU(),

                          #nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

                          #nn.Dropout(p=0.5),

                          nn.Linear(in_features=2048, out_features=5, bias=True),)

            optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=1e-5) 

        modelname = 'resnet50'

        criterion = nn.CrossEntropyLoss()

    elif modelnumber == 2:

        batch_size = 32

        image_size=299

        #training_loader, validation_loader = load_split_train_test(data_dir,batch_size,valsize,image_size)

        training_loader = load_split_train_test(train_data_dir,batch_size,image_size,train_ind=False)

        validation_loader = load_split_train_test(validate_data_dir,batch_size,image_size,train_ind=False)

        #model = models.inception_v3(pretrained=True)

        if pretrained:

            model  = models.inception_v3(pretrained=True)

        else:

            model  = models.inception_v3(pretrained=False)

            checkpoint = torch.load("../input/pretrained-pytorch-models/inception_v3_google-1a9a5a14.pth")

            model.load_state_dict(checkpoint)

        #for param in model5.parameters():

        #    param.requires_grad = False

        #criterion = nn.CrossEntropyLoss() 

        if freezelonlylastlayer == 'yes':    

            for param in model.parameters():

                param.requires_grad = False

            model.fc = nn.Sequential(

                          nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

                          nn.Dropout(p=0.4),

                          nn.Linear(in_features=2048, out_features=2048, bias=True),

                          nn.LeakyReLU(),

                          #nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

                          #nn.Dropout(p=0.5),

                          nn.Linear(in_features=2048, out_features=5, bias=True),

                         )



            optimizer = torch.optim.Adam(model.fc.parameters(), lr = lr, weight_decay=1e-5) 

        else:

            model.fc = nn.Sequential(

                          nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

                          nn.Dropout(p=0.4),

                          nn.Linear(in_features=2048, out_features=2048, bias=True),

                          nn.LeakyReLU(),

                          #nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

                          #nn.Dropout(p=0.5),

                          nn.Linear(in_features=2048, out_features=5, bias=True),

                         )

            optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=1e-5) 

        modelname = 'inception'

        criterion = nn.CrossEntropyLoss()

    elif modelnumber == 3:

        batch_size = 32

        image_size=224

        training_loader = load_split_train_test(train_data_dir,batch_size,image_size,train_ind=False)

        validation_loader = load_split_train_test(validate_data_dir,batch_size,image_size,train_ind=False)

        #training_loader, validation_loader = load_split_train_test(data_dir,batch_size,valsize,image_size)

        #model = models.densenet161(pretrained=True)

        if pretrained:

            model  = models.densenet161(pretrained=True)

        else:

            model  = models.densenet161(pretrained=False)

            checkpoint = torch.load("../input/pytorch-model-zoo/densenet161-347e6b360.pth")

            model.load_state_dict(checkpoint)

            #model  = models.resnet152(pretrained=False)

            #checkpoint = torch.load("../input/resnet152/resnet152.pth")

            #model.load_state_dict(checkpoint)

        #for param in model5.parameters():

        #    param.requires_grad = False

        #criterion = nn.CrossEntropyLoss()      

        if freezelonlylastlayer == 'yes':    

            for param in model.parameters():

                param.requires_grad = False

            model.classifier = nn.Sequential(

                          nn.BatchNorm1d(2208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

                          nn.Dropout(p=0.4),

                          nn.Linear(in_features=2208, out_features=2208, bias=True),

                          nn.LeakyReLU(),

                          #nn.BatchNorm1d(2208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

                          #nn.Dropout(p=0.5),

                          nn.Linear(in_features=2208, out_features=5, bias=True),

                         )

            optimizer = torch.optim.Adam(model.fc.parameters(), lr = lr, weight_decay=1e-5) 

        else:

            model.classifier = nn.Sequential(                          

                        nn.BatchNorm1d(2208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

                          nn.Dropout(p=0.25),

                          nn.Linear(in_features=2208, out_features=2208, bias=True),

                          nn.LeakyReLU(),

                          #nn.BatchNorm1d(2208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

                          #nn.Dropout(p=0.5),

                          nn.Linear(in_features=2208, out_features=5, bias=True),

                )

            optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=1e-5) 

        modelname = 'densenet161'

        criterion = nn.CrossEntropyLoss()

        #model.to(device)

        #model3.to(device)

    elif modelnumber == 4:

        batch_size = 64

        image_size=224

        training_loader = load_split_train_test(train_data_dir,batch_size,image_size,train_ind=False)

        validation_loader = load_split_train_test(validate_data_dir,batch_size,image_size,train_ind=False)

        #training_loader, validation_loader = load_split_train_test(data_dir,batch_size,valsize,image_size)

        #print(len(training_loader))

        #

        #print(len(validation_loader))

        #model = models.resnet152(pretrained=True)

        if pretrained:

            model  = models.resnet152(pretrained=True)

        else:

            model  = models.resnet152(pretrained=False)

            checkpoint = torch.load("../input/resnet152/resnet152.pth")

            model.load_state_dict(checkpoint)

        #for param in model2.parameters():

        #    param.requires_grad = False



        #criterion = nn.CrossEntropyLoss()

        

        if freezelonlylastlayer == 'yes':    

            for param in model.parameters():

                param.requires_grad = False

            model.fc = nn.Sequential(                          

                        nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

                          nn.Dropout(p=0.25),

                          nn.Linear(in_features=2048, out_features=2048, bias=True),

                          nn.LeakyReLU(),

                          #nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

                          #nn.Dropout(p=0.5),

                          nn.Linear(in_features=2048, out_features=5, bias=True),

                )

            optimizer = torch.optim.Adam(model.fc.parameters(), lr = lr, weight_decay=1e-5) 

        else:

            model.fc = nn.Sequential(                          

                        nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

                          nn.Dropout(p=0.25),

                          nn.Linear(in_features=2048, out_features=2048, bias=True),

                          nn.LeakyReLU(),

                          #nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

                          #nn.Dropout(p=0.5),

                          nn.Linear(in_features=2048, out_features=5, bias=True),

                )

            optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=1e-5) 

        modelname = 'resnet152'

        criterion = nn.CrossEntropyLoss()

    return model,criterion,optimizer,modelname,training_loader, validation_loader,image_size
from sklearn.metrics import cohen_kappa_score

def quadratic_kappa(y_hat, y):

    return torch.tensor(cohen_kappa_score(torch.round(y_hat.type(torch.FloatTensor)), y.type(torch.FloatTensor), weights='quadratic'),device='cpu')
def cyclical_lr(stepsize, min_lr=3e-4, max_lr=3e-3):



    # Scaler: we can adapt this if we do not want the triangular CLR

    scaler = lambda x: 1.



    # Lambda function to calculate the LR

    lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize)

    

    # Additional function to see where on the cycle we are

    def relative(it, stepsize):

        cycle = math.floor(1 + it / (2 * stepsize))

        x = abs(it / stepsize - 2 * cycle + 1)

        #print(max(0, (1 - x)) * scaler(cycle))

        return max(0, (1 - x)) * scaler(cycle)



    return lr_lambda
def modeltrainv2(model,criterion,optimizer,training_loader,validation_loader,epochs = 10,

                 modeltype = 'other',freezer=[0,0,0,0,0,0,0,0,0,0],stepsize=4,factor=2

                 ,end_lr=0.001,sch_lr='sgd_cyclic'):

    running_loss_history = []

    running_corrects_history = []

    val_running_loss_history = []

    val_running_corrects_history = []

    step_size = stepsize*len(training_loader)

    print(f"Step Size id {step_size}")

    if sch_lr == 'cyclic':

        clr = cyclical_lr(step_size, min_lr=end_lr/factor, max_lr=end_lr)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])

    ###elif sch_lr == 'sgd_cyclic' :

        #optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    #    scheduler = torch.optim.CyclicLR(optimizer)

    elif sch_lr == 'cosine' :

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0, last_epoch=-1)

    elif sch_lr == 'zero' :

        pass

    elif sch_lr == 'red_lr_p':

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    elif sch_lr=='cyclic_lr':

        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.005)

    #batch = 0

    #lr_lambda = lambda x: math.exp(x * math.log(end_lr / start_lr) / (lr_find_epochs * len( dataloaders["train"])))

    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    layers=[]

#    for child in model.children():

#        layers.append(child)

    for child in model.children():

        for c in child.children():

#        for c1 in c.children():

            layers.append(c)

    no_of_layers = len(layers)

    print(f"Number of layers in the model is {no_of_layers}")

    for e in range(epochs):

        batch = 0

        running_loss = 0.0

        running_corrects = 0.0

        val_running_loss = 0.0

        val_running_corrects = 0.0

        kappa_score=0.0

        for child in model.children():

            for c in child.children():

                for param in c.parameters():

                    param.requires_grad = True

        layerf = freezer[e]

        print(f"Input freezed layers are {layerf}")

        cntr=0

        freezelayer=0

        if no_of_layers <= freezer[e]:

            print(f"freezer mentioned is {freezer[e]} and total number of layer is {no_of_layers}, which will freeze all layers")

            layerf = no_of_layers-1

            print(f"Changing freezer to {layerf}")

        if layerf > 0:

            for child in model.children():

                for c in child.children():

                    cntr+=1 

                    if cntr < layerf:

                        freezelayer=freezelayer+1

                        for param in c.parameters():

                            param.requires_grad = False

            print(f"Number of layers freezed is {freezelayer}")

 #       if layerf == -1:

 #           #print(f"Number of layers freezed is {cntr}")

 #           for child in model.children():

 #               for c in child.children():

 #                   for param in c.parameters():

 #                       param.requires_grad = True

 #           print(f"Unfreezing all layers")

        for inputs, labels in training_loader:

            #print(len(inputs))

            #print(len(labels))

            #print(labels)

            inputs = inputs.to(device)

            labels = labels.to(device)

            batch = batch + len(inputs)

            #bs, ncrops, c, h, w = inputs.size()

            #outputs = model(input.view(-1, c, h, w))

            #outputs = model(inputs)

            

            optimizer.zero_grad()

            if modeltype == 'inception':

                outputs = model.forward(inputs)[0]

            else:

                outputs = model.forward(inputs)

            #optimizer.zero_grad()

            loss = criterion(outputs, labels)

            loss.backward()

            

            optimizer.step()

            if sch_lr != 'zero' :

                scheduler.step()

            #outputs = torch.exp(outputs)

            _, preds = torch.max(outputs, 1)

            #preds = preds + 1

            #print(preds)

            #print( labels.data)

            #print('-------------------')

            running_loss += loss.item()

            #print(running_loss,loss.item())

            running_corrects += torch.sum(preds == labels.data)

            #print(f"Epoch {e} has accuracy of ")

            #print(torch.sum(preds == labels.data),len(inputs),int(torch.sum(preds == labels.data))/len(inputs))

 

        else:

            valbatch = 0

            with torch.no_grad():

                for val_inputs, val_labels in validation_loader:

                    val_inputs = val_inputs.to(device)

                    val_labels = val_labels.to(device)

                    valbatch = valbatch + len(val_inputs)

                    if modeltype == 'inception':

                        val_outputs = model(val_inputs)[0]

                    else:

                        val_outputs = model(val_inputs)          

                    val_loss = criterion(val_outputs, val_labels)

                    #val_outputs = torch.exp(val_outputs)

                    _, val_preds = torch.max(val_outputs, 1)

                    #val_preds = val_preds + 1

                    val_running_loss += val_loss.item()

                    #print(val_loss.item(),val_running_loss)

                    val_running_corrects += torch.sum(val_preds == val_labels.data)

                    #kappa_running +=(val_preds == val_labels.data)

                    kappa_score+= quadratic_kappa(val_preds, val_labels.data)

        #print(epoch_loss)  

        #print(running_corrects.float())

        #print('-----------')

        epoch_loss = running_loss/len(training_loader)

        epoch_acc = running_corrects.float()/ batch

        running_loss_history.append(epoch_loss)

        running_corrects_history.append(epoch_acc)

        val_epoch_loss = val_running_loss/len(validation_loader)

        val_kappa_score = kappa_score/len(validation_loader)

        val_epoch_acc = val_running_corrects.float()/ valbatch

        val_running_loss_history.append(val_epoch_loss)

        val_running_corrects_history.append(val_epoch_acc)

        print('epoch :', (e+1))

        print('training loss: {:.4f}, acc {:.4f} '.format(epoch_loss, epoch_acc.item()))

        print('validation loss: {:.4f}, validation acc {:.4f} '.format(val_epoch_loss, val_epoch_acc.item()))

        print('Kappa Score: {:.4f} '.format(val_kappa_score))

        print('*'*100)

    modelpth = modeltype + '.pth'

    torch.save(model, modelpth)

    return model
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
classes

sample.head()
newtestfinal = pd.DataFrame(columns=['id_code', 'diagnosis'])

newtestfinal.shape[0]
device
def addtensorcols(val):

    return val['diagnosis1'] + val['diagnosis2']



def extractfilename(val):

    return os.path.split(val)[1]



def maxtensorval(val):

    #ps = torch.exp(val)

    #ps = F.softmax(val,dim=1)

    top_p, top_class = val.topk(1)

    return top_class





newtestfinal = pd.DataFrame(columns=['id_code', 'diagnosis1'])

for modelnum in [0]: 

    #counter = 0

    print('---------------------------------------------------------------------------------')

    print('Defining model and creating data loaders for model number {0}'.format(modelnum))

    #freezer=[45,45,45,45,45,45,45,45,30,30,30,30,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,25,25,25,25,0,0,0,0,0,0,0,40,40] #resnet152

    #freezer=[16,16,16,16,16,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] #densenet201

    freezer=[16,16,16,15,15,15,13,13,13,12,12,12,11,11,11,10,10,10,9,9,9,8,8,8,7,7,7,6,6,6,5,5,5,4,4,4,3,3,3,2,2,2,1,1,1,0,0,0]

    #freezer=[0,6,-1,0,8]

    #freezer=[16,16,16,10,10,10,0,0]

    model,criterion,optimizer,modelname,training_loader, validation_loader,image_size = definemodel(modelnum,pretrained=False,freezelonlylastlayer = 'no',

                                                                                                    lr=1.0)

    #test_transforms = transforms.Compose([transforms.Resize(256),

    #                                    transforms.CenterCrop(image_size),

    #                                  #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),

    #                                transforms.ToTensor(),

    #                                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

    #                              ])

    test_transforms = transforms.Compose([transforms.Resize(256),

                                            transforms.RandomHorizontalFlip(),

                                            transforms.RandomRotation(10),

                                           transforms.CenterCrop(image_size),

                                          transforms.ToTensor(),

                                          transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

                                       ])

    print("Training of model {0} started".format(modelname))

    model.to(device)

    #optimizer = torch.optim.SGD(model.parameters(), lr=1.0, momentum=0.9, nesterov=True)

    model = modeltrainv2(model,criterion,optimizer,training_loader,validation_loader,epochs = len(freezer),

                         modeltype=modelname,freezer=freezer,stepsize=4,factor=2,end_lr=0.001,

                        sch_lr = 'cyclic')   

    basedir = '/kaggle/images/test'

    newtest = pd.DataFrame(columns=['id_code', 'diagnosis2'])

    #destinationfolder = '../images'

    print("Prediction of model {0} started".format(modelname))

    for i,row in test.iterrows():

        pathfile = basedir + '/'+row['id_code']+'.png'

        test_image_name,imagetype = predict(model, pathfile,test_transforms,image_size,modeltype=modelname)

        newtest.loc[i] = [test_image_name,imagetype]

    print("Prediction for model {0} completed".format(modelname))

    newtest['id_code']  = newtest['id_code'].apply(extractfilename)

    del model,criterion,optimizer

    torch.cuda.empty_cache()

    if newtestfinal.shape[0]>0:

        newtestfinal = pd.merge(newtestfinal, newtest, on='id_code')

        newtestfinal['diagnosis1'] = newtestfinal.apply(addtensorcols, axis=1)

        newtestfinal = newtestfinal[['id_code','diagnosis1']]

    else:

        newtestfinal = newtest.copy()

        newtestfinal.columns=['id_code', 'diagnosis1']

    print("Training of model {0} completed".format(modelname))



newtestfinal['diagnosis'] = newtestfinal['diagnosis1'].apply(maxtensorval)

    
#ct = 0

#for child in model.children():

#ct += 1

#if ct < 7:

#    for param in child.parameters():

#        param.requires_grad = False
#    model,criterion,optimizer,modelname,training_loader, validation_loader,image_size = definemodel(4,pretrained=False,freezelonlylastlayer = 'no',

#                                                                                                    lr=0.0001)

#layers=[]

#lsch = list(model.children()[:-2])

#for child in model.children():

#    for c in child.children():

#        for c1 in c.children():

        #layers.append(child)

#        for param in c.parameters():

#            print(param)

#            print('......')

#        layers.append(c)

#    layers.append(c1)
#len(layers)
#model
#layers[-2]
#for i,row in test.iterrows():

#    pathfile = basedir + '/'+row['id_code']+'.png'

#    test_image_name,imagetype = predict(model, pathfile,test_transforms,image_size,modeltype=modelname)

#    newtest.loc[i] = [test_image_name,imagetype]

#print("Prediction for model {0} completed".format(modelname))

#newtest['id_code']  = newtest['id_code'].apply(extractfilename)

#del model,criterion,optimizer

#torch.cuda.empty_cache()

#if newtestfinal.shape[0]>0:

#    newtestfinal = pd.merge(newtestfinal, newtest, on='id_code')

#    newtestfinal['diagnosis1'] = newtestfinal.apply(addtensorcols, axis=1)

#    newtestfinal = newtestfinal[['id_code','diagnosis1']]

#else:

#    newtestfinal = newtest.copy()

#    newtestfinal.columns=['id_code', 'diagnosis1']

#print("Training of model {0} completed".format(modelname))

#newtestfinal['diagnosis'] = newtestfinal['diagnosis1'].apply(maxtensorval)
a = torch.tensor([4, 2, 4, 0, 2, 2, 1, 3, 1, 3, 3, 3, 4, 4, 3, 2, 0, 2, 2, 2, 4, 3, 1, 2,

        0, 2, 1, 0, 3, 1, 3, 2, 3, 2, 1, 3, 3, 1, 4, 4, 3, 0, 2, 2, 3, 0, 1, 2,

        1, 3, 2, 1, 0, 4, 4, 3, 4, 2, 0, 2, 0, 1, 2, 2])

b = torch.tensor([4, 2, 0, 4, 2, 2, 1, 3, 1, 3, 3, 3, 4, 0, 3, 2, 0, 3, 2, 2, 4, 3, 1, 2,

        0, 2, 1, 0, 3, 1, 3, 2, 3, 2, 1, 3, 3, 1, 4, 4, 3, 0, 2, 2, 3, 0, 1, 2,

        1, 3, 2, 1, 4, 4, 4, 3, 4, 2, 0, 2, 4, 1, 2, 2])

c = int(torch.sum(a == b))/len(b)

c
#newtestfinal['category'] = newtestfinal['category1'].apply(maxtensorval)
newtestfinal.head()
newtestfinal['diagnosis'] = newtestfinal['diagnosis'].astype('int')
newtestfinal = newtestfinal[['id_code','diagnosis']]
#for i, x in newtestfinal.iterrows():

#    print(x['category1'])
newtestfinal['id_code_new']=newtestfinal['id_code'].apply(lambda x:x.split('.')[0])
newtestfinal = newtestfinal[['id_code_new','diagnosis']]

newtestfinal.columns = ['id_code','diagnosis']
newtestfinal.head()
newtestfinal['diagnosis'].value_counts()
train['diagnosis'].value_counts()
sample.shape
newtestfinal.shape
newtestfinal.to_csv('submission.csv',index=False)
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):



    def __init__(self, df,

                 transforms=None,

                 labels_=False):

        self.labels = None

        self.transforms = None

        self.df = df

        self.imagename = np.asarray(self.df.iloc[:, 0])

        self.data_len = len(self.df.index)

        if transforms is not None:

            self.transforms = transforms



    def __getitem__(self, index):

        basedir = '../test/dummy/'

        image_name = basedir + self.imagename[index]

        #id_ = self.ids[index]

        img_ = Image.open(image_name).convert('RGB')

        if self.transforms is not None:

            img_ = self.transforms(img_)[:3,:,:]

            label = 0

        return (img_,image_name)



    def __len__(self):

        return self.data_len
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(newtestfinal)