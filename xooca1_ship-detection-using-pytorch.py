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

##!pip install pretrainedmodels

#import pretrainedmodels
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
train['category_label'].value_counts()
a = 2120 - 1217

b = 2120 - 1167

c = 2120 - 916

d = 2120 - 832


    

#trainfilenames = train['image'].tolist()

basedir = '../input/train/images/'

destinationfolder = '../images'

for i,row in train.iterrows():

    currentfileloc = basedir + row['image']

    newdirname = destinationfolder + 'train/' + str(row['category'])

    if not os.path.exists(newdirname):

        os.makedirs(newdirname)

    shutil.copy(currentfileloc, newdirname)

    

    


    

#trainfilenames = train['image'].tolist()

basedir = '../input/train/images/'

destinationfolder = '../test/dummy'

for i,row in test.iterrows():

    currentfileloc = basedir + row['image']

    #newdirname = destinationfolder

    if not os.path.exists(destinationfolder):

        os.makedirs(destinationfolder)

    shutil.copy(currentfileloc, destinationfolder)

    

    
#!wget http://data.lip6.fr/cadene/pretrainedmodels/xception-43020ad28.pth
!ls -ltr ../test/dummy | wc -l
data = '../test/dummy/2821132.jpg'

#data = '../input/train/images/1007700.jpg'

pil_im = Image.open(data, 'r')

imshow(np.asarray(pil_im))
def random_brightness(image):

    # Convert 2 HSV colorspace from RGB colorspace

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Generate new random brightness

    rand = random.uniform(0.6, 1.0)

    hsv[:, :, 2] = rand*hsv[:, :, 2]

    rand = random.uniform(1.0, 1.5)

    hsv[:, :, 1] = rand*hsv[:, :, 1]

    # Convert back to RGB colorspace

    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return new_img





def zoom(image,rows,cols):

    zoom_pix = random.randint(5, 10)

    zoom_factor = 1 + (2*zoom_pix)/rows

    image = cv2.resize(image, None, fx=zoom_factor,

                       fy=zoom_factor, interpolation=cv2.INTER_LINEAR)

    top_crop = (image.shape[0] - rows)//2

#     bottom_crop = image.shape[0] - top_crop - IMAGE_HEIGHT

    left_crop = (image.shape[1] - cols)//2

#     right_crop = image.shape[1] - left_crop - IMAGE_WIDTH

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

            #if op3:

            #    image = cv2.flip(image, 1)

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



                                        #transforms.CenterCrop(274),

                                        #transforms.RandomHorizontalFlip(),

                                        transforms.RandomRotation(5),

                                        transforms.Resize(224),

                                        #transforms.RandomResizedCrop((224,224)),

                                        #transforms.CenterCrop(224),

                                        #transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),

                                       # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),

                                        transforms.ToTensor()

                                        #transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

                                        #transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])

                                       ])

            else:

                changeimg = transforms.Compose([ 



                            #transforms.CenterCrop(274),

                            transforms.RandomHorizontalFlip(),

                            transforms.RandomRotation(10),

                            transforms.Resize(224),

                            #transforms.RandomResizedCrop((224,224)),

                            #transforms.CenterCrop(224),

                            #transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),

                            #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),

                            transforms.ToTensor()

                            #transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

                            #transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])

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
dirname = '../imagestrain/5/'

no_of_images = a

createaugimagesv2(dirname,no_of_images)

dirname = '../imagestrain/2/'

no_of_images = b

createaugimagesv2(dirname,no_of_images)

dirname = '../imagestrain/3/'

no_of_images = 916

createaugimagesv2(dirname,no_of_images)

dirname = '../imagestrain/4/'

no_of_images = 832

createaugimagesv2(dirname,no_of_images)



dirname = '../imagestrain/1/'

no_of_images = 500

createaugimages(dirname,no_of_images)

dirname = '../imagestrain/5/'

no_of_images = 500

createaugimages(dirname,no_of_images)

dirname = '../imagestrain/2/'

no_of_images = 500

createaugimages(dirname,no_of_images)

dirname = '../imagestrain/3/'

no_of_images = 916

createaugimages(dirname,no_of_images)

dirname = '../imagestrain/4/'

no_of_images = 832

createaugimages(dirname,no_of_images)



dirname = '../imagestrain/3/'

no_of_images = 288

createaugimagesv2(dirname,no_of_images)

dirname = '../imagestrain/4/'

no_of_images = 456

createaugimagesv2(dirname,no_of_images)
#dirname = '../imagestrain/3/'

#no_of_images = 288

#createaugimages(dirname,no_of_images)

#dirname = '../imagestrain/4/'

#no_of_images = 600

#createaugimages(dirname,no_of_images)
#dirname = '../imagestrain/1/'

#file3 = resizeall(dirname)

#print(len(file3))

#dirname = '../imagestrain/2/'

#file3 = resizeall(dirname)

#print(len(file3))

#dirname = '../imagestrain/3/'

#file3 = resizeall(dirname)

#print(len(file3))

#dirname = '../imagestrain/4/'

#file3 = resizeall(dirname)

#print(len(file3))

#dirname = '../imagestrain/5/'

#file3 = resizeall(dirname)

#print(len(file3))
!ls ../imagestrain

!ls ../imagestrain/1 | wc -l

!ls ../imagestrain/5 | wc -l

!ls ../imagestrain/2 | wc -l

!ls ../imagestrain/4 | wc -l

!ls ../imagestrain/3 | wc -l
data = '../imagestrain/5/2868105_enh.jpg'

pil_im = Image.open(data, 'r')

imshow(np.asarray(pil_im))
data = '../imagestrain/5/2878253.jpg'

pil_im = Image.open(data, 'r')

imshow(np.asarray(pil_im))
def load_split_train_test(datadir, batch_size=32,valid_size = .3,image_size=224):

    

    #train_transforms = transforms.Compose([transforms.Resize((64,64)),

    #                                  transforms.RandomHorizontalFlip(),

    #                                  transforms.RandomRotation(10),

    #                                  transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),

    #                                  transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.2),

    #                                  transforms.ToTensor(),

    #                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    #                           ])

    #test_transforms = transforms.Compose([transforms.Resize((64,64)),

    #                           transforms.ToTensor(),

    #                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    #                           ])

    #Grayscale(num_output_channels=1)

    #train_transforms = transforms.Compose([transforms.RandomRotation(10),

    #                                       transforms.RandomHorizontalFlip(),

    #                                       transforms.Resize(316),

    #                                       transforms.CenterCrop(256),

    #                                       transforms.RandomResizedCrop((image_size,image_size)),

    #                                       #transforms.RandomCrop(224),

    #                                       #transforms.Resize(256),

    #                                        #transforms.CenterCrop(224),

    #                                       #transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),

    #                                       #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),

    #                                        transforms.ToTensor(),

    #                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

    #                                  ])

    

    #train_transforms = transforms.Compose([ 

    #                    transforms.RandomRotation(10),

    #                    #transforms.RandomResizedCrop((224,224)),

    #                    transforms.RandomHorizontalFlip(),

    #                    #transforms.CenterCrop(224),

    #                    #transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),

    #                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),

    #                    transforms.Scale(256),

    #                    transforms.FiveCrop(224),

    #                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),

    #                    transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])(crop) for crop in crops]))

                        #transforms.ToTensor(),

                        #transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

                        #transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])

    #                   ])

    #test_transforms = transforms.Compose([transforms.Resize(256),

    #                                        transforms.CenterCrop(image_size), 

    #                                      #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),

    #                                      transforms.ToTensor(),

    #                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

    #                                  ])

    train_transforms = transforms.Compose([#transforms.RandomRotation(10),

                                            #transforms.Resize(256),

                                            transforms.RandomResizedCrop(256,scale=(0.8, 1.0),ratio=(0.75, 1.33)),

                                            transforms.RandomRotation(degrees=15),

                                            #transforms.RandomHorizontalFlip(),

                                            transforms.CenterCrop(image_size),

                                           #transforms.RandomHorizontalFlip(),

                                           #transforms.Resize(256),

                                           #transforms.CenterCrop(image_size),

                                           #transforms.RandomResizedCrop((image_size,image_size)),

                                           #transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),

                                           #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),

                                           transforms.ToTensor(),

                                           transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

                                       ])

    test_transforms = transforms.Compose([transforms.Resize(256),

                                        transforms.CenterCrop(image_size), 

                                          #transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.2),

                                          transforms.ToTensor(),

                                          transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

                                      ])

    train_data = datasets.ImageFolder(datadir,transform=train_transforms)

    test_data = datasets.ImageFolder(datadir,transform=test_transforms)

    num_train = len(train_data)

    indices = list(range(num_train))

    split = int(np.floor(valid_size * num_train))

    np.random.shuffle(indices)

    from torch.utils.data.sampler import SubsetRandomSampler

    train_idx, test_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)

    test_sampler = SubsetRandomSampler(test_idx)

    trainloader = torch.utils.data.DataLoader(train_data,

                   sampler=train_sampler, batch_size=batch_size)

    testloader = torch.utils.data.DataLoader(test_data,

                   sampler=test_sampler, batch_size=batch_size)

    return trainloader, testloader
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
classes = ('Cargo', 'Military', 'Carrier', 'Cruise', 'Tankers') 



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device
def definemodel(modelnumber,freezelonlylastlayer = 'yes',lr=0.0001):

    data_dir = '../imagestrain'

    valsize=0.2

    if modelnumber == 0:

        batch_size = 32

        image_size=224

        training_loader, validation_loader = load_split_train_test(data_dir,batch_size,valsize,image_size)

        model  = models.densenet201(pretrained=True)

        #criterion = nn.CrossEntropyLoss()        

        if freezelonlylastlayer == 'yes':    

            for param in model.parameters():

                param.requires_grad = False

            model.classifier = nn.Sequential(nn.Linear(1920, 720),

                                nn.ReLU(),

                                nn.Dropout(0.5),

                                nn.Linear(720, 256),

                                nn.ReLU(),

                                nn.Dropout(0.4),

                                nn.Linear(256, 64),

                                nn.ReLU(),

                                nn.Dropout(0.3),

                                nn.Linear(64, 5))

            optimizer = torch.optim.Adam(model.classifier.parameters(), lr = lr) 

        else:

            model.classifier = nn.Sequential(nn.Linear(1920, 720),

                                nn.ReLU(),

                                nn.Dropout(0.5),

                                nn.Linear(720, 256),

                                nn.ReLU(),

                                nn.Dropout(0.4),

                                nn.Linear(256, 64),

                                nn.ReLU(),

                                nn.Dropout(0.3),

                                nn.Linear(64, 5))

            optimizer = torch.optim.Adam(model.parameters(), lr = lr) 

        criterion = nn.CrossEntropyLoss()

        modelname = 'densenet201'

    elif modelnumber == 1:

        batch_size = 64

        image_size=224

        training_loader, validation_loader = load_split_train_test(data_dir,batch_size,valsize,image_size)

        model = models.resnet50(pretrained=True)

        #for param in model2.parameters():

        #    param.requires_grad = False



        #criterion = nn.CrossEntropyLoss()

        

        if freezelonlylastlayer == 'yes':    

            for param in model.parameters():

                param.requires_grad = False

            model.fc = nn.Sequential(nn.Linear(2048, 720),

                                nn.ReLU(),

                                nn.Dropout(0.5),

                                nn.Linear(720, 256),

                                nn.ReLU(),

                                nn.Dropout(0.4),

                                nn.Linear(256, 64),

                                nn.ReLU(),

                                nn.Dropout(0.3),

                                nn.Linear(64, 5))

            optimizer = torch.optim.Adam(model.fc.parameters(), lr = lr) 

        else:

            model.fc = nn.Sequential(nn.Linear(2048, 720),

                                nn.ReLU(),

                                nn.Dropout(0.5),

                                nn.Linear(720, 256),

                                nn.ReLU(),

                                nn.Dropout(0.4),

                                nn.Linear(256, 64),

                                nn.ReLU(),

                                nn.Dropout(0.3),

                                nn.Linear(64, 5))

            optimizer = torch.optim.Adam(model.parameters(), lr = lr) 

        modelname = 'resnet50'

        criterion = nn.CrossEntropyLoss()

    elif modelnumber == 2:

        batch_size = 32

        image_size=299

        training_loader, validation_loader = load_split_train_test(data_dir,batch_size,valsize,image_size)

        model = models.inception_v3(pretrained=True)

        #for param in model5.parameters():

        #    param.requires_grad = False

        #criterion = nn.CrossEntropyLoss() 

        if freezelonlylastlayer == 'yes':    

            for param in model.parameters():

                param.requires_grad = False

            model.fc = nn.Sequential(nn.Linear(2048, 720),

                                nn.ReLU(),

                                nn.Dropout(0.5),

                                nn.Linear(720, 256),

                                nn.ReLU(),

                                nn.Dropout(0.4),

                                nn.Linear(256, 64),

                                nn.ReLU(),

                                nn.Dropout(0.3),

                                nn.Linear(64, 5))

            optimizer = torch.optim.Adam(model.fc.parameters(), lr = lr) 

        else:

            model.fc = nn.Sequential(nn.Linear(2048, 720),

                                nn.ReLU(),

                                nn.Dropout(0.5),

                                nn.Linear(720, 256),

                                nn.ReLU(),

                                nn.Dropout(0.4),

                                nn.Linear(256, 64),

                                nn.ReLU(),

                                nn.Dropout(0.3),

                                nn.Linear(64, 5))

            optimizer = torch.optim.Adam(model.parameters(), lr = lr) 

        modelname = 'inception'

        criterion = nn.CrossEntropyLoss()

    elif modelnumber == 3:

        batch_size = 32

        image_size=224

        training_loader, validation_loader = load_split_train_test(data_dir,batch_size,valsize,image_size)

        model = models.densenet161(pretrained=True)

        #for param in model5.parameters():

        #    param.requires_grad = False

        #criterion = nn.CrossEntropyLoss()      

        if freezelonlylastlayer == 'yes':    

            for param in model.parameters():

                param.requires_grad = False

            model.classifier = nn.Sequential(nn.Linear(2208, 720),

                                nn.ReLU(),

                                nn.Dropout(0.5),

                                nn.Linear(720, 256),

                                nn.ReLU(),

                                nn.Dropout(0.4),

                                nn.Linear(256, 64),

                                nn.ReLU(),

                                nn.Dropout(0.3),

                                nn.Linear(64, 5))

            optimizer = torch.optim.Adam(model.fc.parameters(), lr = lr) 

        else:

            model.classifier = nn.Sequential(nn.Linear(2208, 720),

                                nn.ReLU(),

                                nn.Dropout(0.5),

                                nn.Linear(720, 256),

                                nn.ReLU(),

                                nn.Dropout(0.4),

                                nn.Linear(256, 64),

                                nn.ReLU(),

                                nn.Dropout(0.3),

                                nn.Linear(64, 5))

            optimizer = torch.optim.Adam(model.parameters(), lr = lr) 

        modelname = 'densenet161'

        criterion = nn.CrossEntropyLoss()

        #model.to(device)

        #model3.to(device)

    elif modelnumber == 4:

        batch_size = 64

        image_size=224

        training_loader, validation_loader = load_split_train_test(data_dir,batch_size,valsize,image_size)

        model = models.resnet152(pretrained=True)

        #for param in model2.parameters():

        #    param.requires_grad = False



        #criterion = nn.CrossEntropyLoss()

        

        if freezelonlylastlayer == 'yes':    

            for param in model.parameters():

                param.requires_grad = False

            model.fc = nn.Sequential(nn.Linear(2048, 720),

                                nn.ReLU(),

                                nn.Dropout(0.5),

                                nn.Linear(720, 256),

                                nn.ReLU(),

                                nn.Dropout(0.4),

                                nn.Linear(256, 64),

                                nn.ReLU(),

                                nn.Dropout(0.3),

                                nn.Linear(64, 5))

            optimizer = torch.optim.Adam(model.fc.parameters(), lr = lr) 

        else:

            model.fc = nn.Sequential(nn.Linear(2048, 720),

                                nn.ReLU(),

                                nn.Dropout(0.5),

                                nn.Linear(720, 256),

                                nn.ReLU(),

                                nn.Dropout(0.4),

                                nn.Linear(256, 64),

                                nn.ReLU(),

                                nn.Dropout(0.3),

                                nn.Linear(64, 5))

            optimizer = torch.optim.Adam(model.parameters(), lr = lr) 

        modelname = 'resnet152'

        criterion = nn.CrossEntropyLoss()

    return model,criterion,optimizer,modelname,training_loader, validation_loader,image_size
def modeltrainv2(model,criterion,optimizer,training_loader,validation_loader,epochs = 10,modeltype = 'other'):

    running_loss_history = []

    running_corrects_history = []

    val_running_loss_history = []

    val_running_corrects_history = []

    #batch = 0

    

    for e in range(epochs):

        batch = 0

        running_loss = 0.0

        running_corrects = 0.0

        val_running_loss = 0.0

        val_running_corrects = 0.0

  

        for inputs, labels in training_loader:

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

        #print(epoch_loss)  

        #print(running_corrects.float())

        #print('-----------')

        epoch_loss = running_loss/len(training_loader)

        epoch_acc = running_corrects.float()/ batch

        running_loss_history.append(epoch_loss)

        running_corrects_history.append(epoch_acc)

        val_epoch_loss = val_running_loss/len(validation_loader)

        val_epoch_acc = val_running_corrects.float()/ valbatch

        val_running_loss_history.append(val_epoch_loss)

        val_running_corrects_history.append(val_epoch_acc)

        print('epoch :', (e+1))

        print('training loss: {:.4f}, acc {:.4f} '.format(epoch_loss, epoch_acc.item()))

        print('validation loss: {:.4f}, validation acc {:.4f} '.format(val_epoch_loss, val_epoch_acc.item()))

    return model
def modeltrain(model,criterion,optimizer,trainloader,testloader,epochs = 10,modeltype = 'others'):

    #epochs = 1

    steps = 0

    running_loss = 0

    print_every = 10

    train_losses, test_losses = [], []

    for epoch in range(epochs):

        for inputs, labels in trainloader:

            #print(labels)

            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            if modeltype == 'inception':

                logps = model.forward(inputs)[0]

            else:

                logps = model.forward(inputs)

            loss = criterion(logps, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()



            if steps % print_every == 0:

                test_loss = 0

                accuracy = 0

                model.eval()

                with torch.no_grad():

                    for inputs, labels in testloader:

                        inputs, labels = inputs.to(device),labels.to(device)

                        logps = model.forward(inputs)

                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        ps = torch.exp(logps)

                        top_p, top_class = ps.topk(1, dim=1)

                        equals = top_class == labels.view(*top_class.shape)

                        accuracy +=torch.mean(equals.type(torch.FloatTensor)).item()

                train_losses.append(running_loss/len(trainloader))

                test_losses.append(test_loss/len(testloader))                    

                print(f"Epoch {epoch+1}/{epochs}.. "

                      f"Train loss: {running_loss/print_every:.3f}.. "

                      f"Test loss: {test_loss/len(testloader):.3f}.. "

                      f"Test accuracy: {accuracy/len(testloader):.3f}")

                running_loss = 0

                model.train()

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
device
def addtensorcols(val):

    return val['category1'] + val['category2']



def extractfilename(val):

    return os.path.split(val)[1]



def maxtensorval(val):

    #ps = torch.exp(val)

    #ps = F.softmax(val,dim=1)

    top_p, top_class = val.topk(1)

    return top_class+1





newtestfinal = pd.DataFrame(columns=['image', 'category1'])

for modelnum in [0,1,2,3,4]: 

    #counter = 0

    print('---------------------------------------------------------------------------------')

    print('Defining model and creating data loaders for model number {0}'.format(modelnum))

    model,criterion,optimizer,modelname,training_loader, validation_loader,image_size = definemodel(modelnum,freezelonlylastlayer = 'no',lr=0.0001)

    #test_transforms = transforms.Compose([transforms.Resize(256),

    #                                    transforms.CenterCrop(image_size),

    #                                  #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),

    #                                transforms.ToTensor(),

    #                                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

    #                              ])

    test_transforms = transforms.Compose([transforms.Resize(256),

                                        transforms.CenterCrop(image_size), 

                                      #transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.2),

                                      transforms.ToTensor(),

                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

                                  ])

    print("Training of model {0} started".format(modelname))

    model.to(device)

    model = modeltrainv2(model,criterion,optimizer,training_loader,validation_loader,epochs = 7,modeltype=modelname)   

    basedir = '../test/dummy/'

    newtest = pd.DataFrame(columns=['image', 'category2'])

    #destinationfolder = '../images'

    print("Prediction of model {0} started".format(modelname))

    for i,row in test.iterrows():

        pathfile = basedir + row['image']

        test_image_name,imagetype = predict(model, pathfile,test_transforms,image_size,modeltype=modelname)

        newtest.loc[i] = [test_image_name,imagetype]

    print("Prediction for model {0} completed".format(modelname))

    newtest['image']  = newtest['image'].apply(extractfilename)

    del model,criterion,optimizer

    torch.cuda.empty_cache()

    if newtestfinal.shape[0]>0:

        newtestfinal = pd.merge(newtestfinal, newtest, on='image')

        newtestfinal['category1'] = newtestfinal.apply(addtensorcols, axis=1)

        newtestfinal = newtestfinal[['image','category1']]

    else:

        newtestfinal = newtest.copy()

        newtestfinal.columns=['image', 'category1']

    print("Training of model {0} completed".format(modelname))



newtestfinal['category'] = newtestfinal['category1'].apply(maxtensorval)

    
newtestfinal.head()
newtestfinal['category'] = newtestfinal['category'].astype('int')
newtestfinal = newtestfinal[['image','category']]
newtestfinal.head()
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