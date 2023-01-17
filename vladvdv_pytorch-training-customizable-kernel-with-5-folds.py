# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

!pip install ../input/pretrainedmodels/pretrainedmodels-0.7.4/pretrainedmodels-0.7.4/ > /dev/null # no output
import os

import gc

import numpy as np

import pandas as pd

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt

import cv2

import albumentations as A

import torch

from skimage.transform import AffineTransform, warp

import warnings

warnings.filterwarnings("ignore")

import torch.nn.functional as F

from torch.utils.data.dataloader import DataLoader

import torch.nn as nn

import pretrainedmodels

from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import StratifiedKFold

#Define paths



BASE_PATH="/kaggle/input/"

TRAIN_IMG_DIR = BASE_PATH+"panda2/train_images/"

TRAIN_MASK_DIR = BASE_PATH+"panda2/train_label_masks/"

train = pd.read_csv(BASE_PATH+"/prostate-cancer-grade-assessment/train.csv").set_index("image_id")

test = pd.read_csv(BASE_PATH+"/prostate-cancer-grade-assessment/test.csv")

sample_submission = pd.read_csv(BASE_PATH+"/prostate-cancer-grade-assessment/sample_submission.csv")



train.head()

test.head()
#Dataset class used both for training and validation, difference consist in passing different transform objects

class PandaDataset:

    def __init__(self, df, transform=None):

        self.image_ids=df.index.values

        self.isup_grade = df.isup_grade.values

        self.transform=transform

        self.df=df

    def __len__(self):

        return len(self.image_ids)

    

    def __getitem__(self, index):

        image = cv2.imread(f"{TRAIN_IMG_DIR}{self.image_ids[index]}.png")

        label = self.isup_grade[index]

        image = (255 - image).astype(np.float32) / 255.

        if self.transform:

            image,_ = self.transform([image,label])

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)



        return torch.tensor(image, dtype=torch.float), torch.tensor(self.isup_grade[index])
#This cells are the implementation for multiple augmentation techniques

#There are implemented:

# * affine transforms (scale, rotation, shear, translation)

# * flip (horizontal, vertical)

# * resize

# * blur (blur, median blur, gaussian blur, motion blur)

# * noise (GaussNoise, MultiplicativeNoise)

# * distort 

# * RandomBrightnessContrast

# * cutout (CoarseDropout)

#Not all techniques are tested, if there are problems with one of them let me know 



def Aug_affine(img, prob):

    if (np.random.uniform()>=prob):

        return img

    # --- scale ---

    min_scale = 0.7

    max_scale = 1.3

    sx = np.random.uniform(min_scale, max_scale)

    sy = np.random.uniform(min_scale, max_scale)



    # --- rotation ---

    max_rot_angle = 20

    rot_angle = np.random.uniform(-max_rot_angle, max_rot_angle) * np.pi / 180.



    # --- shear ---

    max_shear_angle = 7

    shear_angle = np.random.uniform(-max_shear_angle, max_shear_angle) * np.pi / 180.



    # --- translation ---

    max_translation = 20

    tx = np.random.randint(-max_translation, max_translation)

    ty = np.random.randint(-max_translation, max_translation)



    tform = AffineTransform(scale=(sx, sy), rotation=rot_angle, shear=shear_angle,

                            translation=(tx, ty))

    transformed_image = warp(img, tform)



    return transformed_image    







def Aug_flip(img,prob):

    if (np.random.uniform()>=prob):

        return img

    r = np.random.uniform()

    if r < 0.5:

        img = apply_aug(A.HorizontalFlip(p=1), img)

    else:

         img = apply_aug(A.VerticalFlip(p=1), img)



        

    return img    

    

def apply_aug(aug, image):

    return aug(image=image)['image']





def Aug_resize(img,size):

    return cv2.resize(img, size, interpolation=cv2.INTER_AREA )





def Aug_blur(img, prob):

    if (np.random.uniform()>=prob):

        return img

    r = np.random.uniform()

    if r < 0.25:

        img = apply_aug(A.Blur(p=1), img)

    elif r < 0.5:

        img = apply_aug(A.MedianBlur(blur_limit=5, p=1), img)

    elif r < 0.75:

        img = apply_aug(A.GaussianBlur(p=1), img)

    else:

        img = apply_aug(A.MotionBlur(p=1), img)    

        

    return img





def Aug_noise(img, prob):

    if (np.random.uniform()>=prob):

        return img

    r = np.random.uniform()

    if r < 0.50:

        img = apply_aug(A.GaussNoise(var_limit=5. / 255., p=1), img)

    else:

        img = apply_aug(A.MultiplicativeNoise(p=1), img)    

    return img

        

def Aug_distort(img, prob):

    img = apply_aug(A.GridDistortion(p=prob), img)    

    return img

    

def Aug_brightness(img, prob):

    img = apply_aug(A.RandomBrightnessContrast(p=prob), img)     

    return img



def Aug_coarseDropout(img,prob):

    img = apply_aug(A.CoarseDropout(max_holes=4, max_height=30, max_width=30, p=prob), img)

    return img


#augmentation manager class

class Transform:

    def __init__(self, size=(512, 512),

                 normalize=True, train=True,

                 blurProb=0, noiseProb=0, distortionProb=0, 

                 elasticDistortionProb=0., randomBrightnessProb=0,

                 affineTransformProb=0, coarseDropout=0, flipProb=0):

        self.size=size

        self.normalize=normalize

        self.train=train

        self.blurProb=blurProb

        self.noiseProb=noiseProb

        self.distortionProb=distortionProb

        self.randomBrightnessProb=randomBrightnessProb

        self.affineTransformProb=affineTransformProb

        self.coarseDropout=coarseDropout

        self.flipProb=flipProb

        

    def __call__(self, example):

        if self.train:

            x, y = example

        else:

            x = example[0]



        # --- Augmentation ---

        x = Aug_affine(x.astype(np.float32), prob=self.affineTransformProb)

  

        x = Aug_resize(x.astype(np.float32), size=self.size)



        x = Aug_blur(x.astype(np.float32), prob=self.blurProb)



        x = Aug_flip(x.astype(np.float32),  prob=self.flipProb)

        

        x = Aug_noise(x.astype(np.float32), prob=self.noiseProb)

        

        x = Aug_distort(x.astype(np.float32), prob=self.distortionProb)

        

        x = Aug_brightness(x.astype(np.float32), prob=self.randomBrightnessProb)

        

        x = Aug_coarseDropout(x.astype(np.float32),prob=self.coarseDropout)



        # normalizing

        x = (x.astype(np.float32) - 0.0692) / 0.2051

        

        if self.train:

            return x, y

        else:

            return x, None
#label smoothing



def onehot_encoding(label, n_classes):

    return torch.zeros(label.size(0), n_classes).to(label.device).scatter_(

        1, label.view(-1, 1), 1)

def cross_entropy_loss(input, target, reduction):

    logp = F.log_softmax(input, dim=1)

    loss = torch.sum(-logp * target, dim=1)

    if reduction == 'none':

        return loss

    elif reduction == 'mean':

        return loss.mean()

    elif reduction == 'sum':

        return loss.sum()

    else:

        raise ValueError(

            '`reduction` must be one of \'none\', \'mean\', or \'sum\'.')

        

def label_smoothing_criterion(epsilon=0.1, reduction='mean'):

    def _label_smoothing_criterion(preds, targets):

        n_classes = preds.size(1)

        device = preds.device



        onehot = onehot_encoding(targets, n_classes).float().to(device)

        targets = onehot * (1 - epsilon) + torch.ones_like(onehot).to(

            device) * epsilon / n_classes

        loss = cross_entropy_loss(preds, targets, reduction)

        if reduction == 'none':

            return loss

        elif reduction == 'mean':

            return loss.mean()

        elif reduction == 'sum':

            return loss.sum()

        else:

            raise ValueError(

                '`reduction` must be one of \'none\', \'mean\', or \'sum\'.')



    return _label_smoothing_criterion
# QWK metric function (competition metric)

def qwk3(a1, a2, max_rat):

    assert(len(a1) == len(a2))

    a1 = np.asarray(a1, dtype=int)

    a2 = np.asarray(a2, dtype=int)



    hist1 = np.zeros((max_rat + 1, ))

    hist2 = np.zeros((max_rat + 1, ))



    o = 0

    for k in range(a1.shape[0]):

        i, j = a1[k], a2[k]

        hist1[i] += 1

        hist2[j] += 1

        o +=  (i - j) * (i - j)



    e = 0

    for i in range(max_rat + 1):

        for j in range(max_rat + 1):

            e += hist1[i] * hist2[j] * (i - j) * (i - j)



    e = e / a1.shape[0]



    return 1 - o / e
# class of the model arhitecture

#you can play with this class a lot(change model arhitecture, add more linear layers, etc)



class ModelArhitecture(nn.Module):

    def __init__(self,pretrainedModelArhitecture='se_resnet50', pretrainedModelWeights='imagenet'):

        super(ModelArhitecture, self).__init__()

        self.base_model = pretrainedmodels.__dict__[pretrainedModelArhitecture](pretrained=pretrainedModelWeights).to(device)

        self.final1_1 = nn.Linear(in_features=524288, out_features=6, bias=True).to(device)

                

    def forward(self,x):

        self.do_pooling=False

        h=self.base_model.features(x)

#        print (h.shape)

        if self.do_pooling:

            h = torch.sum(h, dim=(-1, -2))

        else:

            bs, ch, height, width = h.shape

            h = h.view(bs, ch*height*width)



        h1=self.final1_1(h)



        

        return h1
#class for the general algorithm (takes as input the model, trains and output prediction and statistics about training)

class PandaAlgorithm(nn.Module):

    

    def __init__(self, model, fold):        

        super(PandaAlgorithm, self).__init__()

        self.model = model

        self.lossTrain=[]

        self.lossVal=[]

        self.qwkTrain=[]

        self.qwkVal=[]

        self.fold=fold

        self.criterion=label_smoothing_criterion()

        

    def forward(self,x, optimizer,y=None, phase='train'):



        inputs = x.to(device)

        labels = y.to(device)



        outputs =self.model(inputs)

        loss = self.criterion(outputs, labels)                   



        if phase == 'train':

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()



        return [loss.data.cpu().numpy(), outputs, labels]

    

    def train(self,dataloaders,num_epochs=2):

        self.dataloaders=dataloaders

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, min_lr=1e-10)

        BestQwkScore=0

        for epoch in range(num_epochs):

           

            print('Epoch {}/{}'.format(epoch+1, num_epochs))

            print('-' * 10)     

            for phase in ['train', 'validation']:

                epoch_metrics={'CEloss':0}

                print (phase)

                outputList=[]

                labelList =[]

                if phase == 'train':

                    self.model.train()

                else:

                    self.model.eval()

                for inputs, labels in tqdm(self.dataloaders[phase]):   

                    MSELossPerBatch,outputsNP,labels2=self.forward(inputs,optimizer,labels,phase)

                    

                    for i in range(len(labels2)):

                        labelList.append(labels2[i].item())

                    for j in range(len(outputsNP)):

                        outputList.append(torch.argmax(F.softmax(outputsNP[j])).item())

                    epoch_metrics['CEloss']+=MSELossPerBatch





                loss,qwkScore=self.TrainingStats(epoch_metrics,phase, outputList, labelList)

                

                if phase=='validation':

                    scheduler.step(loss)

                    print ('learning rate:',scheduler.optimizer.param_groups[0]['lr']) 

                    

            if (qwkScore>BestQwkScore):

                BestQwkScore=qwkScore

                torch.save(self.model.state_dict(), 'model_fold'+str(self.fold)+"_epoch"+str(epoch)+'_Qwk'+str(BestQwkScore)+'_v3Beta.pth') 

                    

                

        return self.model

    

    def TrainingStats(self,epoch_metrics,phase, outputList, labels):



        epoch_metrics['CEloss']=epoch_metrics['CEloss']/ len(self.dataloaders[phase])



        labelsNP=np.array(labels)



        outputsNP=np.array(outputList)

        qwkScore=qwk3(labelsNP,outputsNP,6)



        if phase == 'train':

            self.lossTrain.append(epoch_metrics['CEloss'])

            self.qwkTrain.append(qwkScore)

#            uncomment plot functions if you are running on your local machine and want to plot loss and qwk for train and valid            

#            plt.plot(self.lossTrain,color='blue')

#            plt.show()



#            plt.plot(self.qwkTrain,color='red')

#            plt.show()

            

            

        else:

            

            self.lossVal.append(epoch_metrics['CEloss'])  

            self.qwkVal.append(qwkScore)

                              

#            plt.plot(self.lossVal,color='blue')

#            plt.show()                    



#            plt.plot(self.qwkVal,color='red')

#            plt.show()            

        print (epoch_metrics)

        print ("qwkScore: ",qwkScore)

                           

        

        return epoch_metrics['CEloss'], qwkScore
import random

seed=41

random.seed(seed)

os.environ['PYTHONHASHSEED'] = str(seed)

np.random.seed(seed)

torch.manual_seed(seed)

torch.cuda.manual_seed(seed)

torch.backends.cudnn.deterministic = True



#here I defined 5 folds splits keeping the histogram of labels constant (StratifiedKFold)

#change the batch_size, num_epochs according to you needs (my advice is to use minim 40 epochs)

mskf = StratifiedKFold(n_splits=5, random_state=seed)

mskf.get_n_splits(train, train.isup_grade.values)





num_epochs=1 #don't forget to change this (I suggest >50 epochs)

fold=0

for train_index, test_index in mskf.split(train, train.isup_grade.values):

    print("fold:", fold)

    print ("Train data: ",len(train_index))

    print ("Valid data: ",len(test_index))



    

    

    trainData = train.iloc[train_index]

    validData = train.iloc[test_index]

    

    #here is where you change the augmentation probabilities and features

    transformTrain=Transform(noiseProb=0.1, affineTransformProb=0.3, flipProb= 0.2)

    #for validation we don't augment

    transformValid=Transform(train=False)

    

    

    

    trainDataset = PandaDataset(trainData,transform=transformTrain)

    validDataset = PandaDataset(validData,transform=transformValid)

    out = trainDataset.__getitem__(100)

    out

    

    

    print('train_dataset', len(trainDataset), 'valid_dataset', len(validDataset))

    

    image_datasets={'train': trainDataset,

                    'validation':validDataset}

    batch_size=4

    train_loader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)

    valid_loader = DataLoader(validDataset, batch_size=batch_size, shuffle=False)

    

    

    dataloaders = {

        'train':train_loader,

        'validation':valid_loader 

        }

            

    # select running device        

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    

     

    fullModelArhitecture=ModelArhitecture()                

    clf=PandaAlgorithm(model=fullModelArhitecture, fold=fold)

    model=clf.train(dataloaders,num_epochs=num_epochs)

    fold+=1

    torch.cuda.empty_cache()

    gc.collect()

    del model