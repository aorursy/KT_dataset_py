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





#install pretrained models library

!pip install ../input/pretrainedmodels/pretrainedmodels-0.7.4/pretrainedmodels-0.7.4/ > /dev/null # no output
import numpy as np

import pandas as pd

from tqdm import tqdm_notebook as tqdm

import matplotlib.pyplot as plt

import cv2

import albumentations as A

import torch

from skimage.transform import AffineTransform, warp

import warnings

warnings.filterwarnings("ignore")

import torch.nn.functional as F

from torch.utils.data.dataloader import DataLoader

from torchvision import datasets, models, transforms

import torch.nn as nn

from torch.nn import functional as F

import pretrainedmodels

import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import ReduceLROnPlateau

import skimage.io


#Dataset for inference set

#Image will be loaded -> it will be normalized to 0-1 interval ->  it be send to the transform augmentation class -> switch dimensions to (2,0,1) -> convert to tensor



class PandaDataset:

    def __init__(self, df, transform=None,image_dir=" "):

        self.image_ids=df.index.values

        self.transform=transform

        self.df=df

        self.image_dir=image_dir

    def __len__(self):

        return len(self.image_ids)

    

    def __getitem__(self, index):



        imgPath= f"{self.image_dir}{self.image_ids[index]}.tiff"

        img = skimage.io.MultiImage(imgPath)

        image = img[-1]

        image = (255 - image).astype(np.float32) / 255.

        if self.transform:

            image,_ = self.transform(image)

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)



        return torch.tensor(image, dtype=torch.float)


#augmentation class for inference set

#here we perform resize to (512,512) and normalize (mean=0.0692, std=0.2051)

#also there is a flag in order to check if TTA should be taken into account and if so what type of augmentation



def Aug_resize(img,size):

    return cv2.resize(img, size, interpolation=cv2.INTER_AREA )





class Transform:

    def __init__(self, size=(512, 512),normalize=True, TTAconfig=0):

        self.size=size

        self.normalize=normalize

        self.TTAconfig=TTAconfig

        

    def __call__(self, x):



        # --- Augmentation ---



        x = Aug_resize(x, size=self.size)



        x = (x.astype(np.float32) - 0.0692) / 0.2051

        

        if (self.TTAconfig==1):

            x = cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE)

            

        if (self.TTAconfig==2):

            x = cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE)

            

        if (self.TTAconfig==3):

            x = cv2.rotate(x, cv2.ROTATE_180)

        

        return x, None
#model design for the inference of the model V1 based in densenet121

class ModelArhitectureV1(nn.Module):

    def __init__(self,pretrainedModelArhitecture='se_resnext101_32x4d', pretrainedModelWeights=None):

        super(ModelArhitectureV1, self).__init__()

        self.base_model = pretrainedmodels.__dict__[pretrainedModelArhitecture](pretrained=pretrainedModelWeights).to(device)

        self.final1_1 = nn.Linear(in_features=524288, out_features=6, bias=True).to(device)

                

    def forward(self,x):

        self.do_pooling=False

        h=self.base_model.features(x)

        if self.do_pooling:

            h = torch.sum(h, dim=(-1, -2))

        else:

            bs, ch, height, width = h.shape

            h = h.view(bs, ch*height*width)



        h1=self.final1_1(h)      

        return h1



#model design for the inference of the model V2 based in se_resnet50

class ModelArhitectureV2(nn.Module):

    def __init__(self,pretrainedModelArhitecture='se_resnet50', pretrainedModelWeights=None):

        super(ModelArhitectureV2, self).__init__()

        self.base_model = pretrainedmodels.__dict__[pretrainedModelArhitecture](pretrained=pretrainedModelWeights).to(device)

        self.final1_1 = nn.Linear(in_features=524288, out_features=6, bias=True).to(device)

                

    def forward(self,x):

        self.do_pooling=False

        h=self.base_model.features(x)

        if self.do_pooling:

            h = torch.sum(h, dim=(-1, -2))

        else:

            bs, ch, height, width = h.shape

            h = h.view(bs, ch*height*width)



        h1=self.final1_1(h)



        

        return h1
#general algorithm class made for inference 

#the output is a list with all the predictions after applying softmax to the vector, example [[0.2, 0.3, 0.2, 0.05, 0.25],[0.1, 0.4, 0.1, 0.15, 0.25], ...]

class PandaAlgorithm(nn.Module):

    

    def __init__(self, model):        

        super(PandaAlgorithm, self).__init__()

        self.model=model

    def forward(self,x):



        inputs = x.to(device)

        outputs =self.model(inputs)



        return outputs

    

    def getPredictions(self, dataloader):

        predList=[]

        with torch.no_grad():

            for inputs in tqdm(dataloader):

                rawPreds = self.forward(inputs)

                predList.extend(F.softmax(rawPreds))

        return predList

# a manager function that takes a dict with the training params and performs inference

def doInference(train_args_dict):

    if train_args_dict.get('model_arhitecture_base')=='v1':

        arhitecture = ModelArhitectureV1()

    else:

        arhitecture = ModelArhitectureV2()

        

        

    if (TTAmode):

        transformTest0 = Transform(size = train_args_dict.get('image_size'),TTAconfig=0)

        test_dataset0 = PandaDataset(df,transformTest0, image_dir)

        test_loader0 = DataLoader(test_dataset0, batch_size=batch_size, shuffle=False)

        

        transformTest1 = Transform(size = train_args_dict.get('image_size'),TTAconfig=1)

        test_dataset1 = PandaDataset(df,transformTest1, image_dir)

        test_loader1 = DataLoader(test_dataset1, batch_size=batch_size, shuffle=False)

        

        transformTest2 = Transform(size = train_args_dict.get('image_size'),TTAconfig=2)

        test_dataset2 = PandaDataset(df,transformTest2, image_dir)

        test_loader2 = DataLoader(test_dataset2, batch_size=batch_size, shuffle=False)

        

        transformTest3 = Transform(size = train_args_dict.get('image_size'),TTAconfig=3)

        test_dataset3 = PandaDataset(df,transformTest3, image_dir)

        test_loader3 = DataLoader(test_dataset3, batch_size=batch_size, shuffle=False)



        arhitecture.load_state_dict(torch.load(train_args_dict.get('model_path'),map_location='cuda:0'))

        pandaAlgorithm = PandaAlgorithm(arhitecture)

        

        results0= pandaAlgorithm.getPredictions(dataloader=test_loader0)

        results1= pandaAlgorithm.getPredictions(dataloader=test_loader1)

        results2= pandaAlgorithm.getPredictions(dataloader=test_loader2)

        results3= pandaAlgorithm.getPredictions(dataloader=test_loader3)

        

        results=[results0,results1,results2,results3]

        return results

    

    else:

        transformTest = Transform(size = train_args_dict.get('image_size'),TTAconfig=0)

        test_dataset = PandaDataset(df,transformTest, image_dir)

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



        arhitecture.load_state_dict(torch.load(train_args_dict.get('model_path'),map_location='cuda:0'))

        pandaAlgorithm = PandaAlgorithm(arhitecture)

        results= pandaAlgorithm.getPredictions(dataloader=test_loader)

    

    return [results]
#flag for enabling TTA

TTAmode=False

#cuda mode ON

device='cuda:0'

device = torch.device(device)

#batch size for inference, it does not matter for accuracy, just for speed (watch out for Out Of Memory in case of increasing)

batch_size = 6

#number of total models, here we have 2 models (fold 1 from each arhitecture)

number_of_models=6

model_preds_list = []

submission = pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/sample_submission.csv')



if os.path.exists('../input/prostate-cancer-grade-assessment/test_images'):

    submissionMode=True

else:

    submissionMode=False

    

    

if (submissionMode):    

    image_dir= '/kaggle/input/prostate-cancer-grade-assessment/test_images/'



    df = pd.read_csv("/kaggle/input/prostate-cancer-grade-assessment/test.csv").set_index("image_id")





    for i in range (number_of_models):

        train_args_dict={}  



        if (i==0):

            train_args_dict.update({

                'model_path': '/kaggle/input/model2fold0v1/model_fold0_epoch45_Qwk0.6675334622527956_v3Beta.pth',                                    

                'image_size': (512,512),

                'model_arhitecture_base':'v2'                

            })  



        if (i==1):

            train_args_dict.update({

                'model_path': '/kaggle/input/model2fold1v0/model_fold1_epoch77_Qwk0.6080847762871844_v3Beta.pth',                                    

                'image_size': (512,512),

                'model_arhitecture_base':'v2'                

            })  



        if (i==2):

            train_args_dict.update({

                'model_path': '/kaggle/input/model2fold2v0/model_fold2_epoch12_Qwk0.6049939916560642_v3Beta.pth',                                    

                'image_size': (512,512),

                'model_arhitecture_base':'v2'                

            })  

          

        if (i==3):

            train_args_dict.update({

                'model_path': '/kaggle/input/model2fold3v0/model_fold3_epoch26_Qwk0.6141516886726479_v3Beta.pth',                                    

                'image_size': (512,512),

                'model_arhitecture_base':'v2'                

            })  



        if (i==4):

            train_args_dict.update({

                'model_path': '/kaggle/input/model2fold4v0/model_fold4_epoch21_Qwk0.57012076934546_v3Beta.pth',                                    

                'image_size': (512,512),

                'model_arhitecture_base':'v2'                

            })  



        if (i==5):

            train_args_dict.update({

                'model_path': '/kaggle/input/model1fold0v0/model_fold0_epoch36_Qwk0.6627780473749447_v3Gamma.pth',                                    

                'image_size': (512,512),

                'model_arhitecture_base':'v1'                

            })  



        test_preds = doInference(train_args_dict)

        

        for i in range(len(test_preds)):

            model_preds_list.append(test_preds[i])



    









if (submissionMode):  

    if TTAmode:

        weightLists=[1/number_of_models] * number_of_models * 4

    else:

        weightLists = [1/number_of_models] * number_of_models

    

    results=[]

    for i in range(len(model_preds_list[0])):

        klist=[]

        for j in range(len(model_preds_list[0][i])):

            temp=0

            for m in range(len(model_preds_list)):

                temp+=(model_preds_list[m][i][j].item())*weightLists[m]

            klist.append(temp)

        results.append(klist)



    predictions=[]    

    for i in range(len(results)):

         predictions.append(np.argmax(results[i]))

else:

    predictions=[0,0,0]
submission.isup_grade = predictions

submission.isup_grade = submission['isup_grade'].astype(int)

submission.to_csv('submission.csv',index=False)
submission.isup_grade