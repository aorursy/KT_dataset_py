!pip install git+https://github.com/qubvel/segmentation_models.pytorch
import pandas as pd

import matplotlib.pyplot as plt

from PIL import Image

import numpy as np

import random

from torch.utils.data import Dataset

import os

import cv2

import albumentations as albu

from sklearn.model_selection import train_test_split

import segmentation_models_pytorch as smp

from torch.utils.data import DataLoader

import torch

from catalyst.dl.runner import SupervisedRunner

from catalyst.dl.callbacks import DiceCallback, InferCallback, CheckpointCallback,JaccardCallback,IouCallback

from torch.optim.lr_scheduler import ReduceLROnPlateau

import tqdm

from catalyst.contrib.criterion import DiceLoss, IoULoss, FocalLossBinary

import torch.nn as nn

import pandas as pd

from catalyst.dl import utils

import gc
#plot descriptive stats

def DescriptiveStats(trainData):

    #check how many fish,flower,gravel and sugar valid data are in the training set

    occuranceDict={"Fish":0,"Flower":0,"Gravel":0,"Sugar":0}

    for i in range(len(trainData['EncodedPixels'])):

        if (pd.isnull(trainData['EncodedPixels'][i]))==False:

            cloudType=trainData["Image_Label"][i].split('_')[1]

            occuranceDict[cloudType]+=1   

    print("how many fish,flower,gravel and sugar valid data are in the training set")

    labels = occuranceDict.keys()

    sizes = occuranceDict.values()

    explode = (0.1, 0.1, 0.1, 0.1) 

    

    fig1, ax1 = plt.subplots()

    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

            shadow=True, startangle=90)

    ax1.axis('equal')  

    plt.show()

    print("how many clouds are per picture")

    #check how many clouds are per picture in the training set

    occurancePerPic=trainData.loc[trainData['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[0]).value_counts().value_counts()

    labels2 = occurancePerPic.keys() 

    fig2, ax2 = plt.subplots()

    ax2.pie(occurancePerPic, explode=explode, labels=labels2, autopct='%1.1f%%',

            shadow=True, startangle=90)

    ax2.axis('equal')  

    plt.show()
#show a specific image

def ShowImg(trainData,imgName):

    image = Image.open(path+"/train_images/"+imgName)

    print("Original Img")

    plt.imshow(image)

    plt.show()

    ss=trainData.loc[trainData['im_id']==imgName, 'EncodedPixels']

    for row in ss:

        print(trainData.loc[trainData['EncodedPixels']==row, "label"])

        try: # label might not be there!

            mask = rle_decode(row)

        except Exception as exception:

            mask = np.zeros((1400, 2100))

            continue

            

        plt.imshow(image)

        plt.imshow(mask, alpha=0.3, cmap='gray')

        plt.show()
# create a custom loss (total loss=alfa*IoULoss + beta*DiceLoss + gamma*FocalLossBinary)

class CustomLoss(nn.Module):

    def __init__(self, alpha=.7, beta=1.5, gamma=.4):

        super().__init__()

        self.alpha=alpha

        self.beta=beta

        self.gamma=gamma

        self.lossIOU=IoULoss()

        self.lossDice=DiceLoss()

        self.lossFocal=FocalLossBinary()

    def forward(self, input, target):

        loss=self.alpha*self.lossIOU(input.cpu(), target.cpu()) + self.beta*self.lossDice(input.cpu(), target.cpu()) + self.beta*self.lossFocal(input.cpu(), target.cpu())

        return loss.mean()/(alpha+beta+gamma)
#Create mask based on df, image name and shape

def make_mask(df, image_name= 'img.jpg',

              shape= (1400, 2100)):

    encoded_masks = df.loc[df['im_id'] == image_name, 'EncodedPixels']

    masks = np.zeros((shape[0], shape[1], 4), dtype=np.float32)



    for idx, label in enumerate(encoded_masks.values):

        if label is not np.nan:

            mask = rle_decode(label)

            masks[:, :, idx] = mask



    return masks
#Decode rle encoded mask    

def rle_decode(mask_rle='', shape=(1400, 2100)):

    s = mask_rle.split()

    starts, lengths = [np.asarray(x, dtype=int)

                       for x in (s[0:][::2], s[1:][::2])]

    starts -= 1

    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):

        img[lo:hi] = 1

    return img.reshape(shape, order='F')
def mask2rle(img):

    pixels = img.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)
#reshape after using albumentation

def ConvertToTensorFormat(x, **kwargs):

    return x.transpose(2, 0, 1).astype('float32')
def post_process(probability, threshold, min_size):

    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]

    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))

    predictions = np.zeros((350, 525), np.float32)

    num = 0

    for c in range(1, num_component):

        p = (component == c)

        if p.sum() > min_size:

            predictions[p] = 1

            num += 1

    return predictions, num
def dice(img1, img2):

    img1 = np.asarray(img1).astype(np.bool)

    img2 = np.asarray(img2).astype(np.bool)



    intersection = np.logical_and(img1, img2)



    return 2. * intersection.sum() / (img1.sum() + img2.sum())



def sigmoid(x): return 1/(1+np.exp(-x))
class CloudDataset(Dataset):

    def __init__(self,data,dataSetType,transforms,img_ids,preprocessing):

        self.data=data

        self.preprocessing=preprocessing

        self.transforms=transforms

        self.img_ids=img_ids

        if (dataSetType=="train"):

            self.imgFolder=path+"/train_images"

        if (dataSetType=="test"):

            self.imgFolder=path+"/test_images"



    def __getitem__(self, idx):

        image_name = self.img_ids[idx]



        mask = make_mask(self.data, image_name)

        image_path = os.path.join(self.imgFolder, image_name)

        

        img = cv2.imread(image_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        

        augmented = self.transforms(image=img, mask=mask)

        img = augmented['image']

        mask = augmented['mask']

        

        if self.preprocessing:

            preprocessed = self.preprocessing(image=img, mask=mask)

            img = preprocessed['image']

            mask = preprocessed['mask']

            

            

        return img, mask



    def __len__(self):

        return len(self.img_ids)      
#preprocess for specific network used

def get_preprocessing(preprocessing_fn=None):

    if preprocessing_fn is not None:

        _transform = [

            albu.Lambda(image=preprocessing_fn),

            albu.Lambda(image=ConvertToTensorFormat, mask=ConvertToTensorFormat),

        ]

    else:

        _transform = [

            albu.Normalize(),

            albu.Lambda(image=ConvertToTensorFormat, mask=ConvertToTensorFormat),

        ]

    return albu.Compose(_transform)

#augmentation for training data

def get_training_augmentation(p=0.5):

    train_transform = [

        albu.Resize(320, 640),

        albu.HorizontalFlip(p=0.25),

        albu.VerticalFlip(p=0.25),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0)

    ]

    return albu.Compose(train_transform)



#for validation dataset it is just resize

def get_validation_augmentation():

    train_transform = [

        albu.Resize(320, 640)

    ]

    return albu.Compose(train_transform)
path = '../input/understanding_cloud_organization'



#read files

trainData = pd.read_csv(path+'/train.csv')

subSample = pd.read_csv(path+'/sample_submission.csv')



#We can see that are not major imbalance in the dataset

DescriptiveStats(trainData)



#rearrange dataframe

trainData['label'] = trainData['Image_Label'].apply(lambda x: x.split('_')[1])

trainData['im_id'] = trainData['Image_Label'].apply(lambda x: x.split('_')[0])



subSample['label'] = subSample['Image_Label'].apply(lambda x: x.split('_')[1])

subSample['im_id'] = subSample['Image_Label'].apply(lambda x: x.split('_')[0])
#plot a random image

imageToPlot=random.choice(trainData['im_id'])

ShowImg(trainData,imageToPlot)

uniqueImgId=trainData.im_id.unique()

#split in train-test

train_ids, valid_ids = train_test_split(

        uniqueImgId,

        random_state=42,

        test_size=0.1)

#using efficientnet-b3 with imagenet weights

ENCODER = 'efficientnet-b2'

ENCODER_WEIGHTS = 'imagenet'

DEVICE = 'cuda'

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

num_workers = 0

bs = 1



ACTIVATION = None

model = smp.Unet(

    encoder_name=ENCODER, 

    encoder_weights=ENCODER_WEIGHTS, 

    classes=4, 

    activation=ACTIVATION,

)
train_dataset = CloudDataset(data=trainData, dataSetType='train', img_ids=train_ids, transforms = get_training_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))

valid_dataset = CloudDataset(data=trainData, dataSetType='train', img_ids=valid_ids, transforms = get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))



train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)

valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)

loaders = {

    "train": train_loader,

    "valid": valid_loader

}
num_epochs = 10

logdir = "./logs/CloudsSegmentation"



optimizer = torch.optim.Adam([

    {'params': model.decoder.parameters(), 'lr': 5e-3}, 

    {'params': model.encoder.parameters(), 'lr': 7e-4},  

])

            

#criterion=CustomLoss()

criterion=smp.utils.losses.BCEDiceLoss()

scheduler = ReduceLROnPlateau(optimizer, factor=0.3, patience=5)



runner = SupervisedRunner()

torch.cuda.empty_cache()

gc.collect()    
runner.train(

    model=model,

    criterion=criterion,

    optimizer=optimizer,

    scheduler=scheduler,

    loaders=loaders,

    #callbacks=[DiceCallback(),InferCallback()],

    logdir=logdir,

    num_epochs=num_epochs,

    verbose=True

)
#load best checkpoint



encoded_pixels = []

loaders = {"infer": valid_loader}

runner.infer(

    model=model,

    loaders=loaders,

    callbacks=[

        CheckpointCallback(

            resume=f"{logdir}/checkpoints/best.pth"),

        InferCallback()

    ],

)
resizedMasks=[]

probabilities = np.zeros((len(valid_dataset)*4, 350, 525))

# for each valid set and prediction on valid set:

# the predictions should be scaled down to a 350 x 525 pixel image

# make a resizedMasks of the valid elements

#make a probabilities mask with 4*len(valid_dataset) (labels)

for i in range(len(valid_dataset)):

    batch=valid_dataset[i]

    output=runner.callbacks[0].predictions["logits"][i]

    image, mask = batch

    for m in mask:

        m = cv2.resize(m, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)

        resizedMasks.append(m)

        

    for j in range(len(output)):

        probability = cv2.resize(output[j], dsize=(525, 350), interpolation=cv2.INTER_LINEAR)

        probabilities[i * 4 + j, :, :] = probability
#find the optimum threshold for each class



class_params = {}

for class_id in range(4):

    print("Calculating optimum threshold for class:",class_id)

    attempts = []

    for t in range(0, 100, 30):

        t /= 100

        for ms in [10000]:

            masks = []

            for i in range(class_id, len(probabilities), 4):

                probability = probabilities[i]

                predict, num_predict = post_process(sigmoid(probability), t, ms)

                masks.append(predict)



            d = []

            for i, j in zip(masks, resizedMasks[class_id::4]):

                if (i.sum() == 0) & (j.sum() == 0):

                    d.append(1)

                else:

                    d.append(dice(i, j))



            attempts.append((t, ms, np.mean(d)))



    attempts_df = pd.DataFrame(attempts, columns=['threshold', 'size', 'dice'])





    attempts_df = attempts_df.sort_values('dice', ascending=False)

    print(attempts_df.head())

    best_threshold = attempts_df['threshold'].values[0]

    best_size = attempts_df['size'].values[0]

    

    class_params[class_id] = (best_threshold, best_size)

        

        

print(class_params)     
del probabilities

del resizedMasks

del attempts_df

torch.cuda.empty_cache()

gc.collect()   
torch.cuda.empty_cache()

gc.collect()    



test_ids = subSample['Image_Label'].apply(lambda x: x.split('_')[0]).drop_duplicates().values



test_dataset = CloudDataset(data=subSample, dataSetType='test', img_ids=test_ids, transforms = get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)



loaders = {"test": test_loader}  





encoded_pixels = []

image_id = 0

for i, test_batch in enumerate(tqdm.tqdm(loaders['test'])):

    runner_out = runner.predict_batch({"features": test_batch[0].cuda()})['logits']

    for i, batch in enumerate(runner_out):

        for probability in batch:

            

            probability = probability.cpu().detach().numpy()

            if probability.shape != (350, 525):

                probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)

            predict, num_predict = post_process(sigmoid(probability), class_params[image_id % 4][0], class_params[image_id % 4][1])

            if num_predict == 0:

                encoded_pixels.append('')

            else:

                r = mask2rle(predict)

                encoded_pixels.append(r)

            image_id += 1
subSample['EncodedPixels'] = encoded_pixels

subSample.to_csv('sub12.csv', columns=['Image_Label', 'EncodedPixels'], index=False)