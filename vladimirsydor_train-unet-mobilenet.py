!pip install segmentation-models-pytorch
import os

import numpy as np

import pandas as pd

import torch

import cv2

import collections

import segmentation_models_pytorch as smp

import albumentations as albu



from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from glob import glob

from os import path

from PIL import Image

from tqdm import tqdm

from matplotlib import pyplot as plt



%matplotlib inline
ROOT_PATH_TRAIN = '/kaggle/input/imaterialist-fashion-2019-FGVC6/train'

DF_PATH_TRAIN = '/kaggle/input/imaterialist-fashion-2019-FGVC6/train.csv'

PATH_TO_MODEL_WEIGHTS = '/kaggle/input/za-cho-takoe-testovoe/best_model.pth'



IMAGE_PREDICTION_SIZE = (256, 256)

N_CLASSES = 46
def rle_decode(mask_rle, shape):

    '''

    mask_rle: run-length as string formated: [start0] [length0] [start1] [length1]... in 1d array

    shape: (height,width) of array to return

    Returns numpy array according to the shape, 1 - mask, 0 - background

    '''

    shape = (shape[1], shape[0])

    s = mask_rle.split()

    # gets starts & lengths 1d arrays

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]

    starts -= 1

    # gets ends 1d array

    ends = starts + lengths

    # creates blank mask image 1d array

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    # sets mark pixles

    for lo, hi in zip(starts, ends):

        img[lo:hi] = 1

    # reshape as a 2d mask image

    return img.reshape(shape).T  # Needed to align to RLE direction
def create_one_represent_class(df_param):

    v_c_df = df_param['CategoryId'].value_counts().reset_index()

    one_represent = v_c_df.loc[v_c_df['CategoryId'] == 1, 'index'].tolist()

    df_param.loc[df_param['CategoryId'].isin(one_represent), 'CategoryId'] = 'one_represent'

    return df_param



def custom_train_test_split(df_param):

    

    df_param['CategoryId'] = df_param.ClassId.apply(lambda x: str(x).split("_")[0])

    

    img_categ = train_df.groupby('ImageId')['CategoryId'].apply(list).reset_index()

    img_categ['CategoryId'] = img_categ['CategoryId'].apply(lambda x: ' '.join(sorted(x)))

    

    img_categ = create_one_represent_class(img_categ)

    

    img_train, img_val  = train_test_split(img_categ, test_size=0.2, random_state=42, stratify=img_categ['CategoryId'])

    

    df_param = df_param.drop(columns='CategoryId')

    

    df_train = df_param[df_param['ImageId'].isin(img_train['ImageId'])].reset_index(drop=True)

    df_val = df_param[df_param['ImageId'].isin(img_val['ImageId'])].reset_index(drop=True)

    

    return df_train, df_val
train_df = pd.read_csv(DF_PATH_TRAIN)

train_df.head()
train_df, val_df = custom_train_test_split(train_df)
class UnetDataset(torch.utils.data.Dataset):

    def __init__(self, image_dir, df, height, width, augmentation=None, preprocessing=None):

        

        self.preprocessing = preprocessing

        self.augmentation = augmentation

        

        self.image_dir = image_dir

        self.df = df

        

        self.height = height

        self.width = width

        

        self.image_info = collections.defaultdict(dict)

        

        self.df['CategoryId'] = self.df.ClassId.apply(lambda x: str(x).split("_")[0])

        self.num_classes = self.df['CategoryId'].nunique()

        

        temp_df = self.df.groupby('ImageId')['EncodedPixels', 'CategoryId'].agg(lambda x: list(x)).reset_index()

        size_df = self.df.groupby('ImageId')['Height', 'Width'].mean().reset_index()

        temp_df = temp_df.merge(size_df, on='ImageId', how='left')

        

        for index, row in tqdm(temp_df.iterrows(), total=len(temp_df)):

            image_id = row['ImageId']

            image_path = os.path.join(self.image_dir, image_id)

            self.image_info[index]["image_id"] = image_id

            self.image_info[index]["image_path"] = image_path

            self.image_info[index]["width"] = self.width

            self.image_info[index]["height"] = self.height

            self.image_info[index]["labels"] = row["CategoryId"]

            self.image_info[index]["orig_height"] = row["Height"]

            self.image_info[index]["orig_width"] = row["Width"]

            self.image_info[index]["annotations"] = row["EncodedPixels"]



    def __getitem__(self, idx):

        

        img_path = self.image_info[idx]["image_path"]

        

        img = cv2.imread(img_path)

        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (self.width, self.height))



        info = self.image_info[idx]

        

        mask = np.zeros((self.width, self.height, self.num_classes))

        

        for annotation, label in zip(info['annotations'], info['labels']):

            cur_mask = rle_decode(annotation, (info['orig_height'], info['orig_width']))

            mask[:, :, int(label)] += cv2.resize(cur_mask, (self.width, self.height))

            

        mask = (mask > 0.5).astype(np.float32)

        

        # apply augmentations

        if self.augmentation is not None:

            sample = self.augmentation(image=img, mask=mask)

            img, mask = sample['image'], sample['mask']

        

        # apply preprocessing

        if self.preprocessing is not None:

            sample = self.preprocessing(image=img, mask=mask)

            img, mask = sample['image'], sample['mask']

            

        return img, mask



    def __len__(self):

        return len(self.image_info)
def get_training_augmentation():

    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.VerticalFlip(p=0.5),

    ]

    return albu.Compose(train_transform)
def to_tensor(x, **kwargs):

    return x.transpose(2, 0, 1).astype('float32')



def get_preprocessing(preprocessing_fn):

    """Construct preprocessing transform

    

    Args:

        preprocessing_fn (callbale): data normalization function 

            (can be specific for each pretrained neural network)

    Return:

        transform: albumentations.Compose

    

    """

    

    _transform = [

        albu.Lambda(image=preprocessing_fn),

        albu.Lambda(image=to_tensor, mask=to_tensor),

    ]

    return albu.Compose(_transform)

ENCODER = 'mobilenet_v2'

ENCODER_WEIGHTS = 'imagenet'

DEVICE = 'cuda'



ACTIVATION = 'sigmoid'
model = smp.Unet(

    encoder_name=ENCODER, 

    encoder_weights=ENCODER_WEIGHTS, 

    classes=N_CLASSES, 

    activation=ACTIVATION,

)



preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
train_dataset = UnetDataset(

    ROOT_PATH_TRAIN,

    train_df,

    IMAGE_PREDICTION_SIZE[0],

    IMAGE_PREDICTION_SIZE[1], 

    preprocessing=get_preprocessing(preprocessing_fn),

    augmentation=get_training_augmentation()

)



valid_dataset = UnetDataset(

    ROOT_PATH_TRAIN,

    val_df,

    IMAGE_PREDICTION_SIZE[0],

    IMAGE_PREDICTION_SIZE[1], 

    preprocessing=get_preprocessing(preprocessing_fn),

)





train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)

valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=2)
loss = smp.utils.losses.DiceLoss()

metrics = [

    smp.utils.metrics.IoU(threshold=0.5),

]



optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# create epoch runners 

# it is a simple loop of iterating over dataloader`s samples

train_epoch = smp.utils.train.TrainEpoch(

    model, 

    loss=loss, 

    metrics=metrics, 

    optimizer=optimizer,

    device=DEVICE,

    verbose=True,

)



valid_epoch = smp.utils.train.ValidEpoch(

    model, 

    loss=loss, 

    metrics=metrics, 

    device=DEVICE,

    verbose=True,

)
torch.save(model.state_dict(), 'best_model.pth')

max_score = 0



for i in range(0, 2):

    

    print('\nEpoch: {}'.format(i))

    train_logs = train_epoch.run(train_loader)

    valid_logs = valid_epoch.run(valid_loader)

    

    # do something (save model, change lr, etc.)

    if max_score < valid_logs['iou_score']:

        max_score = valid_logs['iou_score']

        torch.save(model.state_dict(), 'best_model.pth')

        print('Model saved!')