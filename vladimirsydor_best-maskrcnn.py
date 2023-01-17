!pip install segmentation-models-pytorch
import os

import numpy as np

import pandas as pd

import torch

import cv2

import sys

import collections

import albumentations as albu

import torchvision

import segmentation_models_pytorch as smp



from torch.utils.data import DataLoader

from glob import glob

from os import path

from PIL import Image

from tqdm import tqdm

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor



%matplotlib inline
IMG_WIDTH = 256

IMG_HEIGHT = 256

NUM_CLASSES = 46



BATCH_SIZE = 16

N_WORKERS = 2



DEVICE = 'cuda'



root_path_train = '/kaggle/input/imaterialist-fashion-2019-FGVC6/train'

df_path_train = '/kaggle/input/imaterialist-fashion-2019-FGVC6/train.csv'
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







def get_unique_class_id_df(inital_df):

    temp_df = inital_df.groupby(['ImageId','ClassId'])['EncodedPixels'].agg(lambda x: ' '.join(list(x))).reset_index()

    size_df = inital_df.groupby(['ImageId','ClassId'])['Height', 'Width'].mean().reset_index()

    temp_df = temp_df.merge(size_df, on=['ImageId','ClassId'], how='left')

    

    return temp_df
train_df = pd.read_csv(df_path_train)

train_df.head()
train_df, val_df = custom_train_test_split(train_df)
train_df, val_df = get_unique_class_id_df(train_df), get_unique_class_id_df(val_df)
class FashionDataset(torch.utils.data.Dataset):

    def __init__(self, image_dir, df, height, width, transforms=None):

        self.transforms = transforms

        self.image_dir = image_dir

        self.df = df

        self.height = height

        self.width = width

        self.image_info = collections.defaultdict(dict)

        self.df['CategoryId'] = self.df.ClassId.apply(lambda x: str(x).split("_")[0])

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

            

        self.img2tensor = torchvision.transforms.ToTensor()



    def __getitem__(self, idx):

        # load images ad masks

        img_path = self.image_info[idx]["image_path"]

        img = Image.open(img_path).convert("RGB")

        img = img.resize((self.width, self.height), resample=Image.BILINEAR)



        info = self.image_info[idx]

        mask = np.zeros((len(info['annotations']), self.width, self.height), dtype=np.uint8)

        labels = []

        for m, (annotation, label) in enumerate(zip(info['annotations'], info['labels'])):

            sub_mask = rle_decode(annotation, (info['orig_height'], info['orig_width']))

            sub_mask = Image.fromarray(sub_mask)

            sub_mask = sub_mask.resize((self.width, self.height), resample=Image.BILINEAR)

            mask[m, :, :] = sub_mask

            labels.append(int(label) + 1)



        num_objs = len(labels)

        boxes = []

        new_labels = []

        new_masks = []



        for i in range(num_objs):

            try:

                pos = np.where(mask[i, :, :])

                xmin = np.min(pos[1])

                xmax = np.max(pos[1])

                ymin = np.min(pos[0])

                ymax = np.max(pos[0])

                if abs(xmax - xmin) >= 20 and abs(ymax - ymin) >= 20:

                    boxes.append([xmin, ymin, xmax, ymax])

                    new_labels.append(labels[i])

                    new_masks.append(mask[i, :, :])

            except ValueError:

                continue



        if len(new_labels) == 0:

            boxes.append([0, 0, 20, 20])

            new_labels.append(0)

            new_masks.append(mask[0, :, :])



        nmx = np.zeros((len(new_masks), self.width, self.height), dtype=np.uint8)

        for i, n in enumerate(new_masks):

            nmx[i, :, :] = n



        target = {}

        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)

        target["labels"] = torch.as_tensor(new_labels, dtype=torch.int64)

        target["masks"] = torch.as_tensor(nmx, dtype=torch.uint8)

        

        img = self.img2tensor(img)



        if self.transforms is not None:

            img, target = self.transforms(img, target)



        return img, target



    def __len__(self):

        return len(self.image_info)

def custom_collate(batch):

    images = []

    labels = []

    for img, label in batch:

        images.append(img)

        labels.append(label)

        

    return images, labels
datset = FashionDataset(image_dir=root_path_train, 

                        df=train_df, 

                        height=IMG_HEIGHT, 

                        width=IMG_WIDTH)
data_loader = torch.utils.data.DataLoader(

    datset, batch_size=4, shuffle=True, num_workers=2,

    collate_fn=custom_collate)
class MyEpoch(smp.utils.train.Epoch):

    def _to_device(self):

        self.model.to(self.device)

        

    def run(self, dataloader):



        self.on_epoch_start()



        logs = {}

        loss_meter = smp.utils.meter.AverageValueMeter()

        

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:

            for x, y in iterator:

                x = list(map(lambda x_el: x_el.to(self.device), x))

                y = list(map(lambda y_el: {k:v.to(self.device) for k,v in y_el.items()}, y))

                loss = self.batch_update(x, y)



                # update loss logs

                loss_value = loss.cpu().detach().numpy()

                loss_meter.add(loss_value)

                loss_logs = {'loss': loss_meter.mean}

                logs.update(loss_logs)



                if self.verbose:

                    s = self._format_logs(logs)

                    iterator.set_postfix_str(s)



        return logs

    

class TrainEpoch(MyEpoch):



    def __init__(self, model, loss, metrics, optimizer, device='cpu', verbose=True):

        super().__init__(

            model=model,

            loss=loss,

            metrics=metrics,

            stage_name='train',

            device=device,

            verbose=verbose,

        )

        self.optimizer = optimizer



    def on_epoch_start(self):

        self.model.train()



    def batch_update(self, x, y):

        self.optimizer.zero_grad()

        loss = self.model(x, y)

        loss = sum(l for l in loss.values())

        loss.backward()

        self.optimizer.step()

        return loss

num_classes = NUM_CLASSES + 1

device = torch.device(DEVICE)



model_ft = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

in_features = model_ft.roi_heads.box_predictor.cls_score.in_features

model_ft.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

in_features_mask = model_ft.roi_heads.mask_predictor.conv5_mask.in_channels

hidden_layer = 256

model_ft.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
for param in model_ft.parameters():

    param.requires_grad = True
optimizer = torch.optim.Adam(model_ft.parameters(), lr=0.001)
train_epoch = TrainEpoch(

    model_ft, 

    loss=None, 

    metrics=None, 

    optimizer=optimizer,

    device=DEVICE,

    verbose=True,

)

torch.save(model_ft.state_dict(), 'best_model.pth')
train_epoch.run(data_loader)
torch.save(model_ft.state_dict(), 'best_model.pth')