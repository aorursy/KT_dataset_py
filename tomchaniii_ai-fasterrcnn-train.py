import pandas as pd

import numpy as np

import cv2

import os

import re



from PIL import Image



import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2



import torch

import torchvision



from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection import FasterRCNN

from torchvision.models.detection.rpn import AnchorGenerator



from torch.utils.data import DataLoader, Dataset

from torch.utils.data.sampler import SequentialSampler



from matplotlib import pyplot as plt



DIR_INPUT = '/kaggle/input/global-wheat-detection'

DIR_TRAIN = f'{DIR_INPUT}/train'

DIR_TEST = f'{DIR_INPUT}/test'
# Download TorchVision repo to use some files from

# references/detection

!git clone https://github.com/pytorch/vision.git && cd vision && git checkout v0.3.0



!cp vision/references/detection/utils.py ./

!cp vision/references/detection/transforms.py ./

!cp vision/references/detection/coco_eval.py ./

!cp vision/references/detection/engine.py ./

!cp vision/references/detection/coco_utils.py ./

!rm -rf vision
!pip install cython

# Install pycocotools, the version by default in Colab

# has a bug fixed in https://github.com/cocodataset/cocoapi/pull/354

!pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
train_df = pd.read_csv(f'{DIR_INPUT}/train.csv')

train_df.shape
train_df['x'] = -1

train_df['y'] = -1

train_df['w'] = -1

train_df['h'] = -1



def expand_bbox(x):

    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))

    if len(r) == 0:

        r = [-1, -1, -1, -1]

    return r



train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))

train_df.drop(columns=['bbox'], inplace=True)

train_df['x'] = train_df['x'].astype(np.float)

train_df['y'] = train_df['y'].astype(np.float)

train_df['w'] = train_df['w'].astype(np.float)

train_df['h'] = train_df['h'].astype(np.float)
image_ids = train_df['image_id'].unique()

valid_ids = image_ids[-500:]

train_ids = image_ids[:-500]
valid_df = train_df[train_df['image_id'].isin(valid_ids)]

train_df = train_df[train_df['image_id'].isin(train_ids)]
valid_df.shape, train_df.shape
import os

from PIL import Image

import math



class WheatDataset(Dataset):



    def __init__(self, dataframe, image_dir, transforms=None, noise=0, size_ratio=1):

        super().__init__()



        self.image_ids = dataframe['image_id'].unique()

        self.df = dataframe

        self.image_dir = image_dir

        self.transforms = transforms

        self.noise = noise

        self.size_ratio = size_ratio



    def __getitem__(self, index: int):

        

        if self.noise > 0:

            is_noise = index % 2

            index = math.floor(index / 2)



        image_id = self.image_ids[index]

        records = self.df[self.df['image_id'] == image_id]

        image = Image.open(os.path.join(self.image_dir, f'{image_id}.jpg')).convert("RGB")

        image = image.resize((round(image.width * self.size_ratio), round(image.height * self.size_ratio)))

        

        if self.noise > 0:

            if is_noise > 0:

                noise_image = Image.effect_noise(image.size, 127).convert("RGB")

                image = Image.blend(image, noise_image, self.noise)

                



        boxes = records[['x', 'y', 'w', 'h']].values

        

        boxes[:, 2] = np.clip(np.round((boxes[:, 0] + boxes[:, 2]) * self.size_ratio), 0, image.width)

        boxes[:, 3] = np.clip(np.round((boxes[:, 1] + boxes[:, 3]) * self.size_ratio), 0, image.height)

        boxes[:, 0] = np.clip(np.round(boxes[:, 0] * self.size_ratio), 0, image.width)

        boxes[:, 1] = np.clip(np.round(boxes[:, 1] * self.size_ratio), 0, image.height)

        

        

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        area = torch.as_tensor(area, dtype=torch.float32)



        # there is only one class

        labels = torch.ones((records.shape[0],), dtype=torch.int64)

        

        # suppose all instances are not crowd

        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        

        target = {}

        target['boxes'] = torch.as_tensor(boxes, dtype=torch.int64)

        target['labels'] = labels

        target['image_id'] = torch.tensor([index])

        target['area'] = area

        target['iscrowd'] = iscrowd



        if self.transforms is not None:

            image, target = self.transforms(image, target)



        return image, target



    def __len__(self) -> int:

        if self.noise > 0:

            return self.image_ids.shape[0] * 2

        else:

            return self.image_ids.shape[0]
# helper functions for data augmentation / transformation

from engine import train_one_epoch, evaluate

import transforms as T





def get_transform(train):

    transforms = []

    # converts the image, a PIL image, into a PyTorch Tensor

    transforms.append(T.ToTensor())

    if train:

        # during training, randomly flip the training images

        # and ground-truth for data augmentation

        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)

# load a model; pre-trained on COCO

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2  # 1 class (wheat) + background



# get number of input features for the classifier

in_features = model.roi_heads.box_predictor.cls_score.in_features



# replace the pre-trained head with a new one

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
import utils



train_dataset = WheatDataset(train_df, DIR_TRAIN, get_transform(train=True), noise=0.2, size_ratio=1)

valid_dataset = WheatDataset(valid_df, DIR_TRAIN, get_transform(train=False))



train_data_loader = DataLoader(

    train_dataset,

    batch_size=16,

    shuffle=True,

    num_workers=4,

    collate_fn=utils.collate_fn

)



valid_data_loader = DataLoader(

    valid_dataset,

    batch_size=16,

    shuffle=True,

    num_workers=4,

    collate_fn=utils.collate_fn

)
sample, target = train_dataset[0]

fig, ax = plt.subplots(1, 1, figsize=(16, 8))



sample = sample.permute(1,2,0).cpu().numpy()

boxes = target['boxes'].data.cpu().numpy().astype(np.int32)



sample = cv2.UMat(sample).get()



for box in boxes:

    cv2.rectangle(

        sample, 

        (box[0], box[1]), 

        (box[2], box[3]),

        (220, 0, 0), 2)



ax.set_axis_off()

ax.imshow(sample)



sample, target = train_dataset[1]

fig, ax = plt.subplots(1, 1, figsize=(16, 8))



sample = sample.permute(1,2,0).cpu().numpy()

boxes = target['boxes'].data.cpu().numpy().astype(np.int32)



sample = cv2.UMat(sample).get()



for box in boxes:

    cv2.rectangle(

        sample, 

        (box[0], box[1]), 

        (box[2], box[3]),

        (220, 0, 0), 2)



ax.set_axis_off()

ax.imshow(sample)
#for gpu

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

torch.cuda.empty_cache()
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(params, lr=0.00349, momentum=0.9, weight_decay=0.0004)

# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

lr_scheduler = None



num_epochs = 8
import sys

sys.path.insert(0, "/kaggle/working")



from engine import train_one_epoch, evaluate



for epoch in range(num_epochs):

    # train for one epoch, printing every 10 iterations

    train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=20)

    

    # update the learning rate

    if lr_scheduler is not None:

        lr_scheduler.step()

        

    # evaluate on the test dataset

    evaluate(model, valid_data_loader, device=device)
# save model

torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn.pth')
#remove downloaded modules

!rm utils.py

!rm transforms.py

!rm coco_eval.py

!rm engine.py

!rm coco_utils.py