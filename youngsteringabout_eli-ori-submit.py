# Import required packages

import numpy as np

import os

import pandas as pd

import re

from PIL import Image

from PIL import ImageDraw

import gc

import warnings

import time

from numba import jit

import glob





import albumentations as A



warnings.filterwarnings('ignore')



# PyTorch libraries

import torch

import torchvision



from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection import FasterRCNN



from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm



# OpenCV

import cv2



# Plotting libraries

import seaborn as sns

import matplotlib.pyplot as plt



# %matplotlib inline

# sns.set_style('whitegrid')

# sns.set_palette('muted')

# pd.set_option('display.max_columns', 500)
BATCH = 4

# Define the root directory

root_dir = '../input/global-wheat-detection'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

DATA_ROOT_PATH = '../input/global-wheat-detection/test/'

# DATA_ROOT_PATH = '../input/test222/'
# # Add columns for bbox coordinates

# train_data['xmin'] = -1

# train_data['ymin'] = -1

# train_data['width'] = -1

# train_data['height'] = -1



# # join lists along `axis=0` 

# print(train_data['bbox'].apply(lambda x: bbox_coordinates(x)).shape)

# coordinates = np.stack(train_data['bbox'].apply(lambda x: bbox_coordinates(x)))

# print(coordinates.shape)



# # Updating the newly created columns

# train_data[['xmin', 'ymin', 'width', 'height']] = coordinates



# # dropping the bbox column

# train_data.drop(columns='bbox', inplace=True)

# train_data.head()
class WheatDataset(Dataset):

    def __init__(self, image_ids, transforms=None):

        super().__init__()

#         self.data = data_dir

        self.image_ids = image_ids

        self.transforms = transforms



        

    def __getitem__(self, index :int):

        image_id = self.image_ids[index]

        image = cv2.imread(f'{DATA_ROOT_PATH}/{image_id}.jpg', cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        image /= 255.0

        

#         image_id = self.images[idx].split('.')[0]

#         path = os.path.join(self.data, image_id) + '.jpg'

#         image = Image.open(path).convert("RGB")

        

        if self.transforms is not None:

            image = self.transforms(image)

          

        return image, image_id

#         return image, target, image_id

    

    def __len__(self):

#         return len(self.images)

        return self.image_ids.shape[0]
# # helper class to keep track of loss and loss per iteration

# # source: https://www.kaggle.com/pestipeti/pytorch-starter-fasterrcnn-train

# class Averager:

#     def __init__(self):

#         self.current_total = 0

#         self.iterations = 0

        

#     def send(self, value):

#         self.current_total += value

#         self.iterations += 1

        

#     @property

#     def value(self):

#         if self.iterations == 0:

#             return 0

#         else:

#             return 1.0 * self.current_total / self.iterations

        

#     def reset(self):

#         self.current_total = 0

#         self.iterations = 0
def get_model(num_classes):

    # load the object detection model pre-trained on COCO

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)

    

    # get the input features in the classifier

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    

    # replace the input features of pretrained head with the num_classes

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    

    return model
# creating some transforms

# source: https://github.com/pytorch/vision/blob/master/references/detection/transforms.py

# source: https://github.com/microsoft/computervision-recipes/blob/master/utils_cv/detection/dataset.py





class Compose(object):

    def __init__(self, transforms):

        self.transforms = transforms

        

    def __call__(self, image):

        for t in self.transforms:

            image = t(image)

        return image



class RandomHorizontalFlip(object):

    """

    Wrapper for torchvision's HorizontalFlip

    """

    def __init__(self, prob):

        self.prob = prob

        

    def __call__(self, image):

        if random.random() < self.prob:

            height, width = image.shape[-2:]  # image must be a ndarray or torch tensor (PIL.Image has no attribute `.shape`)

            image = image.flip(-1)

#             bbox = target['boxes']    # bbox coordinates MUST be of form [xmin, ymin, xmax, ymax]

#             bbox[:, [0, 2]] = width - bbox[:, [2, 0]]

#             target['boxes'] = bbox

        return image



class ColorJitterTransform(object):

    """

    Wrapper for torchvision's ColorJitter

    """

    def __init__(self, brightness, contrast, saturation, hue):

        self.brightness = brightness

        self.contrast = contrast

        self.saturation = saturation

        self.hue = hue

    

    def __call__(self, image):

        image = ColorJitter(

            brightness=self.brightness,

            contrast=self.contrast,

            saturation=self.saturation,

            hue=self.hue

        )(image)

        return image

        

class ToTensor(object):

    def __call__(self, image):

        image = F.to_tensor(image)   # normalizes the image and converts PIL image to torch.tensor

        return image



class Resize(object):

    def __init__(self, height, width):

        self.height = height

        self.width = width

    

    def __call__(self, image):

        image = torchvision.transforms.Resize((self.height, self.width))

        return image

        

class eli_bright(object):

    def __init__(self, threshold, value):

        self.threshold = threshold

        self.value = value

    

    def __call__(self, image):

        image[image > self.threshold] = self.value

        

        return image

    

class gamma(object):

    def __init__(self, gama, gain):

        self.gama = gama

        self.gain = gain

    

    def __call__(self, image):

        image = F.adjust_gamma(image, self.gama, self.gain)

        

        return image

        
import random

from torchvision.transforms import functional as F

from torchvision.transforms import ColorJitter



# merge all the images in a batch

def collate_fn(batch):

    return tuple(zip(*batch))



# add some image augmentation operations

def get_transform(train):

    transforms = []

    

#     transforms.append(A.Resize(height=1024, width=1024, p=1.0))

#     if train:

#     transforms.append(Resize(1024, 1024))

        

#     if train:

        # needs the image to be a PIL image

#     transforms.append(ColorJitterTransform(brightness=0.2, contrast=0.2, saturation=0.4, hue=0.05))

        

#     if train:

#         transforms.append(gamma(2, 1))

        

    # converts a PIL image to pytorch tensor

    transforms.append(ToTensor())

    

#     if train:

        # randomly flip the images, bboxes and ground truth (only during training)

#     transforms.append(RandomHorizontalFlip(0.5))  # this operation needs image to be a torch tensor

    

#     if train:

#         transforms.append(eli_bright(0.5, 0))

#     return transforms.append(A.Compose([

#             A.Resize(height=1024, width=1024, p=1.0),

#             ToTensorV2(p=1.0),

#         ], p=1.0))

    

        

    return Compose(transforms)
def format_prediction_string(boxes, scores):

    pred_strings = []

    for j in zip(scores, boxes):

        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))



    return " ".join(pred_strings)
# data_dir = os.path.join(root_dir, 'test')

# dataset_train = WheatDataset(data_dir=data_dir, 

#                              transforms=get_transform(train=False))

dataset_train = WheatDataset(np.array([path.split('/')[-1][:-4] for path in glob.glob(f'{DATA_ROOT_PATH}/*.jpg')]), 

                             transforms=get_transform(train=False))

dataloader_train = DataLoader(dataset_train, batch_size=BATCH, shuffle=True, 

                                   num_workers=4, collate_fn=collate_fn)
batch = iter(dataloader_train)

imgs,img_id = next(batch)

images = list(img.to(device) for img in imgs)
num_classes = 2

model = get_model(num_classes)

saved_model_path = '../input/the-model/fasterrcnn_resnet50_fpn.pth'

model.load_state_dict(torch.load(saved_model_path))

model.eval()

model.to(device)

# predictions = model(images)
# predictions[0]['boxes']
results = []

for images, image_ids in dataloader_train:

    images = torch.stack(images).float().cuda()

    preds = model(images)



    for i,img in enumerate(images):

        if len(preds[i]['boxes']) > 0:

            boxes = preds[i]['boxes'].cpu().detach().numpy().astype(np.int32).clip(min=0, max=1023)

            scores = preds[i]['scores'].detach().cpu().numpy()

    #         boxes_s = boxes.round().astype(np.int32).clip(min=0, max=1023)







            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]

            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

            

            score_threshold = 0.5

            indexes = np.where(scores > score_threshold)[0]

            boxes = boxes[indexes]

            image_id = image_ids[i]

            

            print(len(boxes))

            

            result = {

              'image_id': image_id,

              'PredictionString': format_prediction_string(boxes,scores)

            }

    

        else:

            image_id = image_ids[i]

            result = {

              'image_id': image_id,

              'PredictionString': ''

            }

            

        results.append(result)
test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

test_df.to_csv('submission.csv', index=False)

test_df.head(10)



# SUBMISSION_PATH = '/kaggle/working'

# submission_id = 'submission'

# cur_submission_path = os.path.join(SUBMISSION_PATH, '{}.csv'.format(submission_id))

# sample_submission = pd.DataFrame(results, columns=["image_id","PredictionString"])

# sample_submission.to_csv(cur_submission_path, index=False)

# submission_df = pd.read_csv(cur_submission_path)
# submission_df.head(10)
# def show_bbox_single_image(images, targets):

#     score_threshold = 0.4

#     image = Image.fromarray(images[0].mul(255).permute(1,2,0).cpu().byte().numpy())

#     boxes = targets[0]['boxes'].cpu().detach().numpy().astype(np.int64)

# #     boxes[:, 2] = boxes[:, 2] - boxes[:, 0]

# #     boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

    

#     indexes = np.where(scores > score_threshold)[0]

#     boxes = boxes[indexes]

#     draw = ImageDraw.Draw(image)



#     for box in boxes:

#         coord1 = (box[0], box[1])

#         coord2 = (box[2], box[3])

#         draw.rectangle([coord1, coord2], outline=(220,20,60), width=3)

#     return image



# images, image_ids = next(iter(dataloader_train))

# images = list(img.to(device) for img in images)

# targets = model(images)

# image = show_bbox_single_image(images, targets)

# image
# def createSubmission(models, device, data_loader, detection_threshold=0.9, iou_thr=0.5):

#     SUBMISSION_PATH = '/kaggle/working'

#     submission_id = 'submission'

#     final_csv = []

#     preds = model(images)

#     for (image, boxes, scores, image_path) in results:

#         boxes = boxes.detach().cpu().numpy()

#         if boxes.shape[0] > 0 :

#             boxes[:, 2] = boxes[:, 2] - boxes[:, 0]

#             boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

#             image_id = image_path.split("/")[-1]

#             result = [image_id,format_prediction_string(boxes, scores)]

#             final_csv.append(result)



#     cur_submission_path = os.path.join(SUBMISSION_PATH, '{}.csv'.format(submission_id))

#     sample_submission = pd.DataFrame(final_csv, columns=["image_id","PredictionString"])

#     sample_submission.to_csv(cur_submission_path, index=False)

#     submission_df = pd.read_csv(cur_submission_path)

#     return submission_df

# submission = createSubmission([resnet50,desnet121], device, test_loader, detection_threshold=0.5, iou_thr=0.3)