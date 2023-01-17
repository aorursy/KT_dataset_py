import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sys

sys.path.insert(0, "../input/weightedboxesfusion")



from ensemble_boxes import *

import cv2

import os

from tqdm.notebook import tqdm

from glob import glob

from PIL import Image

import torch

import torchvision

from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor

from torchvision.models.detection.rpn import AnchorGenerator

import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2

from torch.utils.data import DataLoader, Dataset

import re

from matplotlib import pyplot as plt
# def read_bbox_from_str(s):

#     s = s.replace("[", "").replace("]", "")

#     char_bbox = s.split(",")

#     bbox = [float(c) for c in char_bbox]

#     return bbox





def format_prediction_string(boxes, scores):

    pred_strings = []

    for j in zip(scores, boxes):

        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)
class WheatTestDataset(Dataset):

    def __init__(self, img_ids, img_dir, transforms):

        self.img_ids = img_ids

        self.img_dir = img_dir

        self.transforms = transforms

        

    def __getitem__(self, idx):

        img_id = self.img_ids[idx]

        img_fn = f"{self.img_dir}/{img_id}.jpg"

        img = cv2.imread(img_fn)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        img /= 255.0

        if self.transforms:

            sample = {

                'image': img,

            }

            sample = self.transforms(**sample)

            img = sample['image']

        return img_id, img

        

    def __len__(self):

        return len(self.img_ids)

    

    

def collate_fn(batch):

    return tuple(zip(*batch))


test_df = pd.read_csv("/kaggle/input/global-wheat-detection/sample_submission.csv")

img_ids = test_df["image_id"]

print("number of image:",  len(img_ids))

img_dir = "/kaggle/input/global-wheat-detection/test"



test_transform = A.Compose([ToTensorV2(p=1.0)])

test_dataset = WheatTestDataset(img_ids, img_dir, test_transform)

test_dataloader = DataLoader(

    test_dataset,

    batch_size=8,

    shuffle=False,

    num_workers=2,

    collate_fn=collate_fn

)
nb_class = 2



weight_file = "/kaggle/input/modev5/model_800_0.82.pth"

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)

in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, nb_class)



if os.path.isfile(weight_file):

    print("loading ...")

    model.load_state_dict(torch.load(weight_file))

    

params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(params, lr=0.005, momentum=0, weight_decay=0.0005)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model.to(device)

model.eval()

print("")
def run_wbf(prediction, image_size=1024, iou_thr=0.44, skip_box_thr=0.43, weights=None):

    boxes = [(prediction['boxes']/(image_size-1)).tolist()]

    scores = [prediction['scores'].tolist()]

    labels = [np.ones(prediction['scores'].shape[0]).tolist()]

    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    boxes = boxes*(image_size-1)

    return boxes, scores, labels
thresh = 0.5

results = []



for image_ids, images in tqdm(test_dataloader):

    images = list(image.to(device) for image in images)

    predictions = model(images)



    for i, image in enumerate(images):

#         boxes, scores, _ = run_wbf(predictions[i], iou_thr=0.4)

        boxes = predictions[i]["boxes"]

        scores = predictions[i]["scores"]

    

#         indicies = torchvision.ops.nms(boxes, scores, iou_threshold=0.5)

#         boxes = boxes[indicies].data.cpu().numpy().astype(np.int32).clip(min=0, max=1024)

#         scores = scores[indicies].data.cpu().numpy()



        boxes = boxes.data.cpu().numpy().astype(np.int32).clip(min=0, max=1024)

        scores = scores.data.cpu().numpy()

        boxes = boxes[scores >= thresh]

        scores = scores[scores >= thresh]

        

        boxes[:, 2] = (boxes[:, 2] - boxes[:, 0])

        boxes[:, 3] = (boxes[:, 3] - boxes[:, 1])



#         boxes[:, 0] = boxes[:,0] - boxes[:,2]*0.1

#         boxes[:, 1] = boxes[:,1] - boxes[:,3]*0.1

        result = {

            'image_id': image_ids[i],

            'PredictionString': format_prediction_string(boxes, scores)

        }

        results.append(result)
test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

test_df.head()
test_df.to_csv('submission.csv', index=False)