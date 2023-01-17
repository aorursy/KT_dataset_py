!pip install --no-index ../input/ensemble-boxes/ensemble_boxes-1.0.4-py3-none-any.whl
import os, json, time, random, operator, functools



from tqdm.notebook import tqdm

from IPython import display

from ipywidgets import Output



import pandas as pd

import numpy as np

from skimage import io

from PIL import Image, ImageDraw 

import matplotlib.pyplot as plt



import torch

import torch.utils

import torchvision 

from torchvision import transforms

import torch.nn as nn

import torch.optim as optim



import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2



import ensemble_boxes
path = '../input/global-wheat-detection'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cpu = torch.device("cpu")
class wheatDataset(object):

    def __init__(self, root, mode, transform = None):

        self.root = root

        self.transform = transform

        self.mode = mode



        self.imgs = sorted(list(os.listdir(os.path.join(root, mode))))

        

        if mode == "train":

          self.table = pd.read_csv(os.path.join(root, "train.csv"))

          self.table.bbox = self.table.bbox.map(lambda x: json.loads(x))

          self.table = self.table.groupby(['image_id'])

          self.ID = self.table.groups.keys()

        #root - train.csv

        #     |- train - image



    def __getitem__(self, idx):

        # load images

        image_name = self.imgs[idx]

        image_id = image_name[:-4]



        if (self.mode == 'train'): 

          if (not(image_id in self.ID)):

              return self.__getitem__(random.randint(0,len(self.imgs)-1))##avoid no wheat

        

        img_path = os.path.join(self.root, self.mode, image_name)

        img = np.asarray(Image.open(img_path).convert("RGB"))



        if (self.mode == 'test'):

          return transforms.ToTensor()(img)



        # get bounding box

        boxes_list = np.array(self.table.get_group(image_id).bbox.to_list())

        boxes_list[:,2] += boxes_list[:,0]#x2 = x1 + box width

        boxes_list[:,3] += boxes_list[:,1]#y2 = y1 + box hight

        



#         # convert everything into a torch.Tensor

#         boxes = torch.as_tensor(boxes_list, dtype=torch.float32)

#         # there is only one class

#         label = torch.ones((len(boxes_list),), dtype=torch.int64)

#         image_id = torch.tensor([idx])

        

        # not crowd

#         iscrowd = torch.zeros((1,), dtype=torch.int64)

        boxes = boxes_list

        label = np.ones((len(boxes_list),), dtype=np.int64)

        

        target = {}

        if self.transform is not None:

            transformed = self.transform(image = img, bboxes=boxes, labels = label)

            while(len(transformed['bboxes']) == 0):

                transformed = self.transform(image = img, bboxes=boxes, labels = label)

            

            img = transforms.ToTensor()(transformed['image'])

            target["boxes"] = torch.as_tensor(np.asarray(transformed['bboxes']), dtype=torch.float32)

            target["labels"] = torch.as_tensor(np.asarray(transformed['labels']), dtype=torch.int64)

        else:

            img = transforms.ToTensor()(img)

            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)

            target["labels"] = torch.as_tensor(label, dtype=torch.int64)

            

        

        return img, target



    def __len__(self):

        return len(self.imgs)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained_backbone=False)

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

num_classes = 2  # wheat + background

model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, num_classes)
model.load_state_dict(torch.load("../input/fastrcnnwheat/Trained-wheat-fastrcnn"))

model.to(device)
dataset_test = wheatDataset(path, "test")

data_loader = torch.utils.data.DataLoader(

        dataset_test, batch_size=10, shuffle=False, num_workers=5, pin_memory = True,

        collate_fn=lambda batch: batch)
def test_numpy():

  result = list()

  for images in tqdm(data_loader):

    images = list(image.to(device) for image in images)

    model.eval()

    predictions = model(images)

    

    images = list(image.to(cpu).detach().numpy() for image in images)

    predictions = list({k:v.to(cpu).detach().numpy() for k, v in p.items()} for p in predictions)

    

    result.append(predictions)

#     for x,y in zip(images, predictions):

#       drawBondingBox_test(x,y['boxes'])

  return result



result = test_numpy()
result = functools.reduce(operator.add, result)
IDs = list(map(lambda n: n[:-4], sorted(list(os.listdir(os.path.join(path, 'test'))))))
def drawBondingBox_test(image_name, listOfbox):

    img_path = os.path.join(path, 'test', image_name + '.jpg')

    img = Image.open(img_path).convert("RGB")

    draw = ImageDraw.Draw(img) 

    for b in listOfbox:

        x0,y0,x1,y1 = b

        draw.rectangle((x0,y0,x1,y1), fill=None, outline="#FA6E1A", width=4)

    plt.figure(figsize = (10,10))

    plt.imshow(img, interpolation='nearest')

    plt.show()
n = 0

submission = pd.DataFrame(columns=['image_id', 'PredictionString'])

for predictions in result:

    y = predictions

    # print(IDs[n])

    

    boxes = [(b/1024).tolist() for b in y['boxes']]

    scores = y['scores'].tolist()

    labels = y['labels'].astype(int).tolist()

    

    

    boxes, scores, labels = ensemble_boxes.ensemble_boxes_wbf.weighted_boxes_fusion([boxes], [scores], [labels], weights=None, iou_thr=0.44, skip_box_thr=0.43)

    boxes = (np.array(boxes)*1024).astype(int)

#     drawBondingBox_test(IDs[n], boxes)

    PredictionString = ""

    for b, s in zip(boxes, scores):

        PredictionString += "{:.4f} {:.0f} {:.0f} {:.0f} {:.0f} ".format(s, b[0], b[1], b[2]-b[0], b[3]-b[1])

    PredictionString = PredictionString[:-1]

    submission.loc[n] = [IDs[n], PredictionString]

    n+=1
submission.to_csv("submission.csv",index=False)