!pip install --no-deps '../input/timm-package/timm-0.1.26-py3-none-any.whl' > /dev/null

!pip install --no-deps '../input/pycocotools/pycocotools-2.0-cp37-cp37m-linux_x86_64.whl' > /dev/null
import sys

sys.path.insert(0, "../input/timm-efficientdet-pytorch")

sys.path.insert(0, "../input/omegaconf")

sys.path.insert(0, "../input/weightedboxesfusion")



from ensemble_boxes import *

import torch

import numpy as np

import pandas as pd

from glob import glob

from torch.utils.data import Dataset,DataLoader

import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2

import cv2

import gc

from matplotlib import pyplot as plt

from effdet import get_efficientdet_config, EfficientDet, DetBenchEval

from effdet.efficientdet import HeadNet
img_size = 1024
def get_valid_transforms():

    return A.Compose([

            A.Resize(height=img_size, width=img_size, p=1.0),

            ToTensorV2(p=1.0),

        ], p=1.0)
DATA_ROOT_PATH = '../input/global-wheat-detection/test'



class DatasetRetriever(Dataset):



    def __init__(self, image_ids, transforms=None):

        super().__init__()

        self.image_ids = image_ids

        self.transforms = transforms



    def __getitem__(self, index: int):

        image_id = self.image_ids[index]

        image = cv2.imread(f'{DATA_ROOT_PATH}/{image_id}.jpg', cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        image /= 255.0

        if self.transforms:

            sample = {'image': image}

            sample = self.transforms(**sample)

            image = sample['image']

        return image, image_id



    def __len__(self) -> int:

        return self.image_ids.shape[0]
dataset = DatasetRetriever(

    image_ids=np.array([path.split('/')[-1][:-4] for path in glob(f'{DATA_ROOT_PATH}/*.jpg')]),

    transforms=get_valid_transforms()

)



def collate_fn(batch):

    return tuple(zip(*batch))



data_loader = DataLoader(

    dataset,

    batch_size=2,

    shuffle=False,

    num_workers=4,

    drop_last=False,

    collate_fn=collate_fn

)
def load_net(checkpoint_path):

    config = get_efficientdet_config('tf_efficientdet_d4')

    net = EfficientDet(config, pretrained_backbone=False)



    config.num_classes = 1

    config.image_size=img_size

    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))



    checkpoint = torch.load(checkpoint_path)

    net.load_state_dict(checkpoint['model_state_dict'])



    del checkpoint

    gc.collect()



    net = DetBenchEval(net, config)

    net.eval();

    return net.cuda()



#net = load_net('../input/wheat-effdet5-fold0-best-checkpoint/fold0-best-all-states.bin')

net = load_net('../input/weights/deaug.bin')
def make_predictions(images, score_threshold=0.2):

    images = torch.stack(images).cuda().float()

    predictions = []

    with torch.no_grad():

        det = net(images, torch.tensor([1]*images.shape[0]).float().cuda())

        for i in range(images.shape[0]):

            boxes = det[i].detach().cpu().numpy()[:,:4]    

            scores = det[i].detach().cpu().numpy()[:,4]

            indexes = np.where(scores > score_threshold)[0]

            boxes = boxes[indexes]

            boxes[:, 2] = boxes[:, 2] + boxes[:, 0]

            boxes[:, 3] = boxes[:, 3] + boxes[:, 1]

            predictions.append({

                'boxes': boxes[indexes],

                'scores': scores[indexes],

            })

    return [predictions]



def run_wbf(predictions, image_index, image_size=img_size, iou_thr=0.44, skip_box_thr=0.43, weights=None):

    boxes = [(prediction[image_index]['boxes']/(image_size-1)).tolist()  for prediction in predictions]

    scores = [prediction[image_index]['scores'].tolist()  for prediction in predictions]

    labels = [np.ones(prediction[image_index]['scores'].shape[0]).tolist() for prediction in predictions]

    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    boxes = boxes*(image_size-1)

    return boxes, scores, labels
import matplotlib.pyplot as plt



for j, (images, image_ids) in enumerate(data_loader):

    break



predictions = make_predictions(images)



i = 0

sample = images[i].permute(1,2,0).cpu().numpy()



boxes, scores, labels = run_wbf(predictions, image_index=i)

boxes = boxes.astype(np.int32).clip(min=0, max=img_size)



fig, ax = plt.subplots(1, 1, figsize=(16, 8))



for box in boxes:

    cv2.rectangle(sample, (box[0], box[1]), (box[2], box[3]), (1, 0, 0), 1)

    

ax.set_axis_off()

ax.imshow(sample);
def format_prediction_string(boxes, scores):

    pred_strings = []

    for j in zip(scores, boxes):

        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)
results = []



for images, image_ids in data_loader:

    predictions = make_predictions(images)

    for i, image in enumerate(images):

        boxes, scores, labels = run_wbf(predictions, image_index=i)

        boxes = boxes.astype(np.int32).clip(min=0, max=1023)

        image_id = image_ids[i]

        

        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]

        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]



        result = {

            'image_id': image_id,

            'PredictionString': format_prediction_string(boxes, scores)

        }

        results.append(result)
test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

test_df.to_csv('submission.csv', index=False)

test_df.head()