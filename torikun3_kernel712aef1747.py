!pip install --no-deps '../input/pycocotools/pycocotools-2.0-cp37-cp37m-linux_x86_64.whl' > /dev/null

!pip install --no-deps '../input/effdet012/effdet-0.1.2-py3-none-any.whl' > /dev/null

!pip install --no-deps '../input/timm-lab/timm-0.1.30-py3-none-any.whl' > /dev/null
import sys

sys.path.append('../input/effdet/omegaconf/')

sys.path.append('../input/ensemble/')

import omegaconf

import timm
from effdet import DetBenchPredict

IMG_SIZE = 512
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

from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict

from effdet.efficientdet import HeadNet
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def load_net(checkpoint_path):

    config = get_efficientdet_config('tf_efficientdet_d7')

    net = EfficientDet(config, pretrained_backbone=False)



    config.num_classes = 1

    config.image_size = 512

    net.class_net =  HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01),act_layer=timm.models.layers.activations.Mish)



    checkpoint = torch.load(checkpoint_path)

    net.load_state_dict(checkpoint['model_state_dict'])



    del checkpoint

    gc.collect()



    net = DetBenchPredict(net, config)

    net.eval();

    return net.cuda()



models = [load_net('../input/weight2/fold0-best2.bin'),

        load_net('../input/weight2/fold1-best2.bin'),

        load_net('../input/weight2/fold2-best2.bin'),

        load_net('../input/weight2/fold3-best2.bin'),

        load_net('../input/weight2/fold4-best2.bin')]
def load_net(checkpoint_path):

    config = get_efficientdet_config('tf_efficientdet_d7')

    net = EfficientDet(config, pretrained_backbone=False)



    config.num_classes = 1

    config.image_size = 512

    net.class_net =  HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))



    checkpoint = torch.load(checkpoint_path)

    net.load_state_dict(checkpoint['model_state_dict'])



    del checkpoint

    gc.collect()



    net = DetBenchPredict(net, config)

    net.eval();

    return net.cuda()



models2 = [load_net('../input/weight1/fold0-best1.bin'),

        load_net('../input/weight1/fold1-best1.bin'),

        load_net('../input/weight1/fold2-best1.bin'),

        load_net('../input/weight1/fold3-best1.bin'),

        load_net('../input/weight1/fold4-best1.bin')]
models=models+models2
DATA_ROOT_PATH = '../input/global-wheat-detection/test'



class TestDatasetRetriever(Dataset):



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



def get_valid_transforms():

    return A.Compose(

        [

            A.Resize(height=512, width=512, p=1.0),

            ToTensorV2(p=1.0),

        ], 

        p=1.0, 

    )



dataset = TestDatasetRetriever(

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



def make_predictions(

    images, 

    score_threshold=0.25,

):

    

    predictions = []

    images = torch.stack(images).to(device).float()

    img_scale = torch.tensor([[1.] for _ in range(10)]).to(device)

    img_size = torch.tensor([(IMG_SIZE, IMG_SIZE) for _ in range(10)]).to(device)

        

    for fold_number, net in enumerate(models):

        with torch.no_grad():

            det = net(images,img_scales = img_scale,img_size = img_size)

            result = []

            for i in range(images.shape[0]):

                boxes = det[i].detach().cpu().numpy()[:,:4]    

                scores = det[i].detach().cpu().numpy()[:,4]

                indexes = np.where(scores > score_threshold)[0]

                boxes = boxes[indexes]

                boxes[:, 2] = boxes[:, 2] + boxes[:, 0]

                boxes[:, 3] = boxes[:, 3] + boxes[:, 1]

                result.append({

                    'boxes': boxes[indexes],

                    'scores': scores[indexes],

                })

            predictions.append(result)

    return predictions





def run_wbf(predictions, image_index, image_size=512, iou_thr=0.44, skip_box_thr=0.43, weights=None):

    boxes = [(prediction[image_index]['boxes']/(image_size-1)).tolist()  for prediction in predictions]

    scores = [prediction[image_index]['scores'].tolist()  for prediction in predictions]

    labels = [np.ones(prediction[image_index]['scores'].shape[0]).tolist() for prediction in predictions]

    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    boxes = boxes*(image_size-1)

    return boxes, scores, labels





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

        boxes = (boxes*2).astype(np.int32).clip(min=0, max=1023)

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

test_df