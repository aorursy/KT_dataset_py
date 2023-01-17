!pip install '/kaggle/input/pytorch-15/torch-1.5.0cu101-cp37-cp37m-linux_x86_64.whl'

!pip install '/kaggle/input/pytorch-15/torchvision-0.6.0cu101-cp37-cp37m-linux_x86_64.whl'

!pip install '/kaggle/input/pytorch-15/yacs-0.1.7-py3-none-any.whl'

!pip install '/kaggle/input/pytorch-15/fvcore-0.1.1.post200513-py3-none-any.whl'

!pip install '/kaggle/input/pytorch-15/pycocotools-2.0-cp37-cp37m-linux_x86_64.whl'

!pip install '/kaggle/input/pytorch-15/detectron2-0.1.3cu101-cp37-cp37m-linux_x86_64.whl'
!nvidia-smi
# install dependencies: (use cu101 because colab has CUDA 10.1)

# !pip install -U torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html 

# !pip install cython pyyaml==5.1

# !pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# import torch, torchvision

# print(torch.__version__, torch.cuda.is_available())

# !gcc --version







# # install detectron2:

# !pip install detectron2==0.1.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html
import numpy as np

import pandas as pd

import torch

import os

import random



import matplotlib.pyplot as plt

from matplotlib import patches

import seaborn as sns

%matplotlib inline



import cv2

import itertools



import detectron2

from detectron2.utils.logger import setup_logger

setup_logger()

from detectron2 import model_zoo

from detectron2.engine import DefaultPredictor, DefaultTrainer

from detectron2.config import get_cfg

from detectron2.utils.visualizer import Visualizer

from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader

from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from detectron2.structures import BoxMode
DATA_DIR  = '../input/global-wheat-detection/train/'

TEST_DIR  = '../input/global-wheat-detection/test/'

List_Data_dir = os.listdir(DATA_DIR)
raw = pd.read_csv('../input/global-wheat-detection/train.csv')

raw
# Extract bbox column to xmin, ymin, width, height, then create xmax, ymax, and area columns



raw[['xmin','ymin','w','h']] = pd.DataFrame(raw.bbox.str.strip('[]').str.split(',').tolist()).astype(float)

raw['xmax'], raw['ymax'], raw['area'] = raw['xmin'] + raw['w'], raw['ymin'] + raw['h'], raw['w'] * raw['h']

raw
# split train, val

unique_files = raw.image_id.unique()



train_files = set(np.random.choice(unique_files, int(len(unique_files) * 0.90), replace = False))

train_df = raw[raw.image_id.isin(train_files)]

test_df = raw[~raw.image_id.isin(train_files)]
def custom_dataset(df, dir_image):

    

    dataset_dicts = []

    

    for img_id, img_name in enumerate(df.image_id.unique()):

        

        record = {}

        image_df = df[df['image_id'] == img_name]

        img_path = dir_image + img_name + '.jpg'

        

        record['file_name'] = img_path

        record['image_id'] = img_id

        record['height'] = int(image_df['height'].values[0])

        record['width'] = int(image_df['width'].values[0])

                

        objs = []

        for _, row in image_df.iterrows():

            

            x_min = int(row.xmin)

            y_min = int(row.ymin)

            x_max = int(row.xmax)

            y_max = int(row.ymax)

            

            poly = [(x_min, y_min), (x_max, y_min),

                    (x_max, y_max), (x_min, y_max) ]

            

            poly = list(itertools.chain.from_iterable(poly))

            

            obj = {

               "bbox": [x_min, y_min, x_max, y_max],

               "bbox_mode": BoxMode.XYXY_ABS,

               "segmentation": [poly],

               "category_id": 0,

               "iscrowd" : 0

                

                  }

            

            objs.append(obj)

            

        record['annotations'] = objs

        dataset_dicts.append(record)

        

    return dataset_dicts
def register_dataset(df, dataset_label='wheat_train', image_dir = DATA_DIR):

    

    # Register dataset - if dataset is already registered, give it a new name    

    try:

        DatasetCatalog.register(dataset_label, lambda d=df: custom_dataset(df, image_dir))

        MetadataCatalog.get(dataset_label).set(thing_classes = ['wheat'])

    except:

        # Add random int to dataset name to not run into 'Already registered' error

        n = random.randint(1, 1000)

        dataset_label = dataset_label + str(n)

        DatasetCatalog.register(dataset_label, lambda d=df: custom_dataset(df, image_dir))

        MetadataCatalog.get(dataset_label).set(thing_classes = ['wheat'])



    return MetadataCatalog.get(dataset_label), dataset_label
metadata, train_dataset = register_dataset(train_df)

metadata, val_dataset = register_dataset(test_df, dataset_label='wheat_test')



print(metadata, train_dataset)
cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = (train_dataset,)

cfg.DATASETS.TEST = ()

cfg.DATALOADER.NUM_WORKERS = 4

cfg.MODEL.WEIGHTS = "/kaggle/input/retinanet/R-101.pkl"

cfg.SOLVER.IMS_PER_BATCH = 4

cfg.SOLVER.BASE_LR =  0.001

cfg.SOLVER.MAX_ITER = 1500

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256     

cfg.MODEL.RETINANET.NUM_CLASSES = 1



# cfg.SOLVER.WARMUP_ITERS = 1000



# cfg.SOLVER.STEPS = (1000, 1500)

# cfg.SOLVER.GAMMA = 0.05









os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = DefaultTrainer(cfg) 

trainer.resume_or_load(resume=False)
trainer.train()
cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))

cfg.MODEL.WEIGHTS = "output/model_final.pth"

cfg.MODEL.RETINANET.NUM_CLASSES = 1

cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.4

cfg.DATASETS.TEST = ('wheat_test', )

predictor = DefaultPredictor(cfg)
# evaluator = COCOEvaluator(val_dataset, cfg, False, output_dir="./output/")

# val_loader = build_detection_test_loader(cfg, val_dataset)

# inference_on_dataset(trainer.model, val_loader, evaluator)
# CONFIG



font = cv2.FONT_HERSHEY_SIMPLEX     

fontScale = 1 

color = (255, 255, 0)

thickness = 2

results = []



def format_prediction_string(boxes, scores):

    pred_strings = []

    for j in zip(scores, boxes):

        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))



    return " ".join(pred_strings)





def result_show(df, color):

    

    for image_id in df_sub['image_id']:

        im = cv2.imread('{}/{}.jpg'.format(TEST_DIR, image_id))

        boxes = []

        scores = []

        labels = []

        outputs = predictor(im)

        out = outputs["instances"].to("cpu")

        scores = out.get_fields()['scores'].numpy()

        boxes = out.get_fields()['pred_boxes'].tensor.numpy().astype(int)

        labels= out.get_fields()['scores'].numpy()

        boxes = boxes.astype(int)

        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]

        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

        result = {'image_id': image_id,'PredictionString': format_prediction_string(boxes, scores)}

        results.append(result)

        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32)

        im /= 255.0

        

        for b,s in zip(boxes,scores):

            cv2.rectangle(im, (b[0],b[1]), (b[0]+b[2],b[1]+b[3]), color, thickness)

            cv2.putText(im, '{:.2}'.format(s), (b[0],b[1]), font, 1, color, thickness)

                

        plt.figure(figsize=(12,12))

        plt.imshow(im)


df_sub = pd.read_csv('../input/global-wheat-detection/sample_submission.csv')

df_sub



result_show(df_sub['image_id'], color = (255, 255, 255))

# print(results)

test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

test_df.to_csv('submission.csv', index=False)

test_df