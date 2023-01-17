import tensorflow as tf

print(tf.__version__)
import pandas as pd

import numpy as np

import os

import cv2

from PIL import Image, ImageDraw

from ast import literal_eval

import matplotlib.pyplot as plt

import urllib

from tqdm.notebook import tqdm
!cp -r /kaggle/input/efficientdet-code/EfficientDet /kaggle/working/
!ls /kaggle/working/EfficientDet
# !git clone https://github.com/xuannianz/EfficientDet.git

os.chdir('/kaggle/working/EfficientDet/')

!python setup.py build_ext --inplace
from model import efficientdet

from utils import preprocess_image, postprocess_boxes

from utils.draw_boxes import draw_boxes



phi = 3

weighted_bifpn = True

model_path = '/kaggle/input/efficientdet-version1-weights/phi3_stage1_9286.h5'

image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)

image_size = image_sizes[phi]



num_classes = 1

score_threshold = 0.5

_, model = efficientdet(phi=phi,

                            weighted_bifpn=weighted_bifpn,

                            num_classes=num_classes,

                            score_threshold=score_threshold)

model.load_weights(model_path, by_name=True)
import glob

from tqdm import tqdm

from pathlib import Path



DIR_INPUT = '/kaggle/input/global-wheat-detection'

data_dir = '/kaggle/input/global-wheat-detection/test'

submission = pd.read_csv(f'{DIR_INPUT}/sample_submission.csv')



root_image = Path("/kaggle/input/global-wheat-detection/test")

test_images = [root_image / f"{img}.jpg" for img in submission.image_id]



submission = []



def clahe(image, clip_limit=4.0, tile_grid_size=(8, 8)): # image is in RGB already, load_image in csv generator

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    lab_planes = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit,tileGridSize=tile_grid_size)

    lab_planes[0] = clahe.apply(lab_planes[0])

    lab = cv2.merge(lab_planes)

    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)



for image_path in tqdm(test_images):

    prediction_string = []

    

    image = cv2.imread(str(image_path))

    image = image[:, :, ::-1]

    image = clahe(image)

    

    h, w = image.shape[:2]

    image, scale = preprocess_image(image, image_size=image_size)

    

    boxes, scores, labels = model.predict_on_batch([np.expand_dims(image, axis=0)])

    boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)

        

    boxes = postprocess_boxes(boxes=boxes, scale=scale, height=h, width=w)

    

    # select indices which have a score above the threshold

    indices = np.where(scores[:] > score_threshold)[0]



    # select those detections

    boxes = boxes[indices]

    labels = labels[indices]

    

    for idx in range(boxes.shape[0]):

        box,s=boxes[idx],scores[idx]

        x_min, y_min, x_max, y_max = box

        

        x = round(x_min)

        y = round(y_min)

        w = round(x_max-x_min)

        h = round(y_max-y_min)

        prediction_string.append(f"{s} {x} {y} {w} {h}")

        

    prediction_string = " ".join(prediction_string)

    submission.append([image_path.name[:-4],prediction_string])



sample_submission = pd.DataFrame(submission, columns=["image_id","PredictionString"])

sample_submission.to_csv('/kaggle/working/submission.csv', index=False)

sample_submission.head()

# for image_path in glob.glob("/kaggle/input/global-wheat-detection/test/*.jpg"):

    

#     image = cv2.imread(image_path)

#     image = image[:, :, ::-1]

    

#     h, w = image.shape[:2]

#     image, scale = preprocess_image(image, image_size=image_size)

    

#     boxes, scores, labels = model.predict_on_batch([np.expand_dims(image, axis=0)])

#     boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)

        

#     boxes = postprocess_boxes(boxes=boxes, scale=scale, height=h, width=w)

    

#     # select indices which have a score above the threshold

#     indices = np.where(scores[:] > score_threshold)[0]



#     # select those detections

#     boxes = boxes[indices]

#     labels = labels[indices]

    

    

#     for idx in range(boxes.shape[0]):

#         box,score=boxes[idx],scores[idx]

#         imgid.append((image_path.split(".")[0]).split('/')[-1])

#         preds.append("{} {} {} {} {}".format(score, int(box[0]), int(box[1]), int(box[2]-box[0]), int(box[3]-box[1])))
for index, row in sample_submission.iterrows():

    path = '/kaggle/input/global-wheat-detection/test/' + row["image_id"] + '.jpg'

    img = cv2.imread(path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = clahe(img)

    

    bboxs = row['PredictionString'].split(' ')

    for i in range(len(bboxs)//5):

        bbox = bboxs[i*5:(i+1)*5]

        score, xmin, ymin, width, height = bbox

        

        xmin = int(float(xmin))

        ymin = int(float(ymin))

        xmax = xmin + int(float(width))

        ymax = ymin + int(float(height))

        

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255,0,0), 5)

        

    plt.imshow(img)

    plt.show()
sample_submission.head()
!rm -r /kaggle/working/EfficientDet/
# sub = {"image_id": imgid, "PredictionString": preds}

# sub = pd.DataFrame(sub)

# sub_ = sub.groupby(["image_id"])['PredictionString'].apply(lambda x: ' '.join(x)).reset_index()

# print(sub_)
# sub_.to_csv('/kaggle/working/submission.csv', index=False)
# samsub=pd.read_csv("/kaggle/input/global-wheat-detection/sample_submission.csv")

# samsub.head()
# for idx,imgid in enumerate(samsub['image_id']):

#     samsub.iloc[idx,1]=sub_[sub_['image_id']==imgid].values[0,1]

    

# samsub.head()
# samsub.to_csv('/kaggle/working/submission.csv',index=False)
# img = cv2.imread('/kaggle/input/global-wheat-detection/test/' + imgid[0] + '.jpg')

# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# preds = sub_[sub_['image_id']==imgid[0]].PredictionString.values

# preds = preds[0].split(' ')



# for i in range(len(preds)//5):

#     conf, xmin, ymin, width, height = preds[:5]

#     preds = preds[5:]

    

#     xmin = int(xmin)

#     ymin = int(ymin)

#     width = int(width)

#     height = int(height)

    

#     cv2.rectangle(img, (xmin,ymin), (xmin+width, ymin+height), (255, 0, 0), 2)
# import matplotlib.pyplot as plt

# plt.imshow(img)
# import pandas as pd



# sub=pd.read_csv('/kaggle/input/efficientdet-version1-weights/submission.csv')

# sub.to_csv('/kaggle/working/submission.csv',index=False)