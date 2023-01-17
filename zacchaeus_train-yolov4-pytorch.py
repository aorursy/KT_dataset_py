!rm ./* -r
!cp -r ../input/yolov4pytorch/pytorch-YOLOv4/* ./
!pip install -U -r requirements.txt
import numpy as np

import pandas as pd

import os

df = pd.read_csv('../input/global-wheat-detection/train.csv')

bboxs = np.stack(df['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))

for i, column in enumerate(['x', 'y', 'w', 'h']):

    df[column] = bboxs[:,i]

df.drop(columns=['bbox'], inplace=True)

df['x1'] = df['x'] + df['w']

df['y1'] = df['y'] + df['h']

df['classes'] = 0

from tqdm.auto import tqdm

import shutil as sh

df = df[['image_id','x', 'y', 'w', 'h','x1','y1','classes']]

df.head()
def f7(seq):

    seen = set()

    seen_add = seen.add

    return [x for x in seq if not (x in seen or seen_add(x))]

index = f7(df.image_id)

import random

random.Random(42).shuffle(index)

len(index)
source = 'train'

for fold in [0]:

    val_index = index[len(index)*fold//5:len(index)*(fold+1)//5]

    for name,mini in tqdm(df.groupby('image_id')):

        if not os.path.exists('convertor'):

            os.makedirs('convertor')

        sh.copy("../input/global-wheat-detection/{}/{}.jpg".format(source,name),'convertor/{}.jpg'.format(name))

        if name in val_index:

            path2save = 'convertor/val.txt'

        else:

            path2save = 'convertor/train.txt'

        with open(path2save, 'a') as f:

            f.write(f'{name}.jpg')

            row = mini[['x','y','x1','y1','classes']].astype(int).values

            # row = row/1024

            row = row.astype(str)

            for j in range(len(row)):

                text = ' '+','.join(row[j])

                f.write(text)

            f.write('\n')
#!python train.py -l 0.01 -g 0 -classes 1 -dir /kaggle/working/convertor -pretrained ../input/yolov4coco/yolov4.conv.137.pth -optimizer sgd -iou-type giou -train_label_path convertor/train.txt
!rm convertor/*