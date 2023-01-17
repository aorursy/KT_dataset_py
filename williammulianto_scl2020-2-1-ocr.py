!pip install keras_ocr
!pip freeze > requirements.txt
import matplotlib.pyplot as plt

import keras_ocr



import numpy as np

import pandas as pd



from time import time

import re

import itertools

import multiprocessing

import gc
print('Numpy version:', np.__version__)

print('Pandas version:', pd.__version__)
!ls /kaggle/input
df = pd.read_csv('/kaggle/input/shopee-product-detection-student/train.csv', dtype='object')

df['category'] = df['category'].apply(lambda c: str(c).zfill(2))

df.head()
paths = []

for i in df.index:

    paths.append(f'/kaggle/input/shopee-product-detection-student/train/train/train/{df.loc[i, "category"]}/{df.loc[i, "filename"]}')
pipeline = keras_ocr.pipeline.Pipeline()
# part 1 = 0:26347

# part 2 = 26347:52695 (CURRENT)

# part 3 = 52695:79042

# part 4 = 79042:TOTAL_IMAGES

TOTAL_IMAGES = len(paths)

BATCH_PREDICT = 5



START_INDEX = 79042

END_INDEX = TOTAL_IMAGES



list_texts = []
for i in range(START_INDEX, END_INDEX, BATCH_PREDICT):

    try:

        if i + BATCH_PREDICT < END_INDEX:

            images = [keras_ocr.tools.read(p) for p in paths[i:i + BATCH_PREDICT]]

        else:

            images = [keras_ocr.tools.read(p) for p in paths[i:END_INDEX]]



        prediction_groups = pipeline.recognize(images)



        for x in range(len(prediction_groups)):

            texts = []



            for y in range(len(prediction_groups[x])):

                texts.append(prediction_groups[x][y][0])



            list_texts.append(texts)

        gc.collect()

    except:

        if i + BATCH_PREDICT < END_INDEX:

            for j in range(BATCH_PREDICT):

                list_texts.append([])

        else:

            for j in range(END_INDEX - i):

                list_texts.append([])
sr_text = pd.Series(list_texts)

sr_text
sr_text.to_csv('train2d.csv', index=False, header=False)