import os

import pandas as pd
base = "/kaggle/input/landmark-recognition-2020"

os.listdir(base)
train_df = pd.read_csv(os.path.join(base, "train.csv"))

train_df.head()
top_50 = train_df.landmark_id.value_counts()[:50].index.to_list()

top_100 = train_df.landmark_id.value_counts()[:100].index.to_list()

bottom_50 = train_df.landmark_id.value_counts()[-50:].index.to_list()

bottom_100 = train_df.landmark_id.value_counts()[-100:].index.to_list()

extreme_50 = train_df.landmark_id.value_counts()[:25].index.to_list() + train_df.landmark_id.value_counts()[-25:].index.to_list()

extreme_100 = train_df.landmark_id.value_counts()[:50].index.to_list() + train_df.landmark_id.value_counts()[-50:].index.to_list()

random_50 = train_df.landmark_id.sample(n=50)

random_100 = train_df.landmark_id.sample(n=100)

random_500 = train_df.landmark_id.sample(n=500)



dct = {"top_50": top_50, "top_100": top_100, "bottom_50": bottom_50, "bottom_100": bottom_100, 

       "extreme_50": extreme_50, "extreme_100": extreme_100, "random_50": random_50, "random_100": random_100,

       "random_500": random_500}
from tqdm.notebook import tqdm

import cv2

from shutil import copy2

def store_images(name, ids):

    temp = os.path.join("/kaggle/working", name)

    os.makedirs(temp, exist_ok=True)

    for lm_id in tqdm(ids):

        temp_two = os.path.join(temp, str(lm_id))

        os.makedirs(temp_two, exist_ok=True)

        for image_id in train_df[train_df.landmark_id == lm_id].id.to_list():

            src = os.path.join(base, 'train', image_id[0], image_id[1], image_id[2], image_id + '.jpg')

            dst = os.path.join(temp_two, image_id + '.jpg')

            copy2(src, dst)
'''

会空间不够

for key, value in tqdm(dct.items()):

    store_images(key, value)

    

要用哪个的时候调用哪个函数，然后在你的model notebook引用output，这里举得例子是 top_100

'''



store_images('random_50', dct['random_50'])
