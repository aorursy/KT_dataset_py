# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from pathlib import Path

import matplotlib.pyplot as plt
image_path = Path('/kaggle/input/the-oxfordiiit-pet-dataset/images/images')

anno_path = Path('/kaggle/input/the-oxfordiiit-pet-dataset/annotations/annotations')
img = []



for i in range(100):

    path = (image_path/f'samoyed_{i}.jpg')

    

    if path.exists():

        img.append(path)

    if len(img) == 10:

        break
trimaps = []



for i in range(100):

    path = (anno_path/'trimaps'/f'samoyed_{i}.png')

    

    if path.exists():

        trimaps.append(path)

    if len(trimaps) == 10:

        break
fig = plt.figure(figsize=(50, 12))



for idx, pair in enumerate(zip(img, trimaps)):

    image, seg = pair

    ax0 = fig.add_subplot(2, 10, idx+1)

    ax1 = fig.add_subplot(2, 10, idx+11)

    

    ax0.imshow(plt.imread(image))

    ax1.imshow(plt.imread(seg))

    

    
from pathlib import Path

import tensorflow as tf

from tqdm import tqdm

import re



import matplotlib.pyplot as plt



from typing import Dict
image_path = Path('/kaggle/input/the-oxfordiiit-pet-dataset/images/images')

anno_path = Path('/kaggle/input/the-oxfordiiit-pet-dataset/annotations/annotations')
def _byte_feature(value):

    """Returns a bytes_list from a string/bytes"""

    if isinstance(value, type(tf.constant([0]))):

        value = value.numpy()

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def _int64_feature(value):

    """Returns an int64_list from an int/uint"""    

    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))



def serialize_example(data: Dict) -> tf.Tensor:

    

    features = {

        'height': _int64_feature(data['height']),

        'width': _int64_feature(data['width']),

        'channels': _int64_feature(data['channels']),

        'image_raw': _byte_feature(data['image']),

        'segmentation': _byte_feature(data['segmentation']),

        'name': _byte_feature(data['name'])

    }

    

    example_proto = tf.train.Example(features=tf.train.Features(feature=features))

    return example_proto.SerializeToString()
def get_num(filename: Path) -> str:

    pat = r'([^\d]+)([\d]+)'

    filename_split = filename.name.split('.')[0]



    num = re.match(pat, filename_split, re.I).groups()[1]

    

    return int(num)

    

def compiled_data(img: Path, seg: Path) -> Dict:

    """Compile data ready for conversion into TFRecord Example"""

    

    img_bytes = plt.imread(img)



    height, width, channels = img_bytes.shape

    img_bytes = img_bytes.tobytes()

    seg_bytes = plt.imread(seg).tobytes()

    

    data = {

        'height': height,

        'width': width,

        'channels': channels,

        'image': img_bytes,

        'segmentation': seg_bytes,

        'name': img.name.encode()

    }

    

    return data

    



data_filename = 'data.tfrecord'



with tf.io.TFRecordWriter(data_filename) as writer:

    for dog_name in ['great_pyrenees', 'samoyed']:

        sorted_img = sorted(list(image_path.glob(f'{dog_name}_*')), key=get_num)

        sorted_seg = sorted(list((anno_path/'trimaps').glob(f'{dog_name}_*')), key=get_num)

        for img, seg in tqdm(zip(sorted_img, sorted_seg)):

            if img.name.split('.')[0] != seg.name.split('.')[0]:

                continue

            else:

                data = compiled_data(img, seg)

                example = serialize_example(data)

                writer.write(example)

                

print('Done writing!')
from typing import Tuple, Dict



import tensorflow as tf

from pathlib import Path
record_path = Path('/kaggle/input/oxford/data.tfrecord')



raw_dataset = tf.data.TFRecordDataset(str(record_path))
def parse_image_function(example_proto: tf.Tensor) -> Tuple[tf.Tensor, ...]:

    """

    Convert protobuf objects into image and segmentation

    

    Returns tuple of image and segmentations

    """

    

    image_feature_desc = {

        "height": tf.io.FixedLenFeature([], tf.int64),

        "width": tf.io.FixedLenFeature([], tf.int64),

        "channels": tf.io.FixedLenFeature([], tf.int64),

        "image": tf.io.FixedLenFeature([], tf.string),

        "segmentation": tf.io.FixedLenFeature([], tf.string),

        "name": tf.io.FixedLenFeature([], tf.string)

    }

    

    example = tf.io.parse_single_example(example_proto, image_feature_desc)

    

    img_raw = tf.io.decode_raw(example['image'], tf.uint8)

    img_raw = tf.reshape(img_raw, (example['height'], example['width'], example['channels']))

    

    seg_raw = tf.io.decode_raw(example['segmentation'], tf.uint8)

    seg_raw = tf.reshape(seg_raw, (example['height'], example['width'], example['channels']))

    

    return img_raw, seg_raw