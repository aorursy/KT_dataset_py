from sklearn.model_selection import KFold

import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt, cv2

import tensorflow as tf

from glob import glob

import numpy as np # linear algebra  GLR-make-tfrecord-256-0

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from kaggle_datasets import KaggleDatasets

import random



def read_df_0(input_path):

    mapping = {}

    df = pd.read_csv(input_path + 'train.csv')

    

    

    files_paths = glob(input_path + 'train/*/*/*/*')

    for path in files_paths:

        mapping[path.split('/')[-1].split('.')[0]] = path

    

        

    df['path'] = df['id'].map(mapping)

    

    counts_map = dict(df.groupby('landmark_id')['path'].agg(lambda x: len(x)))

    df['counts'] = df['landmark_id'].map(counts_map)

    

    

    uniques = df['landmark_id'].unique()

    uniques_map = dict(zip(uniques, range(len(uniques))))

    df['labels'] = df['landmark_id'].map(uniques_map)

    return df



df = read_df_0('../input/landmark-recognition-2020/')

df.sort_values('counts',inplace=True)

df.head(10)
def _bytes_feature(value):

    """Returns a bytes_list from a string / byte."""

    if isinstance(value, type(tf.constant(0))):

        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))





def _float_feature(value):

    """Returns a float_list from a float / double."""

    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))





def _int64_feature(value):

    """Returns an int64_list from a bool / enum / int / uint."""

    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))





def serialize_example(feature0, feature1):

    feature = {

        'image': _bytes_feature(feature0),

        'label': _int64_feature(feature1)

        #'prob': _float_feature(feature2),



    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

    return example_proto.SerializeToString()







SIZE = 56500

total_num = len(df) // SIZE + int(len(df) % SIZE != 0)

for j in range(0,2):

    print('Writing TFRecord %i of %i...' % (j, total_num))

    num_per_tf = min(SIZE, len(df) - j * SIZE)

    with tf.io.TFRecordWriter('train%.2i-%i.tfrec' % (j, num_per_tf)) as writer:

        for k in range(num_per_tf):

            idx=j * SIZE + k

            

            path,label= df.path[idx], df.labels[idx]

            bits = tf.io.read_file(path)

            img = tf.image.decode_jpeg(bits, channels=3)

            print(img.shape)            



            img = tf.image.resize(img, (512,512), method='bilinear')

            img = tf.cast(img, tf.uint8)

     

            plt.imsave('img.jpg',img.numpy())

            img = open('./img.jpg', 'rb').read()            

            example = serialize_example(img,label)  # ,str.encode(path)ff

            writer.write(example)

            if k % 1000 == 0: print(k, ', ', end='')