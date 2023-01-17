import numpy as np 
import pandas as pd

import tensorflow as tf
from kaggle_datasets import KaggleDatasets
print("Tensorflow version " + tf.__version__)

import IPython.display as display
from PIL import Image
import matplotlib.pyplot as plt
import glob
import os
import random
from sklearn.model_selection import train_test_split
import ast
TEST_FILE_DIR = '../input/shopee-product-detection-student/test/test/test/'
TRAIN_FILE_DIR = '../input/shopee-product-detection-student/train/train/train/'
train_df = pd.read_csv('../input/product-detection-a3/train_df_final.csv')
train_df.drop(columns='Unnamed: 0',inplace=True)
print(len(train_df))
train_df.head()
filename_list = list(train_df['filename'])
category_list = list(train_df['category'])
train_filename_list, val_filename_list, train_category_list, val_category_list = train_test_split(filename_list, category_list, test_size=0.10, random_state=42, stratify=category_list)
DATASET_SIZE = len(train_df)
print(DATASET_SIZE)
print(len(train_filename_list))
print(len(val_filename_list))
# For training/validation data

def get_file_name(file_path):
    return tf.strings.split(file_path, os.path.sep)[-1].numpy()

def get_label(file_path):
    # convert the path to a list of path components
    # The second to last is the ground truth label for the image
    return int(tf.strings.split(file_path, os.path.sep)[-2])

def get_image_btye_string(file_path):
    return open(file_path, 'rb').read()

def get_extra_features(file_name, train_df):
    feature_list = list(train_df[train_df['filename'] == file_name].iloc[0])
    assert(len(feature_list) == 7)
    word_vec =  ast.literal_eval(feature_list[2])
    width_norm = feature_list[3]
    height_norm = feature_list[4]
    hist_mean_norm = feature_list[5] 
    hist_std_norm = feature_list[6] 
    return {"word_vec": word_vec, 
            "width_norm": width_norm, "height_norm": height_norm, 
            "hist_mean_norm":hist_mean_norm, "hist_std_norm": hist_std_norm}

def create_example(file_path, train_df):
    file_name = get_file_name(file_path)
    label = get_label(file_path)
    image_byte_string = get_image_btye_string(file_path)
    features_dict = get_extra_features(file_path.split('/')[-1], train_df)
    
    feature = {
      'file_name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[file_name])),
      'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_byte_string])),
      'word_vec': tf.train.Feature(float_list=tf.train.FloatList(value=features_dict['word_vec'])),
      'width_norm': tf.train.Feature(float_list=tf.train.FloatList(value=[features_dict['width_norm']])),
      'height_norm': tf.train.Feature(float_list=tf.train.FloatList(value=[features_dict['height_norm']])),
      'hist_mean_norm': tf.train.Feature(float_list=tf.train.FloatList(value=[features_dict['hist_mean_norm']])),
      'hist_std_norm': tf.train.Feature(float_list=tf.train.FloatList(value=[features_dict['hist_std_norm']])),
      'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

TRAINING_DATA = glob.glob('/kaggle/input/shopee-code-league-week-2-product-detection/shopee-product-detection-dataset/train/train/*/*.jpg')
NUM_DATA_PER_FILE = len(val_filename_list)
%%time
FILE_NUM = 0

print(f"Getting data for file {FILE_NUM}.")
# filenames = train_filename_list[NUM_DATA_PER_FILE*FILE_NUM:NUM_DATA_PER_FILE*(FILE_NUM+1)]
# categories = train_category_list[NUM_DATA_PER_FILE*FILE_NUM:NUM_DATA_PER_FILE*(FILE_NUM+1)] 
filenames = val_filename_list
categories = val_category_list

record_file = f"val_a3_p{FILE_NUM}.tfrecords"
with tf.io.TFRecordWriter(record_file) as writer:
    for filename, category in zip(filenames, categories):
        file_path = TRAIN_FILE_DIR + str(category).zfill(2) + '/' + filename
        example = create_example(file_path, train_df)
        writer.write(example.SerializeToString())
    writer.close()
!rm train_a3_p6.tfrecords
def check_labeled_tfrecords(record_file):
    "Helper function to check the that TFRecord files for labeled data are correctly implemented."
    
    def _parse_image_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, image_feature_description)
    
    raw_image_dataset = tf.data.TFRecordDataset(record_file)
    image_feature_description = {
        'file_name': tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        'image': tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        'word_vec': tf.io.FixedLenFeature((1200,), tf.float32),
        'width_norm': tf.io.FixedLenFeature([], tf.float32),
        'height_norm': tf.io.FixedLenFeature([], tf.float32),
        'hist_mean_norm': tf.io.FixedLenFeature([], tf.float32),
        'hist_std_norm': tf.io.FixedLenFeature([], tf.float32),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    return raw_image_dataset.map(_parse_image_function)

for data in check_labeled_tfrecords(f"val_a3_p{FILE_NUM}.tfrecords").take(1):
#     print(data)
    print(data['word_vec'])

test_df = pd.read_csv('../input/product-detection-a3/test_df_final.csv')
test_df.drop(columns='Unnamed: 0',inplace=True)
print(len(test_df))
test_df.head()
def create_example(file_path, test_df):
    file_name = get_file_name(file_path)
    image_byte_string = get_image_btye_string(file_path)
    features_dict = get_extra_features(file_path.split('/')[-1], test_df)
    
    feature = {
      'file_name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[file_name])),
      'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_byte_string])),
      'word_vec': tf.train.Feature(float_list=tf.train.FloatList(value=features_dict['word_vec'])),
      'width_norm': tf.train.Feature(float_list=tf.train.FloatList(value=[features_dict['width_norm']])),
      'height_norm': tf.train.Feature(float_list=tf.train.FloatList(value=[features_dict['height_norm']])),
      'hist_mean_norm': tf.train.Feature(float_list=tf.train.FloatList(value=[features_dict['hist_mean_norm']])),
      'hist_std_norm': tf.train.Feature(float_list=tf.train.FloatList(value=[features_dict['hist_std_norm']])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))
%%time

filename_list = list(test_df['filename'])
                                   
record_file = 'test_a3.tfrecords'
with tf.io.TFRecordWriter(record_file) as writer:
    for filename in filename_list:
        file_path = TEST_FILE_DIR + filename
        example = create_example(file_path, test_df)
        writer.write(example.SerializeToString())
    writer.close()
def check_unlabeled_tfrecords(record_file):
    "Helper function to check the that TFRecord files for unlabeled data are correctly implemented."
    
    def _parse_image_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, image_feature_description)
    
    raw_image_dataset = tf.data.TFRecordDataset(record_file)
    image_feature_description = {
        'file_name': tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        'image': tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        'word_vec': tf.io.FixedLenFeature((1200,), tf.float32),
        'width_norm': tf.io.FixedLenFeature([], tf.float32),
        'height_norm': tf.io.FixedLenFeature([], tf.float32),
        'hist_mean_norm': tf.io.FixedLenFeature([], tf.float32),
        'hist_std_norm': tf.io.FixedLenFeature([], tf.float32),
    }
    return raw_image_dataset.map(_parse_image_function)

for data in check_unlabeled_tfrecords('test_a3.tfrecords').take(1):
    print(data)
#     print(data['word_vec'])
for data in check_unlabeled_tfrecords('test_a3.tfrecords').take(1):
    file_name = data['file_name'].numpy().decode("utf-8") 
    print(file_name)
    image_raw = data['image'].numpy()
    display.display(display.Image(data=image_raw))
for data in check_labeled_tfrecords('train_a2_p13.tfrecords').take(1):
    file_name = data['file_name'].numpy().decode("utf-8") 
    print(file_name)
    label = data['label'].numpy()
    print(label)
    image_raw = data['image'].numpy()
    display.display(display.Image(data=image_raw))