!pip install -U ../input/kerasapplications/Keras_Applications-1.0.8-py3-none-any.whl

!pip install -U ../input/efficientnetwhl/efficientnet-1.1.0-py3-none-any.whl

import efficientnet.tfkeras as efn
import tensorflow as tf

print(tf.__version__)

import numpy as np

import math

import pandas as pd

from sklearn import model_selection

import glob

import os

from zipfile import ZipFile

import shutil

import tqdm.notebook as tqdm

import random, re, math, os

from kaggle_datasets import KaggleDatasets



import matplotlib.pyplot as plt

import logging

tf.get_logger().setLevel(logging.ERROR)

import warnings

warnings.filterwarnings("ignore")



import copy

import csv

import gc

import operator

import os

import pathlib



import numpy as np

import PIL

import pydegensac

from scipy import spatial



mixed_precision = False



DEVICE = "TPU"

if DEVICE == "TPU":

    print("connecting to TPU...")

    try:

        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

        print('Running on TPU ', tpu.master())

    except ValueError:

        print("Could not connect to TPU")

        tpu = None



    if tpu:

        try:

            print("initializing  TPU ...")

            tf.config.experimental_connect_to_cluster(tpu)

            tf.tpu.experimental.initialize_tpu_system(tpu)

            strategy = tf.distribute.experimental.TPUStrategy(tpu)

            print("TPU initialized")

        except _:

            print("failed to initialize TPU")

    else:

        DEVICE = "GPU"



if DEVICE != "TPU":

    print("Using default strategy for CPU and single GPU")

    strategy = tf.distribute.get_strategy()



if DEVICE == "GPU":

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))



strategy_num = strategy.num_replicas_in_sync

print(f'strategy.num: {strategy_num}')

    

config = {

    'learning_rate': 5e-3,

    'momentum': 0.9,

    'scale': 30,

    'margin': 0.3,

    'clip_grad': 10.0,

    'n_epochs': 3,

    'input_size': (800,800, 3),

    'n_classes': 81313,

}



BATCH_SIZE=32*strategy_num

AUTO = tf.data.experimental.AUTOTUNE
class AddMarginProduct(tf.keras.layers.Layer):

    '''

    Implements large margin cosine distance.



    References:

        https://arxiv.org/pdf/1801.07698.pdf

        https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/

            blob/master/src/modeling/metric_learning.py

    '''

    def __init__(self, n_classes, s=30, m=0.30, **kwargs):

        super(AddMarginProduct, self).__init__(**kwargs)

        self.n_classes = n_classes

        self.s = s

        self.m = m



    def build(self, input_shape):

        super(AddMarginProduct, self).build(input_shape[0])



        self.W = self.add_weight(

            name='W',

            shape=(int(input_shape[0][-1]), self.n_classes),

            initializer='glorot_uniform',

            dtype='float32',

            trainable=True,

            regularizer=None)



    def call(self, inputs):

        X, y = inputs

        y = tf.cast(y, dtype=tf.int32)

        cosine = tf.matmul(

            tf.math.l2_normalize(X, axis=1),

            tf.math.l2_normalize(self.W, axis=0)

        )

        phi = cosine - self.m

        one_hot = tf.cast(

            tf.one_hot(y, depth=self.n_classes),

            dtype=cosine.dtype

        )

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        output *= self.s

        return output





def create_model_b7_1024(input_shape=config['input_size'],n_classes=config['n_classes'],scale=30,margin=0.0):



    backbone = efn.EfficientNetB7(weights=None, include_top=False, input_shape=input_shape)

    pooling = tf.keras.layers.GlobalAveragePooling2D(name='head/pooling')

    margin = AddMarginProduct(n_classes=n_classes,s=scale,m=margin,name='head/cos_margin',dtype='float32')



    dense = tf.keras.layers.Dense(1024, name='head/dense')

    bn_0 = tf.keras.layers.BatchNormalization(name='head/bn_0')

    softmax = tf.keras.layers.Softmax(dtype='float32')



    image = tf.keras.layers.Input(input_shape, name='input/image')

    label = tf.keras.layers.Input((), name='input/label')



    x = backbone(image)    

    x = pooling(x)

    x=dense(x)

    x=bn_0(x)

    x = margin([x, label])

    

    x = softmax(x)



    return tf.keras.Model(inputs=[image, label], outputs=x)





def create_model_b6_512(input_shape=config['input_size'],n_classes=config['n_classes'],scale=30,margin=0.0):

    backbone = efn.EfficientNetB6(weights=None, include_top=False, input_shape=input_shape)

    

    pooling = tf.keras.layers.GlobalAveragePooling2D(name='head/pooling')

    bn_0 = tf.keras.layers.BatchNormalization(name='head/bn_0')

    dense = tf.keras.layers.Dense(512, name='head/dense')

    bn_1 = tf.keras.layers.BatchNormalization(name='head/bn_1')

    margin = AddMarginProduct(n_classes=n_classes,s=scale,m=margin,name='head/cos_margin',dtype='float32')

    

    softmax = tf.keras.layers.Softmax(dtype='float32')



    image = tf.keras.layers.Input(input_shape, name='input/image')

    label = tf.keras.layers.Input((), name='input/label')



    x = backbone(image)

    x = pooling(x)

    x=bn_0(x)

    x= dense(x)

    x=bn_1(x)

    x = margin([x, label])

    

    x = softmax(x)

    return tf.keras.Model(inputs=[image, label], outputs=x)





def create_model_b7_512(input_shape=config['input_size'],n_classes=config['n_classes'],scale=30,margin=0.0):

    backbone = efn.EfficientNetB7(weights=None, include_top=False, input_shape=input_shape)

    

    pooling = tf.keras.layers.GlobalAveragePooling2D(name='head/pooling')

    bn_0 = tf.keras.layers.BatchNormalization(name='head/bn_0')

    dense = tf.keras.layers.Dense(512, name='head/dense')

    bn_1 = tf.keras.layers.BatchNormalization(name='head/bn_1')

    margin = AddMarginProduct(n_classes=n_classes,s=scale,m=margin,name='head/cos_margin',dtype='float32')

    

    softmax = tf.keras.layers.Softmax(dtype='float32')



    image = tf.keras.layers.Input(input_shape, name='input/image')

    label = tf.keras.layers.Input((), name='input/label')



    x = backbone(image)

    x = pooling(x)

    x=bn_0(x)

    x= dense(x)

    x=bn_1(x)

    x = margin([x, label])

    

    x = softmax(x)

    return tf.keras.Model(inputs=[image, label], outputs=x)



def create_model_b7_2560(input_shape=config['input_size'],n_classes=config['n_classes'],scale=30,margin=0.0):

    backbone =efn.EfficientNetB7(weights=None, include_top=False,input_shape=input_shape)

    pooling = tf.keras.layers.GlobalAveragePooling2D(name='head/pooling')

    margin = AddMarginProduct(n_classes=n_classes,s=scale,m=margin,name='head/cos_margin',dtype='float32')

    softmax = tf.keras.layers.Softmax(dtype='float32')



    image = tf.keras.layers.Input(input_shape, name='input/image')

    label = tf.keras.layers.Input((), name='input/label')



    x = backbone(image)

    x = pooling(x)

    x = margin([x, label])

    x = softmax(x)

    return tf.keras.Model(inputs=[image, label], outputs=x)
model_b7_1024 = create_model_b7_1024()

model_b7_1024.load_weights('../input/glr2-sub-model/Acc-0.994-efnb7-model.h5')

model_b7_1024.trainable=False

model_b7_1024 = tf.keras.Model(

    inputs=model_b7_1024.get_layer('input/image').input,

    outputs=model_b7_1024.get_layer('head/bn_0').output)



model_b7_1024.summary()



model_b6_512 = create_model_b6_512()

model_b6_512.load_weights('../input/glr2-sub-model/Acc-0.997-effb6-512-model.h5')

model_b6_512.trainable=False

model_b6_512 = tf.keras.Model(

    inputs=model_b6_512.get_layer('input/image').input,

    outputs=model_b6_512.get_layer('head/bn_1').output)



model_b6_512.summary()



model_b7_512 = create_model_b7_512()

model_b7_512.load_weights('../input/glr2-sub-model/Acc-0.985-efnb7-512-model.h5')

model_b7_512.trainable=False

model_b7_512 = tf.keras.Model(

    inputs=model_b7_512.get_layer('input/image').input,

    outputs=model_b7_512.get_layer('head/bn_1').output)



model_b7_512.summary()



model_b7_2560 = create_model_b7_2560()

model_b7_2560.load_weights('../input/glr2-sub-model/Acc-0.996-efnb7-2560-model.h5')

model_b7_2560.trainable=False

model_b7_2560 = tf.keras.Model(

    inputs=model_b7_2560.get_layer('input/image').input,

    outputs=model_b7_2560.get_layer('head/pooling').output)



model_b7_2560.summary()
def read_test_file():

    files_paths = glob.glob('../input/landmark-recognition-2020' + '/test/*/*/*/*')

    mapping = {}

    for path in files_paths:

        mapping[path.split('/')[-1].split('.')[0]] = path

    df = pd.read_csv('../input/landmark-recognition-2020/sample_submission.csv')

    df['path'] = df['id'].map(mapping)

    return df



df=read_test_file()

print(len(df))



def load_labelmap():

    with open('../input/landmark-recognition-2020/train.csv', mode='r') as csv_file:

        csv_reader = csv.DictReader(csv_file)

        labelmap = {row['id']: row['landmark_id'] for row in csv_reader}



    return labelmap



def read_train_file():

    labelmap=load_labelmap()

    

    files_paths = glob.glob('../input/landmark-recognition-2020' + '/train/*/*/*/*')



    mapping = {}

    for path in files_paths:

        image_id=path.split('/')[-1].split('.')[0]

        mapping[path] = image_id

    return mapping



df_train=read_train_file()

print(len(df_train))
AUTO = tf.data.experimental.AUTOTUNE



def read_image(img_id,image_path):

    image = tf.io.read_file(image_path)

    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.resize(image, config['input_size'][0:2], method='bilinear')

    image = tf.cast(image, tf.float32)

    image /= 255.

    return image,img_id



def create_dataset(df,batch_size):

    image_paths= df.path

    img_id=df.id

    dataset = tf.data.Dataset.from_tensor_slices((img_id,image_paths))

    dataset = dataset.map(read_image,tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(AUTO)

    return dataset



def create_train_dataset(df,batch_size):

    image_paths= list(df_train.keys())

    img_id=list(df_train.values())

    dataset = tf.data.Dataset.from_tensor_slices((img_id,image_paths))

    dataset = dataset.map(read_image,tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(AUTO)

    return dataset





# Dataset parameters:

INPUT_DIR = os.path.join('..', 'input')



DATASET_DIR = os.path.join(INPUT_DIR, 'landmark-recognition-2020')

TEST_IMAGE_DIR = os.path.join(DATASET_DIR, 'test')

TRAIN_IMAGE_DIR = os.path.join(DATASET_DIR, 'train')

TRAIN_LABELMAP_PATH = os.path.join(DATASET_DIR, 'train.csv')



# DEBUGGING PARAMS:

NUM_PUBLIC_TRAIN_IMAGES = 1580470 # Used to detect if in session or re-run.

MAX_NUM_EMBEDDINGS = -1  # Set to > 1 to subsample dataset while debugging.



# Retrieval & re-ranking parameters:

NUM_TO_RERANK = 3

TOP_K = 3 #Number of retrieved images used to make prediction for a test image.



# RANSAC parameters:

MAX_INLIER_SCORE = 35

MAX_REPROJECTION_ERROR = 7.0

MAX_RANSAC_ITERATIONS = 70000

HOMOGRAPHY_CONFIDENCE = 0.99
# #############################1_model

# @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None, 3],dtype=tf.float32,name='input_image')])

# def serving_train(input_image):

#     outputs0 = model_b7_1024(input_image)

#     outputs0 = tf.math.l2_normalize(outputs0,axis=-1)



#     return outputs0



# def get_train_ids_labels_and_scores():



#     train_dataset=create_train_dataset(df_train,16)



#     features_train=[]

#     ID_marks=[]

#     train_dataset = tqdm.tqdm(train_dataset)

#     for k, inputs in enumerate(train_dataset):



#         ID_marks+=list(inputs[1].numpy())

#         out=serving_train(inputs[0])

#         features_train.append(out)



#         #if k==20:break

#         del inputs

#         del out

#         gc.collect()

    

#     del train_dataset

#     gc.collect()

    

#     features_train=tf.concat(features_train,axis=0)

#     ID_marks=np.array(ID_marks)

    

#     W=tf.concat(features_train,axis=0)

#     W = tf.math.l2_normalize(W,axis=-1)

#     W=tf.transpose(W)

#     @tf.function(input_signature=[tf.TensorSpec(shape=[None,None, None, 3],dtype=tf.float32,name='input_image')])

#     def serving(input_image):

#         outputs = model_b7_1024(input_image)

#         outputs = tf.math.l2_normalize(outputs,axis=-1)

#         res=tf.matmul(outputs,W)

#         return res





#     num_k=NUM_TO_RERANK

#     test_dataset=create_dataset(df,16)

#     ID_test=list(df.id)



#     test_dataset = tqdm.tqdm(test_dataset)

#     train_ids_labels_and_scores=[]

#     labelmap=load_labelmap()

#     for k, inputs in enumerate(test_dataset):

#         out=serving(inputs[0]).numpy()

#         for j in range(inputs[1].shape[0]):

#             top_k = np.argpartition(out[j], -num_k)[-num_k:]

#             top_k_score=out[j][top_k]

#             top_k_id=[str(x)[2:-1] for x in ID_marks[top_k]]

#             top_k_label=[labelmap[x] for x in  top_k_id]           

#             train_ids_labels_and_scores.append(list(zip(top_k_id,top_k_label,top_k_score)))

#         #if k ==3: break



#         del inputs

#         del out

#         gc.collect()

#     del W

#     del test_dataset

#     gc.collect()

#     return ID_test[0:len(train_ids_labels_and_scores)],train_ids_labels_and_scores#
# #############################2_model

# @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None, 3],dtype=tf.float32,name='input_image')])

# def serving_train(input_image):

#     outputs0 = model_b7_1024(input_image)

#     outputs0 = tf.math.l2_normalize(outputs0,axis=-1)

    

#     outputs1 = model_b7_512(input_image)

#     outputs1 = tf.math.l2_normalize(outputs1,axis=-1)

#     return outputs0,outputs1



# def get_train_ids_labels_and_scores():



#     train_dataset=create_train_dataset(df_train,16)



#     features_b7_1024=[]

#     features_b7_512=[]

#     ID_marks=[]

#     train_dataset = tqdm.tqdm(train_dataset)

#     for k, inputs in enumerate(train_dataset):



#         ID_marks+=list(inputs[1].numpy())

#         out=serving_train(inputs[0])

#         features_b7_1024.append(out[0])

#         features_b7_512.append(out[1])

        

#         #if k==100:break



#         del inputs

#         del out

#         gc.collect()

    

#     del train_dataset

#     gc.collect()

    

#     features_b7_1024=tf.concat(features_b7_1024,axis=0)

#     features_b7_512=tf.concat(features_b7_512,axis=0)

#     ID_marks=np.array(ID_marks)

    

#     W_b7_1024=tf.concat(features_b7_1024,axis=0)

#     W_b7_1024=tf.math.l2_normalize(W_b7_1024,axis=-1)

#     W_b7_1024=tf.transpose(W_b7_1024)

#     print(W_b7_1024.shape)

    

#     W_b7_512=tf.concat(features_b7_512,axis=0)

#     W_b7_512=tf.math.l2_normalize(W_b7_512,axis=-1)

#     W_b7_512=tf.transpose(W_b7_512)

#     print(W_b7_512.shape)

    

#     del features_b7_1024

#     del features_b7_512

#     gc.collect()

    

#     @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None, 3],dtype=tf.float32,name='input_image')])

#     def serving(input_image):

#         outputs0 = model_b7_1024(input_image)

#         outputs0 = tf.math.l2_normalize(outputs0,axis=-1)

#         res0=tf.matmul(outputs0,W_b7_1024)

        

#         outputs1 = model_b7_512(input_image)

#         outputs1 = tf.math.l2_normalize(outputs1,axis=-1)

#         res1=tf.matmul(outputs1,W_b7_512)

#         return res0,res1



#     num_k=NUM_TO_RERANK

#     labelmap=load_labelmap()

    

    

#     test_dataset=create_dataset(df,16)

#     ID_test=list(df.id)



#     test_dataset = tqdm.tqdm(test_dataset)

#     train_ids_labels_and_scores=[]

    

#     for k, inputs in enumerate(test_dataset):

#         out=serving(inputs[0])

#         out=(out[0].numpy(),out[1].numpy())

#         for j in range(inputs[1].shape[0]): 

# #             top_k=list(np.argpartition(out[0][j], -num_k)[-num_k:])+list(np.argpartition(out[1][j], -num_k)[-num_k:])

# #             top_k=list(set(top_k))            

# #             top_k_score=np.max(np.stack([out[0][j][top_k],out[1][j][top_k]]),axis=0)



#             top_k0=list(np.argpartition(out[0][j], -num_k)[-num_k:])

#             top_k1=list(np.argpartition(out[1][j], -num_k)[-num_k:])

#             top_k=top_k0+top_k1 

            

#             top_k_score=list(out[0][j][top_k0])+list(out[1][j][top_k1])            

#             top_k_id=[str(x)[2:-1] for x in ID_marks[top_k]]

#             top_k_label=[labelmap[x] for x in  top_k_id]

            

# #             aggregate_scores = {}

# #             for i in range(len(top_k_id)):

# #                 if top_k_id[i] not in aggregate_scores:

# #                     aggregate_scores[top_k_id[i]] = 0

# #                 aggregate_scores[top_k_id[i]] += top_k_score[i]

            

# #             top_k_id=list(aggregate_scores.keys())

# #             top_k_label=[labelmap[x] for x in  top_k_id]

# #             top_k_score=list(aggregate_scores.values())

            

#             res_out=list(zip(top_k_id,top_k_label,top_k_score))

#             train_ids_labels_and_scores.append(res_out)

#         #if k ==50: break



#         del inputs

#         del out

#         gc.collect()

#     del W_b7_1024

#     del W_b7_512

#     del test_dataset

#     gc.collect()

#     return ID_test[0:len(train_ids_labels_and_scores)],train_ids_labels_and_scores#
# #############################2_merge_model

# @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None, 3],dtype=tf.float32,name='input_image')])

# def serving_train(input_image):

#     outputs0 = model_b7_1024(input_image)

#     outputs0 = tf.math.l2_normalize(outputs0,axis=-1)

    

#     outputs1 = model_b7_512(input_image)

#     outputs1 = tf.math.l2_normalize(outputs1,axis=-1)

#     return outputs0,outputs1



# def get_train_ids_labels_and_scores():



#     train_dataset=create_train_dataset(df_train,16)



#     features_b7_1024=[]

#     features_b7_512=[]

#     ID_marks=[]

#     train_dataset = tqdm.tqdm(train_dataset)

#     for k, inputs in enumerate(train_dataset):



#         ID_marks+=list(inputs[1].numpy())

#         out=serving_train(inputs[0])

#         features_b7_1024.append(out[0])

#         features_b7_512.append(out[1])

        

#         #if k==100:break



#         del inputs

#         del out

#         gc.collect()

    

#     del train_dataset

#     gc.collect()

    

#     features_b7_1024=tf.concat(features_b7_1024,axis=0)

#     features_b7_512=tf.concat(features_b7_512,axis=0)

#     ID_marks=np.array(ID_marks)

    

#     W_b7_1024=tf.concat(features_b7_1024,axis=0)

#     W_b7_1024=tf.transpose(W_b7_1024)

#     print(W_b7_1024.shape)

    

#     W_b7_512=tf.concat(features_b7_512,axis=0)

#     W_b7_512=tf.transpose(W_b7_512)

#     print(W_b7_512.shape)

    

#     del features_b7_1024

#     del features_b7_512

#     gc.collect()

    

#     @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None, 3],dtype=tf.float32,name='input_image')])

#     def serving(input_image):

#         outputs0 = model_b7_1024(input_image)

#         outputs0 = tf.math.l2_normalize(outputs0,axis=-1)

#         res0=tf.matmul(outputs0,W_b7_1024)

        

#         outputs1 = model_b7_512(input_image)

#         outputs1 = tf.math.l2_normalize(outputs1,axis=-1)

#         res1=tf.matmul(outputs1,W_b7_512)

#         return res0,res1



#     num_k=NUM_TO_RERANK

#     labelmap=load_labelmap()

    

    

#     test_dataset=create_dataset(df,16)

#     ID_test=list(df.id)



#     test_dataset = tqdm.tqdm(test_dataset)

#     train_ids_labels_and_scores=[]

    

#     for k, inputs in enumerate(test_dataset):

#         out=serving(inputs[0])

#         out=(out[0].numpy(),out[1].numpy())

#         for j in range(inputs[1].shape[0]): 

# #             top_k=list(np.argpartition(out[0][j], -num_k)[-num_k:])+list(np.argpartition(out[1][j], -num_k)[-num_k:])

# #             top_k=list(set(top_k))            

# #             top_k_score=np.max(np.stack([out[0][j][top_k],out[1][j][top_k]]),axis=0)



#             top_k0=list(np.argpartition(out[0][j], -num_k)[-num_k:])

#             top_k1=list(np.argpartition(out[1][j], -num_k)[-num_k:])

#             top_k=top_k0+top_k1 

            

#             top_k_score=list(out[0][j][top_k0])+list(out[1][j][top_k1])            

#             top_k_id=[str(x)[2:-1] for x in ID_marks[top_k]]

#             top_k_label=[labelmap[x] for x in  top_k_id]

            

# #             aggregate_scores = {}

# #             for i in range(len(top_k_id)):

# #                 if top_k_id[i] not in aggregate_scores:

# #                     aggregate_scores[top_k_id[i]] = 0

# #                 aggregate_scores[top_k_id[i]] += top_k_score[i]

            

# #             top_k_id=list(aggregate_scores.keys())

# #             top_k_label=[labelmap[x] for x in  top_k_id]

# #             top_k_score=list(aggregate_scores.values())

            

#             res_out=list(zip(top_k_id,top_k_label,top_k_score))

#             train_ids_labels_and_scores.append(res_out)

#         #if k ==50: break



#         del inputs

#         del out

#         gc.collect()

#     del W_b7_1024

#     del W_b7_512

#     del test_dataset

#     gc.collect()

#     return ID_test[0:len(train_ids_labels_and_scores)],train_ids_labels_and_scores#
# #############################3_model

# @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None, 3],dtype=tf.float32,name='input_image')])

# def serving_train(input_image):

#     outputs0 = model_b7_1024(input_image)

#     outputs0 = tf.math.l2_normalize(outputs0,axis=-1)

    

#     outputs1 = model_b6_512(input_image)

#     outputs1 = tf.math.l2_normalize(outputs1,axis=-1)

    

#     outputs2 = model_b7_512(input_image)

#     outputs2 = tf.math.l2_normalize(outputs2,axis=-1)

#     return outputs0,outputs1,outputs2



# def get_train_ids_labels_and_scores():



#     train_dataset=create_train_dataset(df_train,16)



#     features_b7_1024=[]

#     features_b6_512=[]

#     features_b7_512=[]

        

#     ID_marks=[]

#     train_dataset = tqdm.tqdm(train_dataset)

#     for k, inputs in enumerate(train_dataset):



#         ID_marks+=list(inputs[1].numpy())

#         out=serving_train(inputs[0])

#         features_b7_1024.append(out[0])

#         features_b6_512.append(out[1])

#         features_b7_512.append(out[2])

        

#         #if k==100:break



#         del inputs

#         del out

#         gc.collect()

    

#     del train_dataset

#     gc.collect()

    

#     features_b7_1024=tf.concat(features_b7_1024,axis=0)

#     features_b6_512=tf.concat(features_b6_512,axis=0)

#     features_b7_512=tf.concat(features_b7_512,axis=0)

    

#     ID_marks=np.array(ID_marks)

    

#     W_b7_1024=tf.concat(features_b7_1024,axis=0)

#     W_b7_1024=tf.math.l2_normalize(W_b7_1024,axis=-1)

#     W_b7_1024=tf.transpose(W_b7_1024)

#     print(W_b7_1024.shape)

    

#     W_b6_512=tf.concat(features_b6_512,axis=0)

#     W_b6_512=tf.math.l2_normalize(W_b6_512,axis=-1)

#     W_b6_512=tf.transpose(W_b6_512)

#     print(W_b6_512.shape)

    

#     W_b7_512=tf.concat(features_b7_512,axis=0)

#     W_b7_512=tf.math.l2_normalize(W_b7_512,axis=-1)

#     W_b7_512=tf.transpose(W_b7_512)

#     print(W_b7_512.shape)

    

#     del features_b7_1024

#     del features_b6_512

#     del features_b7_512

#     gc.collect()

    

#     @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None, 3],dtype=tf.float32,name='input_image')])

#     def serving(input_image):

#         outputs0 = model_b7_1024(input_image)

#         outputs0 = tf.math.l2_normalize(outputs0,axis=-1)

#         res0=tf.matmul(outputs0,W_b7_1024)

        

#         outputs1 = model_b6_512(input_image)

#         outputs1 = tf.math.l2_normalize(outputs1,axis=-1)

#         res1=tf.matmul(outputs1,W_b6_512)

        

#         outputs2 = model_b7_512(input_image)

#         outputs2 = tf.math.l2_normalize(outputs2,axis=-1)

#         res2=tf.matmul(outputs2,W_b7_512)

#         return res0,res1,res2



#     num_k=NUM_TO_RERANK

#     labelmap=load_labelmap()

    

    

#     test_dataset=create_dataset(df,16)

#     ID_test=list(df.id)



#     test_dataset = tqdm.tqdm(test_dataset)

#     train_ids_labels_and_scores=[]

    

#     for k, inputs in enumerate(test_dataset):

#         out=serving(inputs[0])

#         out=(out[0].numpy(),out[1].numpy(),out[2].numpy())

#         for j in range(inputs[1].shape[0]): 



#             top_k0=list(np.argpartition(out[0][j], -num_k)[-num_k:])

#             top_k1=list(np.argpartition(out[1][j], -num_k)[-num_k:])

#             top_k2=list(np.argpartition(out[2][j], -num_k)[-num_k:])

#             top_k=top_k0+top_k1+top_k2 

            

#             top_k_score=list(out[0][j][top_k0])+list(0.8*out[1][j][top_k1])+list(out[2][j][top_k2])            

#             top_k_id=[str(x)[2:-1] for x in ID_marks[top_k]]

            

# #             aggregate_scores = {}

# #             for i in range(len(top_k_id)):

# #                 if top_k_id[i] not in aggregate_scores:

# #                     aggregate_scores[top_k_id[i]] = 0

# #                 aggregate_scores[top_k_id[i]] += top_k_score[i]

            

# #             top_k_id=list(aggregate_scores.keys())            

# #             top_k_score=list(aggregate_scores.values())

            

#             top_k_label=[labelmap[x] for x in  top_k_id]

#             res_out=list(zip(top_k_id,top_k_label,top_k_score))

#             train_ids_labels_and_scores.append(res_out)

#         #if k ==100: break



#         del inputs

#         del out

#         gc.collect()

        

#     del W_b7_1024

#     del W_b6_512

#     del W_b7_512

#     del test_dataset

#     gc.collect()

#     return ID_test[0:len(train_ids_labels_and_scores)],train_ids_labels_and_scores#
# #############################3_merge_model

# @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None, 3],dtype=tf.float32,name='input_image')])

# def serving_train(input_image):

#     outputs0 = model_b7_1024(input_image)

#     outputs0 = tf.math.l2_normalize(outputs0,axis=-1)

    

#     outputs1 = model_b6_512(input_image)

#     outputs1 = tf.math.l2_normalize(outputs1,axis=-1)

    

#     outputs2 = model_b7_512(input_image)

#     outputs2 = tf.math.l2_normalize(outputs2,axis=-1)

    

#     outputs=tf.concat([outputs0,outputs1,outputs2],axis=-1)

#     return outputs



# def get_train_ids_labels_and_scores():



#     train_dataset=create_train_dataset(df_train,16)

#     features_2048=[]        

#     ID_marks=[]

#     train_dataset = tqdm.tqdm(train_dataset)

#     for k, inputs in enumerate(train_dataset):



#         ID_marks+=list(inputs[1].numpy())

#         out=serving_train(inputs[0])

#         features_2048.append(out)

        

#         if k==5:break



#         del inputs

#         del out

#         gc.collect()

    

#     del train_dataset

#     gc.collect()

    

#     features_2048=tf.concat(features_2048,axis=0)    

#     ID_marks=np.array(ID_marks)

    

#     features_2048=tf.concat(features_2048,axis=0)

#     features_2048=tf.transpose(features_2048)

#     print(features_2048.shape)

    

#     @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None, 3],dtype=tf.float32,name='input_image')])

#     def serving(input_image):

#         outputs0 = model_b7_1024(input_image)

#         outputs0 = tf.math.l2_normalize(outputs0,axis=-1)

        

#         outputs1 = model_b6_512(input_image)

#         outputs1 = 0.7*tf.math.l2_normalize(outputs1,axis=-1)

        

#         outputs2 = model_b7_512(input_image)

#         outputs2 = tf.math.l2_normalize(outputs2,axis=-1)

        

#         outputs=tf.concat([outputs0,outputs1,outputs2],axis=-1)

#         outputs=tf.matmul(outputs,features_2048)

#         return outputs



#     num_k=NUM_TO_RERANK

#     labelmap=load_labelmap()

    

    

#     test_dataset=create_dataset(df,16)

#     ID_test=list(df.id)



#     test_dataset = tqdm.tqdm(test_dataset)

#     train_ids_labels_and_scores=[]

    

#     for k, inputs in enumerate(test_dataset):

#         out=serving(inputs[0]).numpy()

#         for j in range(inputs[1].shape[0]):

#             top_k = np.argpartition(out[j], -num_k)[-num_k:]

#             top_k_score=out[j][top_k]/3.0

#             top_k_id=[str(x)[2:-1] for x in ID_marks[top_k]]

#             top_k_label=[labelmap[x] for x in  top_k_id]           

#             train_ids_labels_and_scores.append(list(zip(top_k_id,top_k_label,top_k_score)))

#         if k ==3: break



#         del inputs

#         del out

#         gc.collect()

        

#     del features_2048

#     del test_dataset

#     gc.collect()

#     return ID_test[0:len(train_ids_labels_and_scores)],train_ids_labels_and_scores#
#############################3_merge_model

@tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None, 3],dtype=tf.float32,name='input_image')])

def serving_train(input_image):

    outputs0 = model_b7_1024(input_image)

    outputs0 = tf.math.l2_normalize(outputs0,axis=-1)

    

    outputs1 = model_b6_512(input_image)

    outputs1 = tf.math.l2_normalize(outputs1,axis=-1)

    

    outputs2 = model_b7_512(input_image)

    outputs2 = tf.math.l2_normalize(outputs2,axis=-1)

    

    outputs3 = model_b7_2560(input_image)

    outputs3 = tf.math.l2_normalize(outputs3,axis=-1)

    

    outputs=tf.concat([outputs0,outputs1,outputs2,outputs3],axis=-1)

    return outputs



def get_train_ids_labels_and_scores():



    train_dataset=create_train_dataset(df_train,8)

    features_2048=[]        

    ID_marks=[]

    train_dataset = tqdm.tqdm(train_dataset)

    for k, inputs in enumerate(train_dataset):



        ID_marks+=list(inputs[1].numpy())

        out=serving_train(inputs[0])

        features_2048.append(out)

        

        #if k==20:break



        del inputs

        del out

        gc.collect()

    

    del train_dataset

    gc.collect()

    

    features_2048=tf.concat(features_2048,axis=0)    

    ID_marks=np.array(ID_marks)

    

    features_2048=tf.concat(features_2048,axis=0)

    features_2048=tf.transpose(features_2048)

    print(features_2048.shape)

    

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None, 3],dtype=tf.float32,name='input_image')])

    def serving(input_image):

        outputs0 = model_b7_1024(input_image)

        outputs0 = tf.math.l2_normalize(outputs0,axis=-1)

        

        outputs1 = model_b6_512(input_image)

        outputs1 = 0.7*tf.math.l2_normalize(outputs1,axis=-1)

        

        outputs2 = model_b7_512(input_image)

        outputs2 = tf.math.l2_normalize(outputs2,axis=-1)

        

        outputs3 = model_b7_2560(input_image)

        outputs3 = tf.math.l2_normalize(outputs3,axis=-1)

        

        outputs=tf.concat([outputs0,outputs1,outputs2,outputs3],axis=-1)

        outputs=tf.matmul(outputs,features_2048)

        return outputs



    num_k=NUM_TO_RERANK

    labelmap=load_labelmap()

    

    

    test_dataset=create_dataset(df,8)

    ID_test=list(df.id)



    test_dataset = tqdm.tqdm(test_dataset)

    train_ids_labels_and_scores=[]

    

    for k, inputs in enumerate(test_dataset):

        out=serving(inputs[0]).numpy()

        for j in range(inputs[1].shape[0]):

            top_k = np.argpartition(out[j], -num_k)[-num_k:]

            top_k_score=out[j][top_k]/4.0

            top_k_id=[str(x)[2:-1] for x in ID_marks[top_k]]

            top_k_label=[labelmap[x] for x in  top_k_id]           

            train_ids_labels_and_scores.append(list(zip(top_k_id,top_k_label,top_k_score)))

        #if k ==20: break



        del inputs

        del out

        gc.collect()

        

    del features_2048

    del test_dataset

    gc.collect()

    return ID_test[0:len(train_ids_labels_and_scores)],train_ids_labels_and_scores#
# DELG model:

SAVED_MODEL_DIR = '../input/delg-saved-models/local_and_global'

DELG_MODEL = tf.saved_model.load(SAVED_MODEL_DIR)

DELG_IMAGE_SCALES_TENSOR = tf.convert_to_tensor([0.70710677, 1.0, 1.4142135])

DELG_SCORE_THRESHOLD_TENSOR = tf.constant(175.)

DELG_INPUT_TENSOR_NAMES = [

    'input_image:0', 'input_scales:0', 'input_abs_thres:0'

]



# Global feature extraction:

GLOBAL_FEATURE_EXTRACTION_FN = DELG_MODEL.prune(DELG_INPUT_TENSOR_NAMES,['global_descriptors:0'])



# Local feature extraction:

LOCAL_FEATURE_NUM_TENSOR = tf.constant(1000)

LOCAL_FEATURE_EXTRACTION_FN = DELG_MODEL.prune(

    DELG_INPUT_TENSOR_NAMES + ['input_max_feature_num:0'],

    ['boxes:0', 'features:0'])
def get_image_path(subset, image_id):

    name =image_id

    return os.path.join(DATASET_DIR, subset, name[0], name[1], name[2],'{}.jpg'.format(name))





def load_image_tensor(image_path):

    return tf.convert_to_tensor(

        np.array(PIL.Image.open(image_path).convert('RGB')))





def extract_local_features(image_path):

    """Extracts local features for the given `image_path`."""

    image_tensor = load_image_tensor(image_path)    

    #print(image_tensor.shape)

    features = LOCAL_FEATURE_EXTRACTION_FN(image_tensor, DELG_IMAGE_SCALES_TENSOR,

                                           DELG_SCORE_THRESHOLD_TENSOR,

                                           LOCAL_FEATURE_NUM_TENSOR)

    #print(features[0].shape, features[1].shape)

    # Shape: (N, 2)

    keypoints = tf.divide(

        tf.add(

            tf.gather(features[0], [0, 1], axis=1),

            tf.gather(features[0], [2, 3], axis=1)), 2.0).numpy()



    # Shape: (N, 128)

    descriptors = tf.nn.l2_normalize(

        features[1], axis=1, name='l2_normalization').numpy()

    

    #print(keypoints.shape, descriptors.shape)

    return keypoints, descriptors





def get_putative_matching_keypoints(test_keypoints,

                                    test_descriptors,

                                    train_keypoints,

                                    train_descriptors,

                                    max_distance=0.99):

    """Finds matches from `test_descriptors` to KD-tree of `train_descriptors`."""

    

    train_descriptor_tree = spatial.cKDTree(train_descriptors)

    _, matches = train_descriptor_tree.query(

        test_descriptors, distance_upper_bound=max_distance)



    test_kp_count = test_keypoints.shape[0]

    train_kp_count = train_keypoints.shape[0]



    test_matching_keypoints = np.array([

        test_keypoints[i,]

        for i in range(test_kp_count)

        if matches[i] != train_kp_count

    ])

    train_matching_keypoints = np.array([

        train_keypoints[matches[i],]

        for i in range(test_kp_count)

        if matches[i] != train_kp_count

    ])

    

    return test_matching_keypoints, train_matching_keypoints





def get_num_inliers(test_keypoints, test_descriptors, train_keypoints,

                    train_descriptors):

    """Returns the number of RANSAC inliers."""



    test_match_kp, train_match_kp = get_putative_matching_keypoints(

        test_keypoints, test_descriptors, train_keypoints, train_descriptors)



    if test_match_kp.shape[0] <= 4:  # Min keypoints supported by `pydegensac.findHomography()`

        return 0



    try:

        _, mask = pydegensac.findHomography(test_match_kp, train_match_kp,

                                            MAX_REPROJECTION_ERROR,

                                            HOMOGRAPHY_CONFIDENCE,

                                            MAX_RANSAC_ITERATIONS)

    except np.linalg.LinAlgError:  # When det(H)=0, can't invert matrix.

        return 0



    return int(copy.deepcopy(mask).astype(np.float32).sum())





def get_total_score(num_inliers, global_score):

    local_score = min(num_inliers, MAX_INLIER_SCORE) / MAX_INLIER_SCORE

    return local_score + 0.8*global_score





def rescore_and_rerank_by_num_inliers(test_image_id,

                                      train_ids_labels_and_scores):

    """Returns rescored and sorted training images by local feature extraction."""



    test_image_path = get_image_path('test', test_image_id)

    test_keypoints, test_descriptors = extract_local_features(test_image_path)



    for i in range(len(train_ids_labels_and_scores)):

        train_image_id, label, global_score = train_ids_labels_and_scores[i]



        train_image_path = get_image_path('train', train_image_id)

        train_keypoints, train_descriptors = extract_local_features(

            train_image_path)



        num_inliers = get_num_inliers(test_keypoints, test_descriptors,

                                      train_keypoints, train_descriptors)

        total_score = get_total_score(num_inliers, global_score)

        train_ids_labels_and_scores[i] = (train_image_id, label, total_score)



    train_ids_labels_and_scores.sort(key=lambda x: x[2], reverse=True)



    return train_ids_labels_and_scores





def get_prediction_map(test_ids, train_ids_labels_and_scores):

    """Makes dict from test ids and ranked training ids, labels, scores."""



    prediction_map = dict()



    for test_index, test_id in enumerate(test_ids):

        hex_test_id = test_id



        aggregate_scores = {}

        for _, label, score in train_ids_labels_and_scores[test_index][:TOP_K]:

            if label not in aggregate_scores:

                aggregate_scores[label] = 0

            aggregate_scores[label] += score



        label, score = max(aggregate_scores.items(), key=operator.itemgetter(1))



        prediction_map[hex_test_id] = {'score': score, 'class': label}



    return prediction_map





def get_predictions(labelmap):



    

    test_ids, train_ids_labels_and_scores=get_train_ids_labels_and_scores()

    pre_verification_predictions = get_prediction_map(test_ids, train_ids_labels_and_scores)



    #return None, pre_verification_predictions########################################

    #print(test_ids,train_ids_labels_and_scores)

    

    test_ids = tqdm.tqdm(test_ids)

    for test_index, test_id in enumerate(test_ids):

        train_ids_labels_and_scores[test_index] = rescore_and_rerank_by_num_inliers(

            test_id, train_ids_labels_and_scores[test_index])

        

    #print(test_ids,train_ids_labels_and_scores)

    post_verification_predictions = get_prediction_map(

        test_ids, train_ids_labels_and_scores)



    return pre_verification_predictions, post_verification_predictions



def save_submission_csv(predictions=None):

    

    if predictions is None:

        # Dummy submission!

        shutil.copyfile(

            os.path.join(DATASET_DIR, 'sample_submission.csv'), 'submission.csv')

        return



    with open('submission.csv', 'w') as submission_csv:

        csv_writer = csv.DictWriter(submission_csv, fieldnames=['id', 'landmarks'])

        csv_writer.writeheader()

        for image_id, prediction in predictions.items():

            label = prediction['class']

            score = prediction['score']

            if score>0.3:

                csv_writer.writerow({'id': image_id, 'landmarks': f'{label} {score}'})

            else:

                csv_writer.writerow({'id': image_id, 'landmarks': ''})



def main():

    pass



    labelmap = load_labelmap()

    num_training_images = len(labelmap.keys())

    print(f'Found {num_training_images} training images.')



    if num_training_images == NUM_PUBLIC_TRAIN_IMAGES:

        print(

            f'Found {NUM_PUBLIC_TRAIN_IMAGES} training images. Copying sample submission.'

        )

        save_submission_csv()

        return

    

    _, post_verification_predictions = get_predictions(labelmap)

    save_submission_csv(post_verification_predictions)







if __name__ == '__main__':

    main()