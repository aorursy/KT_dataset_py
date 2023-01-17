import numpy as np

import tensorflow as tf

print(tf.__version__)

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

!pip install -q efficientnet

import efficientnet.tfkeras as efn

warnings.filterwarnings("ignore")





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

    'margin': 0.05,

    'clip_grad': 20.0,

    'n_epochs': 1,

    'input_size': (416,416, 3),

    'n_classes': 81313,

}



BATCH_SIZE=32*strategy_num

AUTO = tf.data.experimental.AUTOTUNE
def decode_image(image_data,image_size=config['input_size'][:2]):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_crop(image,config['input_size'])

    #image=tf.image.resize(image,image_size)

    #image=tf.image.random_jpeg_quality(image,70,100)



    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range

    image = tf.reshape(image, [*image_size, 3]) # explicit size needed for TPU

    return image



def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "label": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    label = example['label']

    return image, label



    

def load_dataset(filenames, ordered=False):  



    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files

    #dataset = dataset.apply(tf.data.experimental.ignore_errors())#########################   bug

        

    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False # disable order, increase speed

    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order

    

    dataset = dataset.map(read_labeled_tfrecord)

    return dataset



def get_training_dataset(files,shuffle):

    dataset = load_dataset(files)

    #dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.shuffle(shuffle)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset





def count_data_items(filenames):

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)



TRAINING_FILENAMES_512=['gs://kds-a55b790383607d61af836bb0eb5d7205752c74256860b08fab6a528d/train00-56500.tfrec', 'gs://kds-a55b790383607d61af836bb0eb5d7205752c74256860b08fab6a528d/train01-56500.tfrec', 'gs://kds-f6a96f7c31dcc7eb248fbd3f1161837fa780a06dc5f7195a5065877b/train02-56500.tfrec', 'gs://kds-f6a96f7c31dcc7eb248fbd3f1161837fa780a06dc5f7195a5065877b/train03-56500.tfrec', 'gs://kds-b93d389ed7a00352828607f39caf5e44a55c2eab43ecc5401b6cc876/train04-56500.tfrec', 'gs://kds-b93d389ed7a00352828607f39caf5e44a55c2eab43ecc5401b6cc876/train05-56500.tfrec', 'gs://kds-c3a597438822868f8b824c2128fbce3bd67c357a220b1810a8ee5502/train06-56500.tfrec', 'gs://kds-c3a597438822868f8b824c2128fbce3bd67c357a220b1810a8ee5502/train07-56500.tfrec', 'gs://kds-067e741475cb90c3e685309c566f8c7e44ae4f023598f222272ac58d/train08-56500.tfrec', 'gs://kds-067e741475cb90c3e685309c566f8c7e44ae4f023598f222272ac58d/train09-56500.tfrec', 'gs://kds-c8af124cca0baadc4f64b22bc2480e47b12613523cb08efc51238fd4/train10-56500.tfrec', 'gs://kds-c8af124cca0baadc4f64b22bc2480e47b12613523cb08efc51238fd4/train11-56500.tfrec', 'gs://kds-5dc5de7098b3289785c6cd3c95e438c1da0a3e8f41a03b3972cc9525/train12-56500.tfrec', 'gs://kds-5dc5de7098b3289785c6cd3c95e438c1da0a3e8f41a03b3972cc9525/train13-56500.tfrec', 'gs://kds-d87e904cbf43dcf45df72f4cf0d478d4eab7fd4c63410c08b8eda1bf/train14-56500.tfrec', 'gs://kds-d87e904cbf43dcf45df72f4cf0d478d4eab7fd4c63410c08b8eda1bf/train15-56500.tfrec', 'gs://kds-100b1dddf193cbcb6362d453e44c280d39eafe42c45f3fcfda65786c/train16-56500.tfrec', 'gs://kds-100b1dddf193cbcb6362d453e44c280d39eafe42c45f3fcfda65786c/train17-56500.tfrec', 'gs://kds-1fa19b8a29c797e5250f2e0def79657336da2aedd1805d6128c9c06e/train18-56500.tfrec', 'gs://kds-1fa19b8a29c797e5250f2e0def79657336da2aedd1805d6128c9c06e/train19-56500.tfrec', 'gs://kds-55dd20e507a75022952f96c49d21ebf87c86594f47bc117dab4621b4/train20-56500.tfrec', 'gs://kds-55dd20e507a75022952f96c49d21ebf87c86594f47bc117dab4621b4/train21-56500.tfrec', 'gs://kds-b1f2931bc191c08451479b71cad082549d79887cd52d215bf8b02ea4/train22-56500.tfrec', 'gs://kds-b1f2931bc191c08451479b71cad082549d79887cd52d215bf8b02ea4/train23-56500.tfrec', 'gs://kds-bf37932df5659c2c5eb226cbccb5a5551a0fccf47acd7b8d8a5eea5a/train24-56500.tfrec', 'gs://kds-bf37932df5659c2c5eb226cbccb5a5551a0fccf47acd7b8d8a5eea5a/train25-56500.tfrec', 'gs://kds-a29b6d24170583b676d51f85c6edde39da7f70bd53a78a2a250ef0ad/train26-56500.tfrec', 'gs://kds-a29b6d24170583b676d51f85c6edde39da7f70bd53a78a2a250ef0ad/train27-54970.tfrec']

TRAINING_FILENAMES_512_order=['gs://kds-e31c59692d1ebfb7eb905ba529436c30610e298e77acdd84166fde16/train00-56500.tfrec', 'gs://kds-e31c59692d1ebfb7eb905ba529436c30610e298e77acdd84166fde16/train01-56500.tfrec', 'gs://kds-5bd8dad7fcf39157ff0d8d968383eaa6f2217fce4cae649efd0aa2fa/train02-56500.tfrec', 'gs://kds-5bd8dad7fcf39157ff0d8d968383eaa6f2217fce4cae649efd0aa2fa/train03-56500.tfrec', 'gs://kds-b0df18831a9081c9585ddbacde1a1b189b22b760cbddc4f5b0f327bd/train04-56500.tfrec', 'gs://kds-b0df18831a9081c9585ddbacde1a1b189b22b760cbddc4f5b0f327bd/train05-56500.tfrec', 'gs://kds-2895efa34adf21ad51403dda50a87d2ded318924e4d283c7f9a35712/train06-56500.tfrec', 'gs://kds-2895efa34adf21ad51403dda50a87d2ded318924e4d283c7f9a35712/train07-56500.tfrec', 'gs://kds-3a927960544646c67c79d906c481af275c85cf19f06bdf34e2cb9f5c/train08-56500.tfrec', 'gs://kds-3a927960544646c67c79d906c481af275c85cf19f06bdf34e2cb9f5c/train09-56500.tfrec', 'gs://kds-e580f172ea7b5404545b57dea8057d3851d7faabcd74a0d49eff747a/train10-56500.tfrec', 'gs://kds-e580f172ea7b5404545b57dea8057d3851d7faabcd74a0d49eff747a/train11-56500.tfrec']

# train_num=count_data_items(TRAINING_FILENAMES_order)

# steps=train_num//BATCH_SIZE

# print('There are %i train images'%train_num,f'need Steps:{steps}/epoch')

# random.shuffle(TRAINING_FILENAMES_order)

# train_dataset = get_training_dataset(TRAINING_FILENAMES_order,10000)



# dist_train_ds = tqdm.tqdm(train_dataset)

# for i, inputs in enumerate(dist_train_ds):

#     print(i,len(inputs),inputs[0].shape,inputs[1].shape)

    

    

# plt.figure(figsize=(20, 12))

# for db in train_dataset:

#     print(db[0].shape,db[1][0:6])

#     for step,img in enumerate(db[0]):

#         plt.subplot(2,3,step+1)

#         plt.axis('off')

#         plt.imshow(img.numpy())

#         if step==5:break

#     break
# class AddMarginProduct(tf.keras.layers.Layer):

#     '''

#     Implements large margin cosine distance.



#     References:

#         https://arxiv.org/pdf/1801.07698.pdf

#         https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/

#             blob/master/src/modeling/metric_learning.py

#     '''

#     def __init__(self, n_classes, s=30, m=0.30, **kwargs):

#         super(AddMarginProduct, self).__init__(**kwargs)

#         self.n_classes = n_classes

#         self.s = s

#         self.m = m



#     def build(self, input_shape):

#         super(AddMarginProduct, self).build(input_shape[0])



#         self.W = self.add_weight(

#             name='W',

#             shape=(int(input_shape[0][-1]), self.n_classes),

#             initializer='glorot_uniform',

#             dtype='float32',

#             trainable=True,

#             regularizer=None)



#     def call(self, inputs):

#         X, y = inputs

#         y = tf.cast(y, dtype=tf.int32)

#         cosine = tf.matmul(

#             tf.math.l2_normalize(X, axis=1),

#             tf.math.l2_normalize(self.W, axis=0)

#         )

#         phi = cosine - self.m

#         one_hot = tf.cast(

#             tf.one_hot(y, depth=self.n_classes),

#             dtype=cosine.dtype

#         )

#         output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

#         output *= self.s

#         return output



class AddMarginProduct(tf.keras.layers.Layer):

    '''

    Implements large margin arc distance.



    Reference:

        https://arxiv.org/pdf/1801.07698.pdf

        https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/

            blob/master/src/modeling/metric_learning.py

    '''

    def __init__(self, n_classes, s=30, m=0.50, easy_margin=False,

                 ls_eps=0.0, **kwargs):



        super(AddMarginProduct, self).__init__(**kwargs)



        self.n_classes = n_classes

        self.s = s

        self.m = m

        self.ls_eps = ls_eps

        self.easy_margin = easy_margin

        self.cos_m = tf.math.cos(m)

        self.sin_m = tf.math.sin(m)

        self.th = tf.math.cos(math.pi - m)

        self.mm = tf.math.sin(math.pi - m) * m



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

        sine = tf.math.sqrt(1.0 - tf.math.pow(cosine, 2))

        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:

            phi = tf.where(cosine > 0, phi, cosine)

        else:

            phi = tf.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = tf.cast(

            tf.one_hot(y, depth=self.n_classes),

            dtype=cosine.dtype

        )

        if self.ls_eps > 0:

            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.n_classes



        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        output *= self.s

        return output



def create_model(input_shape,n_classes,scale=30,margin=0.3):



    #backbone = tf.keras.applications.EfficientNetB6(include_top=False,input_shape=input_shape,weights='imagenet')

    backbone = efn.EfficientNetB7(weights=None, include_top=False, input_shape=input_shape)

    #backbone.summary()

    

    pooling = tf.keras.layers.GlobalAveragePooling2D(name='head/pooling')

    bn_0 = tf.keras.layers.BatchNormalization(name='head/bn_0')

    dense = tf.keras.layers.Dense(512, name='head/dense')

    bn_1 = tf.keras.layers.BatchNormalization(name='head/bn_1')

    margin = AddMarginProduct(n_classes=n_classes,s=scale,m=margin,name='head/cos_margin',dtype='float32')

    

    dropout = tf.keras.layers.Dropout(0.0, name='head/dropout')

    

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





class DistributedModel:



    def __init__(self,

                 input_size,

                 n_classes,

                 finetuned_weights,

                 scale,

                 margin,

                 optimizer,

                 strategy,

                 mixed_precision,

                 clip_grad):



        self.model = create_model(input_shape=input_size, n_classes=n_classes,scale=scale,margin=margin,)

        self.model.summary()

        self.input_size = input_size

        self.n_classes=n_classes



        if finetuned_weights:

            self.model.load_weights(finetuned_weights)



        self.mixed_precision = mixed_precision

        self.optimizer = optimizer

        self.strategy = strategy

        self.clip_grad = clip_grad



        # loss function

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)        



        # metrics

        self.mean_loss_train = tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=False)

        self.mean_accuracy_train = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)



        if self.optimizer and self.mixed_precision:

            self.optimizer =tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, loss_scale='dynamic')



    def _compute_loss(self, labels, probs):

        per_example_loss = self.loss_object(labels, probs)

        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=BATCH_SIZE)



    def _backprop_loss(self, tape, loss, weights):

        gradients = tape.gradient(loss, weights)

        if self.mixed_precision:

            gradients = self.optimizer.get_unscaled_gradients(gradients)

        #clipped, _ = tf.clip_by_global_norm(gradients, clip_norm=self.clip_grad)

        clipped =gradients

        self.optimizer.apply_gradients(zip(clipped, weights))



    def _train_step(self, inputs):

        with tf.GradientTape() as tape:

            probs = self.model(inputs, training=True)

            loss = self._compute_loss(inputs[1], probs)

            if self.mixed_precision:

                loss = self.optimizer.get_scaled_loss(loss)

        self._backprop_loss(tape, loss, self.model.trainable_weights)

        self.mean_loss_train.update_state(inputs[1], probs)

        self.mean_accuracy_train.update_state(inputs[1], probs)

        return loss



    @tf.function

    def _distributed_train_step(self, dist_inputs):

        per_replica_loss = self.strategy.run(self._train_step, args=(dist_inputs,))

        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)



    def train(self, epochs, save_path):        

        acc=0

        model_name=''

        for _EPOCH in range(0):############################# Training Attention please change epoch 0 to 1

            random.shuffle(TRAINING_FILENAMES_512)

            for epoch in range(2):

                train_files=TRAINING_FILENAMES_512[14*epoch:14*(epoch+1)]

                #train_files=TRAINING_FILENAMES_512_order

                for file_name in train_files:

                    print(file_name[-20:],end='')

                train_num=count_data_items(train_files)

                steps=train_num//BATCH_SIZE

                print(' There are %i train images'%train_num,f'need Steps:{steps}/epoch')

                random.shuffle(train_files)

                train_ds = get_training_dataset(train_files,50000)



                dist_train_ds = self.strategy.experimental_distribute_dataset(train_ds)

                dist_train_ds = tqdm.tqdm(dist_train_ds)

                for i, inputs in enumerate(dist_train_ds):

                    loss = self._distributed_train_step(inputs)

                    dist_train_ds.set_description(

                        "TRAIN: Loss {:.3f}, Accuracy {:.3f}".format(

                            self.mean_loss_train.result().numpy(),

                            self.mean_accuracy_train.result().numpy()

                        )

                    )





            acc=self.mean_accuracy_train.result().numpy()

            self.mean_loss_train.reset_states()

            self.mean_accuracy_train.reset_states()

            model_name=f'ep-{_EPOCH}-acc-{acc:.3f}-'

                

            if save_path:

                self.model.save_weights(model_name+save_path)

#         save_model = tf.keras.Model(

#             inputs=self.model.layers[1].input,

#             outputs=self.model.layers[1].output)

#         save_model.save_weights('efnb7-model.h5')
with strategy.scope():



    optimizer = tf.keras.optimizers.SGD(config['learning_rate'], momentum=config['momentum'],decay=1e-5)



    dist_model = DistributedModel(

        input_size=config['input_size'],

        n_classes=config['n_classes'],

        finetuned_weights='',

        scale=config['scale'],

        margin=config['margin'],

        optimizer=optimizer,

        strategy=strategy,

        mixed_precision=mixed_precision,

        clip_grad=config['clip_grad'])



    dist_model.train(

        epochs=config['n_epochs'], 

        save_path='efnb7-512-model.h5')