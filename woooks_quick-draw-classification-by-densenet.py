!pip install tensorflow==1.15
import os

from glob import glob

import re

import ast

import numpy as np 

import pandas as pd

from PIL import Image, ImageDraw 

from tqdm import tqdm

from dask import bag



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Flatten

from tensorflow.keras.layers import Conv2D, MaxPooling2D

from tensorflow.keras.metrics import top_k_categorical_accuracy

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping



from tflearn.layers.conv import global_avg_pool

from tensorflow.contrib.layers import batch_norm, flatten

from tensorflow.contrib.framework import arg_scope
ROOT_DIR = '../input/quickdraw-doodle-recognition/'

TRAIN_DIR = ROOT_DIR + 'train_simplified/'
# Set label dictionary and params

classfiles = os.listdir(TRAIN_DIR)

numstonames = {i: v[:-4].replace(" ", "_") for i, v in enumerate(classfiles)}



num_classes = 340

imheight, imwidth = 32, 32  

ims_per_class = 2000
# Image conversion function

def draw_it(strokes):

    image = Image.new("P", (256,256), color=255)

    image_draw = ImageDraw.Draw(image)

    for stroke in ast.literal_eval(strokes):

        for i in range(len(stroke[0])-1):

            image_draw.line([stroke[0][i], 

                             stroke[1][i],

                             stroke[0][i+1], 

                             stroke[1][i+1]],

                            fill=0, width=5)

    image = image.resize((imheight, imwidth))

    return np.array(image)/255.
# Load and preprocess train data

train_grand = []

class_paths = glob(TRAIN_DIR + '*.csv')

for i,c in enumerate(tqdm(class_paths[0: num_classes])):

    train = pd.read_csv(c, usecols=['drawing', 'recognized'], nrows=ims_per_class*5//4)

    train = train[train.recognized == True].head(ims_per_class)

    imagebag = bag.from_sequence(train.drawing.values).map(draw_it) 

    trainarray = np.array(imagebag.compute())  # PARALLELIZE

    trainarray = np.reshape(trainarray, (ims_per_class, -1))    

    labelarray = np.full((train.shape[0], 1), i)

    trainarray = np.concatenate((labelarray, trainarray), axis=1)

    train_grand.append(trainarray)



train_grand = np.array([train_grand.pop() for i in np.arange(num_classes)]) #less memory than np.concatenate

train_grand = train_grand.reshape((-1, (imheight*imwidth+1)))



del trainarray

del train



np.random.shuffle(train_grand)

y_train, X_train = train_grand[:,0], train_grand[:,1:]

y_train = keras.utils.to_categorical(y_train, num_classes)

print('X_train.shape:', X_train.shape)

print('y_train.shape:', y_train.shape)



del train_grand
# Hyperparameter

growth_k = 12

nb_block = 2 # how many (dense block + Transition Layer) ?

init_learning_rate = 1e-4

epsilon = 1e-8 # AdamOptimizer epsilon

dropout_rate = 0.2



# Momentum Optimizer will use

nesterov_momentum = 0.9

weight_decay = 1e-4



# Label & batch_size

class_num = num_classes

batch_size = 100



# Epoch

total_epochs = 1
def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):

    with tf.name_scope(layer_name):

        network = tf.layers.conv2d(inputs=input, filters=filter, kernel_size=kernel, strides=stride, padding='SAME')

        return network



def Global_Average_Pooling(x, stride=1):

    return global_avg_pool(x, name='Global_avg_pooling')



def Batch_Normalization(x, training, scope):

    with arg_scope([batch_norm],

                   scope=scope,

                   updates_collections=None,

                   decay=0.9,

                   center=True,

                   scale=True,

                   zero_debias_moving_mean=True) :

        return tf.cond(training,

                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),

                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))



def Drop_out(x, rate, training) :

    return tf.layers.dropout(inputs=x, rate=rate, training=training)



def Relu(x):

    return tf.nn.relu(x)



def Average_pooling(x, pool_size=[2,2], stride=2, padding='VALID'):

    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)



def Max_Pooling(x, pool_size=[3,3], stride=2, padding='VALID'):

    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)



def Concatenation(layers) :

    return tf.concat(layers, axis=3)



def Linear(x) :

    return tf.layers.dense(inputs=x, units=class_num, name='linear')





class DenseNet():

    def __init__(self, x, nb_blocks, filters, training):

        self.nb_blocks = nb_blocks

        self.filters = filters

        self.training = training

        self.model = self.Dense_net(x)





    def bottleneck_layer(self, x, scope):

        with tf.name_scope(scope):

            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')

            x = Relu(x)

            x = conv_layer(x, filter=4 * self.filters, kernel=[1,1], layer_name=scope+'_conv1')

            x = Drop_out(x, rate=dropout_rate, training=self.training)



            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch2')

            x = Relu(x)

            x = conv_layer(x, filter=self.filters, kernel=[3,3], layer_name=scope+'_conv2')

            x = Drop_out(x, rate=dropout_rate, training=self.training)



            return x



    def transition_layer(self, x, scope):

        with tf.name_scope(scope):

            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')

            x = Relu(x)

            in_channel = int(x.shape[-1])

            x = conv_layer(x, filter=in_channel*0.5, kernel=[1,1], layer_name=scope+'_conv1')

            x = Drop_out(x, rate=dropout_rate, training=self.training)

            x = Average_pooling(x, pool_size=[2,2], stride=2)



            return x



    def dense_block(self, input_x, nb_layers, layer_name):

        with tf.name_scope(layer_name):

            layers_concat = list()

            layers_concat.append(input_x)



            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))



            layers_concat.append(x)



            for i in range(nb_layers - 1):

                x = Concatenation(layers_concat)

                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))

                layers_concat.append(x)



            x = Concatenation(layers_concat)



            return x



    def Dense_net(self, input_x):

        x = conv_layer(input_x, filter=2 * self.filters, kernel=[7,7], stride=2, layer_name='conv0')

        x = Max_Pooling(x, pool_size=[3,3], stride=2)



        for i in range(self.nb_blocks) :

            # 6 -> 12 -> 48

            x = self.dense_block(input_x=x, nb_layers=4, layer_name='dense_'+str(i))

            x = self.transition_layer(x, scope='trans_'+str(i))

        x = self.dense_block(input_x=x, nb_layers=32, layer_name='dense_final')



        # 100 Layer

        x = Batch_Normalization(x, training=self.training, scope='linear_batch')

        x = Relu(x)

        x = Global_Average_Pooling(x)

        x = flatten(x)

        x = Linear(x)



        return x
tf.reset_default_graph()

x = tf.placeholder(tf.float32, shape=[None, imheight * imwidth])

batch_images = tf.reshape(x, [-1, imheight, imwidth, 1])

label = tf.placeholder(tf.float32, shape=[None, class_num])



training_flag = tf.placeholder(tf.bool)

learning_rate = tf.placeholder(tf.float32, name='learning_rate')

logits = DenseNet(x=batch_images, nb_blocks=nb_block, filters=growth_k, training=training_flag).model

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))



optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)

train = optimizer.minimize(cost)



correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess = tf.Session()

sess.run(tf.global_variables_initializer())



epoch_learning_rate = init_learning_rate

for epoch in range(total_epochs):

    if epoch == (total_epochs * 0.5) or epoch == (total_epochs * 0.75):

        epoch_learning_rate = epoch_learning_rate / 10



    total_batch = int(len(X_train) / batch_size)

    for step in range(total_batch):

        start, end = batch_size * step, batch_size * step + batch_size

        batch_x, batch_y = X_train[start:end], y_train[start:end]

        train_feed_dict = {

            x: batch_x,

            label: batch_y,

            learning_rate: epoch_learning_rate,

            training_flag: True

        }



        _, loss = sess.run([train, cost], feed_dict=train_feed_dict)

        if step % 1000 == 0:

            train_accuracy = sess.run([accuracy], feed_dict=train_feed_dict)

            print("Epoch:", epoch, "Step:", step, "Loss:", loss, "Training accuracy:", train_accuracy)
ttvlist = []

reader = pd.read_csv(ROOT_DIR + 'test_simplified.csv', index_col=['key_id'], chunksize=2048)

for chunk in tqdm(reader, total=55):

    imagebag = bag.from_sequence(chunk.drawing.values).map(draw_it)

    testarray = np.array(imagebag.compute()).reshape([-1, 1024])

    test_feed_dict = {x: testarray, training_flag : False}

    testpreds = sess.run(logits, feed_dict=test_feed_dict)

    ttvs = np.argsort(-testpreds)[:, 0:3]  # top 3

    ttvlist.append(ttvs)



ttvarray = np.concatenate(ttvlist)
preds_df = pd.DataFrame({'first': ttvarray[:,0], 'second': ttvarray[:,1], 'third': ttvarray[:,2]})

preds_df = preds_df.replace(numstonames)

preds_df['words'] = preds_df['first'] + " " + preds_df['second'] + " " + preds_df['third']



sub = pd.read_csv(ROOT_DIR + 'sample_submission.csv', index_col=['key_id'])

sub['word'] = preds_df.words.values

sub.to_csv('densenet_submission.csv')

sub.head()