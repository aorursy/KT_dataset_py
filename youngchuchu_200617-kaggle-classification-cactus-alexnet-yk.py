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
import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

import random

import pandas as pd

import os



from tensorflow.keras import layers

from glob import glob

from zipfile import ZipFile



#matplotlib inline

train_zip = '../input/aerial-cactus-identification/train.zip'



if not os.path.exists('train'):

    print('No train data. Extracting zip file starts')

    with ZipFile(train_zip, 'r') as zip_obj:

        zip_obj.extractall()
os.listdir('train')

train_csv_path = '../input/aerial-cactus-identification/train.csv'

train_dir = os.path.join( os.getcwd(), 'train')

df = pd.read_csv(train_csv_path)

df.keys()

train_data = [ (os.path.join(train_dir, path), label) for path, label in zip(df['id'], df['has_cactus'])]
random.shuffle(train_data)
train_ratio = 0.8

val_data = train_data[int(train_ratio*len(train_data)):]

train_data = train_data[:int(train_ratio*len(train_data))]

len(train_data), len(val_data)
class_nums = tf.constant(['0','1'])

path = train_data[0]

path



def read_data(path):

    gfile = tf.io.read_file(path[0])

    image = tf.io.decode_image(gfile)

    image = tf.cast(image, tf.float32)/255



    onehot = tf.cast(class_nums == path[1], tf.uint8)

    return image, onehot

batch_size = 32

train_ds = tf.data.Dataset.from_tensor_slices(np.array(train_data))

train_ds = train_ds.map(read_data)

train_ds = train_ds.shuffle(1000)

train_ds = train_ds.batch(batch_size)

train_ds = train_ds.repeat()



val_ds = tf.data.Dataset.from_tensor_slices(np.array(val_data))

val_ds = val_ds.map(read_data)

val_ds = val_ds.batch(batch_size)

val_ds = val_ds.repeat()
image, label = next(iter(train_ds))

image.shape, label.shape
input_shape =(32,32,3)

inputs = layers.Input(input_shape)



net = layers.Conv2D(32,3,strides=1, padding='SAME')(inputs)

net = layers.Activation('relu')(net)

net = layers.Conv2D(32,3,strides=1, padding='SAME')(net)

net = layers.Activation('relu')(net)

net = layers.MaxPool2D((2,2))(net)

net = layers.Dropout(0.5)(net)





net = layers.Conv2D(64,3,strides=1, padding='SAME')(net)

net = layers.Activation('relu')(net)

net = layers.Conv2D(64,3,strides=1, padding='SAME')(net)

net = layers.Activation('relu')(net)

net = layers.MaxPool2D((2,2))(net)

net = layers.Dropout(0.5)(net)



net = layers.Flatten()(net)

net = layers.Dense(512)(net)

net = layers.Activation('relu')(net)

net = layers.Dropout(0.5)(net)

net = layers.Dense(2)(net)

net = layers.Activation('softmax')(net)



model = tf.keras.Model(inputs=inputs, outputs=net, name='basic_cnn')

model.compile(loss=tf.keras.losses.categorical_crossentropy,

             optimizer=tf.keras.optimizers.Adam(),

             metrics=['accuracy'])
steps_per_epoch = len(train_data) // batch_size

validation_steps = len(val_data) // batch_size
hist = model.fit(train_ds, 

                 steps_per_epoch=steps_per_epoch,

                 validation_data=val_ds,

                 validation_steps=validation_steps,

                 epochs=20)
histories = hist.history

plt.subplot(121)

plt.plot(histories['loss'])

plt.title('Loss')

plt.subplot(122)

plt.plot(histories['accuracy'])

plt.title('Accuracy')

plt.ylim([0,1])

plt.show()

test_zip = ('../input/aerial-cactus-identification/test.zip')



if not os.path.exists('test'):

    print('Train folder does not exist. Extracting zip file starts')    

    with ZipFile(test_zip, 'r') as zip_obj:

        zip_obj.extractall()
test_paths = glob('test/*.jpg')

test_nums = len(test_paths)

x_test = np.zeros([test_nums, 32, 32, 3])



for idx in range(len(test_paths)):

    path = test_paths[idx]

    gfile = tf.io.read_file(path)

    image = tf.io.decode_image(gfile)    

    x_test[idx] = image

test_paths
test_prediction = model.predict(x_test)
logits = np.argmax(test_prediction,-1)

logits
path = test_paths[0]

path.split('/')
test_id = [ path.split('/')[-1] for path in test_paths]

test_id
dict_test = {'id':test_id, 'has_cactus':logits}

submit_file = pd.DataFrame(dict_test)

submit_file.to_csv('submission.csv',index=False)

submit_file.head()
os.listdir()