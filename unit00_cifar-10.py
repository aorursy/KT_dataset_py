from pathlib import Path
from typing import List, Dict

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for idx, filename in enumerate(filenames, start=1):
        print(os.path.join(dirname, filename))
        
        if idx % 4 == 0:
            print()
            break
import tensorflow as tf
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    if type(value) is np.ndarray:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value.tolist()))
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

image_path = Path('/kaggle/input/cifar10-python/cifar-10-batches-py/')
test_path = image_path.joinpath('test_batch')
def compile_data(batch_path: Path) -> List:
    """
    Compiles CIFAR-10 batches into one
    Returns list of compiled images and labels, along with meta data
    """
    import pickle
    
    compiled_data = []

    with open(batch_path, 'rb') as file:
        batch_dict = pickle.load(file, encoding='bytes')

        labels = batch_dict[b'labels']
        data = batch_dict[b'data']
        filenames = batch_dict[b'filenames']

        num_samples = len(labels)

        for i in tqdm(range(num_samples)):
            label = labels[i]
            datum = data[i]
            filename = filenames[i]

            features = {'data': datum,
                        'label': label,
                        'filename': filename}

            compiled_data.append(features)
            
    return compiled_data
            
train_data = []
for batch_num in range(1, 5):
    filename = image_path.joinpath(f'data_batch_{batch_num}')
    
    train_data.extend(compile_data(filename))
    
    
test_data = compile_data(test_path)
def convert_to_example(image: Dict):
    """Convert Image to TFRecord ready format"""
    feature = {
        'height': _int64_feature(32),
        'width': _int64_feature(32),
        'channels': _int64_feature(3),
        'label': _int64_feature(image['label']),
        'filename': _bytes_feature(image['filename']),
        'image_raw': _bytes_feature(image['data'].tobytes()),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))
train_record_file = 'train.tfrecords'

with tf.io.TFRecordWriter(train_record_file) as writer:
    for image in tqdm(train_data):
        tf_example = convert_to_example(image)
        writer.write(tf_example.SerializeToString())
test_record_file = 'test.tfrecords'

with tf.io.TFRecordWriter(test_record_file) as writer:
    for image in tqdm(test_data):
        tf_example = convert_to_example(image)
        writer.write(tf_example.SerializeToString())
from typing import Tuple, Dict

import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
# metadata_path = Path('/kaggle/input/cifar10-python/cifar-10-batches-py/batches.meta')
# def unpack_metadata(mdp):
#     import pickle
#     with open(mdp, 'rb') as f:
#         file = pickle.load(f, encoding='bytes')
#     return file
record_path = Path('/kaggle/input/cifar-10/')
train_record_file = record_path.joinpath('train.tfrecords')
test_record_file = record_path.joinpath('test.tfrecords')

raw_train_dataset = tf.data.TFRecordDataset(str(train_record_file))
raw_test_dataset = tf.data.TFRecordDataset(str(train_record_file))
def parse_image_function(ex_proto: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Parse protobuf messages into proper dataset objects
    Returns image and labels
    """
    image_feature_desc = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'channels': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'filename': tf.io.FixedLenFeature([], tf.string),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }
    
    example = tf.io.parse_single_example(ex_proto, image_feature_desc)
    
    label = example['label']
    
    img_raw = tf.io.decode_raw(example['image_raw'], tf.uint8) #convert into uint8
    img_raw = tf.reshape(img_raw, (example['channels'], example['height'], example['width'])) #reshape image into 3 by 32 by 32
    img_raw = tf.transpose(img_raw, (1, 2, 0)) #transpose image into (32, 32, 3), tf format

    return img_raw, label
def process_image(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Returns processed image
    """
    
    image = tf.image.per_image_standardization(image)
    image = tf.math.divide(image, 255)
    image = tf.image.resize(image, (64, 64))
    
    return image, label
train_ds = (raw_train_dataset.map(parse_image_function)
                  .map(process_image)
                  .repeat()
                  .shuffle(buffer_size=40000)
                  .batch(batch_size=32)
                  .prefetch(buffer_size=625)
)


test_ds = (raw_test_dataset.map(parse_image_function)
                  .map(process_image)
                  .shuffle(buffer_size=10000)
                  .batch(batch_size=32)
                  .prefetch(buffer_size=150)
)
                
from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, MaxPool2D, BatchNormalization
from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.callbacks import EarlyStopping


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
label_classes = list(map(lambda s: s.decode('utf-8'), [b'airplane',
 b'automobile',
 b'bird',
 b'cat',
 b'deer',
 b'dog',
 b'frog',
 b'horse',
 b'ship',
 b'truck']))
def lenet(input_shape: Tuple[int, ...],
          output_shape: int) -> Model:
    """
    Returns LeNet Keras model
    """
    
    num_classes = output_shape
    
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding='valid'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='valid'))
    model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model
def alexnet(input_shape: Tuple[int, ...],
            output_shape: int) -> Model:
    """
    Returns AlexNet Keras model
    """
    
    model = Sequential([
    Conv2D(filters=128, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=input_shape),
    BatchNormalization(),
    MaxPool2D(pool_size=(2,2)),
    Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    BatchNormalization(),
    MaxPool2D(pool_size=(3,3)),
    Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    BatchNormalization(),
    Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    BatchNormalization(),
    Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    BatchNormalization(),
    MaxPool2D(pool_size=(2,2)),
    Flatten(),
    Dense(1024,activation='relu'),
    Dropout(0.5),
    Dense(1024,activation='relu'),
    Dropout(0.5),
    Dense(output_shape,activation='softmax')])
    
    return model
input_shape = (64, 64, 3)
output_shape = 10
epochs = 50
callbacks = []

early_stopping = EarlyStopping(monitor='val_loss',
                               min_delta=0.01,
                               patience=2,
                               verbose=1,
                               mode='auto')

callbacks.append(early_stopping)
net = lenet(input_shape, output_shape)
net.summary()
with tf.device('GPU:0'):
    net.compile(loss='sparse_categorical_crossentropy',
                optimizer='adam', 
                metrics=['accuracy'])
    history = net.fit(train_ds,
                      epochs=epochs,
                      callbacks=callbacks,
                      validation_data=test_ds,
                      steps_per_epoch=int(40000/32))
e = range(1, len(history.history['loss'])+1)

loss_result = pd.DataFrame({'epochs': e,
              'training_loss': history.history['loss'],
              'validation_loss': history.history['val_loss']
             })

sns.lineplot(data=loss_result, x='epochs', y='training_loss', label='Train')
sns.lineplot(data=loss_result, x='epochs', y='validation_loss', label='Valid')

plt.ylabel('Loss')

plt.title('Training loss vs Validation loss');
e = range(1, len(history.history['loss'])+1)

loss_result = pd.DataFrame({'epochs': e,
              'training_accuracy': history.history['accuracy'],
              'validation_accuracy': history.history['val_accuracy']
             })

sns.lineplot(data=loss_result, x='epochs', y='training_accuracy', label='Train')
sns.lineplot(data=loss_result, x='epochs', y='validation_accuracy', label='Valid')

plt.ylabel('accuracy')
plt.title('Training accuracy vs Validation accuracy');
