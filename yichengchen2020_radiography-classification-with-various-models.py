import tensorflow as tf

import numpy as np

import random

import os

import time

import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import *

from sklearn.metrics import *

from tensorflow.keras.models import load_model



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignore warning :)



# tensorboard = TensorBoard(log_dir='mylog')



gpus = tf.config.experimental.list_physical_devices(device_type='GPU')

for gpu in gpus:

    tf.config.experimental.set_memory_growth(device=gpu, enable=True)

    

train_tfrecord = 'XRay_train.tfrecords'

test_tfrecord = 'XRay_test.tfrecords'

train_percentage = 0.8  # Proportion of training set



random.seed(20)



input_path='/kaggle/input/covid19-radiography-database/COVID-19 Radiography Database/'
learning_rate = 0.001

buffer_size = 512

batch_size = 16

epochs = 20



img_size = 224
def read_directory():

    data_filenames = []

    data_labels = []



    for filename in os.listdir(input_path + 'COVID-19'):

        data_filenames.append(input_path + 'COVID-19/' + filename)

        data_labels.append(0)



    for filename in os.listdir(input_path + 'NORMAL'):

        data_filenames.append(input_path + 'NORMAL/' + filename)

        data_labels.append(1)



    for filename in os.listdir(input_path + 'Viral Pneumonia'):

        data_filenames.append(input_path + 'Viral Pneumonia/' + filename)

        data_labels.append(2)

        

    data_size = len(data_labels)



    tmp_uni = list(zip(data_filenames, data_labels))



    random.shuffle(tmp_uni)



    train_size = int(data_size * train_percentage)

    print('Size of training set：', train_size)

    print('Size of test set：', data_size - train_size)



    train_list = tmp_uni[0:train_size]

    test_list = tmp_uni[train_size:]



    train_filenames, train_labels = zip(*train_list)

    test_filenames, test_labels = zip(*test_list)



    return train_filenames, train_labels, test_filenames, test_labels
def build_train_tfrecord(train_filenames, train_labels):  # Generate TFRecord of training set 

    with tf.io.TFRecordWriter(train_tfrecord)as writer:

        for filename, label in zip(train_filenames, train_labels):

            image = open(filename, 'rb').read()



            feature = {

                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),  # img > Bytes

                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))  # label > Int

            }



            example = tf.train.Example(features=tf.train.Features(feature=feature))

            writer.write(example.SerializeToString())





def build_test_tfrecord(test_filenames, test_labels):  # Generate TFRecord of test set

    with tf.io.TFRecordWriter(test_tfrecord)as writer:

        for filename, label in zip(test_filenames, test_labels):

            image = open(filename, 'rb').read()



            feature = {

                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),

                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))

            }



            example = tf.train.Example(features=tf.train.Features(feature=feature))

            writer.write(example.SerializeToString())
def _parse_example(example_string):

    feature_description = {

        'image': tf.io.FixedLenFeature([], tf.string),

        'label': tf.io.FixedLenFeature([], tf.int64),

    }



    feature_dict = tf.io.parse_single_example(example_string, feature_description)

    feature_dict['image'] = tf.io.decode_png(feature_dict['image'], channels=3)

    feature_dict['image'] = tf.image.resize(feature_dict['image'], [img_size, img_size]) / 255.0

    return feature_dict['image'], feature_dict['label']





def get_train_dataset(train_tfrecord):  # read TFRecord

    raw_train_dataset = tf.data.TFRecordDataset(train_tfrecord)

    train_dataset = raw_train_dataset.map(_parse_example)



    return train_dataset





def get_test_dataset(test_tfrecord):

    raw_test_dataset = tf.data.TFRecordDataset(test_tfrecord)

    test_dataset = raw_test_dataset.map(_parse_example)



    return test_dataset





def data_Preprocessing(train_dataset, test_dataset):

    train_dataset = train_dataset.shuffle(buffer_size)

    train_dataset = train_dataset.batch(batch_size)

    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)



    test_dataset = test_dataset.batch(batch_size)

    test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)



    return train_dataset, test_dataset
class CNN(tf.keras.Model):

    def __init__(self):

        super().__init__()

        self.conv1 = tf.keras.layers.Conv2D(

            filters=32,

            kernel_size=[3, 3],

            padding='same',

            activation=tf.nn.relu,

            input_shape=(img_size, img_size, 3)

        )

        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)

        self.conv2 = tf.keras.layers.Conv2D(

            filters=64,

            kernel_size=[5, 5],

            padding='same',

            activation=tf.nn.relu

        )

        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)

        self.flatten = tf.keras.layers.Reshape(target_shape=(int(img_size/4) * int(img_size/4) * 64,))

        self.dense1 = tf.keras.layers.Dense(units=512, activation=tf.nn.relu)

        self.drop1 = tf.keras.layers.Dropout(0.5)

        self.dense2 = tf.keras.layers.Dense(units=3, activation='softmax')



    def call(self, inputs):

        x = self.conv1(inputs)

        x = self.pool1(x)

        x = self.conv2(x)

        x = self.pool2(x)

        x = self.flatten(x)

        x = self.dense1(x)

        x = self.drop1(x)

        output = self.dense2(x)

        return output
def Xception_model():

    xception = tf.keras.applications.Xception(weights='imagenet',include_top=False, input_shape=[img_size,img_size,3])

    xception.trainable= True

    model = tf.keras.Sequential([

        xception,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(3, activation='softmax')

    ])



    model.compile(

        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),

        loss=tf.keras.losses.sparse_categorical_crossentropy,

        metrics=[tf.keras.metrics.sparse_categorical_accuracy]

    )

    

    return model
def VGG16_model():

    vgg16 = tf.keras.applications.VGG16(weights='imagenet',include_top=False, input_shape=[img_size,img_size,3])

    vgg16.trainable= True

    model = tf.keras.Sequential([

        vgg16,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(3, activation='softmax')

    ])



    model.compile(

        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),

        loss=tf.keras.losses.sparse_categorical_crossentropy,

        metrics=[tf.keras.metrics.sparse_categorical_accuracy]

    )

    

    return model
def ResNet50_model():

    res50 = tf.keras.applications.ResNet50(weights='imagenet',include_top=False, input_shape=[img_size,img_size,3])

    res50.trainable= True

    model = tf.keras.Sequential([

        res50,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(3, activation='softmax')

    ])



    model.compile(

        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),

        loss=tf.keras.losses.sparse_categorical_crossentropy,

        metrics=[tf.keras.metrics.sparse_categorical_accuracy]

    )

    

    return model
def ResNet50V2_model():

    res50v2 = tf.keras.applications.ResNet50V2(weights='imagenet',include_top=False, input_shape=[img_size,img_size,3])

    res50v2.trainable= True

    model = tf.keras.Sequential([

        res50v2,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(3, activation='softmax')

    ])



    model.compile(

        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),

        loss=tf.keras.losses.sparse_categorical_crossentropy,

        metrics=[tf.keras.metrics.sparse_categorical_accuracy]

    )

    

    return model
def InceptionV3_model():

    incpV3 = tf.keras.applications.InceptionV3(weights='imagenet',include_top=False, input_shape=[img_size,img_size,3])

    incpV3.trainable= True

    model = tf.keras.Sequential([

        incpV3,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(3, activation='softmax')

    ])



    model.compile(

        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),

        loss=tf.keras.losses.sparse_categorical_crossentropy,

        metrics=[tf.keras.metrics.sparse_categorical_accuracy]

    )

    

    return model
def MobileNetV2_model():

    mobV2 = tf.keras.applications.MobileNetV2(weights='imagenet',include_top=False, input_shape=[img_size,img_size,3])

    mobV2.trainable= True

    model = tf.keras.Sequential([

        mobV2,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(3, activation='softmax')

    ])



    model.compile(

        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),

        loss=tf.keras.losses.sparse_categorical_crossentropy,

        metrics=[tf.keras.metrics.sparse_categorical_accuracy]

    )

    

    return model
def DenseNet121_model():

    den121 = tf.keras.applications.DenseNet121(weights='imagenet',include_top=False, input_shape=[img_size,img_size,3])

    den121.trainable= True

    model = tf.keras.Sequential([

        den121,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(3, activation='softmax')

    ])



    model.compile(

        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),

        loss=tf.keras.losses.sparse_categorical_crossentropy,

        metrics=[tf.keras.metrics.sparse_categorical_accuracy]

    )

    

    return model
def ResNet101V2_model():

    res101v2 = tf.keras.applications.ResNet101V2(weights='imagenet',include_top=False, input_shape=[img_size,img_size,3])

    res101v2.trainable= True

    model = tf.keras.Sequential([

        res101v2,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(3, activation='softmax')

    ])



    model.compile(

        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),

        loss=tf.keras.losses.sparse_categorical_crossentropy,

        metrics=[tf.keras.metrics.sparse_categorical_accuracy]

    )

    

    return model
def InceptionResNetV2_model():

    incp_res_v2 = tf.keras.applications.InceptionResNetV2(weights='imagenet',include_top=False, input_shape=[img_size,img_size,3])

    incp_res_v2.trainable= True

    model = tf.keras.Sequential([

        incp_res_v2,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(3, activation='softmax')

    ])



    model.compile(

        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),

        loss=tf.keras.losses.sparse_categorical_crossentropy,

        metrics=[tf.keras.metrics.sparse_categorical_accuracy]

    )

    

    return model
def scheduler(epoch,lr):

    if epoch<8:

        return lr

    else:

        return lr*0.9

callback = tf.keras.callbacks.LearningRateScheduler(scheduler,verbose=1)



# class_weight = {0:6.14, 1:1., 2:1.}
def train():

    time_start = time.time()

    

#     model = Xception_model()

#     model = VGG16_model()

#     model = ResNet50_model()

#     model = ResNet50V2_model()

#     model = InceptionV3_model()

#     model = MobileNetV2_model()

#     model = DenseNet121_model()

#     model = ResNet101V2_model()

    model = InceptionResNetV2_model()

    

    model.summary()

    

    train_history = model.fit(train_dataset, epochs=epochs, callbacks=[callback])



    model.save('mymodel.h5')

    

    print('Model saved.')

    

    time_end = time.time()

    print('Training Time:', time_end - time_start)

    print('\n')



    return train_history
def show_train_history(train_history, index):

    plt.plot(train_history.history[index])

    plt.title('Train History')

    plt.ylabel(index)

    plt.xlabel('Epoch')

    plt.show()
def test(test_labels):

    test_labels = np.array(test_labels)

    model = load_model('/kaggle/working/mymodel.h5')

    

    print('Testing:')

    

    model.evaluate(test_dataset)

    

    predIdxs = model.predict(test_dataset)

    predIdxs = np.argmax(predIdxs, axis=1) 



    target_names = ['COVID-19', 'NORMAL', 'Viral Pneumonia']

    print('\n')

    print(classification_report(test_labels, predIdxs, target_names=target_names, digits=5))
if __name__ == "__main__":

    train_filenames, train_labels, test_filenames, test_labels = read_directory()



    build_train_tfrecord(train_filenames, train_labels)

    build_test_tfrecord(test_filenames, test_labels)



    train_dataset = get_train_dataset(train_tfrecord)

    test_dataset = get_test_dataset(test_tfrecord)



    print('Info of train_dataset', type(train_dataset))

    print('Info of test_dataset', type(test_dataset))



    train_dataset, test_dataset = data_Preprocessing(train_dataset, test_dataset) 



    train_history = train()

    

    test(test_labels)

    

    show_train_history(train_history, 'sparse_categorical_accuracy')

    

    # Avoid filling up the disk, if you want to save the tfrecords, just '#' these lines:

    for filename in os.listdir('/kaggle/working'): 

        if 'X' in filename: # Delete the tfrecord.

            os.remove('/kaggle/working/' + filename)