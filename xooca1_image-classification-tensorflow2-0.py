!pip install -q efficientnet
import os

import collections

from datetime import datetime, timedelta

from functools import partial

import math, re, os

import tensorflow as tf

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

#from kaggle_datasets import KaggleDatasets

from torchvision import datasets, transforms ,utils,models

import efficientnet.tfkeras as efn



from tensorflow.keras.applications import ResNet50V2,ResNet101V2,ResNet152V2,DenseNet201

import cv2

import torch.nn.functional as F

import torch.nn as nn

import os

import shutil

import torch

from torchvision import transforms

from PIL import Image

import pathlib

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from sklearn.model_selection import train_test_split



print("Tensorflow version " + tf.__version__)
!ls
# Create strategy from tpu

istpu=False

if istpu:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

    BATCH_SIZE = 16 * strategy.num_replicas_in_sync
AUTO = tf.data.experimental.AUTOTUNE



# Configuration

IMAGE_SIZE = [512, 512]

EPOCHS = 20

test = pd.read_csv('../input/test_ApKoW4T.csv')

sample = pd.read_csv('../input/sample_submission_ns2btKE.csv')

train = pd.read_csv('../input/train/train.csv')
convertlabeldict = {1: 'Cargo', 

2:'Military', 

3:'Carrier', 

4:'Cruise', 

5:'Tankers'}

train['category_label'] = train['category'].map(convertlabeldict)
train.head()
train, holdout = train_test_split(train, test_size=0.2, random_state=0, 

                               stratify=train['category'])
train = train.reset_index(drop=True)

train.head()
holdout = holdout.reset_index(drop=True)

holdout.head()
test = test.reset_index(drop=True)

test.head()
train['category'].value_counts()
holdout['category'].value_counts()
def copy_files_local(df,basedir,destinationfolder):

    for i,row in df.iterrows():

        currentfileloc = basedir + row['image']

        if not os.path.exists(destinationfolder):

            os.makedirs(destinationfolder)

        shutil.copy(currentfileloc, destinationfolder)
copy_files_local(train,'../input/train/images/','../train/')

copy_files_local(holdout,'../input/train/images/','../holdout/')

copy_files_local(test,'../input/train/images/','../test/')
!ls ../train/ | wc -l

!ls ../holdout/ | wc -l

!ls ../test/ | wc -l

def get_filenames(df,category,image_num):

    return random.sample(df[df.category==category]['image'].tolist(),image_num)



def random_brightness(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    rand = random.uniform(0.6, 1.0)

    hsv[:, :, 2] = rand*hsv[:, :, 2]

    rand = random.uniform(1.0, 1.5)

    hsv[:, :, 1] = rand*hsv[:, :, 1]

    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return new_img





def zoom(image,rows,cols):

    zoom_pix = random.randint(5, 10)

    zoom_factor = 1 + (2*zoom_pix)/rows

    image = cv2.resize(image, None, fx=zoom_factor,

                       fy=zoom_factor, interpolation=cv2.INTER_LINEAR)

    top_crop = (image.shape[0] - rows)//2

    left_crop = (image.shape[1] - cols)//2

    image = image[top_crop: top_crop+rows,

                  left_crop: left_crop+cols]

    return image
import random



def createaugimagesv2(df,category,image_num,dirname):

    filename = get_filenames(df,category,image_num)

    for images in filename:

        if images[-8:]!='_enh.jpg' and images[-9:]!='_enh1.jpg':

            imagepath = dirname + images

            image = cv2.imread(imagepath)

            rows,cols,channel = image.shape

            image = np.fliplr(image)



            op1 = random.randint(0, 1)

            op2 = random.randint(0, 1)

            op3 = random.randint(0, 1)

            if op1:

                image = random_brightness(image)

            if op2:

                image = zoom(image,rows,cols)

            newimagepath = dirname + images.split('.')[0]+'_'+str(category) + '_enh.jpg'

            try:

                image = cv2.resize(image, (224, 224))

                cv2.imwrite(newimagepath, image)

            except:

                print("file {0} is not converted".format(images))
import random

def im_convert(tensor):

    image = tensor.clone().detach().numpy()

    image = image.transpose(1, 2, 0)

    image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))

    image = image.clip(0, 1)

    return image



def createaugimages(df,category,image_num,dirname):

    filename = get_filenames(df,category,image_num)

    for images in filename:

        if images[-9:]!='_enh1.jpg':

            imagepath = dirname + images

            pil_im = Image.open(imagepath, 'r').convert('RGB')

            op1 = random.randint(0, 1)

            if op1 ==1:

                changeimg = transforms.Compose([ 

                                        transforms.RandomRotation(5),

                                        transforms.Resize(224),

                                        transforms.ToTensor()

                                       ])

            else:

                changeimg = transforms.Compose([ 

                            transforms.RandomHorizontalFlip(),

                            transforms.RandomRotation(10),

                            transforms.Resize(224),

                            transforms.ToTensor()

                           ])



            img = changeimg(pil_im)

            newimagepath = dirname + images.split('.')[0]+'_'+str(category) + '_enh1.jpg'

            utils.save_image(img,newimagepath)   
def resizeall(dirname):

    filename = os.listdir(dirname)

    non3channel = []

    for images in filename:

        imagepath = dirname + images

        image = cv2.imread(imagepath)

        if image.shape[2] !=3:

            non3channel.append(images)

    return non3channel
convertlabeldict = {1: 'Cargo', 

2:'Military', 

3:'Carrier', 

4:'Cruise', 

5:'Tankers'}
a = 1908 - 1095

b = 1908 - 1050

c = 1908 - 824

d = 1908 - 749
#1    1696

#5     973

#2     933

#3     733

#4     666
createaugimagesv2(train,5,1696 - 1095,'../train/')

createaugimagesv2(train,2,1696 - 1050,'../train/')

createaugimagesv2(train,3,733,'../train/')

createaugimagesv2(train,4,666,'../train/')
createaugimages(train,1,500,'../train/')

createaugimages(train,5,500,'../train/')

createaugimages(train,2,500,'../train/')

createaugimages(train,3,733,'../train/')

createaugimages(train,4,666,'../train/')
createaugimagesv2(train,3,288,'../train/')

createaugimagesv2(train,4,400,'../train/')
train.shape
data_dir = '../train/'

data_dir = pathlib.Path(data_dir)

enh_files = list(data_dir.glob('*_enh.jpg'))

enh_files1 = list(data_dir.glob('*_enh1.jpg'))

allfiles = enh_files+enh_files1
len(allfiles),len(set(allfiles))
train = train.reset_index(drop=False)

train.head()
holdout = holdout.reset_index(drop=False)

holdout.head()
test = test.reset_index(drop=False)

test.head()
enh_df = pd.DataFrame()

i = max(train['index'])

for file in allfiles:

    st = str(file)

    cat = st.split('_')[1]

    st = st.split('/')[-1]

    i = i+1

    enh_df = enh_df.append({'index': int(i), 'image': st, 'category': int(cat),'category_label':convertlabeldict[int(cat)]}, ignore_index=True)
enh_df['index']=enh_df['index'].apply(int)

train = pd.concat([train,enh_df]).reset_index(drop=True)

train.head()
def null_finder(df):

    for col in df.columns.tolist():

        print("For column {0} : NULLS are {1}".format(col,sum(df[col].isnull())))
null_finder(train)
null_finder(holdout)
null_finder(test)


!ls ../train/ | wc -l

!ls ../holdout/ | wc -l

!ls ../test/ | wc -l

data_dir = '../train/'

data_dir = pathlib.Path(data_dir)

roses = list(data_dir.glob('*_enh.jpg'))

import IPython.display as display

from PIL import Image

import numpy as np

import matplotlib.pyplot as plt

import os

for image_path in roses[:3]:

    print(image_path)

    display.display(Image.open(str(image_path)))
BATCH_SIZE = 64

img_size = 224

train_image_count = 10546

hold_image_count = 1251

STEPS_PER_EPOCH = np.ceil(train_image_count/BATCH_SIZE)

STEPS_PER_EPOCH_HOLD = np.ceil(hold_image_count/BATCH_SIZE)

STEPS_PER_EPOCH,STEPS_PER_EPOCH_HOLD
def decode_img(img):

    # convert the compressed string to a 3D uint8 tensor

    img = tf.image.decode_jpeg(img, channels=3)

    #img = tf.cast(image, tf.float32) / 255.0 

    # Use `convert_image_dtype` to convert to floats in the [0,1] range.

    img = tf.image.convert_image_dtype(img, tf.float32)

    # resize the image to the desired size.

    return tf.image.resize(img, [img_size, img_size])



def get_images_label(image,label):

    img = tf.io.read_file(image)

    img = decode_img(img)

    label = tf.cast(label, tf.int32)

    return img,label



def convert_cat(image,label):

    label = tf.convert_to_tensor(tf.keras.utils.to_categorical(label, num_classes=5, dtype='float32'))

    return image,label



def get_images_id(image,ids):

    img = tf.io.read_file(image)

    img = decode_img(img)

    ids = tf.cast(ids, tf.int32)

    return img,ids



def data_augment(image, label, seed=2020):

    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),

    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part

    # of the TPU while the TPU itself is computing gradients.

    image = tf.image.random_flip_left_right(image, seed=seed)

#    image = tf.image.random_flip_up_down(image, seed=seed)

#    image = tf.image.random_brightness(image, 0.1, seed=seed)

#    image = tf.image.random_jpeg_quality(image, 85, 100, seed=seed)

#    image = tf.image.resize(image, [256, 256])

#    image = tf.image.central_crop(image, [224, 224])

#    image = tf.image.random_crop(image, [224, 224], seed=seed)

    #image = tf.image.random_saturation(image, 0, 2)

    return image, label 



def data_reshape(image,label,seed=2020):

#    image = tf.image.resize(image, [256, 256])

    #image = tf.image.central_crop(image, [224, 224])

    image = tf.image.random_crop(image, [img_size, img_size], seed=seed)

    return image,label

    



def get_training_dataset(dataset):

    dataset = dataset.map(get_images_label, num_parallel_calls=AUTOTUNE)

    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

    #dataset = dataset.map(convert_cat, num_parallel_calls=AUTO)

    dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.shuffle(BATCH_SIZE * 50)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_validation_dataset(dataset):

    dataset = dataset.map(get_images_label, num_parallel_calls=AUTOTUNE)

    #dataset = dataset.map(data_reshape, num_parallel_calls=AUTO)

    #dataset = dataset.map(convert_cat, num_parallel_calls=AUTO)

    dataset = dataset.batch(BATCH_SIZE)

    #dataset = dataset.cache()

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_test_dataset(dataset):

    dataset = dataset.map(get_images_id, num_parallel_calls=AUTOTUNE)

    #dataset = dataset.map(data_reshape, num_parallel_calls=AUTO)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset
#BATCH_SIZE = 64

AUTOTUNE = tf.data.experimental.AUTOTUNE

CLASSES =[1,2,3,4,5]



dataset_train = tf.data.Dataset.from_tensor_slices(('../train/'+train['image'].values,np.array(pd.get_dummies(train['category'].values))))

dataset_holdout = tf.data.Dataset.from_tensor_slices(('../holdout/'+holdout['image'].values,np.array(pd.get_dummies(holdout['category'].values))))

dataset_test = tf.data.Dataset.from_tensor_slices(('../test/'+test['image'].values,test['index'].values))
final_ds_train = get_training_dataset(dataset_train)

final_ds_holdout = get_training_dataset(dataset_holdout)

final_ds_test = get_test_dataset(dataset_test)
def show_batch(image_batch, label_batch):

    plt.figure(figsize=(10,10))

    for n in range(25):

        ax = plt.subplot(5,5,n+1)

        plt.imshow(image_batch[n])

        value = label_batch.numpy().tolist()

        value = value[n].index(1) + 1

        plt.title(convertlabeldict[value])

        plt.axis('off')
image_batch, label_batch = next(iter(final_ds_train))


show_batch(image_batch, label_batch)
for image,caregory in final_ds_holdout.take(5):

    print(image.shape,caregory.shape)
convertlabeldict
def macro_f1(y, y_hat, thresh=0.5):

    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)

    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)

    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)

    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)

    f1 = 2*tp / (2*tp + fn + fp + 1e-16)

    macro_f1 = tf.reduce_mean(f1)

    return macro_f1
#with strategy.scope():

enet=False

if enet:

    features_model = efn.EfficientNetB7(

        input_shape=(256, 256, 3),

        weights='imagenet',

        include_top=False

    )

else:

    features_model = DenseNet201(

        input_shape=(img_size, img_size, 3),

        weights='imagenet',

        include_top=False

    )



model = tf.keras.Sequential([

    features_model,

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dense(5, activation='softmax')

])



model.compile(

    #optimizer=tf.keras.optimizers.Adam(lr=0.0001),

    optimizer=tf.keras.optimizers.Adam(lr=0.00001),

    #optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9),

    #loss = 'categorical_crossentropy',

    #loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),

    #loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),

    #metrics=[tf.keras.metrics.SparseCategoricalCrossentropy()]

    #metrics = tf.keras.metrics.sparse_categorical_accuracy()

    loss = 'categorical_crossentropy',

    #loss = macro_f1,

    metrics=['categorical_accuracy']

)

model.summary()
gpu_name = tf.test.gpu_device_name()

gpu_name
def exponential_decay(lr0, s):

    def exponential_decay_fn(epoch):

        return lr0 * 0.1**(epoch / s)

    return exponential_decay_fn



exponential_decay_fn = exponential_decay(lr0=0.01, s=20)

lr_schedule = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)



history = model.fit(

    final_ds_train, 

    steps_per_epoch=STEPS_PER_EPOCH,

    epochs=15, 

    validation_data=final_ds_holdout,

    validation_steps=STEPS_PER_EPOCH_HOLD,

    #validation_data=dataset_holdout

    callbacks=[lr_schedule]

)
history.history
from collections import Counter


probabilities = model.predict(final_ds_test)

predictions = np.argmax(probabilities, axis=-1) +1

print(Counter(predictions))
print('Generating submission.csv file...')

test_ids_ds = final_ds_test.map(lambda image, idnum: idnum).unbatch()

test_ids = next(iter(test_ids_ds.batch(2680))).numpy().astype('U') # all in one batch

np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')
def display_training_curves(training, validation, title, subplot):

    if subplot%10==1: # set up the subplots on the first call

        plt.subplots(figsize=(10,10), facecolor='#F0F0F0')

        plt.tight_layout()

    ax = plt.subplot(subplot)

    ax.set_facecolor('#F8F8F8')

    ax.plot(training)

    ax.plot(validation)

    ax.set_title('model '+ title)

    ax.set_ylabel(title)

    #ax.set_ylim(0.28,1.05)

    ax.set_xlabel('epoch')

    ax.legend(['train', 'valid.'])
display_training_curves(history.history['loss'], history.history['loss'], 'loss', 211)

display_training_curves(history.history['categorical_accuracy'], history.history['categorical_accuracy'], 'accuracy', 212)