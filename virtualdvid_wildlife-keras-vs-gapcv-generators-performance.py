%%capture

# install tensorflow 2.0 alpha

!pip install -q tensorflow-gpu==2.0.0-alpha0



#install GapCV

!pip install -q gapcv
import os

import time

import gc

import shutil

import numpy as np



import tensorflow as tf

from tensorflow.keras import backend as K

from tensorflow.keras.models import Sequential, load_model

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import layers

from tensorflow.keras.optimizers import SGD, Adam

from tensorflow.keras import regularizers



import gapcv

from gapcv.vision import Images

from gapcv.utils.img_tools import ImgUtils



import warnings

warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



print('tensorflow version: ', tf.__version__)

print('keras version: ', tf.keras.__version__)

print('gapcv version: ', gapcv.__version__)
os.makedirs('model', exist_ok=True)

print(os.listdir('../input'))

print(os.listdir('./'))
def plot_sample(imgs_set, labels_set, img_size=(12,12), columns=4, rows=4, random=False):

    """

    Plot a sample of images

    """

    

    fig=plt.figure(figsize=img_size)

    

    for i in range(1, columns*rows + 1):

        

        if random:

            img_x = np.random.randint(0, len(imgs_set))

        else:

            img_x = i-1

        

        img = imgs_set[img_x]

        ax = fig.add_subplot(rows, columns, i)

        ax.set_title(str(labels_set[img_x]))

        plt.axis('off')

        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.show()
wildlife_filter = ['black_bear', 'bald_eagle', 'cougar', 'elk', 'gray_wolf']



for folder in os.scandir('../input/oregon_wildlife/oregon_wildlife'):

    if folder.name in wildlife_filter:

        shutil.copytree(folder.path, './oregon_wildlife/{}'.format(folder.name))

        print('{} copied from main data set'.format(folder.name))
!ls -l oregon_wildlife
data_set = 'wildlife'

data_set_folder = './oregon_wildlife'

img_height = 128 

img_width = 128

batch_size = 32

nb_epochs = 50
def model_def(img_height, img_width):

    return Sequential([

        layers.Conv2D(filters=128, kernel_size=(4, 4), activation='tanh', input_shape=(img_height, img_width, 3)),

        layers.MaxPool2D(pool_size=(2,2)),

        layers.Dropout(0.22018745727040784),

        layers.Conv2D(filters=64, kernel_size=(4, 4), activation='relu'),

        layers.MaxPool2D(pool_size=(2,2)),

        layers.Dropout(0.02990527559235584),

        layers.Conv2D(filters=32, kernel_size=(4, 4), activation='tanh'),

        layers.MaxPool2D(pool_size=(2,2)),

        layers.Dropout(0.0015225556862044631),

        layers.Conv2D(filters=32, kernel_size=(4, 4), activation='tanh'),

        layers.MaxPool2D(pool_size=(2,2)),

        layers.Dropout(0.1207251417283281),

        layers.Flatten(),

        layers.Dense(256, activation='relu'),

        layers.Dropout(0.4724418446300173),

        layers.Dense(len(wildlife_filter), activation='softmax')

    ])
image_datagen = ImageDataGenerator(

    rescale=1./255,

    validation_split=0.2 # set validation split

)



train_generator = image_datagen.flow_from_directory(

    data_set_folder,

    target_size=(img_height, img_width),

    batch_size=batch_size,

    class_mode='categorical',

    subset='training' # set as training data

)



validation_generator = image_datagen.flow_from_directory(

    data_set_folder, # same directory as training data

    target_size=(img_height, img_width),

    batch_size=batch_size, # tried to set 703 manually to compare apples with apples but performance was way worse

    class_mode='categorical',

    subset='validation' # set as validation data

)
model = model_def(img_height, img_width)

model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
!free -m
%%time

model.fit_generator(

    train_generator,

    steps_per_epoch=train_generator.samples // batch_size,

    validation_data=validation_generator, 

    validation_steps=validation_generator.samples // batch_size,

    epochs = nb_epochs

)
images = Images(data_set, data_set_folder, config=['resize=({},{})'.format(img_height, img_width), 'store', 'stream'])

# explore directory to see if the h5 file is there

!ls
# stream from h5 file

images = Images(config=['stream'])

images.load(data_set, '../input')
# split data set

images.split = 0.2

X_test, Y_test = images.test



# generator

images.minibatch = batch_size

gap_generator = images.minibatch
total_train_images = images.count - len(X_test)

n_classes = len(images.classes)



print('content:', os.listdir("./"))

print('time to preprocess the data set:', images.elapsed)

print('number of images in data set:', images.count)

print('classes:', images.classes)

print('data type:', images.dtype)
shutil.rmtree(data_set_folder) # delete images folder for kaggle kernel limitations

K.clear_session() # clean previous model

gc.collect()
model = model_def(img_height, img_width)

model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
!free -m
%%time

model.fit_generator(

    generator=gap_generator,

    validation_data=(X_test, Y_test),

    steps_per_epoch=total_train_images // batch_size,

    epochs=nb_epochs,

    verbose=1

)