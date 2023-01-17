# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import cv2

# Any results you write to the current directory are saved as output.
import os

os.getcwd()

from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline

import cv2
train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')
IMG_SIZE = 512

NB_CHANNELS = 3

MAX_TRAIN_STEPS = 1000

BATCH_SIZE = 32

NB_EPOCHS = 40

weight_path = './'



from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

x_train,x_test,y_train,y_test = train_test_split(train['id_code'].values,train['diagnosis'].values,test_size=0.1,random_state=42)

train_image_dir = '../input/aptos2019-blindness-detection/train_images/'

def make_image_gen(img_file_list,class_list, batch_size = 4):

    all_batches = img_file_list

    all_classes = class_list

    out_rgb = []

    yield_rgb = []

    yield_label = []

    out_label = []

    

    

    while True:

        #np.random.shuffle(all_batches)

        out_rgb = []

        out_label = []

        for idx, c_img_id in enumerate(all_batches):

            imgname  = c_img_id + '.png'

            c_img = cv2.imread(os.path.join(train_image_dir,imgname))

            c_img = cv2.cvtColor(c_img, cv2.COLOR_BGR2HSV)

            c_img = cv2.resize(c_img,(IMG_SIZE,IMG_SIZE),interpolation = cv2.INTER_AREA)

            

            label = class_list[idx]

            out_rgb += [c_img]

            out_label += [label]

            if len(out_rgb)>=batch_size:

                yield_rgb = out_rgb

                yield_label = out_label

                out_rgb = []

                out_label = []

                #print("size",sys.getsizeof(out_rgb))

                yield np.stack(yield_rgb, 0)/255.0, to_categorical(np.stack(yield_label, 0),num_classes=5)


train_gen = make_image_gen(x_train,y_train,4)

train_x, train_y = next(train_gen)

print('x', train_x.shape, train_x.min(), train_x.max())

print('y', train_y.shape, train_y.min(), train_y.max())
valid_x, valid_y = next(make_image_gen(x_test,y_test,4))

print(valid_x.shape, valid_y.shape)
from keras.models import Sequential, Model

from keras.layers import Conv2D, MaxPooling2D , Input, GlobalAveragePooling2D

from keras.layers import Activation, Flatten, Dense, Dropout

from keras.preprocessing.image import ImageDataGenerator

from keras.applications.resnet50 import ResNet50

from keras.optimizers import Adam

#from keras.applications.resnet50 import preprocess_input, decode_predictions

import time



from keras.preprocessing.image import ImageDataGenerator

from keras.utils import to_categorical



dg_args = dict(featurewise_center = False, 

                  samplewise_center = False,

                  rotation_range = 15, 

                  width_shift_range = 0.1, 

                  height_shift_range = 0.1, 

                  shear_range = 0.01,

                  zoom_range = [0.9, 1.25],  

                  horizontal_flip = True, 

                  vertical_flip = False,

                  fill_mode = 'reflect',

                   data_format = 'channels_last')



image_gen = ImageDataGenerator(**dg_args)





def create_aug_gen(in_gen, seed = None):

    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))

    for in_x, in_y in in_gen:

        seed = np.random.choice(range(9999))

        # keep the seeds syncronized otherwise the augmentation to the images is different from the masks

        g_x = image_gen.flow(255*in_x, 

                             batch_size = in_x.shape[0], 

                             seed = seed, 

                             shuffle=True)



        g_y = in_y

        yield next(g_x)/255.0 , g_y
train_gen = make_image_gen(x_train,y_train,4)

cur_gen = create_aug_gen(train_gen)

t_x, t_y = next(cur_gen)

print('x', t_x.shape, t_x.dtype, t_x.min(), t_x.max())

print('y', t_y.shape, t_y.dtype, t_y.min(), t_y.max())
#https://www.kaggle.com/mathormad/aptos-resnet50-baseline

function = "softmax"

def create_model(input_shape, n_out):

    input_tensor = Input(shape=input_shape)

    base_model = ResNet50(include_top=False,

                   weights=None,

                   input_tensor=input_tensor)

    base_model.load_weights('../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

    x = GlobalAveragePooling2D()(base_model.output)

    x = Dropout(0.5)(x)

    x = Dense(1024, activation='relu')(x)

    x = Dropout(0.5)(x)

    final_output = Dense(n_out, activation=function, name='final_output')(x)

    model = Model(input_tensor, final_output)    

    return model
input_shape = (IMG_SIZE,IMG_SIZE,NB_CHANNELS)

n_out = 5

model = create_model(input_shape,n_out)

#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
from keras.optimizers import Nadam, adadelta,adagrad,adam,RMSprop,SGD

for layer in model.layers:

    layer.trainable = False



for i in range(-5,0):

    model.layers[i].trainable = True



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()




from keras.callbacks import EarlyStopping , ReduceLROnPlateau , ModelCheckpoint

step_count = min(MAX_TRAIN_STEPS, len(x_train)//BATCH_SIZE)

train_gen = make_image_gen(x_train,y_train,BATCH_SIZE)

aug_gen = create_aug_gen(train_gen)

valid_gen = make_image_gen(x_test,y_test,2)





#earlyStopper = EarlyStopping(monitor="acc", mode="max", patience=15)

#checkPointer = ModelCheckpoint(weight_path, monitor='acc', verbose=1, 

#                             save_best_only=True, mode='max', save_weights_only = True)

checkpoint = ModelCheckpoint('../working/aptos.h5', monitor='val_loss', verbose=1, 

                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, 

                                   verbose=1, mode='min', min_delta=0.0001)

early = EarlyStopping(monitor="val_loss", 

                      mode="min", 

                      patience=9)

callbacks_list = [checkpoint, reduceLROnPlat, early]

model.fit_generator(aug_gen,

                            steps_per_epoch=step_count, 

                            epochs=NB_EPOCHS, 

                            validation_data=valid_gen,

                            validation_steps=len(x_test)//BATCH_SIZE, 

                            callbacks=callbacks_list,

                            workers=1)
#https://www.kaggle.com/mathormad/aptos-resnet50-baseline

from tqdm import tqdm

submit = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

# model.load_weights('../working/Resnet50.h5')

model.load_weights('../working/aptos.h5')

predicted = []

for i, name in tqdm(enumerate(submit['id_code'])):

    path = os.path.join('../input/aptos2019-blindness-detection/test_images/', name+'.png')

    image = cv2.imread(path)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    score_predict = model.predict((image[np.newaxis])/255)

    label_predict = np.argmax(score_predict)

    # label_predict = score_predict.astype(int).sum() - 1

    predicted.append(str(label_predict))

submit['diagnosis'] = predicted

submit.to_csv('submission.csv', index=False)

submit.head()