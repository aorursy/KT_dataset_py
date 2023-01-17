import os

print(os.listdir("../input"))
import pandas as pd

import numpy as np

import os, sys

import matplotlib.pyplot as plt

import skimage.io



from skimage.transform import resize

from imgaug import augmenters as iaa

from tqdm import tqdm

import PIL 



from PIL import Image, ImageOps

import cv2



from sklearn.utils import class_weight, shuffle

from keras.losses import binary_crossentropy



from keras.applications.resnet50 import preprocess_input

import keras.backend as K



import tensorflow as tf

from sklearn.metrics import f1_score, fbeta_score

from keras.utils import Sequence



from keras.utils import to_categorical

from sklearn.model_selection import train_test_split



WORKERS = 8

CHANNEL = 3



import warnings

warnings.filterwarnings("ignore")



#IMG_SIZE = 512

NUM_CLASSES = 5

SEED = 77

TRAIN_NUM = 1000



df_test  = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")

df_test.head()
import matplotlib.pyplot as plt

import skimage.io

from skimage.transform import resize

from imgaug import augmenters as iaa

from tqdm import tqdm

import PIL

from PIL import Image, ImageOps

import cv2

from sklearn.utils import class_weight, shuffle

from keras.losses import binary_crossentropy, categorical_crossentropy

from keras.applications.resnet50 import preprocess_input

import keras.backend as K

import tensorflow as tf

from sklearn.metrics import f1_score, fbeta_score, cohen_kappa_score, accuracy_score

from keras.utils import Sequence

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split



WORKERS = 12

CHANNEL = 3



import warnings

warnings.filterwarnings("ignore")

SIZE = 300

NUM_CLASSES = 5
from keras.legacy import interfaces

from keras.optimizers import Optimizer

from keras import backend as K





class AdamAccumulate_v1(Optimizer):

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,

                 epsilon=None, decay=0., amsgrad=False, accum_iters=20, **kwargs):

        super(AdamAccumulate, self).__init__(**kwargs)

        with K.name_scope(self.__class__.__name__):

            self.iterations = K.variable(0, dtype='int64', name='iterations')

            self.effective_iterations = K.variable(0, dtype='int64', name='effective_iterations')



            self.lr = K.variable(lr, name='lr')

            self.beta_1 = K.variable(beta_1, name='beta_1')

            self.beta_2 = K.variable(beta_2, name='beta_2')

            self.decay = K.variable(decay, name='decay')

        if epsilon is None:

            epsilon = K.epsilon()

        self.epsilon = epsilon

        self.initial_decay = decay

        self.amsgrad = amsgrad

        self.accum_iters = K.variable(accum_iters, dtype='int64')



    @interfaces.legacy_get_updates_support

    def get_updates(self, loss, params):

        grads = self.get_gradients(loss, params)



        self.updates = [K.update(self.iterations, self.iterations + 1)]



        flag = K.equal(self.iterations % self.accum_iters, self.accum_iters - 1)

        flag = K.cast(flag, K.floatx())



        self.updates.append(K.update(self.effective_iterations,

                                     self.effective_iterations + K.cast(flag, 'int64')))



        lr = self.lr

        if self.initial_decay > 0:

            lr = lr * (1. / (1. + self.decay * K.cast(self.effective_iterations,

                                                      K.dtype(self.decay))))



        t = K.cast(self.effective_iterations, K.floatx()) + 1



        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /

                     (1. - K.pow(self.beta_1, t)))



        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        gs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]



        if self.amsgrad:

            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        else:

            vhats = [K.zeros(1) for _ in params]

        self.weights = [self.iterations] + ms + vs + vhats



        for p, g, m, v, vhat, gg in zip(params, grads, ms, vs, vhats, gs):



            gg_t = (1 - flag) * (gg + g)

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * (gg + flag * g) / K.cast(self.accum_iters, K.floatx())

            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(

                (gg + flag * g) / K.cast(self.accum_iters, K.floatx()))



            if self.amsgrad:

                vhat_t = K.maximum(vhat, v_t)

                p_t = p - flag * lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)

                self.updates.append(K.update(vhat, vhat_t))

            else:

                p_t = p - flag * lr_t * m_t / (K.sqrt(v_t) + self.epsilon)



            self.updates.append((m, flag * m_t + (1 - flag) * m))

            self.updates.append((v, flag * v_t + (1 - flag) * v))

            self.updates.append((gg, gg_t))

            new_p = p_t



            # Apply constraints.

            if getattr(p, 'constraint', None) is not None:

                new_p = p.constraint(new_p)



            self.updates.append(K.update(p, new_p))

        return self.updates



    def get_config(self):

        config = {'lr': float(K.get_value(self.lr)),

                  'beta_1': float(K.get_value(self.beta_1)),

                  'beta_2': float(K.get_value(self.beta_2)),

                  'decay': float(K.get_value(self.decay)),

                  'epsilon': self.epsilon,

                  'amsgrad': self.amsgrad}

        base_config = super(AdamAccumulate, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
class AdamAccumulate(Optimizer):

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,

                 epsilon=None, decay=0., amsgrad=False, accum_iters=2, **kwargs):

        if accum_iters < 1:

            raise ValueError('accum_iters must be >= 1')

        super(AdamAccumulate, self).__init__(**kwargs)

        with K.name_scope(self.__class__.__name__):

            self.iterations = K.variable(0, dtype='int64', name='iterations')

            self.lr = K.variable(lr, name='lr')

            self.beta_1 = K.variable(beta_1, name='beta_1')

            self.beta_2 = K.variable(beta_2, name='beta_2')

            self.decay = K.variable(decay, name='decay')

        if epsilon is None:

            epsilon = K.epsilon()

        self.epsilon = epsilon

        self.initial_decay = decay

        self.amsgrad = amsgrad

        self.accum_iters = K.variable(accum_iters, K.dtype(self.iterations))

        self.accum_iters_float = K.cast(self.accum_iters, K.floatx())



    @interfaces.legacy_get_updates_support

    def get_updates(self, loss, params):

        grads = self.get_gradients(loss, params)

        self.updates = [K.update_add(self.iterations, 1)]



        lr = self.lr



        completed_updates = K.cast(K.tf.floor(self.iterations / self.accum_iters), K.floatx())



        if self.initial_decay > 0:

            lr = lr * (1. / (1. + self.decay * completed_updates))



        t = completed_updates + 1



        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) / (1. - K.pow(self.beta_1, t)))



        # self.iterations incremented after processing a batch

        # batch:              1 2 3 4 5 6 7 8 9

        # self.iterations:    0 1 2 3 4 5 6 7 8

        # update_switch = 1:        x       x    (if accum_iters=4)

        update_switch = K.equal((self.iterations + 1) % self.accum_iters, 0)

        update_switch = K.cast(update_switch, K.floatx())



        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        gs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]



        if self.amsgrad:

            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        else:

            vhats = [K.zeros(1) for _ in params]



        self.weights = [self.iterations] + ms + vs + vhats



        for p, g, m, v, vhat, tg in zip(params, grads, ms, vs, vhats, gs):



            sum_grad = tg + g

            avg_grad = sum_grad / self.accum_iters_float



            m_t = (self.beta_1 * m) + (1. - self.beta_1) * avg_grad

            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(avg_grad)



            if self.amsgrad:

                vhat_t = K.maximum(vhat, v_t)

                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)

                self.updates.append(K.update(vhat, (1 - update_switch) * vhat + update_switch * vhat_t))

            else:

                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)



            self.updates.append(K.update(m, (1 - update_switch) * m + update_switch * m_t))

            self.updates.append(K.update(v, (1 - update_switch) * v + update_switch * v_t))

            self.updates.append(K.update(tg, (1 - update_switch) * sum_grad))

            new_p = p_t



            # Apply constraints.

            if getattr(p, 'constraint', None) is not None:

                new_p = p.constraint(new_p)



            self.updates.append(K.update(p, (1 - update_switch) * p + update_switch * new_p))

        return self.updates



    def get_config(self):

        config = {'lr': float(K.get_value(self.lr)),

                  'beta_1': float(K.get_value(self.beta_1)),

                  'beta_2': float(K.get_value(self.beta_2)),

                  'decay': float(K.get_value(self.decay)),

                  'epsilon': self.epsilon,

                  'amsgrad': self.amsgrad}

        base_config = super(AdamAccumulate, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
class My_Generator(Sequence):



    def __init__(self, image_filenames, labels, batch_size, mix=False, is_train=False):

        self.image_filenames, self.labels = image_filenames, labels

        self.batch_size = batch_size

        self.is_train = is_train

        if(self.is_train):

            self.on_epoch_end()

        self.is_mix = mix



    def __len__(self):

        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))



    def __getitem__(self, idx):

        

        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]



        if(self.is_train):

            return self.train_generate(batch_x, batch_y)

        return self.valid_generate(batch_x, batch_y)

    

    def on_epoch_end(self):

        if(self.is_train):

            self.image_filenames, self.labels = shuffle(self.image_filenames, self.labels)

    

    def mix_up(self, x, y):

        lam = np.random.beta(0.2, 0.4)

        ori_index = np.arange(int(len(x)))

        index_array = np.arange(int(len(x)))

        np.random.shuffle(index_array)        

        

        mixed_x = lam * x[ori_index] + (1 - lam) * x[index_array]

        mixed_y = lam * y[ori_index] + (1 - lam) * y[index_array]

        

        return mixed_x, mixed_y

    

    def train_generate(self, batch_x, batch_y):

        batch_images = []

        for (sample, label) in zip(batch_x, batch_y):

            path = f"{sample}"

            

            image = cv2.imread(path)

            #path=f"../data/older_data/diabetic-retinopathy-resized/resized_train/{sample}.jpeg"

            image = load_ben_color(path,sigmaX=30)            

            batch_images.append(cv2.resize(image, (SIZE,SIZE), interpolation=cv2.INTER_CUBIC))



        batch_images = np.array(batch_images, np.float32) / 255

        batch_y = np.array(batch_y, np.float32)

        

        if(self.is_mix):

            batch_images, batch_y = self.mix_up(batch_images, batch_y)

            

        return batch_images, batch_y

    

    def valid_generate(self, batch_x, batch_y):

        batch_images = []

        for (sample, label) in zip(batch_x, batch_y):

            path = f"{sample}"

            

            image = cv2.imread(path)

            #path=f"../data/older_data/diabetic-retinopathy-resized/resized_train/{sample}.jpeg"

            image = load_ben_color(path,sigmaX=30)



                

            batch_images.append(cv2.resize(image, (SIZE,SIZE), interpolation=cv2.INTER_CUBIC))



        batch_images = np.array(batch_images, np.float32) / 255

        batch_y = np.array(batch_y, np.float32)

        return batch_images, batch_y
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential, load_model

from keras.layers import (Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D,

                          BatchNormalization, Input, Conv2D, GlobalAveragePooling2D)

from keras.applications.resnet50 import ResNet50

from keras.callbacks import ModelCheckpoint

from keras import metrics

from keras.optimizers import Adam 

from keras import backend as K

import keras

from keras.models import Model
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
# create callbacks list

from keras.callbacks import (ModelCheckpoint, LearningRateScheduler,

                             EarlyStopping, ReduceLROnPlateau,CSVLogger)



epochs = 30; batch_size = 32

checkpoint = ModelCheckpoint('../working/Resnet50.h5', monitor='val_loss', verbose=1, 

                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, 

                                   verbose=1, mode='min', epsilon=0.0001)

early = EarlyStopping(monitor="val_loss", 

                      mode="min", 

                      patience=9)



csv_logger = CSVLogger(filename='../working/training_log.csv',

                       separator=',',

                       append=True)



model = create_model(

    input_shape=(SIZE,SIZE,3), 

    n_out=NUM_CLASSES)
submit = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

# model.load_weights('../working/Resnet50.h5')

model.load_weights('../input/resnet50trainedwithaptosolddataset/Resnet50_bestqwk.h5')

predicted = []
model.compile(loss='categorical_crossentropy',

            # loss=kappa_loss,

            # loss='binary_crossentropy',

            # optimizer=Adam(lr=1e-4),

            optimizer=AdamAccumulate(lr=1e-4, accum_iters=2),

            metrics=['accuracy'])
model.summary()
def crop_image1(img,tol=7):

    # img is image data

    # tol  is tolerance

        

    mask = img>tol

    return img[np.ix_(mask.any(1),mask.any(0))]



def crop_image_from_gray(img,tol=7):

    if img.ndim ==2:

        mask = img>tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    elif img.ndim==3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        mask = gray_img>tol

        

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0): # image is too dark so that we crop out everything,

            return img # return original image

        else:

            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

    #         print(img1.shape,img2.shape,img3.shape)

            img = np.stack([img1,img2,img3],axis=-1)

    #         print(img.shape)

        return img
def load_ben_color(path, sigmaX=10 ):

    try:

        image = cv2.imread(path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = crop_image_from_gray(image)

        image = cv2.resize(image, (SIZE, SIZE))

        image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)



        return image

    except cv2.error as e:

        print(e)

        print(path)

for i, name in tqdm(enumerate(submit['id_code'])):

    path = os.path.join('../input/aptos2019-blindness-detection/test_images/', name+'.png')

    image = cv2.imread(path)

    image = load_ben_color(path,sigmaX=30)

    image = cv2.resize(image, (SIZE,SIZE), interpolation=cv2.INTER_CUBIC)

    score_predict = model.predict((image[np.newaxis])/255)

    label_predict = np.argmax(score_predict)

    # label_predict = score_predict.astype(int).sum() - 1

    predicted.append(str(label_predict))
submit['diagnosis'] = predicted

submit.to_csv('submission.csv', index=False)

submit.head()