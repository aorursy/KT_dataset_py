import os,sys,random,warnings

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf



from tqdm import tqdm

from itertools import chain



from skimage.morphology import label

from skimage.transform import resize

from skimage.io import imread,imshow,imread_collection,concatenate_images



from keras.models import Sequential,Input,Model,load_model

from keras.utils.vis_utils import model_to_dot

from keras.utils.vis_utils import plot_model

from IPython.display import SVG

from keras.layers import Input

from keras.layers import Dropout,Lambda,Conv2D,Conv2DTranspose,MaxPooling2D,concatenate,SeparableConv2D

from keras.layers import Dense,Flatten,Dropout,Concatenate,GlobalAveragePooling2D

from keras.utils.vis_utils import model_to_dot

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam,SGD

from keras.utils.vis_utils import plot_model

from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard,CSVLogger,ReduceLROnPlateau,LearningRateScheduler

from keras import backend as K
def run_length_encoding(x):

    dots = np.where(x.T.flatten() == 1)[0]

    run_lengths = []

    prev = -2

    for b in dots:

        if (b>prev+1): run_lengths.extend((b + 1, 0))

        run_lengths[-1] += 1

        prev = b

    return run_lengths



def prob_to_rles(x, cutoff=0.5):

    lab_img = label(x > cutoff)

    for i in range(1, lab_img.max() + 1):

        yield run_length_encoding(lab_img == i)

def show_final_history(history):

    fig, ax = plt.subplots(1, 2, figsize=(15,5))

    ax[0].set_title('loss')

    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")

    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")

    ax[1].set_title('iou')

    ax[1].plot(history.epoch, history.history["iou"], label="Train iou")

    ax[1].plot(history.epoch, history.history["val_iou"], label="Validation iou")

    ax[0].legend()

    ax[1].legend()
def iou(y_true, y_pred):

    prec = []

    for t in np.arange(0.5, 1.0, 0.05):

        y_pred_ = tf.to_int32(y_pred > t)

        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)

        K.get_session().run(tf.local_variables_initializer())

        with tf.control_dependencies([up_opt]):

            score = tf.identity(score)

        prec.append(score)

    return K.mean(K.stack(prec), axis=0)
IMG_WIDTH = 128

IMG_HEIGHT = 128

IMG_CHANNELS = 3

train_path = '../input/stage1_train/'

test_path = '../input/stage1_test/'

train_ids = next(os.walk(train_path))[1]

test_ids = next(os.walk(test_path))[1]

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

opt = SGD(lr=1e-4,momentum=0.95)
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

print('Extract and Transform training images and masks')

sys.stdout.flush()

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):

    path = train_path + id_

    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]

    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    X_train[n] = img

    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

    for mask_file in next(os.walk(path + '/masks/'))[2]:

        mask_ = imread(path + '/masks/' + mask_file)

        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 

                                      preserve_range=True), axis=-1)

        mask = np.maximum(mask, mask_)

    Y_train[n] = mask



# Get and resize test images

X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

sizes_test = []

print('Extracting and Transforming test images')

sys.stdout.flush()

for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):

    path = test_path + id_

    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]

    sizes_test.append([img.shape[0], img.shape[1]])

    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    X_test[n] = img
inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

s = Lambda(lambda x: x / 255) (inputs)



c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)

c1 = Dropout(0.1) (c1)

c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)

p1 = MaxPooling2D((2, 2)) (c1)



c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)

c2 = Dropout(0.1) (c2)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)

p2 = MaxPooling2D((2, 2)) (c2)



c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)

c3 = Dropout(0.2) (c3)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)

p3 = MaxPooling2D((2, 2)) (c3)



c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)

c4 = Dropout(0.2) (c4)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)

p4 = MaxPooling2D(pool_size=(2, 2)) (c4)



c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)

c5 = Dropout(0.3) (c5)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)



u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)

u6 = concatenate([u6, c4])

c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)

c6 = Dropout(0.2) (c6)

c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)



u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)

u7 = concatenate([u7, c3])

c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)

c7 = Dropout(0.2) (c7)

c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)



u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)

u8 = concatenate([u8, c2])

c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)

c8 = Dropout(0.1) (c8)

c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)



u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)

u9 = concatenate([u9, c1], axis=3)

c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)

c9 = Dropout(0.1) (c9)

c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)



outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)



model = Model(inputs=[inputs], outputs=[outputs])

model.summary()



SVG(model_to_dot(model).create(prog='dot', format='svg'))

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True, expand_nested=True)
checkpoint = ModelCheckpoint(

    './base.model',

    monitor='val_loss',

    verbose=1,

    save_best_only=True,

    mode='min',

    save_weights_only=False,

    period=1

)

earlystop = EarlyStopping(

    monitor='val_loss',

    min_delta=0.001,

    patience=80,

    verbose=1,

    mode='auto'

)

tensorboard = TensorBoard(

    log_dir = './logs',

    histogram_freq=0,

    batch_size=16,

    write_graph=True,

    write_grads=True,

    write_images=False,

)



csvlogger = CSVLogger(

    filename= "training_csv.log",

    separator = ",",

    append = False

)



#lrsched = LearningRateScheduler(step_decay,verbose=1)



reduce = ReduceLROnPlateau(

    monitor='val_loss',

    factor=0.3,

    patience=5,

    verbose=1, 

    mode='auto',

    cooldown=1 

)



callbacks = [checkpoint,tensorboard,csvlogger,reduce,earlystop]
opt1 = Adam(lr=3e-4)

model.compile(optimizer=opt1, loss='binary_crossentropy', metrics=[iou])



history = model.fit(X_train, Y_train, validation_split=0.1, batch_size=8, epochs=200, 

                    callbacks=callbacks)
show_final_history(history)

print("Validation Loss: " + str(history.history['val_loss'][-1:]))
model = load_model('./base.model', custom_objects={'iou': iou})

preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)

preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)

preds_test = model.predict(X_test, verbose=1)



# Threshold predictions

preds_train_t = (preds_train > 0.5).astype(np.uint8)

preds_val_t = (preds_val > 0.5).astype(np.uint8)

preds_test_t = (preds_test > 0.5).astype(np.uint8)



# Create list of upsampled test masks

preds_test_upsampled = []

for i in range(len(preds_test)):

    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]),(sizes_test[i][0], sizes_test[i][1]), mode='constant', preserve_range=True))
ix = random.randint(0, len(preds_train_t))

imshow(X_train[ix])

plt.show()

imshow(np.squeeze(Y_train[ix]))

plt.show()

imshow(np.squeeze(preds_train_t[ix]))

plt.show()
new_test_ids = []

rles = []

for n, id_ in enumerate(test_ids):

    rle = list(prob_to_rles(preds_test_upsampled[n]))

    rles.extend(rle)

    new_test_ids.extend([id_] * len(rle))
sub = pd.DataFrame()

sub['ImageId'] = new_test_ids

sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))

sub.to_csv('data_bowl_segmentation.csv', index=False)