from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import layers, models, Model, Sequential

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from tqdm import tqdm

from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping,ModelCheckpoint

import tensorflow as tf

import json

import os

import cv2

tqdm.pandas()
EPOCHS = 20

SAMPLE_LEN = 600

IMAGE_PATH = "../input/plant-pathology-2020-fgvc7/images/"

TEST_PATH = "../input/plant-pathology-2020-fgvc7/test.csv"

TRAIN_PATH = "../input/plant-pathology-2020-fgvc7/train.csv"

SUB_PATH = "../input/plant-pathology-2020-fgvc7/sample_submission.csv"



sub = pd.read_csv(SUB_PATH)

test_data = pd.read_csv(TEST_PATH)

train_data = pd.read_csv(TRAIN_PATH)
train_data.head(100)
test_data.head()
def load_image(image_id):

    file_path = image_id + ".jpg"

    image = cv2.imread(IMAGE_PATH + file_path)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



train_images = train_data["image_id"][:SAMPLE_LEN].progress_apply(load_image)
def visualize_leaves(cond=[0, 0, 0, 0], cond_cols=["healthy"], is_cond=True):

    if not is_cond:

        cols, rows = 3, min([3, len(train_images)//3])

        fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(30, rows*20/3))

        for col in range(cols):

            for row in range(rows):

                ax[row, col].imshow(train_images.loc[train_images.index[-row*3-col-1]])

        return None

        

    cond_0 = "healthy == {}".format(cond[0])

    cond_1 = "scab == {}".format(cond[1])

    cond_2 = "rust == {}".format(cond[2])

    cond_3 = "multiple_diseases == {}".format(cond[3])

    

    cond_list = []

    for col in cond_cols:

        if col == "healthy":

            cond_list.append(cond_0)

        if col == "scab":

            cond_list.append(cond_1)

        if col == "rust":

            cond_list.append(cond_2)

        if col == "multiple_diseases":

            cond_list.append(cond_3)

    

    data = train_data.iloc[:600]

    for cond in cond_list:

        data = data.query(cond)

        

    images = train_images.iloc[list(data.index)]

    cols, rows = 3, min([3, len(images)//3])

    

    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(30, rows*20/3))

    for col in range(cols):

        for row in range(rows):

            ax[row, col].imshow(images.loc[images.index[row*3+col]])

    plt.show()
visualize_leaves(cond=[1, 0, 0, 0], cond_cols=["healthy"])
visualize_leaves(cond=[0, 1, 0, 0], cond_cols=["scab"])
visualize_leaves(cond=[0, 0, 1, 0], cond_cols=["rust"])
visualize_leaves(cond=[0, 0, 0, 1], cond_cols=["multiple_diseases"])
from sklearn.model_selection import train_test_split
def format_path(st):

    return '../input/plant-pathology-2020-fgvc7' + '/images/' + st + '.jpg'



test_paths = test_data.image_id.apply(format_path).values

train_paths = train_data.image_id.apply(format_path).values



train_labels = np.float32(train_data.loc[:, 'healthy':'scab'].values)

train_paths, valid_paths, train_labels, valid_labels =train_test_split(train_paths, train_labels, test_size=0.15, random_state=2020)
def decode_image(filename, label=None, image_size=(334, 334)):

    bits = tf.io.read_file(filename)

    image = tf.image.decode_jpeg(bits, channels=3)

    image = tf.cast(image, tf.float32) / 255.0

    image = tf.image.resize(image, image_size)

    

    if label is None:

        return image

    else:

        return image, label



def data_augment(image, label=None):

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_flip_up_down(image)

    

    if label is None:

        return image

    else:

        return image, label
BATCH_SIZE = 32
train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((train_paths,train_labels))

    .map(decode_image, num_parallel_calls=4)

    .map(data_augment, num_parallel_calls=4)

    .repeat()

    .shuffle(len(train_labels))

    .batch(BATCH_SIZE)

    .prefetch(1)

)



valid_dataset = (

    tf.data.Dataset

    .from_tensor_slices((valid_paths, valid_labels))

    .map(decode_image, num_parallel_calls=4)

    .batch(BATCH_SIZE)

    .cache()

    .prefetch(1)

)



test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(test_paths)

    .map(decode_image, num_parallel_calls=4)

    .batch(BATCH_SIZE)

)
def lrfn(epoch):

    LR_START = 0.00001

    LR_MAX = 0.0004

    LR_MIN = 0.00001

    LR_RAMPUP_EPOCHS = 5

    LR_SUSTAIN_EPOCHS = 0

    LR_EXP_DECAY = .8

    

    if epoch < LR_RAMPUP_EPOCHS:

        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START

    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:

        lr = LR_MAX

    else:

        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN

    return lr



rng = [i for i in range(EPOCHS)]

y = [lrfn(x) for x in rng]

plt.plot(rng, y)

print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
model = tf.keras.Sequential([tf.keras.applications.InceptionResNetV2(input_shape=(334,334,3),

                                             weights='imagenet',

                                             include_top=False),

                            tf.keras.layers.GlobalAveragePooling2D(),

                            tf.keras.layers.Dense(train_labels.shape[1],

                                         activation='softmax')])

        

model.compile(optimizer='adam',

              loss = 'categorical_crossentropy',

              metrics=['accuracy'])

model.summary()
TRAIN_STEPS_PER_EPOCH = train_labels.shape[0] // BATCH_SIZE

VALID_STEPS_PER_EPOCH = valid_labels.shape[0] // BATCH_SIZE

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
history = model.fit(x=train_dataset,

                    steps_per_epoch=TRAIN_STEPS_PER_EPOCH,

                    epochs = EPOCHS,

                    validation_data=valid_dataset,

                    validation_steps=VALID_STEPS_PER_EPOCH,

                    callbacks=[lr_schedule])
# plot loss and accuracy image

history_dict = history.history

train_loss = history_dict["loss"]

train_accuracy = history_dict["accuracy"]

val_loss = history_dict["val_loss"]

val_accuracy = history_dict["val_accuracy"]



# figure 1

plt.figure()

plt.plot(range(EPOCHS), train_loss, label='train_loss')

plt.plot(range(EPOCHS), val_loss, label='val_loss')

plt.legend()

plt.xlabel('epochs')

plt.ylabel('loss')



# figure 2

plt.figure()

plt.plot(range(EPOCHS), train_accuracy, label='train_accuracy')

plt.plot(range(EPOCHS), val_accuracy, label='val_accuracy')

plt.legend()

plt.xlabel('epochs')

plt.ylabel('accuracy')

plt.show()
from PIL import Image

import numpy as np
dict = {0:'healthy',1:'multiple_diseases',2:'rust',3:'scab'}
img = []

image = []

img1 = Image.open("../input/plant-pathology-2020-fgvc7/images/Test_0.jpg")

img.append(img1)

img2 = Image.open("../input/plant-pathology-2020-fgvc7/images/Test_1.jpg")

img.append(img2)

img3 = Image.open("../input/plant-pathology-2020-fgvc7/images/Test_2.jpg")

img.append(img3)

img4 = Image.open("../input/plant-pathology-2020-fgvc7/images/Test_3.jpg")

img.append(img4)
for i in range(0,len(img)):

    image.append(img[i])

    img[i] = img[i].resize((334,334))

    img[i] = np.array(img[i]) / 255.

    img[i] = (np.expand_dims(img[i], 0))

    result = np.squeeze(model.predict(img[i]))

    predict_class = np.argmax(result)

    plt.subplot(2,2,i+1)

    plt.title([dict[int(predict_class)],result[predict_class]])

    plt.imshow(image[i])

    plt.xticks([])

    plt.yticks([])