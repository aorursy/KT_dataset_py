!pip install -q efficientnet
import math, re, os

import tensorflow as tf

from tensorflow.keras import layers 

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from kaggle_datasets import KaggleDatasets

import efficientnet.tfkeras as efn

from tensorflow.keras.applications import DenseNet201

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix



print("Tensorflow version " + tf.__version__)
os.listdir("/kaggle/input/pneumonia-tfrecord/")
# Detect hardware, return appropriate distribution strategy

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.



print("REPLICAS: ", strategy.num_replicas_in_sync)
AUTO = tf.data.experimental.AUTOTUNE



# Create strategy from tpu

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu)

tf.tpu.experimental.initialize_tpu_system(tpu)

strategy = tf.distribute.experimental.TPUStrategy(tpu)



# Data access

GCS_DS_PATH = KaggleDatasets().get_gcs_path("pneumonia-tfrecord")



# Configuration

IMAGE_SIZE = [512, 512]##### remember: 512 x 512

EPOCHS = 20

BATCH_SIZE = 16 * strategy.num_replicas_in_sync ### 32x8=128
GCS_PATH = GCS_DS_PATH + '/pneumonia_tfrecord'

TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train/*.tfrecord')

VALIDATION_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/valid/*.tfrecord')

TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/test/*.tfrecord') 
print(TRAINING_FILENAMES)

print(VALIDATION_FILENAMES)

print(TEST_FILENAMES)
CLASSES=["class0","class1","class2"]
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

    ax.legend(['train', 'valid'])
def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range

    image = tf.image.resize(image, [*IMAGE_SIZE]) # explicit size needed for TPU

    return image



def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        "image_raw": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "label": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image_raw'])

    label = tf.cast(example['label'], tf.int32)

    return image, label # returns a dataset of (image, label) pairs



def read_unlabeled_tfrecord(example):

    UNLABELED_TFREC_FORMAT = {

        "image_raw": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "ID": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element

        # class is missing, this competitions's challenge is to predict flower classes for the test dataset

    }

    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)

    image = decode_image(example['image_raw'])

    idnum = example['ID']

    return image, idnum # returns a dataset of image(s)



def load_dataset(filenames, labeled=True, ordered=False):

    # Read from TFRecords. For optimal performance, reading from multiple files at once and

    # disregarding data order. Order does not matter since we will be shuffling the data anyway.



    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False # disable order, increase speed



    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files

    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order

    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)

    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False

    return dataset



def data_augment(image, label, seed=2020):

    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),

    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part

    # of the TPU while the TPU itself is computing gradients.

    image = tf.image.random_flip_left_right(image, seed=seed)

#     image = tf.image.random_flip_up_down(image, seed=seed)

#     image = tf.image.random_brightness(image, 0.1, seed=seed)

    

#     image = tf.image.random_jpeg_quality(image, 85, 100, seed=seed)

#     image = tf.image.resize(image, [530, 530])

#     image = tf.image.random_crop(image, [512, 512], seed=seed)

    #image = tf.image.random_saturation(image, 0, 2)

    return image, label   



def get_training_dataset():

    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)

    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

    dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_validation_dataset(ordered=False):

    dataset = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.cache()

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_train_valid_datasets():

    dataset = load_dataset(TRAINING_FILENAMES + VALIDATION_FILENAMES, labeled=True)

    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

    dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_test_dataset(ordered=False):

    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



# def count_data_items(filenames):

#     # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items

#     n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

#     return np.sum(n)
# data dump

print("Training data shapes:")

for image, label in get_training_dataset().take(3):

    print(image.numpy().shape, label.numpy().shape)

print("Training data label examples:", label.numpy())

print("Validation data shapes:")

for image, label in get_validation_dataset().take(3):

    print(image.numpy().shape, label.numpy().shape)

print("Validation data label examples:", label.numpy())

print("Test data shapes:")

for image, idnum in get_test_dataset().take(3):

    print(image.numpy().shape, idnum.numpy().shape)

print("Test data IDs:", idnum.numpy().astype('U')) # U=unicode string
# NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)

# NUM_VALIDATION_IMAGES = count_data_items(VALIDATION_FILENAMES)

# NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)

# STEPS_PER_EPOCH = (NUM_TRAINING_IMAGES + NUM_VALIDATION_IMAGES) // BATCH_SIZE

# print('Dataset: {} training images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES+NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES))
### hhhh set in person ###

#NUM_TRAINING_IMAGES = 5524

NUM_TRAINING_IMAGES = 18011

#NUM_VALIDATION_IMAGES = 616

NUM_VALIDATION_IMAGES = 2002

#NUM_TEST_IMAGES = 856

NUM_TEST_IMAGES = 6671

TOTAL_STEPS_PER_EPOCH = (NUM_TRAINING_IMAGES + NUM_VALIDATION_IMAGES) // BATCH_SIZE

PART_STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

print('Dataset: {} training images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES+NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES))
def lrfn(epoch):

    LR_START = 0.00001

    LR_MAX = 0.00005 * strategy.num_replicas_in_sync

    LR_MIN = 0.00001

    LR_RAMPUP_EPOCHS = 5

    LR_SUSTAIN_EPOCHS = 0

    LR_EXP_DECAY = .75

    

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
def freeze(model):

    for layer in model.layers:

        layer.trainable = False



def unfreeze(model):

    for layer in model.layers:

        layer.trainable = True
with strategy.scope():

    efficient_net = efn.EfficientNetB7(

        input_shape=(512, 512, 3),

        weights='imagenet',

        include_top=False

    )

    

    inp = layers.Input(shape=(512, 512, 3))

    x = efficient_net(inp)

    gap = layers.GlobalAveragePooling2D(name='GlobalAvgPool')(x)

    gap = layers.Dense(len(CLASSES), activation='linear')(gap)

    

    gmp = layers.GlobalMaxPooling2D(name='GlobalMaxPool')(x)

    gmp = layers.Dense(len(CLASSES), activation='linear')(gmp)

    

    out = layers.add([gap, gmp])

    out = layers.Activation('softmax')(out)

    

    model = tf.keras.Model(inputs=inp, outputs=out)

        

    model.compile(

        optimizer=tf.keras.optimizers.Adam(),

        loss = 'sparse_categorical_crossentropy',

        metrics=['sparse_categorical_accuracy']

    )

    model.summary()
# scheduler = tf.keras.callbacks.ReduceLROnPlateau(patience=3, verbose=1)

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)

#　mcp = tf.keras.callbacks.ModelCheckpoint(filepath='pneumonia_tpu_model_best.hdf5',

#                      monitor="val_sparse_categorical_accuracy", save_best_only=True, save_weights_only=False)

# history = model.fit(

#     get_train_valid_datasets(), 

#     steps_per_epoch=STEPS_PER_EPOCH,

#     epochs=EPOCHS, 

#     callbacks=[lr_schedule]

# )

# history = model.fit(

#     get_training_dataset(), 

#     steps_per_epoch=PART_STEPS_PER_EPOCH,

#     epochs=EPOCHS, 

#     validation_data=get_validation_dataset(),

#     callbacks=[lr_schedule,mcp]

# )

#################### callbacks:chkpt improvement
history = model.fit(

    get_train_valid_datasets(), 

    steps_per_epoch=TOTAL_STEPS_PER_EPOCH,

    epochs=EPOCHS, 

    callbacks=[lr_schedule]

)
# display_training_curves(history.history['loss'], history.history['val_loss'], 'loss', 211)

# display_training_curves(history.history['sparse_categorical_accuracy'], history.history['val_sparse_categorical_accuracy'], 'accuracy', 212)
# model.load_weights("food_tpu_model_best.hdf5")

# model.evaluate(get_validation_dataset())
with strategy.scope():

    rnet = DenseNet201(

        input_shape=(512, 512, 3),

        weights='imagenet',

        include_top=False

    )



    model2 = tf.keras.Sequential([

        rnet,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(len(CLASSES), activation='softmax')

    ])

        

model2.compile(

    optimizer=tf.keras.optimizers.Adam(lr=0.0001),

    loss = 'sparse_categorical_crossentropy',

    metrics=['sparse_categorical_accuracy']

)

model2.summary()
history2 = model2.fit(

    get_train_valid_datasets(), 

    steps_per_epoch=TOTAL_STEPS_PER_EPOCH,

    epochs=EPOCHS, 

    callbacks=[lr_schedule]

)
# def lrate(epoch):

#     lr = 0.00002

#     return lr



# lr_callback = tf.keras.callbacks.LearningRateScheduler(lrate, verbose=True)

# mcp = tf.keras.callbacks.ModelCheckpoint(filepath='food_tpu_model_best2.hdf5',

#                       monitor="val_sparse_categorical_accuracy", save_best_only=True, save_weights_only=False)

# history = model.fit(

#     get_train_valid_datasets(), 

#     steps_per_epoch=TOTAL_STEPS_PER_EPOCH,

#     epochs=5, 

#     validation_data=get_validation_dataset(),

#     callbacks=[lr_callback,mcp]

# )
model.save_weights("pneumonia_tpuB7_model_last.hdf5")
model2.save_weights("pneumonia_tpuDS_model_last.hdf5")
best_alpha = 0.46

print(best_alpha)
# test_ds = get_test_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and ids, order matters.



# print('Computing predictions...')

# test_images_ds = test_ds.map(lambda image, idnum: image)

# probabilities = model.predict(test_images_ds)

# predictions = np.argmax(probabilities, axis=-1)

# # print(predictions)



# print('Generating submission.csv file...')

# test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()

# test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch

# np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')
test_ds = get_test_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and ids, order matters.



print('Computing predictions...')

test_images_ds = test_ds.map(lambda image, idnum: image)

probabilities = best_alpha*model.predict(test_images_ds) + (1-best_alpha)*model2.predict(test_images_ds)

predictions = np.argmax(probabilities, axis=-1)

# print(predictions)



print('Generating submission.csv file...')

test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()

test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch

np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')