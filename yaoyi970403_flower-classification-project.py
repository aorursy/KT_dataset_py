import tensorflow as tf

from kaggle_datasets import KaggleDatasets

import matplotlib.pyplot as plt

import numpy as np



print("Tensorflow version " + tf.__version__)
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



AUTO = tf.data.experimental.AUTOTUNE

print("REPLICAS: ", strategy.num_replicas_in_sync)
GCS_DS_PATH = KaggleDatasets().get_gcs_path() # you can list the bucket with "!gsutil ls $GCS_DS_PATH"
IMAGE_SIZE = [512,512] # at this size, a GPU will run out of memory. Use the TPU

EPOCHS = 15

BATCH_SIZE = 16 * strategy.num_replicas_in_sync



NUM_TRAINING_IMAGES = 12753

NUM_TEST_IMAGES = 7382

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range

    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU

    return image



def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    label = tf.cast(example['class'], tf.int32)

    return image, label # returns a dataset of (image, label) pairs



def read_unlabeled_tfrecord(example):

    UNLABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element

        # class is missing, this competitions's challenge is to predict flower classes for the test dataset

    }

    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    idnum = example['id']

    return image, idnum # returns a dataset of image(s)



def load_dataset(filenames, labeled=True, ordered=False):

    # Read from TFRecords. For optimal performance, reading from multiple files at once and

    # disregarding data order. Order does not matter since we will be shuffling the data anyway.



    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False # disable order, increase speed



    dataset = tf.data.TFRecordDataset(filenames) # automatically interleaves reads from multiple files

    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order

    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord)

    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False

    return dataset





def data_augment(image, label, seed=2020):

    image = tf.image.random_flip_left_right(image, seed=seed)    

    image = tf.image.random_flip_up_down(image, seed=seed)

    #image = tf.image.random_brightness(image, 0.2, seed=seed)

    #image = tf.image.rot90(image)

#     image = tf.image.random_jpeg_quality(image, 85, 100, seed=seed)

#     image = tf.image.resize(image, [530, 530])

#     image = tf.image.random_crop(image, [512, 512], seed=seed)

    #image = tf.image.random_saturation(image, 0, 2)

    return image, label   





def get_training_dataset():

    dataset = load_dataset(tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-512x512/train/*.tfrec'), labeled=True)

    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

    dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    return dataset



def get_validation_dataset():

    dataset = load_dataset(tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-512x512/val/*.tfrec'), labeled=True, ordered=False)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.cache()

    return dataset



def get_test_dataset(ordered=False):

    dataset = load_dataset(tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-512x512/test/*.tfrec'), labeled=False, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    return dataset



training_dataset = get_training_dataset()

validation_dataset = get_validation_dataset()
with strategy.scope():    

    covn_base = tf.keras.applications.Xception(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

    covn_base.trainable = True

    

    #print(len(covn_base.layers))



    #for layers in covn_base.layers[:-50]:

        #layers.trainable = False

    model = tf.keras.Sequential([

        covn_base,

        tf.keras.layers.GlobalAveragePooling2D(),

        #tf.keras.layers.Dense(1024,activation='relu'), #添加全连接层

        #tf.keras.layers.Dropout(rate=0.5),

        tf.keras.layers.Dense(104, activation='softmax')

    ])

    

model.summary()        

model.compile(

    optimizer=tf.keras.optimizers.Adam(),

    loss = 'sparse_categorical_crossentropy',

    metrics=['sparse_categorical_accuracy']

)
def lrfn(epoch):

    LR_START = 0.000001

    LR_MAX = 0.00005 * strategy.num_replicas_in_sync

    LR_MIN = 0.000001

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



fig, ax = plt.subplots(figsize=(24, 6))

plt.plot(rng, y)

print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)



history = model.fit(training_dataset, 

          steps_per_epoch=STEPS_PER_EPOCH, 

          epochs=EPOCHS, 

          validation_data=validation_dataset,

          callbacks=[lr_schedule])


# plot loss and accuracy image

history_dict = history.history

train_loss = history_dict["loss"]

train_accuracy = history_dict["sparse_categorical_accuracy"]

val_loss = history_dict["val_loss"]

val_accuracy = history_dict["val_sparse_categorical_accuracy"]



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
test_ds = get_test_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and ids, order matters.



print('Computing predictions...')

test_images_ds = test_ds.map(lambda image, idnum: image)

probabilities = model.predict(test_images_ds)

predictions = np.argmax(probabilities, axis=-1)

print(predictions)



print('Generating submission.csv file...')

test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()

test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch

np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')