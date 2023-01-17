import math, re, os

import tensorflow as tf

import tensorflow.keras.backend as K

import numpy as np

from matplotlib import pyplot as plt

from kaggle_datasets import KaggleDatasets

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score

import datetime

import tqdm

import json

from collections import Counter

import gc

gc.enable()

print("Tensorflow version " + tf.__version__)

AUTO = tf.data.experimental.AUTOTUNE
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



# Configuration

IMAGE_SIZE = [192, 192]

BATCH_SIZE = 16 * strategy.num_replicas_in_sync



EPOCHS = 20
# Data access

GCS_DS_PATH = KaggleDatasets().get_gcs_path()



GCS_PATH_SELECT = { # available image sizes

    192: GCS_DS_PATH + '/tfrecords-jpeg-192x192',

    224: GCS_DS_PATH + '/tfrecords-jpeg-224x224',

    331: GCS_DS_PATH + '/tfrecords-jpeg-331x331',

    512: GCS_DS_PATH + '/tfrecords-jpeg-512x512'

}



GCS_PATH = GCS_PATH_SELECT[IMAGE_SIZE[0]]



TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train/*.tfrec')

VALIDATION_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')

TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/test/*.tfrec') # predictions on this dataset should be submitted for the competition
CLASSES = ['pink primrose',    'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',     'wild geranium',     'tiger lily',           'moon orchid',              'bird of paradise', 'monkshood',        'globe thistle',         # 00 - 09

           'snapdragon',       "colt's foot",               'king protea',      'spear thistle', 'yellow iris',       'globe-flower',         'purple coneflower',        'peruvian lily',    'balloon flower',   'giant white arum lily', # 10 - 19

           'fire lily',        'pincushion flower',         'fritillary',       'red ginger',    'grape hyacinth',    'corn poppy',           'prince of wales feathers', 'stemless gentian', 'artichoke',        'sweet william',         # 20 - 29

           'carnation',        'garden phlox',              'love in the mist', 'cosmos',        'alpine sea holly',  'ruby-lipped cattleya', 'cape flower',              'great masterwort', 'siam tulip',       'lenten rose',           # 30 - 39

           'barberton daisy',  'daffodil',                  'sword lily',       'poinsettia',    'bolero deep blue',  'wallflower',           'marigold',                 'buttercup',        'daisy',            'common dandelion',      # 40 - 49

           'petunia',          'wild pansy',                'primula',          'sunflower',     'lilac hibiscus',    'bishop of llandaff',   'gaura',                    'geranium',         'orange dahlia',    'pink-yellow dahlia',    # 50 - 59

           'cautleya spicata', 'japanese anemone',          'black-eyed susan', 'silverbush',    'californian poppy', 'osteospermum',         'spring crocus',            'iris',             'windflower',       'tree poppy',            # 60 - 69

           'gazania',          'azalea',                    'water lily',       'rose',          'thorn apple',       'morning glory',        'passion flower',           'lotus',            'toad lily',        'anthurium',             # 70 - 79

           'frangipani',       'clematis',                  'hibiscus',         'columbine',     'desert-rose',       'tree mallow',          'magnolia',                 'cyclamen ',        'watercress',       'canna lily',            # 80 - 89

           'hippeastrum ',     'bee balm',                  'pink quill',       'foxglove',      'bougainvillea',     'camellia',             'mallow',                   'mexican petunia',  'bromelia',         'blanket flower',        # 90 - 99

           'trumpet creeper',  'blackberry lily',           'common tulip',     'wild rose']                                                                                                                                               # 100 - 102
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



def load_dataset(filenames, labeled = True, ordered = False):

    # Read from TFRecords. For optimal performance, reading from multiple files at once and

    # Diregarding data order. Order does not matter since we will be shuffling the data anyway

    

    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False # disable order, increase speed

        

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads = AUTO) # automatically interleaves reads from multiple files

    dataset = dataset.with_options(ignore_order) # use data as soon as it streams in, rather than in its original order

    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls = AUTO) # returns a dataset of (image, label) pairs if labeled = True or (image, id) pair if labeld = False

    return dataset



def get_training_dataset():

    

    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)

    dataset = dataset.repeat() # Since we use custom training loop, we don't need to use repeat() here.

    dataset = dataset.shuffle(20000)

    dataset = dataset.batch(BATCH_SIZE, drop_remainder=False)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    

    return dataset  



def get_validation_dataset():

    

    dataset = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=True)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.cache()

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_test_dataset(ordered=False):

    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def count_data_items(filenames):

    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)



NUM_TRAINING_IMAGES = int(count_data_items(TRAINING_FILENAMES))

NUM_VALIDATION_IMAGES = int(count_data_items(VALIDATION_FILENAMES))

NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE



print('Dataset: {} training images, {} validation images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES))
# Get labels and their countings



def get_training_dataset_raw():



    dataset = load_dataset(TRAINING_FILENAMES, labeled=True, ordered=False)

    return dataset





raw_training_dataset = get_training_dataset_raw()



label_counter = Counter()

for images, labels in raw_training_dataset:

    label_counter.update([labels.numpy()])



del raw_training_dataset    

    

label_counting_sorted = label_counter.most_common()



NUM_TRAINING_IMAGES = sum([x[1] for x in label_counting_sorted])

print("number of examples in the original training dataset: {}".format(NUM_TRAINING_IMAGES))



print("labels in the original training dataset, sorted by occurrence")

label_counting_sorted
# We want each class occur at least (approximately) `TARGET_MIN_COUNTING` times

TARGET_MIN_COUNTING = 100



def get_num_of_repetition_for_class(class_id):

    

    counting = label_counter[class_id]

    if counting >= TARGET_MIN_COUNTING:

        return 1.0

    

    num_to_repeat = TARGET_MIN_COUNTING / counting

    

    return num_to_repeat



numbers_of_repetition_for_classes = {class_id: get_num_of_repetition_for_class(class_id) for class_id in range(104)}



print("number of repetitions for each class (if > 1)")

{k: v for k, v in sorted(numbers_of_repetition_for_classes.items(), key=lambda item: item[1], reverse=True) if v > 1}
# This will be called later in `get_training_dataset_with_oversample()`



keys_tensor = tf.constant([k for k in numbers_of_repetition_for_classes])

vals_tensor = tf.constant([numbers_of_repetition_for_classes[k] for k in numbers_of_repetition_for_classes])

table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), -1)



def get_num_of_repetition_for_example(training_example):

    

    _, label = training_example

    

    num_to_repeat = table.lookup(label)

    num_to_repeat_integral = tf.cast(int(num_to_repeat), tf.float32)

    residue = num_to_repeat - num_to_repeat_integral

    

    num_to_repeat = num_to_repeat_integral + tf.cast(tf.random.uniform(shape=()) <= residue, tf.float32)

    

    return tf.cast(num_to_repeat, tf.int64)
def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):

    # returns 3x3 transformmatrix which transforms indicies

        

    # CONVERT DEGREES TO RADIANS

    rotation = math.pi * rotation / 180.

    shear = math.pi * shear / 180.

    

    # ROTATION MATRIX

    c1 = tf.math.cos(rotation)

    s1 = tf.math.sin(rotation)

    one = tf.constant([1],dtype='float32')

    zero = tf.constant([0],dtype='float32')

    rotation_matrix = tf.reshape( tf.concat([c1,s1,zero, -s1,c1,zero, zero,zero,one],axis=0),[3,3] )

        

    # SHEAR MATRIX

    c2 = tf.math.cos(shear)

    s2 = tf.math.sin(shear)

    shear_matrix = tf.reshape( tf.concat([one,s2,zero, zero,c2,zero, zero,zero,one],axis=0),[3,3] )    

    

    # ZOOM MATRIX

    zoom_matrix = tf.reshape( tf.concat([one/height_zoom,zero,zero, zero,one/width_zoom,zero, zero,zero,one],axis=0),[3,3] )

    

    # SHIFT MATRIX

    shift_matrix = tf.reshape( tf.concat([one,zero,height_shift, zero,one,width_shift, zero,zero,one],axis=0),[3,3] )

    

    return K.dot(K.dot(rotation_matrix, shear_matrix), K.dot(zoom_matrix, shift_matrix))





def transform(image, label):

    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]

    # output - image randomly rotated, sheared, zoomed, and shifted

    DIM = IMAGE_SIZE[0]

    XDIM = DIM%2 #fix for size 331

    

    rot = 15. * tf.random.normal([1],dtype='float32')

    shr = 5. * tf.random.normal([1],dtype='float32') 

    h_zoom = 1.0 + tf.random.normal([1],dtype='float32')/10.

    w_zoom = 1.0 + tf.random.normal([1],dtype='float32')/10.

    h_shift = 16. * tf.random.normal([1],dtype='float32') 

    w_shift = 16. * tf.random.normal([1],dtype='float32') 

  

    # GET TRANSFORMATION MATRIX

    m = get_mat(rot,shr,h_zoom,w_zoom,h_shift,w_shift) 



    # LIST DESTINATION PIXEL INDICES

    x = tf.repeat( tf.range(DIM//2,-DIM//2,-1), DIM )

    y = tf.tile( tf.range(-DIM//2,DIM//2),[DIM] )

    z = tf.ones([DIM*DIM],dtype='int32')

    idx = tf.stack( [x,y,z] )

    

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS

    idx2 = K.dot(m,tf.cast(idx,dtype='float32'))

    idx2 = K.cast(idx2,dtype='int32')

    idx2 = K.clip(idx2,-DIM//2+XDIM+1,DIM//2)

    

    # FIND ORIGIN PIXEL VALUES           

    idx3 = tf.stack( [DIM//2-idx2[0,], DIM//2-1+idx2[1,]] )

    d = tf.gather_nd(image,tf.transpose(idx3))

        

    return tf.reshape(d,[DIM,DIM,3]), label
def get_training_dataset_with_oversample(repeat_dataset=True, oversample=False, augumentation=False):



    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)



    if oversample:

        dataset = dataset.flat_map(lambda image, label: tf.data.Dataset.from_tensors((image, label)).repeat(get_num_of_repetition_for_example((image, label))))



    if augumentation:

        dataset = dataset.map(transform, num_parallel_calls=AUTO)

    

    if repeat_dataset:

        dataset = dataset.repeat() # the training dataset must repeat for several epochs

    

    dataset = dataset.shuffle(20000)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    

    return dataset
oversampled_training_dataset = get_training_dataset_with_oversample(repeat_dataset=False, oversample=True, augumentation=False)



label_counter_2 = Counter()

for images, labels in oversampled_training_dataset:

    label_counter_2.update(labels.numpy())



del oversampled_training_dataset



label_counting_sorted_2 = label_counter_2.most_common()



NUM_TRAINING_IMAGES_OVERSAMPLED = sum([x[1] for x in label_counting_sorted_2])

print("number of examples in the oversampled training dataset: {}".format(NUM_TRAINING_IMAGES_OVERSAMPLED))



print("labels in the oversampled training dataset, sorted by occurrence")

label_counting_sorted_2
# Learning rate schedule for TPU, GPU and CPU.

# Using an LR ramp up because fine-tuning a pre-trained model.

# Starting with a high LR would break the pre-trained weights.



LR_START = 0.00001

LR_MAX = 0.00005 * strategy.num_replicas_in_sync

LR_MIN = 0.00001

LR_RAMPUP_EPOCHS = 5

LR_SUSTAIN_EPOCHS = 0

LR_EXP_DECAY = .8



def lrfn(epoch):

    if epoch < LR_RAMPUP_EPOCHS:

        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START

    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:

        lr = LR_MAX

    else:

        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN

    return lr

    

lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = True)



rng = [i for i in range(25 if EPOCHS<25 else EPOCHS)]

y = [lrfn(x) for x in rng]

plt.plot(rng, y)

print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
valid_ds = get_validation_dataset()



valid_images_ds = valid_ds.map(lambda image, label: image)

valid_labels_ds = valid_ds.map(lambda image, label: label).unbatch()



valid_labels = next(iter(valid_labels_ds.batch(NUM_VALIDATION_IMAGES))).numpy() # get everything as one batch



valid_steps = NUM_VALIDATION_IMAGES // BATCH_SIZE



if NUM_VALIDATION_IMAGES % BATCH_SIZE > 0:

    valid_steps += 1
original_training_dataset = get_training_dataset_with_oversample(repeat_dataset=True, oversample=False, augumentation=False)



with strategy.scope():



    model = tf.keras.Sequential([

        tf.keras.applications.DenseNet201(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), weights='imagenet', include_top=False),

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(len(CLASSES), activation='softmax')

    ])



model.compile(

    optimizer='adam',

    loss = 'sparse_categorical_crossentropy',

    metrics=['sparse_categorical_accuracy']

)



history = model.fit(

    original_training_dataset, 

    steps_per_epoch=NUM_TRAINING_IMAGES // BATCH_SIZE,

    epochs=EPOCHS,

    callbacks=[lr_callback],

    validation_data=valid_ds

)



valid_probs = model.predict(valid_images_ds, steps=valid_steps)

valid_preds = np.argmax(valid_probs, axis=-1)



del model

gc.collect()

tf.keras.backend.clear_session()



val_acc = history.history['val_sparse_categorical_accuracy']



score = f1_score(valid_labels, valid_preds, labels=range(len(CLASSES)), average='macro')

acc = accuracy_score(valid_labels, valid_preds)

precision = precision_score(valid_labels, valid_preds, labels=range(len(CLASSES)), average='macro')

recall = recall_score(valid_labels, valid_preds, labels=range(len(CLASSES)), average='macro')



print("results for training on the original dataset")

print("best 10 validation accuracies during training = {}".format(sorted(val_acc, reverse=True)[:10]))

print('f1 score: {:.6f} | recall: {:.6f} | precision: {:.6f} | acc: {:.6f}'.format(score, recall, precision, acc))
oversampled_training_dataset = get_training_dataset_with_oversample(repeat_dataset=True, oversample=True, augumentation=True)





with strategy.scope():



    model = tf.keras.Sequential([

        tf.keras.applications.DenseNet201(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), weights='imagenet', include_top=False),

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(len(CLASSES), activation='softmax')

    ])



model.compile(

    optimizer='adam',

    loss = 'sparse_categorical_crossentropy',

    metrics=['sparse_categorical_accuracy']

)



history = model.fit(

    oversampled_training_dataset, 

    steps_per_epoch=NUM_TRAINING_IMAGES_OVERSAMPLED // BATCH_SIZE,

    epochs=EPOCHS,

    callbacks=[lr_callback],

    validation_data=valid_ds

)



valid_probs = model.predict(valid_images_ds, steps=valid_steps)

valid_preds = np.argmax(valid_probs, axis=-1)



del model

gc.collect()

tf.keras.backend.clear_session()



val_acc = history.history['val_sparse_categorical_accuracy']



score = f1_score(valid_labels, valid_preds, labels=range(len(CLASSES)), average='macro')

acc = accuracy_score(valid_labels, valid_preds)

precision = precision_score(valid_labels, valid_preds, labels=range(len(CLASSES)), average='macro')

recall = recall_score(valid_labels, valid_preds, labels=range(len(CLASSES)), average='macro')



print("results for training on the oversampled dataset")

print("best 10 validation accuracies during training = {}".format(sorted(val_acc, reverse=True)[:10]))

print('f1 score: {:.6f} | recall: {:.6f} | precision: {:.6f} | acc: {:.6f}'.format(score, recall, precision, acc))
result_1 = {

    "EfficientNetB7": {

        "oversampling": {

            "1": {

                "f1": 0.9163497711078313,

                "recall": 0.9205266060440362,

                "precision": 0.919277486006819,

                "acc": 0.9240301724137931

            },

            "100": {

                "f1": 0.9306967499560146,

                "recall": 0.9385304126936429,

                "precision": 0.9277303112649578,

                "acc": 0.9296875

            },

            "300": {

                "f1": 0.9361016120193594,

                "recall": 0.9394270473352216,

                "precision": 0.939168918056952,

                "acc": 0.9366918103448276

            },

            "800": {

                "f1": 0.9404645585115528,

                "recall": 0.9427588505810944,

                "precision": 0.9440889947725051,

                "acc": 0.9391163793103449

            }

        }

    }

}



print(json.dumps(result_1, ensure_ascii=False, indent=4))
result_2 = {

    "DenseNet201": {

        "oversampling": {

            "1": {

                "f1": 0.9249583259975988,

                "recall": 0.9147673084901355,

                "precision": 0.9413584080076393,

                "acc": 0.9275323275862069

            },

            "100": {

                "f1": 0.9335555346943241,

                "recall": 0.934321682931082,

                "precision": 0.9383416676577636,

                "acc": 0.9337284482758621

            },

            "300": {

                "f1": 0.9343192665802171,

                "recall": 0.9355079525776551,

                "precision": 0.9384803240387035,

                "acc": 0.9331896551724138

            },

            "800": {

                "f1": 0.9389981148396047,

                "recall": 0.9403390782293726,

                "precision": 0.9426095000332035,

                "acc": 0.9348060344827587

            }

        }

    }

}



print(json.dumps(result_2, ensure_ascii=False, indent=4))