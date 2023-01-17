import numpy as np 

import pandas as pd



# NOTE: internet should be ON

from kaggle_datasets import KaggleDatasets

GCS_DS_PATH = KaggleDatasets().get_gcs_path()
import tensorflow as tf

import datetime

import gc



print('TensorFlow version: %s' % tf.__version__)

print('Keras version: %s' % tf.keras.__version__)



# Detect hardware, return appropriate distribution strategy

try:

    # TPU detection. 

    # No parameters necessary if TPU_NAME environment variable is set. 

    # On Kaggle this is always the case.

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  

    print('Running on TPU ', tpu.master())

except ValueError:

    print("TPU is not available")

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    # default distribution strategy in Tensorflow. 

    # Works on CPU and single GPU.

    strategy = tf.distribute.get_strategy() 



print("REPLICAS: ", strategy.num_replicas_in_sync)
# helper log function

import os



dt_start = datetime.datetime.utcnow()

def print_and_log(string):

    seconds = (datetime.datetime.utcnow() - dt_start).seconds

    time_diff = "%d:%02d" % (seconds / 60, seconds % 60)

    print(time_diff, string)

    os.system(f'echo \"{time_diff}\" \"{string}\"')
# at 512,512 size, a GPU will run out of memory. Use the TPU

# 224, 224 is VGG16 input

# 299, 299 is inception v3 input

IMAGE_SIZE = [512, 512] 

IMAGE_CHANNELS = 3



BATCH_SIZE = 16 * strategy.num_replicas_in_sync



GCS_PATH_SELECT = { # available image sizes

    192: GCS_DS_PATH + '/tfrecords-jpeg-192x192',

    224: GCS_DS_PATH + '/tfrecords-jpeg-224x224',

    331: GCS_DS_PATH + '/tfrecords-jpeg-331x331',

    512: GCS_DS_PATH + '/tfrecords-jpeg-512x512'

}

GCS_PATH = GCS_PATH_SELECT[IMAGE_SIZE[0]]



TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train/*.tfrec')

VALIDATION_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')

# predictions on this dataset should be submitted for the competition

TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/test/*.tfrec') 
TENSOR_BOARD = False # CPU/GPU

VERBOSE = 0

IMAGE_AUGMENTATION = True

GRID_MASK = False

USE_CLASS_WEIGHTS = False



# 'EfficientNetB7', 'InceptionV3', 'DenseNet201', 'ResNet152V2', 'VGG16'

# 'imagenet', EfficientNetB7 -> 'noisy-student'



MODEL_NAME = 6 * ['InceptionV3']

WEIGHTS_NAME = 6 * ['imagenet']



selected_optimizer = 'adam'

selected_loss = 'sparse_categorical_crossentropy' 

selected_metrics = 'sparse_categorical_accuracy' 



EPOCHS = 30

FOLDS = len(MODEL_NAME)

SEED = 92



MODEL_FILENAMES = TRAINING_FILENAMES + VALIDATION_FILENAMES
CLASSES = [

    'pink primrose',    'hard-leaved pocket orchid',

    'canterbury bells',                 'sweet pea',

    'wild geranium',                   'tiger lily',           

    'moon orchid',               'bird of paradise',

    'monkshood',                    'globe thistle',# 00 - 09

    'snapdragon',                     "colt's foot",

    'king protea',                  'spear thistle',

    'yellow iris',                   'globe-flower',         

    'purple coneflower',            'peruvian lily',

    'balloon flower',       'giant white arum lily',# 10 - 19

    'fire lily',                'pincushion flower',

    'fritillary',                      'red ginger',

    'grape hyacinth',                  'corn poppy',           

    'prince of wales feathers',  'stemless gentian',

    'artichoke',                    'sweet william',# 20 - 29

    'carnation',                     'garden phlox',

    'love in the mist',                    'cosmos',

    'alpine sea holly',      'ruby-lipped cattleya', 

    'cape flower',               'great masterwort',

    'siam tulip',                     'lenten rose',# 30 - 39

    'barberton daisy',                   'daffodil',

    'sword lily',                      'poinsettia',

    'bolero deep blue',                'wallflower',

    'marigold',                         'buttercup',

    'daisy',                     'common dandelion',# 40 - 49

    'petunia',                         'wild pansy',

    'primula',                          'sunflower',

    'lilac hibiscus',          'bishop of llandaff',

    'gaura',                             'geranium',

    'orange dahlia',           'pink-yellow dahlia',# 50 - 59

    'cautleya spicata',          'japanese anemone',

    'black-eyed susan',                'silverbush',

    'californian poppy',             'osteospermum',         

    'spring crocus',                         'iris',

    'windflower',                      'tree poppy',# 60 - 69

    'gazania',                             'azalea',

    'water lily',                            'rose',

    'thorn apple',                  'morning glory',     

    'passion flower',                       'lotus',

    'toad lily',                        'anthurium',# 70 - 79

    'frangipani',                        'clematis',

    'hibiscus',                         'columbine',

    'desert-rose',                    'tree mallow',      

    'magnolia',                         'cyclamen ',

    'watercress',                      'canna lily',# 80 - 89

    'hippeastrum ',                      'bee balm',

    'pink quill',                        'foxglove',

    'bougainvillea',                     'camellia',        

    'mallow',                     'mexican petunia',

    'bromelia',                    'blanket flower',# 90 - 99

    'trumpet creeper',            'blackberry lily',

    'common tulip',                     'wild rose']# 100 - 102
# https://www.kaggle.com/cdeotte/rotation-augmentation-gpu-tpu-0-96

import math

import tensorflow.keras.backend as K



def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):

    # returns 3x3 transformmatrix which transforms indicies

        

    # CONVERT DEGREES TO RADIANS

    rotation = math.pi * rotation / 180.

    shear = math.pi * shear / 180.



    one = tf.constant([1],dtype='float32')

    zero = tf.constant([0],dtype='float32')

    

    # ROTATION MATRIX

    c1 = tf.math.cos(rotation)

    s1 = tf.math.sin(rotation)

    rotation_matrix = tf.reshape( 

        tf.concat([

              c1,  s1, zero,

             -s1,  c1, zero, 

            zero,zero, one],

        axis=0),

        [3,3] 

    )

        

    # SHEAR MATRIX

    c2 = tf.math.cos(shear)

    s2 = tf.math.sin(shear)

    shear_matrix = tf.reshape( 

        tf.concat([

            one,   s2, zero, 

            zero,  c2, zero, 

            zero,zero, one],

        axis=0),

        [3,3] 

    )    

    

    # ZOOM MATRIX

    zoom_matrix = tf.reshape( 

        tf.concat([

          one/height_zoom,  zero,  zero, 

          zero,   one/width_zoom,  zero,

          zero,             zero,  one],

        axis=0),

        [3,3] 

    )

    

    # SHIFT MATRIX

    shift_matrix = tf.reshape( 

        tf.concat([

          one,  zero, height_shift, 

          zero,  one,  width_shift, 

          zero, zero,         one],

        axis=0),

        [3,3]

    )

    

    return K.dot(

            K.dot(rotation_matrix, shear_matrix), 

            K.dot(zoom_matrix, shift_matrix))



def data_rotate(image,label):

    # input image - is one image of size [dim,dim,3] 

    #               not a batch of [b,dim,dim,3]

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

        

    return tf.reshape(d,[DIM,DIM,3]),label
# https://www.kaggle.com/xiejialun/gridmask-data-augmentation-with-tensorflow



# todo: switch to pytorch

def GridMask(image_height, image_width, d1, d2, rotate_angle=1, ratio=0.5):

    #

    def mask_transform(image, inv_mat, image_shape):

        h, w, c = image_shape

        cx, cy = w//2, h//2



        new_xs = tf.repeat( tf.range(-cx, cx, 1), h)

        new_ys = tf.tile( tf.range(-cy, cy, 1), [w])

        new_zs = tf.ones([h*w], dtype=tf.int32)



        old_coords = tf.matmul(inv_mat, 

                        tf.cast(tf.stack([new_xs, new_ys, new_zs]), tf.float32))

        

        old_coords_x  = tf.round(old_coords[0, :] + w//2)

        old_coords_y  = tf.round(old_coords[1, :] + h//2)



        clip_mask_x = tf.logical_or(old_coords_x<0, old_coords_x>w-1)

        clip_mask_y = tf.logical_or(old_coords_y<0, old_coords_y>h-1)

        clip_mask = tf.logical_or(clip_mask_x, clip_mask_y)



        old_coords_x = tf.boolean_mask(old_coords_x, tf.logical_not(clip_mask))

        old_coords_y = tf.boolean_mask(old_coords_y, tf.logical_not(clip_mask))

        new_coords_x = tf.boolean_mask(new_xs+cx, tf.logical_not(clip_mask))

        new_coords_y = tf.boolean_mask(new_ys+cy, tf.logical_not(clip_mask))



        old_coords = tf.cast(tf.stack([old_coords_y, old_coords_x]), tf.int32)

        new_coords = tf.cast(tf.stack([new_coords_y, new_coords_x]), tf.int64)

        rotated_image_values = tf.gather_nd(image, tf.transpose(old_coords))

        rotated_image_channel = list()

        for i in range(c):

            vals = rotated_image_values[:,i]

            sparse_channel = tf.SparseTensor(tf.transpose(new_coords), vals, [h, w])

            rotated_image_channel.append(tf.sparse.to_dense(sparse_channel, 

                                            default_value=0, validate_indices=False))



        return tf.transpose(tf.stack(rotated_image_channel), [1,2,0])

    #

    def mask_random_rotate(image, angle, image_shape):

        def get_rotation_mat_inv(angle):

              #transform to radian

            angle = math.pi * angle / 180



            cos_val = tf.math.cos(angle)

            sin_val = tf.math.sin(angle)

            one = tf.constant([1], tf.float32)

            zero = tf.constant([0], tf.float32)



            rot_mat_inv = tf.concat([cos_val, sin_val, zero,

                                         -sin_val, cos_val, zero,

                                         zero, zero, one], axis=0)

            rot_mat_inv = tf.reshape(rot_mat_inv, [3,3])

            return rot_mat_inv



        angle = float(angle) * tf.random.normal([1],dtype='float32')

        rot_mat_inv = get_rotation_mat_inv(angle)

        return mask_transform(image, rot_mat_inv, image_shape)    

    #

    h, w = image_height, image_width

    hh = int(np.ceil(np.sqrt(h*h+w*w)))

    hh = hh+1 if hh%2==1 else hh

    d = tf.random.uniform(shape=[], minval=d1, maxval=d2, dtype=tf.int32)

    l = tf.cast(tf.cast(d,tf.float32)*ratio+0.5, tf.int32)



    st_h = tf.random.uniform(shape=[], minval=0, maxval=d, dtype=tf.int32)

    st_w = tf.random.uniform(shape=[], minval=0, maxval=d, dtype=tf.int32)



    y_ranges = tf.range(-1 * d + st_h, -1 * d + st_h + l)

    x_ranges = tf.range(-1 * d + st_w, -1 * d + st_w + l)



    for i in range(0, hh//d+1):

        s1 = i * d + st_h

        s2 = i * d + st_w

        y_ranges = tf.concat([y_ranges, tf.range(s1,s1+l)], axis=0)

        x_ranges = tf.concat([x_ranges, tf.range(s2,s2+l)], axis=0)



    x_clip_mask = tf.logical_or(x_ranges < 0 , x_ranges > hh-1)

    y_clip_mask = tf.logical_or(y_ranges < 0 , y_ranges > hh-1)

    clip_mask = tf.logical_or(x_clip_mask, y_clip_mask)



    x_ranges = tf.boolean_mask(x_ranges, tf.logical_not(clip_mask))

    y_ranges = tf.boolean_mask(y_ranges, tf.logical_not(clip_mask))



    hh_ranges = tf.tile(tf.range(0,hh), 

                        [tf.cast(tf.reduce_sum(tf.ones_like(x_ranges)), tf.int32)])

    x_ranges = tf.repeat(x_ranges, hh)

    y_ranges = tf.repeat(y_ranges, hh)



    y_hh_indices = tf.transpose(tf.stack([y_ranges, hh_ranges]))

    x_hh_indices = tf.transpose(tf.stack([hh_ranges, x_ranges]))



    y_mask_sparse = tf.SparseTensor(tf.cast(y_hh_indices, tf.int64), 

                                    tf.zeros_like(y_ranges), [hh, hh])

    y_mask = tf.sparse.to_dense(y_mask_sparse, 1, False)



    x_mask_sparse = tf.SparseTensor(tf.cast(x_hh_indices, tf.int64), 

                                    tf.zeros_like(x_ranges), [hh, hh])

    x_mask = tf.sparse.to_dense(x_mask_sparse, 1, False)



    mask = tf.expand_dims( tf.clip_by_value(x_mask + y_mask, 0, 1), axis=-1)



    mask = mask_random_rotate(mask, rotate_angle, [hh, hh, 1])

    mask = tf.image.crop_to_bounding_box(mask, (hh-h)//2, (hh-w)//2, 

                                         image_height, image_width)



    return mask



def data_gridmask(image,label):

    AugParams = {

        'd1' : 100,

        'd2': 160,

        'rotate' : 45,

        'ratio' : 0.3

    }

    mask = GridMask(IMAGE_SIZE[0], IMAGE_SIZE[1], AugParams['d1'], 

                    AugParams['d2'], AugParams['rotate'], AugParams['ratio'])

    if IMAGE_CHANNELS == 3:

        mask = tf.concat([mask, mask, mask], axis=-1)

    return image * tf.cast(mask,tf.float32), label
import re



AUTO = tf.data.experimental.AUTOTUNE



def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    # convert image to floats in [0, 1] range

    image = tf.cast(image, tf.float32) / 255.0  

    # explicit size needed for TPU

    image = tf.reshape(image, [*IMAGE_SIZE, IMAGE_CHANNELS]) 

    return image



def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        # tf.string means bytestring

        "image": tf.io.FixedLenFeature([], tf.string), 

        # shape [] means single element

        "class": tf.io.FixedLenFeature([], tf.int64),  

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    label = tf.cast(example['class'], tf.int32)

    # returns a dataset of (image, label) pairs

    return image, label



def read_unlabeled_tfrecord(example):

    UNLABELED_TFREC_FORMAT = {

        # tf.string means bytestring

        "image": tf.io.FixedLenFeature([], tf.string), 

        # shape [] means single element

        "id": tf.io.FixedLenFeature([], tf.string),  

        # class is missing, this competitions's challenge 

        # is to predict flower classes for the test dataset

    }

    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    idnum = example['id']

    return image, idnum 



def load_dataset(filenames, labeled=True, ordered=False):

    # Read from TFRecords. For optimal performance, 

    # reading from multiple files at once and

    # disregarding data order. Order does not matter 

    # since we will be shuffling the data anyway.



    ignore_order = tf.data.Options()

    if not ordered:

        # disable order, increase speed

        ignore_order.experimental_deterministic = False 



    # automatically interleaves reads from multiple files

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) 

    # uses data as soon as it streams in, rather than in its original order

    dataset = dataset.with_options(ignore_order) 

    dataset = dataset.map(read_labeled_tfrecord if labeled else 

                          read_unlabeled_tfrecord, num_parallel_calls=AUTO)

    # returns a dataset of (image, label) pairs if labeled=True 

    # or (image, id) pairs if labeled=False

    return dataset



def data_augment(image, label):

    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in 

    # the next function (below), this happens essentially for free on TPU. 

    # Data pipeline code is executed on the "CPU" part of the TPU while 

    # the TPU itself is computing gradients.

    image = tf.image.random_flip_left_right(image)

    #image = tf.image.random_saturation(image, 0, 2)

    return image, label   



def get_training_dataset(dataset, do_aug=False, do_grid=False):

    # dataset = load_dataset(filenames, labeled=True)

    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

    # Rotation Augmentation GPU/TPU

    if do_aug: dataset = dataset.map(data_rotate, num_parallel_calls=AUTO)

    # grid mask

    if do_grid: dataset = dataset.map(data_gridmask, num_parallel_calls=AUTO)   

    # the training dataset must repeat for several epochs

    dataset = dataset.repeat() 

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    # prefetch next batch while training (autotune prefetch buffer size)

    dataset = dataset.prefetch(AUTO) 

    return dataset



def get_validation_dataset(dataset):

    # dataset = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.cache()

    # prefetch next batch while training (autotune prefetch buffer size)

    dataset = dataset.prefetch(AUTO) 

    return dataset



def get_test_dataset(ordered=False):

    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    # prefetch next batch while training (autotune prefetch buffer size)

    dataset = dataset.prefetch(AUTO) 

    return dataset



def count_data_items(filenames):

    # the number of data items is written in the name of the .tfrec files,

    # i.e. flowers00-230.tfrec = 230 data items

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)



NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)

NUM_VALIDATION_IMAGES = count_data_items(VALIDATION_FILENAMES)

NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

print('Dataset: {} training images, {} validation images, {} unlabeled test images'.

      format(NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES))
from sklearn.utils import class_weight



def read_tfrecord_label(example):

    LABELED_TFREC_FORMAT = {

        # tf.string means bytestring

        "image": tf.io.FixedLenFeature([], tf.string), 

        # shape [] means single element

        "class": tf.io.FixedLenFeature([], tf.int64),  

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    #image = decode_image(example['image'])

    label = tf.cast(example['class'], tf.int32)

    # returns a dataset of (image, label) pairs

    return label



def get_class_weights(filenames):

    count = count_data_items(filenames)

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)

    dataset = dataset.map(read_tfrecord_label, num_parallel_calls=AUTO)

    labels = next(iter(dataset.batch(count))).numpy()

    return class_weight.compute_class_weight('balanced', [x for x in range(len(CLASSES))], labels)
def model_select(name, weights = 'imagenet'):

    if name == 'EfficientNetB7':

        try:

            import efficientnet.tfkeras as efn

        except ImportError:

            !pip install -q efficientnet

            import efficientnet.tfkeras as efn

        pretrained_model = efn.EfficientNetB7(

            weights=weights, 

            include_top=False,

            input_shape=[*IMAGE_SIZE, IMAGE_CHANNELS])

        # model fine tuning

        pretrained_model.trainable = True

        return tf.keras.Sequential([

            pretrained_model,

            tf.keras.layers.GlobalAveragePooling2D(),

            # tf.keras.layers.Dropout(0.2, name="dropout_out"),

            tf.keras.layers.Dense(len(CLASSES), activation='softmax')

        ])

    elif name == 'InceptionV3':

        pretrained_model = tf.keras.applications.InceptionV3(

            weights=weights, 

            include_top=False,

            input_shape=[*IMAGE_SIZE, IMAGE_CHANNELS])

        # model fine tuning

        pretrained_model.trainable = True

        return tf.keras.Sequential([

            pretrained_model,

            tf.keras.layers.GlobalAveragePooling2D(),

            tf.keras.layers.Dense(4096, activation='relu'), 

            tf.keras.layers.Dense(4096, activation='relu'), 

            tf.keras.layers.Dense(len(CLASSES), activation='softmax')

        ])       

    elif name == 'DenseNet201':

        pretrained_model = tf.keras.applications.DenseNet201(

            weights=weights, 

            include_top=False,

            input_shape=[*IMAGE_SIZE, IMAGE_CHANNELS])

        # model fine tuning

        pretrained_model.trainable = True

        return tf.keras.Sequential([

            pretrained_model,

            tf.keras.layers.GlobalAveragePooling2D(),

#             tf.keras.layers.Dense(4096, activation='relu'), 

#             tf.keras.layers.Dense(4096, activation='relu'), 

            tf.keras.layers.Dense(len(CLASSES), activation='softmax')

        ])

    elif name == 'ResNet152V2':

        pretrained_model = tf.keras.applications.ResNet152V2(

            weights=weights, 

            include_top=False,

            input_shape=[*IMAGE_SIZE, IMAGE_CHANNELS])

        # model fine tuning

        pretrained_model.trainable = True

        return tf.keras.Sequential([

            pretrained_model,

            tf.keras.layers.GlobalAveragePooling2D(),

            tf.keras.layers.Dense(4096, activation='relu'), 

            tf.keras.layers.Dense(4096, activation='relu'),            

            tf.keras.layers.Dense(len(CLASSES), activation='softmax')

        ])  

    elif name == 'VGG16':

        pretrained_model = tf.keras.applications.VGG16(

            weights=weights, 

            include_top=False,

            input_shape=[*IMAGE_SIZE, IMAGE_CHANNELS])

        # model fine tuning

        pretrained_model.trainable = True

        return tf.keras.Sequential([

            pretrained_model,

            tf.keras.layers.GlobalAveragePooling2D(),

            tf.keras.layers.Dense(4096, activation='relu'), 

            tf.keras.layers.Dense(4096, activation='relu'), 

            tf.keras.layers.Dense(len(CLASSES), activation='softmax')

        ])

    else:

        print('Unknown model: %s' % name)

        raise

    

def create_model(name, weights = 'imagenet'):

    # transfer learning

    with strategy.scope():

        # https://www.tensorflow.org/tutorials/keras/overfit_and_underfit

        # add dropout, weight regularization to decrease overfitting

        model = model_select(name, weights)

    

        # v25 => lr=0.01, decay=1e-6, momentum=0.9, nesterov=True 

        # v27 => lr=0.001, decay=1e-5, momentum=0.8, nesterov=True 

        #opt = tf.keras.optimizers.SGD(lr=0.01, 

        #                              decay=1e-6, 

        #                              momentum=0.9, 

        #                              nesterov=True)

        #opt = 'adam' #tf.keras.optimizers.Adam(0.0001)

        model.compile(

            optimizer = selected_optimizer,

            loss = selected_loss,

            metrics=[selected_metrics]

        )

        return model
# save configuration for post processing

import json



def save_config():

    # Data to be written 

    config = { 

        "IMAGE_SIZE" : IMAGE_SIZE,

        "IMAGE_CHANNELS" : IMAGE_CHANNELS,

        "BATCH_SIZE" : BATCH_SIZE,

        "MODEL_NAME" : MODEL_NAME,

        "WEIGHTS_NAME" : WEIGHTS_NAME,

        "EPOCHS" : EPOCHS,

        "FOLDS" : FOLDS,

        "TENSOR_BOARD" : TENSOR_BOARD, 

        "VERBOSE" : VERBOSE, 

        "IMAGE_AUGMENTATION" : IMAGE_AUGMENTATION, 

        "GRID_MASK" : GRID_MASK,

        "USE_CLASS_WEIGHTS" : USE_CLASS_WEIGHTS,

        "SEED" : SEED

    } 

    # Serializing json  

    # json_object = json.dumps(config, indent = 4) 

    with open('config.json', 'w') as outfile:

        json.dump(config, outfile)

        

save_config()        
from sklearn.model_selection import KFold



# Learning rate schedule for TPU, GPU and CPU.

# Using an LR ramp up because fine-tuning a pre-trained model.

# Starting with a high LR would break the pre-trained weights.



LR_START = 0.00001

LR_MAX = 0.00005 * strategy.num_replicas_in_sync

LR_MIN = 0.00001

LR_RAMPUP_EPOCHS = 4

LR_SUSTAIN_EPOCHS = 6

LR_EXP_DECAY = .8



def lrfn(epoch):

    if epoch < LR_RAMPUP_EPOCHS:

        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START

        #lr = (LR_MAX - LR_START) * (epoch/LR_RAMPUP_EPOCHS)**2 + LR_START

    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:

        lr = LR_MAX

    else:

        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - 

                        LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN

    return lr



# Clear any logs from previous runs

!rm -rf ./logs/



# define tensorboard log root

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") 



def callbacks(ifold, board = False):

    # prepare callbacks

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = False)

    if board:

        #writer = tf.summary.create_file_writer(log_dir + f"/fold{ifold}")

        tensorboard_callback = tf.keras.callbacks.TensorBoard(

            log_dir=log_dir + f"/fold{ifold}", histogram_freq=1)

        return [lr_callback, tensorboard_callback]

    else:

        return [lr_callback]

    

kfold = KFold(FOLDS, shuffle = True, random_state = SEED)



# https://www.kaggle.com/c/flower-classification-with-tpus/discussion/131876

# array of weights works on TPU with the standard sparse_categorical_crossentropy loss

CLASS_WEIGHTS = get_class_weights(MODEL_FILENAMES) if USE_CLASS_WEIGHTS else None



# since we are splitting the dataset and iterating separately on 

# images and ids, order matters.

test_ds = get_test_dataset(ordered=True)



def train_cross_validate_predict(filenames):

    print(f'Start training {FOLDS} folds {EPOCHS} epochs, img_aug {IMAGE_AUGMENTATION} grid {GRID_MASK} class weights {USE_CLASS_WEIGHTS}')

    histories = []

    for ifold, (trn_ind, val_ind) in enumerate(kfold.split(filenames)):

        # select files

        tdf = list(pd.DataFrame({'FILENAMES': filenames}).loc[trn_ind]['FILENAMES'])

        vdf = list(pd.DataFrame({'FILENAMES': filenames}).loc[val_ind]['FILENAMES'])

        steps_per_epoch = count_data_items(tdf) // BATCH_SIZE

        train_dataset = load_dataset(tdf, labeled = True)

        val_dataset = load_dataset(vdf, labeled = True, ordered = True)

        validation_data = get_validation_dataset(val_dataset)

        # run fit

        print_and_log(f'# FOLD: {ifold+1} {steps_per_epoch}');

        # Recreate the exact same model purely from the file

        #model = tf.keras.models.load_model('my_model.h5')

        model = create_model(MODEL_NAME[ifold], WEIGHTS_NAME[ifold])

        history = model.fit(

            get_training_dataset(train_dataset, IMAGE_AUGMENTATION, GRID_MASK), 

            steps_per_epoch = steps_per_epoch,

            epochs = EPOCHS,

            callbacks = callbacks(ifold, TENSOR_BOARD),

            validation_data = validation_data,

            class_weight = CLASS_WEIGHTS,

            verbose = VERBOSE

        )

        test_images_ds = test_ds.map(lambda image, idnum: image)

        predict = model.predict(test_images_ds)

        np.save(f'fold{ifold+1}.npy', predict)

        #

        cm_predict = model.predict(validation_data)

        np.save(f'cm_fold{ifold+1}.npy', cm_predict)

        histories.append(history)

        # to avoid memory issues with TPU

        # https://www.kaggle.com/c/flower-classification-with-tpus/discussion/131045

        del model, train_dataset, val_dataset, validation_data, predict, cm_predict

        #K.clear_session()

        gc.collect()

        if tpu:

            tf.tpu.experimental.initialize_tpu_system(tpu)

    return histories #, predictions
# run train

histories = train_cross_validate_predict(MODEL_FILENAMES)
from matplotlib import pyplot as plt



def display_training_curves(training, validation, title, subplot):

    if subplot%10==1: # set up the subplots on the first call

        plt.subplots(figsize=(10,8), facecolor='#F0F0F0')

        plt.tight_layout()

    ax = plt.subplot(subplot)

    ax.set_facecolor('#F8F8F8')

    ax.plot(training)

    if(validation):

        ax.plot(validation)

    if len(title) > 0:

        ax.set_title(title)

    ax.set_ylabel(title)

    #ax.set_ylim(0.28,1.05)

    ax.set_xlabel('epoch')

    ax.legend(['train', 'valid.'])



if not TENSOR_BOARD:  

    # Learning rate schedule graph

    lrfn_rng = [i for i in range(25 if EPOCHS<25 else EPOCHS)]

    lrfn_y = [lrfn(x) for x in lrfn_rng]

    a = plt.plot(lrfn_rng, lrfn_y)



    print_and_log('show loss and accuracy')

    for ifold, history in enumerate(histories):

        display_training_curves(

            history.history['loss'], 

            history.history['val_loss'], 

            f'kfold{ifold+1} {MODEL_NAME[ifold]} loss', 221)

        display_training_curves(

            history.history[selected_metrics], 

            history.history['val_' + selected_metrics], 

            f'kfold{ifold+1} {MODEL_NAME[ifold]} accuracy', 222)
# logs

#!tar -zcvf tensorboard.tar.gz logs



# # Upload to colab 

# !tar -xvf tensorboard.tar.gz

# # run tensor board

# %tensorboard --logdir logs/fit
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from matplotlib.colors import LinearSegmentedColormap



def custom_cmap():

    # custom cmap, replace 0 with gray

    cmap = plt.cm.Reds

    cmaplist = [cmap(i) for i in range(cmap.N)]

    cmaplist[0] = (0, 0, 0, 0.2)

    return LinearSegmentedColormap.from_list('mcm' ,cmaplist, cmap.N)



def display_confusion_matrix(cmat, score, precision, recall):

    plt.figure(figsize=(15,15))

    ax = plt.gca()

    ax.matshow(cmat, cmap=custom_cmap())

    ax.set_xticks(range(len(CLASSES)))

    ax.set_xticklabels(CLASSES, fontdict={'fontsize': 7})

    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")

    ax.set_yticks(range(len(CLASSES)))

    ax.set_yticklabels(CLASSES, fontdict={'fontsize': 7})

    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    titlestring = ""

    if score is not None:

        titlestring += 'f1 = {:.3f} '.format(score)

    if precision is not None:

        titlestring += '\nprecision = {:.3f} '.format(precision)

    if recall is not None:

        titlestring += '\nrecall = {:.3f} '.format(recall)

    if len(titlestring) > 0:

        ax.text(101, 1, titlestring, fontdict={'fontsize': 18, 

            'horizontalalignment':'right', 'verticalalignment':'top', 'color':'#804040'})

    plt.show()

    

def plot_confusion_matrix(model_predictions):

    all_labels = []; all_pred = []



    for j, (trn_ind, val_ind) in enumerate( kfold.split(MODEL_FILENAMES) ):

        vdf = list(pd.DataFrame({'FILENAMES': MODEL_FILENAMES}).loc[val_ind]['FILENAMES'])

        NUM_VALIDATION_IMAGES = count_data_items(vdf)

        val_dataset = load_dataset(vdf, labeled = True, ordered = True)

        cmdataset = get_validation_dataset(val_dataset)

        labels_ds = cmdataset.map(lambda image, label: label).unbatch()

        all_labels.append(next(iter(labels_ds.batch(NUM_VALIDATION_IMAGES))).numpy())

        prob = model_predictions[j]

        all_pred.append( np.argmax(prob, axis=-1) )



    cm_correct_labels = np.concatenate(all_labels)

    cm_predictions = np.concatenate(all_pred)



    cmat = confusion_matrix(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)))

    score = f1_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

    precision = precision_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

    recall = recall_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

    display_confusion_matrix(cmat, score, precision, recall)



    print('f1 score: {:.3f}, precision: {:.3f}, recall: {:.3f}'.format(score, precision, recall))

    
#load predictions

fold_predictions = []

try:

    for ifold in range(FOLDS):

        predict = np.load(f'cm_fold{ifold+1}.npy')

        fold_predictions.append(predict)

except:

    print('KFold processing issue')



print_and_log('show confusion matrix')

plot_confusion_matrix(fold_predictions)
def get_predictions(preds):

    #test_images_ds = test_ds.map(lambda image, idnum: image)

    print('Computing predictions...')

    # get the mean probability of the folds models

    probabilities = np.average([preds[i] for i in range(len(preds))], axis = 0)

    predictions = np.argmax(probabilities, axis=-1)

    print(predictions)

    return predictions



def save_prediction(predictions):

    print('Generating submission.csv file...')

    test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()

    # all in one batch

    test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') 

    np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), 

               fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')
#load predictions

predictions = []

try:

    for ifold in range(FOLDS):

        predict = np.load(f'fold{ifold+1}.npy')

        predictions.append(predict)

except:

    print('KFold processing issue')

    

print_and_log('save prediction')

save_prediction(get_predictions(predictions))



# print head

!head submission.csv