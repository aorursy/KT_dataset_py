import numpy as np

import pandas as pd



import tensorflow as tf

from kaggle_datasets import KaggleDatasets

print("Tensorflow version " + tf.__version__)



import IPython.display as display

from PIL import Image

import matplotlib.pyplot as plt

import glob



import random, re, math

import numpy as np, pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

import tensorflow as tf, tensorflow.keras.backend as K

from kaggle_datasets import KaggleDatasets

print('Tensorflow version ' + tf.__version__)

from sklearn.model_selection import KFold
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
GCS_DS_PATH = KaggleDatasets().get_gcs_path('product-detection') # you can list the bucket with "!gsutil ls $GCS_DS_PATH"

GCS_DS_PATH
IMAGE_SIZE = [512, 512] # at this size, a GPU will run out of memory. Use the TPU

EPOCHS = 20

BATCH_SIZE = 16 * strategy.num_replicas_in_sync



NUM_TRAINING_IMAGES = 94852

NUM_VAL_IMAGES = 10540

NUM_TEST_IMAGES = 12192

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

AUTO = tf.data.experimental.AUTOTUNE



MIXED_PRECISION = True

XLA_ACCELERATE = True



# https://www.tensorflow.org/guide/mixed_precision

# used to reduce size of model and increase training speed for GPU/TPU.



if MIXED_PRECISION:

    from tensorflow.keras.mixed_precision import experimental as mixed_precision

    if tpu: policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')

    else: policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')

    mixed_precision.set_policy(policy)

    print('Mixed precision enabled')



# https://www.tensorflow.org/xla

# reduce memory usuage and and processing speed for GPU.

if XLA_ACCELERATE:

    tf.config.optimizer.set_jit(True)

    print('Accelerated Linear Algebra enabled')
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





def transform(image,label):

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

        

    return tf.reshape(d,[DIM,DIM,3]),label



def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range

    image = tf.image.resize(image, [IMAGE_SIZE[0], IMAGE_SIZE[1]])

    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU. MUST HAVE THIS TO USE TPU.

    return image



def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "label": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    label = tf.cast(example['label'], tf.int32)

    return image, label # returns a dataset of (image, label) pairs



def read_unlabeled_tfrecord(example):

    UNLABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element

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



    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files

    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order

    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)

    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False

    return dataset



def onehot(image,label):

    CLASSES = 42

    return image,tf.one_hot(label,CLASSES)



def data_augment(image, label):

#     image = tf.image.random_flip_left_right(image)

#     image = tf.image.random_flip_up_down(image)

    image, label = gridmask(image, label, mask)

    image, label = onehot(image, label)

    return image, label  





def get_training_dataset():

    dataset = load_dataset(tf.io.gfile.glob(GCS_DS_PATH + '/train_*.tfrecords'))

    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

#     dataset = dataset.map(transform, num_parallel_calls=AUTO)

    dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_validation_dataset():

    dataset = load_dataset(tf.io.gfile.glob(GCS_DS_PATH + '/val_p0.tfrecords'))

    dataset = dataset.map(onehot, num_parallel_calls=AUTO)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.cache()

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_test_dataset(ordered):

    dataset = load_dataset(tf.io.gfile.glob(GCS_DS_PATH + '/test_p0.tfrecords'), labeled=False, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset
def transform(image, inv_mat, image_shape):

    h, w, c = image_shape

    cx, cy = w//2, h//2

    new_xs = tf.repeat( tf.range(-cx, cx, 1), h)

    new_ys = tf.tile( tf.range(-cy, cy, 1), [w])

    new_zs = tf.ones([h*w], dtype=tf.int32)

    old_coords = tf.matmul(inv_mat, tf.cast(tf.stack([new_xs, new_ys, new_zs]), tf.float32))

    old_coords_x, old_coords_y = tf.round(old_coords[0, :] + w//2), tf.round(old_coords[1, :] + h//2)

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

        rotated_image_channel.append(tf.sparse.to_dense(sparse_channel, default_value=0, validate_indices=False))

    return tf.transpose(tf.stack(rotated_image_channel), [1,2,0])





def random_rotate(image, angle, image_shape):

    def get_rotation_mat_inv(angle):

        # transform to radian

        angle = math.pi * angle / 180

        cos_val = tf.math.cos(angle)

        sin_val = tf.math.sin(angle)

        one = tf.constant([1], tf.float32)

        zero = tf.constant([0], tf.float32)

        rot_mat_inv = tf.concat([cos_val, sin_val, zero, -sin_val, cos_val, zero, zero, zero, one], axis=0)

        rot_mat_inv = tf.reshape(rot_mat_inv, [3,3])

        return rot_mat_inv

    angle = float(angle) * tf.random.normal([1],dtype='float32')

    rot_mat_inv = get_rotation_mat_inv(angle)

    return transform(image, rot_mat_inv, image_shape)





def GridMask(image_height, image_width, d1, d2, rotate_angle=1, ratio=0.5):

    h, w = image_height, image_width

    hh = int(np.ceil(np.sqrt(h*h+w*w))) # round up length of the image's diagonal

    hh = hh+1 if hh%2==1 else hh # make value even

    d = tf.random.uniform(shape=[], minval=d1, maxval=d2, dtype=tf.int32) # d is a value drawn from a uniform distribution in [d1, d2]

    l = tf.cast(tf.cast(d,tf.float32)*ratio+0.5, tf.int32) # l = d * r + 0.5 



    st_h = tf.random.uniform(shape=[], minval=0, maxval=d, dtype=tf.int32) # st_h is a value drawn from a uniform distribution in [0, d]

    st_w = tf.random.uniform(shape=[], minval=0, maxval=d, dtype=tf.int32) # st_w is a value drawn from a uniform distribution in [0, d]



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



    hh_ranges = tf.tile(tf.range(0,hh), [tf.cast(tf.reduce_sum(tf.ones_like(x_ranges)), tf.int32)])

    x_ranges = tf.repeat(x_ranges, hh)

    y_ranges = tf.repeat(y_ranges, hh)



    y_hh_indices = tf.transpose(tf.stack([y_ranges, hh_ranges]))

    x_hh_indices = tf.transpose(tf.stack([hh_ranges, x_ranges]))



    y_mask_sparse = tf.SparseTensor(tf.cast(y_hh_indices, tf.int64),  tf.zeros_like(y_ranges), [hh, hh])

    y_mask = tf.sparse.to_dense(y_mask_sparse, 1, False)



    x_mask_sparse = tf.SparseTensor(tf.cast(x_hh_indices, tf.int64), tf.zeros_like(x_ranges), [hh, hh])

    x_mask = tf.sparse.to_dense(x_mask_sparse, 1, False)



    mask = tf.expand_dims( tf.clip_by_value(x_mask + y_mask, 0, 1), axis=-1)



    mask = random_rotate(mask, rotate_angle, [hh, hh, 1])

    mask = tf.image.crop_to_bounding_box(mask, (hh-h)//2, (hh-w)//2, image_height, image_width)



    return mask





AugParams = {

        'd1' : 100,

        'd2': 160,

        'rotate' : 45,

        'ratio' : 0.2

}

mask = GridMask(IMAGE_SIZE[0], IMAGE_SIZE[1], AugParams['d1'], AugParams['d2'], AugParams['rotate'], AugParams['ratio'])

    

def apply_grid_mask(image, image_shape, mask):



    if image_shape[-1] == 3:

        mask = tf.concat([mask, mask, mask], axis=-1)

    return image * tf.cast(mask,tf.float32)





def gridmask(image, label, mask):

    return apply_grid_mask(image, (IMAGE_SIZE[0], IMAGE_SIZE[1], 3), mask), label
training_dataset = get_training_dataset()

validation_dataset = get_validation_dataset()
def label_smoothing(y_true,y_pred):

    return tf.keras.losses.categorical_crossentropy(y_true,y_pred,label_smoothing=0.2)

    

def create_model():

    with strategy.scope():    

        base_model = tf.keras.applications.DenseNet201(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), weights='imagenet', include_top=False)

        base_model.trainable = True

        model = tf.keras.Sequential([

          base_model,

          tf.keras.layers.GlobalAveragePooling2D(),

          tf.keras.layers.Dense(42, activation='softmax',dtype='float32')

        ])

        model.compile(optimizer='adam',

              loss=label_smoothing,

              metrics=['categorical_accuracy'])

#     model.load_weights('/kaggle/input/weights/weights.0.77.h5')

    return model

LR_START = 0.00001

LR_MAX = 0.00001 * strategy.num_replicas_in_sync

LR_MIN = 0.000001

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







model = create_model()

# rlr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.00001)

lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = True)

es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5)

mc_callback = tf.keras.callbacks.ModelCheckpoint('weights.{val_categorical_accuracy:.2f}.h5', 

                                        monitor='val_categorical_accuracy', 

                                        mode='max', 

                                        save_best_only=True,

                                        save_weights_only=True)    

history = model.fit(training_dataset,

                steps_per_epoch=STEPS_PER_EPOCH, 

                epochs=EPOCHS,

                validation_data=validation_dataset,

                callbacks=[mc_callback, lr_callback, es_callback])

def get_labeled_dataset(ordered):

    dataset = load_dataset(tf.io.gfile.glob(GCS_DS_PATH + '/train_*.tfrecords') + tf.io.gfile.glob(GCS_DS_PATH + '/val_p0.tfrecords'),ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset
# model = create_model()

# model.load_weights('../input/weights/weights.0.82.h5')





# Get submission.csv file based on all the images in test folder.

test_ds = get_test_dataset(ordered=True)

# test_ds = get_labeled_dataset(ordered=True)



print('Computing predictions...')

test_images_ds = test_ds.map(lambda image, idnum: image)

probabilities = model.predict(test_images_ds)

predictions = np.argmax(probabilities, axis=-1)

print(predictions)



print('Generating submission.csv file...')

test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()

test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch

np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')
# Edit submission.csv to include only images indicated in test.csv.



test_df = pd.read_csv('../input/product-detection/test.csv')

submission_df = pd.read_csv('submission.csv')



submission_df['id'] = submission_df['id'].apply(lambda x: x.split('/')[-1])

submission_df['label'] = submission_df['label'].apply(lambda x: "{:02}".format(x))



combined_df = pd.merge(test_df, submission_df,  right_on='id', left_on='filename')

combined_df = combined_df[['filename', 'label']].rename(columns={'label':'category'})



combined_df.to_csv('submission.csv', index=False)