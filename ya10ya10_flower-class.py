!pip install --quiet efficientnet



import numpy as np

import pandas as pd

import seaborn as sns

import os, re, math, warnings, random

from matplotlib import pyplot as plt

from kaggle_datasets import KaggleDatasets

from sklearn.model_selection import KFold

from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf

import tensorflow.keras.layers as L

import tensorflow.keras.backend as K

from tensorflow.keras import optimizers, applications, Sequential, losses

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

import efficientnet.tfkeras as efn



def seed_everything(seed=0):

    random.seed(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'



seed = 0

seed_everything(seed)

warnings.filterwarnings('ignore')
# TPU or GPU detection

# Detect hardware, return appropriate distribution strategy

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print(f'Running on TPU {tpu.master()}')

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy()



AUTO = tf.data.experimental.AUTOTUNE

REPLICAS = strategy.num_replicas_in_sync

print(f'REPLICAS: {REPLICAS}')
BATCH_SIZE = 16 * REPLICAS

LEARNING_RATE = 3e-5 * REPLICAS

EPOCHS = 20

HEIGHT = 331

WIDTH = 331

CHANNELS = 3

N_CLASSES = 104

ES_PATIENCE = 5

N_FOLDS = 5

FOLDS_USED = 5
GCS_PATH = KaggleDatasets().get_gcs_path() + '/tfrecords-jpeg-%sx%s' % (HEIGHT, WIDTH)



TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train/*.tfrec') + tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')

TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/test/*.tfrec')



CLASSES = [

    'pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 

    'wild geranium', 'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood', 

    'globe thistle', 'snapdragon', "colt's foot", 'king protea', 'spear thistle', 

    'yellow iris', 'globe-flower', 'purple coneflower', 'peruvian lily', 

    'balloon flower', 'giant white arum lily', 'fire lily', 'pincushion flower', 

    'fritillary', 'red ginger', 'grape hyacinth', 'corn poppy', 

    'prince of wales feathers', 'stemless gentian', 'artichoke', 'sweet william', 

    'carnation', 'garden phlox', 'love in the mist', 'cosmos',  'alpine sea holly', 

    'ruby-lipped cattleya', 'cape flower', 'great masterwort',  'siam tulip', 

    'lenten rose', 'barberton daisy', 'daffodil',  'sword lily', 'poinsettia', 

    'bolero deep blue',  'wallflower', 'marigold', 'buttercup', 'daisy', 

    'common dandelion', 'petunia', 'wild pansy', 'primula',  'sunflower', 

    'lilac hibiscus', 'bishop of llandaff', 'gaura',  'geranium', 'orange dahlia', 

    'pink-yellow dahlia', 'cautleya spicata',  'japanese anemone', 

    'black-eyed susan', 'silverbush', 'californian poppy',  'osteospermum', 

    'spring crocus', 'iris', 'windflower',  'tree poppy', 'gazania', 'azalea', 

    'water lily',  'rose', 'thorn apple', 'morning glory', 'passion flower',  

    'lotus', 'toad lily', 'anthurium', 'frangipani',  'clematis', 'hibiscus', 

    'columbine', 'desert-rose', 'tree mallow', 'magnolia', 'cyclamen ', 

    'watercress',  'canna lily', 'hippeastrum ', 'bee balm', 'pink quill',  

    'foxglove', 'bougainvillea', 'camellia', 'mallow',  'mexican petunia',  

    'bromelia', 'blanket flower', 'trumpet creeper',  'blackberry lily', 

    'common tulip', 'wild rose']
def count_data_items(filenames):

    # the number of data items is written in the name of the .tfrec files.

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)



# Train data

NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)

print(f'Number of training images {NUM_TRAINING_IMAGES}')



# Test data

NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)

print(f'Number of test images {NUM_TEST_IMAGES}')
def random_blockout(img, sl=0.1, sh=0.2, rl=0.4):

    p=random.random()

    if p>=0.25:

        w, h, c = HEIGHT, WIDTH, 3

        origin_area = tf.cast(h*w, tf.float32)



        e_size_l = tf.cast(tf.round(tf.sqrt(origin_area * sl * rl)), tf.int32)

        e_size_h = tf.cast(tf.round(tf.sqrt(origin_area * sh / rl)), tf.int32)



        e_height_h = tf.minimum(e_size_h, h)

        e_width_h = tf.minimum(e_size_h, w)



        erase_height = tf.random.uniform(shape=[], minval=e_size_l, maxval=e_height_h, dtype=tf.int32)

        erase_width = tf.random.uniform(shape=[], minval=e_size_l, maxval=e_width_h, dtype=tf.int32)



        erase_area = tf.zeros(shape=[erase_height, erase_width, c])

        erase_area = tf.cast(erase_area, tf.uint8)



        pad_h = h - erase_height

        pad_top = tf.random.uniform(shape=[], minval=0, maxval=pad_h, dtype=tf.int32)

        pad_bottom = pad_h - pad_top



        pad_w = w - erase_width

        pad_left = tf.random.uniform(shape=[], minval=0, maxval=pad_w, dtype=tf.int32)

        pad_right = pad_w - pad_left



        erase_mask = tf.pad([erase_area], [[0,0],[pad_top, pad_bottom], [pad_left, pad_right], [0,0]], constant_values=1)

        erase_mask = tf.squeeze(erase_mask, axis=0)

        erased_img = tf.multiply(tf.cast(img,tf.float32), tf.cast(erase_mask, tf.float32))



        return tf.cast(erased_img, img.dtype)

    else:

        return tf.cast(img, img.dtype)
def dropout(image, DIM = HEIGHT, PROBABILITY = 1, CT = 8, SZ = 0.2):

    

    prob = tf.cast( tf.random.uniform([],0,1)<PROBABILITY, tf.int32)

    if (prob==0)|(CT==0)|(SZ==0): return image

    

    for k in range(CT):



        x = tf.cast( tf.random.uniform([],0,DIM),tf.int32)

        y = tf.cast( tf.random.uniform([],0,DIM),tf.int32)

        WIDTH = tf.cast( SZ*DIM,tf.int32) * prob

        ya = tf.math.maximum(0,y-WIDTH//2)

        yb = tf.math.minimum(DIM,y+WIDTH//2)

        xa = tf.math.maximum(0,x-WIDTH//2)

        xb = tf.math.minimum(DIM,x+WIDTH//2)



        one = image[ya:yb,0:xa,:]

        two = tf.zeros([yb-ya,xb-xa,3]) 

        three = image[ya:yb,xb:DIM,:]

        middle = tf.concat([one,two,three],axis=1)

        image = tf.concat([image[0:ya,:,:],middle,image[yb:DIM,:,:]],axis=0)

            

    image = tf.reshape(image,[DIM,DIM,3])

    

    return image
def data_augment(image, label):

    p_rotation = tf.random.uniform([], 0, 1.0, dtype=tf.float32)

    p_spatial = tf.random.uniform([], 0, 1.0, dtype=tf.float32)

    p_rotate = tf.random.uniform([], 0, 1.0, dtype=tf.float32)

    p_pixel = tf.random.uniform([], 0, 1.0, dtype=tf.float32)    

    p_shear = tf.random.uniform([], 0, 1.0, dtype=tf.float32)

    p_shift = tf.random.uniform([], 0, 1.0, dtype=tf.float32)

    p_crop = tf.random.uniform([], 0, 1.0, dtype=tf.float32)

    

    

    # Flips

    if p_spatial >= .2:

        image = tf.image.random_flip_left_right(image)

        image = tf.image.random_flip_up_down(image)

        

    # Rotates

    if p_rotate > .75:

        image = tf.image.rot90(image, k=3) # rotate 270º

    elif p_rotate > .5:

        image = tf.image.rot90(image, k=2) # rotate 180º

    elif p_rotate > .25:

        image = tf.image.rot90(image, k=1) # rotate 90º

    

    if p_rotation >= .3: # Rotation

        image = transform_rotation(image, height=HEIGHT, rotation=45.)

    if p_shift >= .3: # Shift

        image = transform_shift(image, height=HEIGHT, h_shift=15., w_shift=15.)

    if p_shear >= .3: # Shear

        image = transform_shear(image, height=HEIGHT, shear=20.)

        

    # Crops

    if p_crop > .4:

        crop_size = tf.random.uniform([], int(HEIGHT*.7), HEIGHT, dtype=tf.int32)

        image = tf.image.random_crop(image, size=[crop_size, crop_size, CHANNELS])

    elif p_crop > .7:

        if p_crop > .9:

            image = tf.image.central_crop(image, central_fraction=.7)

        elif p_crop > .8:

            image = tf.image.central_crop(image, central_fraction=.8)

        else:

            image = tf.image.central_crop(image, central_fraction=.9)

            

    image = tf.image.resize(image, size=[HEIGHT, WIDTH])

        

    # Pixel-level transforms

    if p_pixel >= .2:

        if p_pixel >= .8:

            image = tf.image.random_saturation(image, lower=0, upper=2)

        elif p_pixel >= .6:

            image = tf.image.random_contrast(image, lower=.8, upper=2)

        elif p_pixel >= .4:

            image = tf.image.random_brightness(image, max_delta=.2)

        else:

            image = tf.image.adjust_gamma(image, gamma=.6)



    # Blockout    

    #image = random_blockout(image)

    

    # Dropout

    image = dropout(image)

    

    return image, label
# data augmentation @cdeotte kernel: https://www.kaggle.com/cdeotte/rotation-augmentation-gpu-tpu-0-96

def transform_rotation(image, height, rotation):

    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]

    # output - image randomly rotated

    DIM = height

    XDIM = DIM%2 #fix for size 331

    

    rotation = rotation * tf.random.uniform([1],dtype='float32')

    # CONVERT DEGREES TO RADIANS

    rotation = math.pi * rotation / 180.

    

    # ROTATION MATRIX

    c1 = tf.math.cos(rotation)

    s1 = tf.math.sin(rotation)

    one = tf.constant([1],dtype='float32')

    zero = tf.constant([0],dtype='float32')

    rotation_matrix = tf.reshape(tf.concat([c1,s1,zero, -s1,c1,zero, zero,zero,one],axis=0),[3,3])



    # LIST DESTINATION PIXEL INDICES

    x = tf.repeat( tf.range(DIM//2,-DIM//2,-1), DIM )

    y = tf.tile( tf.range(-DIM//2,DIM//2),[DIM] )

    z = tf.ones([DIM*DIM],dtype='int32')

    idx = tf.stack( [x,y,z] )

    

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS

    idx2 = K.dot(rotation_matrix,tf.cast(idx,dtype='float32'))

    idx2 = K.cast(idx2,dtype='int32')

    idx2 = K.clip(idx2,-DIM//2+XDIM+1,DIM//2)

    

    # FIND ORIGIN PIXEL VALUES 

    idx3 = tf.stack( [DIM//2-idx2[0,], DIM//2-1+idx2[1,]] )

    d = tf.gather_nd(image, tf.transpose(idx3))

        

    return tf.reshape(d,[DIM,DIM,3])



def transform_shear(image, height, shear):

    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]

    # output - image randomly sheared

    DIM = height

    XDIM = DIM%2 #fix for size 331

    

    shear = shear * tf.random.uniform([1],dtype='float32')

    shear = math.pi * shear / 180.

        

    # SHEAR MATRIX

    one = tf.constant([1],dtype='float32')

    zero = tf.constant([0],dtype='float32')

    c2 = tf.math.cos(shear)

    s2 = tf.math.sin(shear)

    shear_matrix = tf.reshape(tf.concat([one,s2,zero, zero,c2,zero, zero,zero,one],axis=0),[3,3])    



    # LIST DESTINATION PIXEL INDICES

    x = tf.repeat( tf.range(DIM//2,-DIM//2,-1), DIM )

    y = tf.tile( tf.range(-DIM//2,DIM//2),[DIM] )

    z = tf.ones([DIM*DIM],dtype='int32')

    idx = tf.stack( [x,y,z] )

    

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS

    idx2 = K.dot(shear_matrix,tf.cast(idx,dtype='float32'))

    idx2 = K.cast(idx2,dtype='int32')

    idx2 = K.clip(idx2,-DIM//2+XDIM+1,DIM//2)

    

    # FIND ORIGIN PIXEL VALUES 

    idx3 = tf.stack( [DIM//2-idx2[0,], DIM//2-1+idx2[1,]] )

    d = tf.gather_nd(image, tf.transpose(idx3))

        

    return tf.reshape(d,[DIM,DIM,3])



def transform_shift(image, height, h_shift, w_shift):

    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]

    # output - image randomly shifted

    DIM = height

    XDIM = DIM%2 #fix for size 331

    

    height_shift = h_shift * tf.random.uniform([1],dtype='float32') 

    width_shift = w_shift * tf.random.uniform([1],dtype='float32') 

    one = tf.constant([1],dtype='float32')

    zero = tf.constant([0],dtype='float32')

        

    # SHIFT MATRIX

    shift_matrix = tf.reshape(tf.concat([one,zero,height_shift, zero,one,width_shift, zero,zero,one],axis=0),[3,3])



    # LIST DESTINATION PIXEL INDICES

    x = tf.repeat( tf.range(DIM//2,-DIM//2,-1), DIM )

    y = tf.tile( tf.range(-DIM//2,DIM//2),[DIM] )

    z = tf.ones([DIM*DIM],dtype='int32')

    idx = tf.stack( [x,y,z] )

    

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS

    idx2 = K.dot(shift_matrix,tf.cast(idx,dtype='float32'))

    idx2 = K.cast(idx2,dtype='int32')

    idx2 = K.clip(idx2,-DIM//2+XDIM+1,DIM//2)

    

    # FIND ORIGIN PIXEL VALUES 

    idx3 = tf.stack( [DIM//2-idx2[0,], DIM//2-1+idx2[1,]] )

    d = tf.gather_nd(image, tf.transpose(idx3))

        

    return tf.reshape(d,[DIM,DIM,3])
# Datasets utility functions

def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=CHANNELS)

    image = tf.cast(image, tf.float32) / 255.0

    image = tf.reshape(image, [HEIGHT, WIDTH, CHANNELS])

    return image



def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element

    }

    

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    label = tf.cast(example['class'], tf.int32)

    return image, label



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

    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False # disable order, increase speed



    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files

    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order

    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)

    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False

    return dataset



def get_dataset(filenames, labeled=True, ordered=True, repeated=False, shufled=False, augmented=False):

    dataset = load_dataset(filenames, labeled=labeled, ordered=ordered)

    if augmented:

        dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

    if repeated:

        dataset = dataset.repeat() # the training dataset must repeat for several epochs

    if shufled:

        dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset
# Visualization utility functions

np.set_printoptions(threshold=15, linewidth=80)



def plot_metrics(history, metric_list):

    fig, axes = plt.subplots(len(metric_list), 1, sharex='col', figsize=(24, 12))

    axes = axes.flatten()

    

    for index, metric in enumerate(metric_list):

        axes[index].plot(history[metric], label='Train %s' % metric)

        axes[index].plot(history['val_%s' % metric], label='Validation %s' % metric)

        axes[index].legend(loc='best', fontsize=16)

        axes[index].set_title(metric)



    plt.xlabel('Epochs', fontsize=16)

    sns.despine()

    plt.show()



    

def dataset_to_numpy_util(dataset, N):

    dataset = dataset.unbatch().batch(N)

    for images, labels in dataset:

        numpy_images = images.numpy()

        numpy_labels = labels.numpy()

        break;  

    return numpy_images, numpy_labels



def title_from_label_and_target(label, correct_label):

    label = np.argmax(label, axis=-1)

    correct = (label == correct_label)

    return "{} [{}{}{}]".format(CLASSES[label], str(correct), ', shoud be ' if not correct else '',

                                CLASSES[correct_label] if not correct else ''), correct



def display_one_flower_eval(image, title, subplot, red=False):

    plt.subplot(subplot)

    plt.axis('off')

    plt.imshow(image)

    plt.title(title, fontsize=14, color='red' if red else 'black')

    return subplot+1



def display_9_images_with_predictions(images, predictions, labels):

    subplot=331

    plt.figure(figsize=(10, 10))

    for i, image in enumerate(images):

        title, correct = title_from_label_and_target(predictions[i], labels[i])

        subplot = display_one_flower_eval(image, title, subplot, not correct)

        if i >= 8:

            break;

              

    plt.tight_layout()

    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    plt.show()

    

    

# Methods to display images

def batch_to_numpy_images_and_labels(data):

    images, labels = data

    numpy_images = images.numpy()

    numpy_labels = labels.numpy()

    if numpy_labels.dtype == object: # binary string in this case, these are image ID strings

        numpy_labels = [None for _ in enumerate(numpy_images)]

    # If no labels, only image IDs, return None for labels (this is the case for test data)

    return numpy_images, numpy_labels



def title_from_label_and_target_(label, correct_label):

    if correct_label is None:

        return CLASSES[label], True

    correct = (label == correct_label)

    return "{} [{}{}{}]".format(CLASSES[label], 'OK' if correct else 'NO', u"\u2192" if not correct else '',

                                CLASSES[correct_label] if not correct else ''), correct



def display_one_flower(image, title, subplot, red=False, titlesize=16):

    plt.subplot(*subplot)

    plt.axis('off')

    plt.imshow(image)

    if len(title) > 0:

        plt.title(title, fontsize=int(titlesize) if not red else int(titlesize/1.2), color='red' if red else 'black', fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))

    return (subplot[0], subplot[1], subplot[2]+1)



def display_batch_of_images(databatch, predictions=None):

    """This will work with:

    display_batch_of_images(images)

    display_batch_of_images(images, predictions)

    display_batch_of_images((images, labels))

    display_batch_of_images((images, labels), predictions)

    """

    # data

    images, labels = batch_to_numpy_images_and_labels(databatch)

    if labels is None:

        labels = [None for _ in enumerate(images)]

        

    # auto-squaring: this will drop data that does not fit into square or square-ish rectangle

    rows = int(math.sqrt(len(images)))

    cols = len(images)//rows

        

    # size and spacing

    FIGSIZE = 13.0

    SPACING = 0.1

    subplot=(rows,cols,1)

    if rows < cols:

        plt.figure(figsize=(FIGSIZE,FIGSIZE/cols*rows))

    else:

        plt.figure(figsize=(FIGSIZE/rows*cols,FIGSIZE))

    

    # display

    for i, (image, label) in enumerate(zip(images[:rows*cols], labels[:rows*cols])):

        title = '' if label is None else CLASSES[label]

        correct = True

        if predictions is not None:

            title, correct = title_from_label_and_target_(predictions[i], label)

        dynamic_titlesize = FIGSIZE*SPACING/max(rows,cols)*40+3 # magic formula tested to work from 1x1 to 10x10 images

        subplot = display_one_flower(image, title, subplot, not correct, titlesize=dynamic_titlesize)

    

    #layout

    plt.tight_layout()

    if label is None and predictions is None:

        plt.subplots_adjust(wspace=0, hspace=0)

    else:

        plt.subplots_adjust(wspace=SPACING, hspace=SPACING)

    plt.show()
train_dataset_aug = get_dataset(TRAINING_FILENAMES, labeled=True, ordered=False, repeated=True, shufled=True, augmented=True)

display_batch_of_images(next(iter(train_dataset_aug.unbatch().batch(20))))

display_batch_of_images(next(iter(train_dataset_aug.unbatch().batch(20))))

display_batch_of_images(next(iter(train_dataset_aug.unbatch().batch(20))))
def exponential_schedule_with_warmup(epoch):

    '''

    Create a schedule with a learning rate that decreases exponentially after linearly increasing during a warmup period.

    '''

    

    warmup_epochs=3

    hold_max_epochs=0

    lr_start=1e-6

    lr_max=LEARNING_RATE

    lr_min=1e-6

    decay=0.8

        

        

    if epoch < warmup_epochs:

        lr = (lr_max - lr_start) / warmup_epochs * epoch + lr_start

    elif epoch < warmup_epochs + hold_max_epochs:

        lr = lr_max

    else:

        lr = lr_max * (decay ** (epoch - warmup_epochs - hold_max_epochs))

        if lr_min is not None:

            lr = tf.math.maximum(lr_min, lr)

            

    return lr



    

rng = [i for i in range(EPOCHS)]

y = [exponential_schedule_with_warmup(x) for x in rng]



sns.set(style='whitegrid')

fig, ax = plt.subplots(figsize=(20, 6))

plt.plot(rng, y)



print(f'{EPOCHS} total epochs and {NUM_TRAINING_IMAGES//BATCH_SIZE} steps per epoch')

print(f'Learning rate schedule: {y[0]:.3g} to { max(y):.3g} to { y[-1]:.3g}')
def create_model(input_shape, N_CLASSES):

    base_model = efn.EfficientNetB4(weights='noisy-student', 

                                    include_top=False,

                                    input_shape=input_shape)



    model = tf.keras.Sequential([

                base_model,

                L.GlobalAveragePooling2D(),

                L.Dense(N_CLASSES, activation='softmax')

            ])

    

    

    optimizer = optimizers.Adam(lr=LEARNING_RATE)

    model.compile(optimizer=optimizer, 

                  loss=losses.SparseCategoricalCrossentropy(), 

                  metrics=['sparse_categorical_accuracy'])

    

    return model
kfold = KFold(N_FOLDS, shuffle=True, random_state=seed)

history_list = []



# Datasets

complete_dataset = get_dataset(TRAINING_FILENAMES, labeled=True, ordered=True)

test_dataset = get_dataset(TEST_FILENAMES, labeled=False, ordered=True)

x_complete = complete_dataset.map(lambda image, label: image)

y_complete = next(iter(complete_dataset.unbatch().map(lambda image, label: label).batch(NUM_TRAINING_IMAGES))).numpy()

x_test = test_dataset.map(lambda image, idnum: image)

# Predictions

complete_preds = np.zeros((NUM_TRAINING_IMAGES, N_CLASSES))

test_preds = np.zeros((NUM_TEST_IMAGES, N_CLASSES))





for n_fold, (trn_ind, val_ind) in enumerate(kfold.split(TRAINING_FILENAMES)):

    if n_fold >= FOLDS_USED:

        break

        

    print(f'\nFOLD: {n_fold+1}')

    tf.tpu.experimental.initialize_tpu_system(tpu)

    

    ### Data

    fold_train_filenames = np.asarray(TRAINING_FILENAMES)[trn_ind]

    fold_valid_filenames = np.asarray(TRAINING_FILENAMES)[val_ind]

    train_size = count_data_items(fold_train_filenames)

    validation_size = count_data_items(fold_valid_filenames)

    STEPS_PER_EPOCH = train_size // BATCH_SIZE



    ### Train model

    K.clear_session()

    model_path = f'model_{HEIGHT}x{WIDTH}_fold_{n_fold+1}.h5'



    with strategy.scope():

        model = create_model((None, None, CHANNELS), N_CLASSES)



    es = EarlyStopping(monitor='val_loss', mode='min', patience=ES_PATIENCE, restore_best_weights=True, verbose=1)

    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', mode='min', save_best_only=True)

    lr_callback = LearningRateScheduler(exponential_schedule_with_warmup, verbose=0)





    history = model.fit(x=get_dataset(fold_train_filenames, labeled=True, ordered=False, repeated=True, shufled=True, augmented=True), 

                        validation_data=get_dataset(fold_valid_filenames, labeled=True, ordered=True), 

                        callbacks=[checkpoint, es, lr_callback], 

                        steps_per_epoch=STEPS_PER_EPOCH, 

                        epochs=EPOCHS, 

                        verbose=2).history

    

    history_list.append(history)

    complete_preds += model.predict(x_complete) / FOLDS_USED

    test_preds += model.predict(x_test) / FOLDS_USED

    

complete_preds = np.argmax(complete_preds, axis=-1)

test_preds = np.argmax(test_preds, axis=-1)
for index, history in enumerate(history_list):

    print(f'FOLD {index+1}')

    plot_metrics(history, metric_list=['loss', 'sparse_categorical_accuracy'])
print(classification_report(y_complete, complete_preds, target_names=CLASSES))
fig, ax = plt.subplots(1, 1, figsize=(20, 45))

cfn_matrix = confusion_matrix(y_complete, complete_preds, labels=range(len(CLASSES)))

cfn_matrix = (cfn_matrix.T / cfn_matrix.sum(axis=1)).T

df_cm = pd.DataFrame(cfn_matrix, index=CLASSES, columns=CLASSES)

ax = sns.heatmap(df_cm, cmap='Blues').set_title('Labels', fontsize=30)

plt.show()
x_samp, y_samp = dataset_to_numpy_util(complete_dataset, 9)

samp_preds = model.predict(x_samp, batch_size=9)

display_9_images_with_predictions(x_samp, samp_preds, y_samp)
test_ids_ds = test_dataset.map(lambda image, idnum: idnum).unbatch()

test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U')



submission = pd.DataFrame(test_ids, columns=['id'])

submission['label'] = test_preds

submission.to_csv('submission.csv', index=False)

display(submission.head(10))