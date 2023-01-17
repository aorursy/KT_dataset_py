import numpy as np 

import pandas as pd

import tensorflow as tf



# NOTE: internet should be ON

from kaggle_datasets import KaggleDatasets

GCS_DS_PATH = KaggleDatasets().get_gcs_path('flower-classification-with-tpus')
IMAGE_SIZE = [192, 192] 

IMAGE_CHANNELS = 3



# related notebook output is using TPU

BATCH_SIZE = 16 * 8 #strategy.num_replicas_in_sync



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

    'common tulip',                     'wild rose']#100 -103
import re



# this notebook is running on CPU 

# hardcode AUTO for dataset to load in same way as it done in 

# kernel-mashup notebook



AUTO = 24 # tf.data.experimental.AUTOTUNE



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

    #image = tf.image.random_flip_left_right(image)

    #image = tf.image.random_saturation(image, 0, 2)

    return image, label   



def get_training_dataset(dataset, do_aug=False, do_grid=False):

    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

    # Rotation Augmentation GPU/TPU

    #if do_aug: dataset = dataset.map(data_rotate, num_parallel_calls=AUTO)

    # grid mask

    #if do_grid: dataset = dataset.map(data_gridmask, num_parallel_calls=AUTO)   

    # the training dataset must repeat for several epochs

    dataset = dataset.repeat() 

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    # prefetch next batch while training (autotune prefetch buffer size)

    dataset = dataset.prefetch(AUTO) 

    return dataset



def get_validation_dataset(dataset):

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.cache()

    # prefetch next batch while training (autotune prefetch buffer size)

    dataset = dataset.prefetch(AUTO) 

    return dataset



def count_data_items(filenames):

    # the number of data items is written in the name of the .tfrec files,

    # i.e. flowers00-230.tfrec = 230 data items

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) 

         for filename in filenames]

    return np.sum(n)



NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)

NUM_VALIDATION_IMAGES = count_data_items(VALIDATION_FILENAMES)

NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

print('Dataset: {} training images, {} validation images, {} unlabeled test images'.

      format(NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES))
# load model names

mashup_path = "../input/kernel-mashup" 

MODEL_NAME = ['Model1', 'Model2', 'Model3']



try:

    MODEL_NAME = np.load(f'{mashup_path}/model_names.npy')

except:

    print('model names processing issue')
from sklearn.model_selection import KFold



FOLDS = len(MODEL_NAME)

SEED = 92

MODEL_FILENAMES = TRAINING_FILENAMES + VALIDATION_FILENAMES



kfold = KFold(FOLDS, shuffle = True, random_state = SEED)
#load predictions

fold_predictions = []

try:

    for ifold in range(FOLDS):

        predict = np.load(f'{mashup_path}/cm_fold{ifold+1}.npy')

        fold_predictions.append(predict)

except:

    print('KFold processing issue')
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from matplotlib import pyplot as plt

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

    

def get_labels(idx):

    tst_labels = []; val_labels = []



    j, (trn_ind, val_ind) = list(enumerate(kfold.split(MODEL_FILENAMES)))[idx]



    tdf = list(pd.DataFrame({'FILENAMES': MODEL_FILENAMES}).loc[trn_ind]['FILENAMES'])

    NUM_TRAIN_IMAGES = count_data_items(tdf)

    train_dataset = load_dataset(tdf, labeled = True)

    tdf_dataset = get_training_dataset(train_dataset, True)

    tst_labels_ds = tdf_dataset.map(lambda image, label: label).unbatch()

    tst_labels = next(iter(tst_labels_ds.batch(NUM_TRAIN_IMAGES))).numpy()

    

    vdf = list(pd.DataFrame({'FILENAMES': MODEL_FILENAMES}).loc[val_ind]['FILENAMES'])

    NUM_VALIDATION_IMAGES = count_data_items(vdf)

    val_dataset = load_dataset(vdf, labeled = True, ordered = True)

    cm_val_dataset = get_validation_dataset(val_dataset)

    labels_ds = cm_val_dataset.map(lambda image, label: label).unbatch()

    val_labels = next(iter(labels_ds.batch(NUM_VALIDATION_IMAGES))).numpy()

   

    return tst_labels, val_labels

    

def get_model_confusion_matrix_info(idx):

    all_pred = []



    tst_labels, val_labels = get_labels(idx)

    predictions = np.argmax(fold_predictions[idx], axis=-1)

    

    cmat = confusion_matrix(val_labels, predictions, 

                     labels=range(len(CLASSES)))

    score = f1_score(val_labels, predictions, 

                     labels=range(len(CLASSES)), average='macro')

    precision = precision_score(val_labels, predictions, 

                     labels=range(len(CLASSES)), average='macro')

    recall = recall_score(val_labels, predictions, 

                     labels=range(len(CLASSES)), average='macro')

   

    return (cmat, score, precision, recall)

    

def get_confusion_matrix_info():

    labels = []; pred = []



    for j, (trn_ind, val_ind) in enumerate( kfold.split(MODEL_FILENAMES) ):

        vdf = list(pd.DataFrame({'FILENAMES': MODEL_FILENAMES}).loc[val_ind]['FILENAMES'])

        NUM_VALIDATION_IMAGES = count_data_items(vdf)

        val_dataset = load_dataset(vdf, labeled = True, ordered = True)

        cmdataset = get_validation_dataset(val_dataset)

        labels_ds = cmdataset.map(lambda image, label: label).unbatch()

        try:

            pred.append( np.argmax(fold_predictions[j], axis=-1) )

            labels.append(next(iter(labels_ds.batch(NUM_VALIDATION_IMAGES))).numpy())

        except:

            print("unable to process ", j, " fold")



    cm_labels = np.concatenate(labels)

    cm_preds = np.concatenate(pred)



    cmat = confusion_matrix(cm_labels, cm_preds, labels=range(len(CLASSES)))

    score = f1_score(cm_labels, cm_preds, labels=range(len(CLASSES)), average='macro')

    precision = precision_score(cm_labels, cm_preds, labels=range(len(CLASSES)), average='macro')

    recall = recall_score(cm_labels, cm_preds, labels=range(len(CLASSES)), average='macro')

   

    return (cmat, score, precision, recall)

    
def count_fold_dataset(idx):

    ta, va = get_labels(idx)

    #

    t_df = pd.DataFrame(data = ta ) 

    tc_df = pd.DataFrame(data = t_df[0].value_counts()).sort_index()

    #

    v_df = pd.DataFrame(data = va ) 

    vc_df = pd.DataFrame(data = v_df[0].value_counts()).sort_index()

    # append pred

    ts = pd.Series(va != np.argmax(fold_predictions[idx], axis=-1), dtype='int')

    v_df.insert(1, "error", ts)

    ec_df = v_df.groupby([0]).sum().replace(0, np.nan)

    # Data

    return pd.DataFrame({'x': range(len(CLASSES)), 'test': tc_df[0], 

                         'validation': vc_df[0], 'error': ec_df['error']})
def try_get_data(idx):

    try:

        return count_fold_dataset(idx)

    except:

        return None   



df = []    

for i in range(FOLDS):

    df.append(try_get_data(i))
# multiple line plot

plt.figure(figsize=(30,10))



if df[0] is not None:

    plt.plot( 'x', 'test', data=df[0], marker='o', markerfacecolor='skyblue', 

             markersize=6, color='skyblue', linewidth=1)

    plt.plot( 'x', 'validation', data=df[0], marker='o', markerfacecolor='olive', 

             markersize=6, color='olive', linewidth=1)



if df[1] is not None:

    plt.plot( 'x', 'test', data=df[1], marker='o', markerfacecolor='teal', 

             markersize=6, color='teal', linewidth=1)

    plt.plot( 'x', 'validation', data=df[1], marker='o', markerfacecolor='gold', 

             markersize=6, color='gold', linewidth=1)



# if df2 is not None:

#     plt.plot( 'x', 'test', data=df2, marker='o', markerfacecolor='teal', 

#             markersize=6, color='teal', linewidth=1)

#     plt.plot( 'x', 'validation', data=df2, marker='o', markerfacecolor='gold', 

#             markersize=6, color='gold', linewidth=1)



plt.legend()
def plot_dataset_counts(idx):



    font = {'family': 'serif',

        'color':  'darkred',

        'weight': 'normal',

        'size': 16,

        }

    

    if df[idx] is not None:

        # multiple line plot

        plt.figure(figsize=(30,10))

        plt.plot( 'x', 'test', data=df[idx], marker='o', 

                 markerfacecolor='blue', 

                 markersize=6, color='skyblue', linewidth=1)

        plt.plot( 'x', 'validation', data=df[idx], marker='o', 

                 markerfacecolor='olive', 

                 markersize=6, color='olive', linewidth=1)

        plt.plot( 'x', 'error', data=df[idx], marker='o', 

                 markerfacecolor='red', 

                 markersize=6, color='red', linewidth=1)

        plt.xlabel(MODEL_NAME[idx], fontdict=font)

        plt.legend()
for idx in range(FOLDS):

    plot_dataset_counts(idx)
def try_get_model_confusion_matrix_info(idx):

    try:

        return get_model_confusion_matrix_info(idx)

    except:

        return None  
for j in range(FOLDS):

    info = try_get_model_confusion_matrix_info(j)

    if info is not None:

        print('{4} f1 score: {1:.3f}, precision: {2:.3f}, recall: {3:.3f}'.format(*info, MODEL_NAME[j]))

        display_confusion_matrix(*info)
info = get_confusion_matrix_info()

print('f1 score: {1:.3f}, precision: {2:.3f}, recall: {3:.3f}'.format(*info))

display_confusion_matrix(*info)

# numpy and matplotlib defaults

import math

np.set_printoptions(threshold=15, linewidth=80)



def batch_to_numpy_images_and_labels(data):

    images, labels = data

    numpy_images = images.numpy()

    numpy_labels = labels.numpy()

    # binary string in this case, these are image ID strings

    if numpy_labels.dtype == object: 

        numpy_labels = [None for _ in enumerate(numpy_images)]

    # If no labels, only image IDs, return None for labels 

    # (this is the case for test data)

    return numpy_images, numpy_labels



def title_from_label_and_target(label, correct_label):

    if correct_label is None:

        return CLASSES[label], True

    correct = (label == correct_label)

    ch = u"\u2192"

    if correct:

        return f"{CLASSES[label]} [OK]", correct

    else:

        return f"{CLASSES[label]} [OK]\n[NO{ch}{CLASSES[correct_label]}]", correct



def display_one_flower(image, title, subplot, red=False, titlesize=16):

    plt.subplot(*subplot)

    plt.axis('off')

    plt.imshow(image)

    if len(title) > 0:

        if not red:

            plt.title(title, fontsize=int(titlesize), color='black', 

              fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))

        else:

            plt.title(title, fontsize=int(titlesize/1.2), color='red', 

              fontdict={'verticalalignment':'center'}, pad=int(titlesize/2))

            

    return (subplot[0], subplot[1], subplot[2]+1)



def display_batch_of_images(images, labels = None, predictions=None, squaring=True):

    if labels is None:

        labels = [None for _ in enumerate(images)]

        

    # auto-squaring: this will drop data that does not fit into square 

    # or square-ish rectangle

    rows = int(math.sqrt(len(images)))

    cols = len(images)//rows

    

    if not squaring:

        cols = 5

        rows = len(images)//cols

        # limit by 100

        if rows > 100:

            rows = 100



    # size and spacing

    FIGSIZE = 13.0

    SPACING = 0.1

    subplot=(rows,cols,1)

    # figsize(width, height)

    if rows < cols:

        plt.figure(figsize=(FIGSIZE,FIGSIZE/cols*rows))

    elif not squaring:

        plt.figure(figsize=(FIGSIZE,FIGSIZE/cols*rows))

    else:

        plt.figure(figsize=(FIGSIZE/rows*cols,FIGSIZE))

    

    # display

    for i, (image, label) in enumerate(zip(images[:rows*cols], labels[:rows*cols])):

        title = '' if label is None else CLASSES[label]

        correct = True

        if predictions is not None:

            title, correct = title_from_label_and_target(predictions[i], label)

        # magic formula tested to work from 1x1 to 10x10 images

        dynamic_titlesize = FIGSIZE*SPACING/max(rows,cols)*40+3 

        if not squaring:

            dynamic_titlesize = FIGSIZE*SPACING/cols*40+3 

        subplot = display_one_flower(image, title, subplot, not correct, 

                                     titlesize=dynamic_titlesize)

    

    #layout

    plt.tight_layout()

    if label is None and predictions is None:

        plt.subplots_adjust(wspace=0, hspace=0)

    else:

        plt.subplots_adjust(wspace=SPACING, hspace=SPACING)

    plt.show()
def display_model_images(idx, count=20):

    j, (trn_ind, val_ind) = list(enumerate(kfold.split(MODEL_FILENAMES)))[idx]

    vdf = list(pd.DataFrame({'FILENAMES': MODEL_FILENAMES}).loc[val_ind]['FILENAMES'])

    val_dataset = load_dataset(vdf, labeled = True, ordered = True)

    

    dataset = get_validation_dataset(val_dataset)

    dataset = dataset.unbatch().batch(count)

    batch = iter(dataset)

    

    # run this cell again for next set of images

    images, labels = next(batch)

    predictions = np.argmax(fold_predictions[idx], axis=-1)[:count]



    # data

    images, labels = batch_to_numpy_images_and_labels((images, labels))

    

    display_batch_of_images(images, labels, predictions)

    

def display_model_errors(idx, count=20):

    j, (trn_ind, val_ind) = list(enumerate(kfold.split(MODEL_FILENAMES)))[idx]

    vdf = list(pd.DataFrame({'FILENAMES': MODEL_FILENAMES}).loc[val_ind]['FILENAMES'])

    NUM_VALIDATION_IMAGES = count_data_items(vdf)

    

    val_dataset = load_dataset(vdf, labeled = True, ordered = True)

    cm_val_dataset = get_validation_dataset(val_dataset)



    images_ds = cm_val_dataset.map(lambda image, label: image).unbatch()

    val_images = next(iter(images_ds.batch(NUM_VALIDATION_IMAGES))).numpy()

    

    labels_ds = cm_val_dataset.map(lambda image, label: label).unbatch()

    val_labels = next(iter(labels_ds.batch(NUM_VALIDATION_IMAGES))).numpy()

    

    prediction = np.argmax(fold_predictions[idx], axis=-1)

    

    images = []; labels = []; predictions = [];

    

    for i in range(NUM_VALIDATION_IMAGES):

        if val_labels[i] != prediction[i]:

            images.append(val_images[i])

            labels.append(val_labels[i])

            predictions.append(prediction[i])

            if count != None and len(predictions) >= count:

                break

    

    display_batch_of_images(images, labels, predictions, count != None)    
try:

    for idx in range(FOLDS):

        print(f"KFold{idx+1} model {MODEL_NAME[idx]} 20 images")    

        display_model_images(idx)

        print(f"KFold{idx+1} model {MODEL_NAME[idx]} errors ")    

        display_model_errors(idx, None)

except:

    print('image processing issue')    
