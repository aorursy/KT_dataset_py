!pip install -q efficientnet
!pip install iterative-stratification
import time

import os

import random

import numpy as np

import pandas as pd

from collections import namedtuple

from sklearn import metrics

from matplotlib import pyplot as plt

from kaggle_datasets import KaggleDatasets

import efficientnet.tfkeras as efn

from sklearn.model_selection import train_test_split, GroupKFold

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from glob import glob

import tensorflow as tf

print(tf.__version__)

from tensorflow.keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

import logging

# no extensive logging 

logging.getLogger().setLevel(logging.NOTSET)



AUTO = tf.data.experimental.AUTOTUNE
DROPOUT = 0.5 # use aggressive dropout

# BATCH_SIZE = 16 # per TPU core



EPOCHS = 15

### Different learning rate for transformer and head ###

# LR_EFNET = 1e-2

LR_HEAD = 1e-3
def append_path(pre):

    return np.vectorize(lambda file: os.path.join(GCS_DS_PATH, pre, file))



def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)



def int_div_round_up(a, b):

    return (a + b - 1) // b



def onehot(size, target):

    vec = np.zeros(size, dtype=np.float32)

    vec[target] = 1.

    return vec



seed_everything(42)
def connect_to_TPU():

    """Detect hardware, return appropriate distribution strategy"""

    try:

        # TPU detection. No parameters necessary if TPU_NAME environment variable is

        # set: this is always the case on Kaggle.

        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

        print('Running on TPU ', tpu.master())

    except ValueError:

        tpu = None



    if tpu:

        tf.config.experimental_connect_to_cluster(tpu)

        tf.tpu.experimental.initialize_tpu_system(tpu)

        strategy = tf.distribute.experimental.TPUStrategy(tpu)

    else:

        # Default distribution strategy in Tensorflow. Works on CPU and single GPU.

        strategy = tf.distribute.get_strategy()



    global_batch_size = 16 * strategy.num_replicas_in_sync



    return tpu, strategy, global_batch_size





tpu, strategy, BATCH_SIZE = connect_to_TPU()

print("REPLICAS: ", strategy.num_replicas_in_sync)



# Data access

GCS_DS_PATH = KaggleDatasets().get_gcs_path()
%%time



dataset = []



for label, kind in enumerate(['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']):

    for path in glob('../input/alaska2-image-steganalysis/Cover/*.jpg'):

        dataset.append({

            'kind': kind,

            'image_name': path.split('/')[-1],

            'label': label

        })



random.shuffle(dataset)

dataset = pd.DataFrame(dataset)



gkf = GroupKFold(n_splits=5)



dataset.loc[:, 'fold'] = 0

for fold_number, (train_index, val_index) in enumerate(gkf.split(X=dataset.index, y=dataset['label'], groups=dataset['image_name'])):

    dataset.loc[dataset.iloc[val_index].index, 'fold'] = fold_number



# fold_gkf = pd.read_csv('../input/alaska2-public-baseline/groupkfold_by_shonenkov.csv')

fold_gkf = dataset.copy()

fold_gkf.head()
fold_number = 0

# train_df = fold_gkf[fold_gkf['fold'] != fold_number]

train_df = fold_gkf[fold_gkf['fold'] == fold_number]

train_df.shape
mskf = MultilabelStratifiedKFold(n_splits=8, random_state=42)



train_data = None

valid_data = None



for train_idx, val_idx in mskf.split(train_df['image_name'], train_df[['label', 'kind']]):

    

    train_data = train_df.iloc[train_idx]

    valid_data = train_df.iloc[val_idx]

    break
sub = pd.read_csv('/kaggle/input/alaska2-image-steganalysis/sample_submission.csv')

# train_filenames = np.array(os.listdir("/kaggle/input/alaska2-image-steganalysis/Cover/"))
%%time



train_paths = []

train_labels = []



for i in range(len(train_data['kind'])):

    kind = train_data['kind'].iloc[i]

    im_id = train_data['image_name'].iloc[i]

    label = onehot(4, train_data['label'].iloc[i])

    path = os.path.join(GCS_DS_PATH, kind, im_id)

    

    train_paths.append(path)

    train_labels.append(label)

    

len(train_paths), len(train_labels)
%%time



valid_paths = []

valid_labels = []



for i in range(len(valid_data['kind'])):

    kind = valid_data['kind'].iloc[i]

    im_id = valid_data['image_name'].iloc[i]

    label = onehot(4, valid_data['label'].iloc[i])

    path = os.path.join(GCS_DS_PATH, kind, im_id)

    

    valid_paths.append(path)

    valid_labels.append(label)

    

# len(valid_paths), len(valid_labels)
test_paths = append_path('Test')(sub.Id.values)
train_paths = np.array(train_paths[0:1000])

train_labels = np.array(train_labels[0:1000])

valid_paths = np.array(valid_paths[0:1000])

valid_labels = np.array(valid_labels[0:1000])
LABEL_MAP = {"Cover": 0,

            "JMiPOD": 1,

            "JUNIWARD": 2,

            "UERD": 3}



def decode_image(filename, label, image_size=(512, 512)):

    bits = tf.io.read_file(filename)

    image = tf.image.decode_jpeg(bits, channels=3)

    image = tf.cast(image, tf.float32) / 255.0

    image = tf.image.resize(image, image_size)

    

    if label is None:

        return image

    else:

        return image, label



def decode_test_image(filename, image_size=(512, 512)):

    bits = tf.io.read_file(filename)

    image = tf.image.decode_jpeg(bits, channels=3)

    image = tf.cast(image, tf.float32) / 255.0

    image = tf.image.resize(image, image_size)

    return image

    

def data_augment(image, label=None):

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_flip_up_down(image)

    

    if label is None:

        return image

    else:

        return image, label
def get_training_dataset():

    return (tf.data.Dataset

    .from_tensor_slices((train_paths, train_labels))

    .map(decode_image, num_parallel_calls=AUTO)

    .repeat()

    .shuffle(2048)

    .batch(BATCH_SIZE, drop_remainder=True)

    .cache()

    .prefetch(AUTO))



def get_validation_dataset(repeated=False):

    return (tf.data.Dataset

    .from_tensor_slices((valid_paths, valid_labels))

    .map(decode_image, num_parallel_calls=AUTO)

    .repeat()

    .shuffle(2048)

    .batch(BATCH_SIZE, drop_remainder=repeated)

    .cache()

    .prefetch(AUTO))



def get_test_dataset(ordered=False):

    return (tf.data.Dataset

        .from_tensor_slices(test_paths)

        .map(decode_test_image, num_parallel_calls=AUTO)

        .batch(BATCH_SIZE)

        .prefetch(AUTO))
train_dataset  = get_training_dataset()

valid_dataset  = get_validation_dataset(repeated=True)

test_dataset  = get_test_dataset()
%%time



def build_model():

    base_model = efn.EfficientNetB0(weights='imagenet',include_top=False, input_shape=(512, 512, 3))

    base_model.trainable = False

    

    inputs = Input(shape=(512, 512, 3))

    efnet_feat = base_model(inputs)

    x = GlobalAveragePooling2D()(efnet_feat)

    outputs = Dense(4, activation='softmax', name='custome_head')(x)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    

    return model





with strategy.scope():               

    model = build_model()

    optimizer_head = Adam(learning_rate=LR_HEAD)

    model.summary()
def define_losses_and_metrics():

    with strategy.scope():

        loss_object = tf.keras.losses.CategoricalCrossentropy(

            reduction=tf.keras.losses.Reduction.NONE, from_logits=False)



        def compute_loss(labels, predictions):

            per_example_loss = loss_object(labels, predictions)

            loss = tf.nn.compute_average_loss(

                per_example_loss, global_batch_size = BATCH_SIZE)

            return loss

        train_accuracy_metric = tf.keras.metrics.AUC(name='training_AUC')

        valid_accuracy_metric = tf.keras.metrics.AUC(name='val_AUC')

    return compute_loss, train_accuracy_metric, valid_accuracy_metric



train_loss, train_accuracy_metric, valid_accuracy_metric = define_losses_and_metrics()
STEPS_PER_TPU_CALL = len(train_paths) // 128 

VALIDATION_STEPS_PER_TPU_CALL = len(valid_paths) // 128



@tf.function

def train_step(data_iter):

    def train_step_fn(inputs):

        features, labels = inputs



        # calculate the 2 gradients ( note persistent, and del)

        with tf.GradientTape(persistent=True) as tape:

            predictions = model(features, training=True)

            loss = train_loss(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)

        del tape # not sure if we should delete it.



        ### make the gradients step

        optimizer_head.apply_gradients(zip(gradients, 

                                           model.trainable_variables))



        train_accuracy_metric.update_state(labels, predictions)

        

    # this loop runs on the TPU

    for _ in tf.range(STEPS_PER_TPU_CALL):

        strategy.run(train_step_fn, args=(next(data_iter),))



def predict(dataset):  

    predictions = []

    for tensor in dataset:

        predictions.append(distributed_prediction_step(tensor))

    ### stack replicas and batches

    predictions = np.vstack(list(map(np.vstack,predictions)))

    return predictions



@tf.function

def distributed_prediction_step(data):

    predictions = strategy.run(prediction_step, args=(data,))

    return strategy.experimental_local_results(predictions)



def prediction_step(inputs):

    features = inputs  # note datasets used in prediction do not have labels

    predictions = model(features, training=False)

    return predictions
@tf.function

def valid_step(data_iter):

    def valid_step_fn(images, labels):

        probabilities = model(images, training=False)

        

        # update metrics

        valid_accuracy_metric.update_state(labels, probabilities)

    # this loop runs on the TPU

    for _ in tf.range(VALIDATION_STEPS_PER_TPU_CALL):

        strategy.run(valid_step_fn, next(data_iter))
start_time = epoch_start_time = time.time()

STEPS_PER_EPOCH = len(train_paths) // BATCH_SIZE # we can use BATCH_SIZE instead this is for exp for now





History = namedtuple('History', 'history')

history = History(history={'loss': [], 'categorical_auc': [], 'val_categorical_auc': []})



print("Training steps per epoch:", STEPS_PER_EPOCH, "in increments of", STEPS_PER_TPU_CALL)

print("Validation images:", len(valid_paths),

      "Batch size:", BATCH_SIZE,

      "Validation steps:", len(valid_paths) // BATCH_SIZE, "in increments of", VALIDATION_STEPS_PER_TPU_CALL)



epoch = 0

train_data_iter = iter(train_dataset)

valid_data_iter = iter(valid_dataset)



step = 0

epoch_steps = 0

best_weights = None



while True:

    

    # run training step

    train_step(train_data_iter)

    epoch_steps += STEPS_PER_TPU_CALL

    step += STEPS_PER_TPU_CALL

    print('=', end='', flush=True)

        

    # validation run at the end of each epoch

    if (step // STEPS_PER_EPOCH) > epoch:

        print('|', end='', flush=True)

    

        # validation run

        valid_epoch_steps = 0

        val_preds = []

        val_lables = []

        for _ in range(1):

            valid_step(valid_data_iter)

            valid_epoch_steps += VALIDATION_STEPS_PER_TPU_CALL

            print('=', end='', flush=True)

    

        # compute metrics

        history.history['categorical_auc'].append(train_accuracy_metric.result().numpy())

        history.history['val_categorical_auc'].append(valid_accuracy_metric.result().numpy())



        ## save weights if it is the best yet

        if history.history['val_categorical_auc'][-1] == max(history.history['val_categorical_auc']):

            best_weights = model.get_weights()

        

        ### Restore best weighths ###

        model.set_weights(best_weights)

        

        epoch_time = time.time() - epoch_start_time

        print('\nEPOCH {:d}/{:d}'.format(epoch+1, EPOCHS))

        print('time: {:0.1f}s'.format(epoch_time),

             'auc: {:0.4f}'.format(history.history['categorical_auc'][-1]),

              'val_auc: {:0.4f}'.format(history.history['val_categorical_auc'][-1]),

              'steps/val_steps: {:d}/{:d}'.format(epoch_steps, valid_epoch_steps), flush=True)

        

        ### Reset (train) metrics ###

        train_accuracy_metric.reset_states()

        valid_accuracy_metric.reset_states()

        

        # set up next epoch

        epoch = step // STEPS_PER_EPOCH

        epoch_steps = 0

        epoch_start_time = time.time()

        if epoch >= EPOCHS:

            break



optimized_ctl_training_time = time.time() - start_time

print("OPTIMIZED CTL TRAINING TIME: {:0.1f}s".format(optimized_ctl_training_time))
model.save("efnetB0_exp_1.h5")
%%time

preds = predict(test_dataset)
preds.shape
preds = 1 - preds[:,0]
preds.shape
s = 0

final_preds = np.zeros((5000))

for i in range(8):

    end = s + 5000

    final_preds += preds[s:end]

    s = end
final_preds.shape
sub.Label = final_preds / 8

sub.to_csv('submission.csv', index=False)

sub.head(n=15)
sub['Label'].hist(bins=100)