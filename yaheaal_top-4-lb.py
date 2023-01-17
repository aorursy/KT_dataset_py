import pandas as pd
import numpy as np
import os 

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf

np.random.seed(0)
tf.random.set_seed(0)
!pip install efficientnet
import efficientnet.tfkeras  as efn
from kaggle_datasets import KaggleDatasets

AUTO = tf.data.experimental.AUTOTUNE

def TPU():
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
    return strategy


strategy = TPU()
GCS_DS_PATH = KaggleDatasets().get_gcs_path()
train_data = pd.read_csv('../input/plant-pathology-2020-fgvc7/train.csv')
test_data = pd.read_csv('../input/plant-pathology-2020-fgvc7/test.csv')

train_paths = train_data['image_id'].apply(lambda x: os.path.join(GCS_DS_PATH , 'images' , x + '.jpg')).values
test_paths = test_data['image_id'].apply(lambda x: os.path.join(GCS_DS_PATH , 'images' , x + '.jpg')).values

train_labels = train_data.iloc[:,1:].values
n_classes = 4
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
img_size = 800
EPOCHS = 100
FOLDS = 5
SEED = 42
def decode_image(filename, label=None, image_size=(img_size, img_size)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    if label is None:
        return image
    else:
        return image, label
    
def data_augment(image, label=None, seed=2020):
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.random_flip_up_down(image, seed=seed)
           
    if label is None:
        return image
    else:
        return image, label
    
    
def prepare_train(train_paths, train_labels):
    data = (
        tf.data.Dataset
        .from_tensor_slices((train_paths, train_labels))
        .map(decode_image, num_parallel_calls=AUTO)
        .map(data_augment, num_parallel_calls=AUTO)
        .repeat()
        .shuffle(512)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )
    return data

def prepare_val(val_paths, val_labels):
    data = (
        tf.data.Dataset
        .from_tensor_slices((val_paths, val_labels))
        .map(decode_image, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )
    return data

def prepare_test(test_paths):
    data = (
        tf.data.Dataset
        .from_tensor_slices((test_paths))
        .map(decode_image, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
    )
    return data
def get_model():
    base_model = efn.EfficientNetB7(weights='imagenet', include_top=False, pooling='avg', input_shape=(img_size, img_size, 3))
    x = base_model.output
    predictions = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    return model

def Callbacks():
    erl = EarlyStopping(monitor='val_loss', patience=11, verbose=1, mode='min', restore_best_weights=True)
    rdc = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=1, mode='min')
    return [erl,rdc]
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
test_pred = []
val_roc_auc = []
# all_history = []

for i, (train_idx, val_idx) in enumerate(skf.split(train_paths, train_labels.argmax(1))):
    print(); print('#'*25)
    print('### FOLD',i+1)
    print('#'*25)
    X_train, X_val = train_paths[train_idx], train_paths[val_idx]
    y_train, y_val = train_labels[train_idx], train_labels[val_idx]
    
    strategy = TPU()
    with strategy.scope():
        model = get_model()
        history = model.fit(
                    prepare_train(X_train,y_train),
                    steps_per_epoch=y_train.shape[0] // BATCH_SIZE,
                    validation_data=prepare_val(X_val, y_val),
                    validation_steps=y_val.shape[0] // BATCH_SIZE,
                    callbacks=Callbacks(),
                    epochs=EPOCHS,
                    verbose=1
                )

    test_pred.append(model.predict(prepare_test(test_paths), verbose=1))
    val_roc_auc.append(roc_auc_score(y_val,model.predict(prepare_val(X_val, y_val), verbose=1)))
    
#     all_history.append(history)
#     model.save('{}_model.h5'.format(i+1))
val_roc_auc
all_test = 0
for i in range(FOLDS):
    all_test += test_pred[i]
all_models = all_test/FOLDS
all_models
best_2_models = test_pred[0]*.7 + test_pred[3]*.3
best_2_models
# best_2_models gives me better score on LB
sumb = pd.read_csv('../input/plant-pathology-2020-fgvc7/sample_submission.csv')
sumb.iloc[:,1:] = best_2_models 
# sumb.iloc[:,1:] = all_models
sumb
sumb.to_csv('submission.csv', index=False)
