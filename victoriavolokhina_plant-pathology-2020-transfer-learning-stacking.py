import numpy as np 
import pandas as pd 
import os
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as L
from keras.models import Model

import warnings
warnings.filterwarnings("ignore")
from kaggle_datasets import KaggleDatasets

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

def seed_everything(seed=0):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed = 2048
seed_everything(seed)
print("REPLICAS: ", strategy.num_replicas_in_sync)

# Data access
GCS_DS_PATH = KaggleDatasets().get_gcs_path()

# Configuration
#BATCH_SIZE = 8 * strategy.num_replicas_in_sync
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
EPOCHS = 100
#image_size1 = 800
image_size1 = 533
image_size2 = 800
def format_path(st):
    return GCS_DS_PATH + '/images/' + st + '.jpg'
from sklearn.model_selection import train_test_split

train = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/train.csv')
test = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/test.csv')
sub = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv')

train_paths = train.image_id.apply(format_path).values
test_paths = test.image_id.apply(format_path).values
train_labels = train.loc[:, 'healthy':].values

valid_dataset = []
SPLIT_VALIDATION = False
if SPLIT_VALIDATION:
    train_paths, valid_paths, train_labels, valid_labels =train_test_split(train_paths, train_labels, test_size=0.15, random_state=seed)
    valid_labels_df = pd.DataFrame({'healthy': valid_labels[:, 0], 
                                  'multiple_diseases': valid_labels[:, 1], 
                                  'rust': valid_labels[:, 2], 
                                  'scab': valid_labels[:, 3]})
    valid_labels_df.to_csv('valid_labels.csv', index=False)
    
STEPS_PER_EPOCH = train_labels.shape[0] // BATCH_SIZE
from matplotlib import pyplot as plt

img = plt.imread('../input/plant-pathology-2020-fgvc7/images/Train_500.jpg')
print(img.shape)
plt.imshow(img)
def decode_image(filename, label=None, image_size=(image_size1, image_size2)):
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
    
    #image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    #image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    #Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    if label is None:
        return image
    else:
        return image, label
train_dataset = (
tf.data.Dataset
    .from_tensor_slices((train_paths, train_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .cache()
    .map(data_augment, num_parallel_calls=AUTO)
    .repeat()
    .shuffle(512)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

if SPLIT_VALIDATION:
    valid_dataset = (
        tf.data.Dataset
        .from_tensor_slices((valid_paths, valid_labels))
        .map(decode_image, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .cache()
        .prefetch(AUTO)
    )

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(test_paths)
    .map(decode_image, num_parallel_calls=AUTO)
    .map(data_augment, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
)
import matplotlib.pyplot as plt

def plot_learning(history):
    acc = history.history['categorical_accuracy']
    loss = history.history['loss']
    if SPLIT_VALIDATION: 
        val_acc = history.history['val_categorical_accuracy']
        val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    if SPLIT_VALIDATION: plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Accuracy')

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training Loss')
    if SPLIT_VALIDATION: plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Loss')
    plt.legend()

    plt.show()
    
def save_preds(model_name, pred_test, pred_val=None):
    
    sub.loc[:, 'healthy':] = pred_test
    filename_test = 'preds_' + model_name + '_test.csv'
    sub.to_csv(filename_test, index=False)

    if SPLIT_VALIDATION:    
        valid_labels_df.loc[:, 'healthy':] = pred_val
        filename_val = 'preds_' + model_name + '_val.csv'
        valid_labels_df.to_csv(filename_val, index=False)
LR_START = 0.00001
LR_MAX = 0.0001 * strategy.num_replicas_in_sync
LR_MIN = 0.00001
LR_RAMPUP_EPOCHS = 15
LR_SUSTAIN_EPOCHS = 3
LR_EXP_DECAY = .8

def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr
    
lr_callback = [
  #tf.keras.callbacks.EarlyStopping(patience=7),
  tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)
]
!pip install efficientnet
import efficientnet.tfkeras as efn
with strategy.scope():
    model_EffNetB7 = tf.keras.Sequential([ efn.EfficientNetB7( input_shape=(image_size1, image_size2, 3), 
                                                              weights='imagenet', 
                                                              include_top=False, 
                                                              pooling='avg'), 
                                                    L.Dense(4, activation='softmax')
                                                    ])
    
    model_EffNetB7.compile(optimizer='adam', 
                           loss='categorical_crossentropy',
                           metrics=['categorical_accuracy'])
history = model_EffNetB7.fit(
        train_dataset, 
        steps_per_epoch=STEPS_PER_EPOCH,
        callbacks=lr_callback,
        epochs=EPOCHS,
        validation_data=valid_dataset if SPLIT_VALIDATION else None
    )
pred_test_EffNetB7 = model_EffNetB7.predict(test_dataset)

if SPLIT_VALIDATION:
    pred_val_EffNetB7 = model_EffNetB7.predict(valid_dataset)
    save_preds('EffNetB7', pred_test_EffNetB7, pred_val_EffNetB7)
else:
    save_preds('EffNetB7', pred_test_EffNetB7)
plot_learning(history)
tf.tpu.experimental.initialize_tpu_system(tpu) # Clear TPU Memory
from tensorflow.keras.applications.resnet_v2 import ResNet50V2

with strategy.scope():
    model_ResNet50V2 = tf.keras.Sequential([
                    ResNet50V2(
                        input_shape=(image_size1, image_size2, 3),
                        weights='imagenet',
                        include_top=False
                    ),
                    L.GlobalMaxPooling2D(),

                    L.Dense(1024, activation='relu'),
                    L.Dropout(0.5),
                    L.BatchNormalization(),

                    L.Dense(4, activation='softmax')
                ])
        
    model_ResNet50V2.compile(
        optimizer = 'adam',
        loss = 'categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
history = model_ResNet50V2.fit(
    train_dataset, 
    epochs=EPOCHS, 
    callbacks=lr_callback,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=valid_dataset if SPLIT_VALIDATION else None
)
pred_test_ResNet50V2 = model_ResNet50V2.predict(test_dataset)
pred_val_ResNet50V2 = model_ResNet50V2.predict(valid_dataset)
save_preds('ResNet50V2', pred_test_ResNet50V2, pred_val_ResNet50V2)
plot_learning(history)
tf.tpu.experimental.initialize_tpu_system(tpu) # Clear TPU Memory
from tensorflow.keras.applications import InceptionResNetV2

with strategy.scope():
    model_IncResNetV2 = tf.keras.Sequential([
                InceptionResNetV2(
                    input_shape=(image_size1, image_size2, 3),
                    weights='imagenet',
                    include_top=False
                ),
                L.GlobalMaxPooling2D(),

                L.Dense(512, activation='relu'),
                L.Dropout(0.5),
                L.BatchNormalization(),

                L.Dense(4, activation='softmax')
            ])
        
    model_IncResNetV2.compile(
        optimizer = 'adam',
        loss = 'categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
history = model_IncResNetV2.fit(
    train_dataset, 
    epochs=EPOCHS, 
    callbacks=lr_callback,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=valid_dataset if SPLIT_VALIDATION else None,
)
pred_test_IncResNetV2 = model_IncResNetV2.predict(test_dataset)
pred_val_IncResNetV2 = model_IncResNetV2.predict(valid_dataset)
save_preds('IncResNetV2', pred_test_IncResNetV2, pred_val_IncResNetV2)
plot_learning(history)
tf.tpu.experimental.initialize_tpu_system(tpu) # Clear TPU Memory
from tensorflow.keras.applications.inception_v3 import InceptionV3

with strategy.scope(): 
    model_IncV3 = tf.keras.Sequential([ InceptionV3( input_shape=(image_size1, image_size2, 3), 
                                                                       weights='imagenet', 
                                                                       include_top=False ), 
                                                    L.GlobalMaxPooling2D(), 
                                                    L.Dense(4, activation='softmax')
                                                    ])
    model_IncV3.compile(optimizer='adam',
                  loss = 'categorical_crossentropy', 
                  metrics=['categorical_accuracy'])
history = model_IncV3.fit(
    train_dataset, 
    epochs=EPOCHS, 
    callbacks=lr_callback,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=valid_dataset if SPLIT_VALIDATION else None
)
pred_test_IncV3 = model_IncV3.predict(test_dataset)

if SPLIT_VALIDATION:
    pred_val_IncV3 = model_IncV3.predict(valid_dataset)
    save_preds('IncV3', pred_test_IncV3, pred_val_IncV3)
else:
    save_preds('IncV3', pred_test_IncV3)
plot_learning(history)
tf.tpu.experimental.initialize_tpu_system(tpu) # Clear TPU Memory
from tensorflow.keras.applications import Xception

with strategy.scope(): 
    
    model_Xcept = tf.keras.Sequential([Xception(input_shape=(image_size1, image_size2, 3),
                                                            weights='imagenet',
                                                            include_top=False),
                                             L.GlobalAveragePooling2D(),
                                             L.Dense(4, activation='softmax')
                                             ])
        
    model_Xcept.compile(loss="categorical_crossentropy", 
                        optimizer= 'adam', 
                        metrics=["categorical_accuracy"])
history = model_Xcept.fit(
    train_dataset, 
    epochs=EPOCHS, 
    callbacks=lr_callback,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=valid_dataset if SPLIT_VALIDATION else None
)
pred_test_Xcept = model_Xcept.predict(test_dataset)

if SPLIT_VALIDATION:
    pred_val_Xcept = model_Xcept.predict(valid_dataset)
    save_preds('Xcept', pred_test_Xcept, pred_val_Xcept)
else:
    save_preds('Xcept', pred_test_Xcept)
plot_learning(history)
tf.tpu.experimental.initialize_tpu_system(tpu) # Clear TPU Memory
from tensorflow.keras.applications import ResNet152V2

with strategy.scope():
    model_ResNet152V2 = tf.keras.Sequential([ResNet152V2(input_shape=(image_size1, image_size2, 3),
                                                            weights='imagenet',
                                                            include_top=False),
                                             L.GlobalAveragePooling2D(),
                                             L.Dense(4, activation='softmax')
                                             ])
    
    model_ResNet152V2.compile(loss="categorical_crossentropy", 
                              optimizer= 'adam', 
                              metrics=["categorical_accuracy"])
history = model_ResNet152V2.fit(
    train_dataset, 
    epochs=EPOCHS, 
    callbacks=lr_callback,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=valid_dataset if SPLIT_VALIDATION else None
)
pred_test_ResNet152V2 = model_ResNet152V2.predict(test_dataset)

if SPLIT_VALIDATION:
    pred_val_ResNet152V2 = model_ResNet152V2.predict(valid_dataset)
    save_preds('ResNet152V2', pred_test_ResNet152V2, pred_val_ResNet152V2)
else:
    save_preds('ResNet152V2', pred_test_ResNet152V2)
plot_learning(history)
tf.tpu.experimental.initialize_tpu_system(tpu) # Clear TPU Memory
from tensorflow.keras.applications.nasnet import NASNetLarge
with strategy.scope():    
    model_NASNet = tf.keras.Sequential([NASNetLarge( input_shape=(image_size1, image_size2, 3), 
                                                                       weights='imagenet', 
                                                                       include_top=False ), 
                                                    L.GlobalMaxPooling2D(), 
                                                    L.Dense(4, activation='softmax')
                                                    ])
    model_NASNet.compile(optimizer='adam',
                  loss = 'categorical_crossentropy', 
                  metrics=['categorical_accuracy'])
history = model_NASNet.fit(
    train_dataset, 
    epochs=EPOCHS, 
    callbacks=lr_callback,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=valid_dataset if SPLIT_VALIDATION else None
)
pred_test_NASNet = model_NASNet.predict(test_dataset)

if SPLIT_VALIDATION:
    pred_val_NASNet = model_NASNet.predict(valid_dataset)
    save_preds('NASNet', pred_test_NASNet, pred_val_NASNet)
else:
    save_preds('NASNet', pred_test_NASNet)
from tensorflow.keras.applications import DenseNet201

with strategy.scope():
    model_DenseNet201 = tf.keras.Sequential([DenseNet201(input_shape=(image_size1, image_size2, 3),
                                                            weights='imagenet',
                                                            include_top=False),
                                             L.GlobalAveragePooling2D(),
                                             L.Dense(4, activation='softmax')
                                             ])
    
    model_DenseNet201.compile(loss="categorical_crossentropy", 
                              optimizer= 'adam', 
                              metrics=["categorical_accuracy"])
history = model_DenseNet201.fit(
    train_dataset, 
    epochs=EPOCHS, 
    callbacks=lr_callback,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=valid_dataset if SPLIT_VALIDATION else None
)
pred_test_DenseNet201 = model_DenseNet201.predict(test_dataset)

if SPLIT_VALIDATION:
    pred_val_DenseNet201 = model_DenseNet201.predict(valid_dataset)
    save_preds('DenseNet201', pred_test_DenseNet201, pred_val_DenseNet201)
else:
    save_preds('DenseNet201', pred_test_DenseNet201)
plot_learning(history)
tf.tpu.experimental.initialize_tpu_system(tpu) # Clear TPU Memory
preds_avg = (pred_test_EffNetB7 + pred_test_IncV3 + pred_test_Xcept) / 3
sub.loc[:, 'healthy':] = preds_avg
sub.to_csv('submission_avg_3model_NoSplit_800-533.csv', index=False)
sub.head()
# LB 0.98
# pred_train = np.concatenate((pred_val_EffNetB7, pred_val_ResNet50V2, pred_val_IncResNetV2, pred_val_IncV3, pred_val_Xcept, pred_val_ResNet152V2), axis=1)
# pred_train.shape
# pred_test = np.concatenate((pred_test_EffNetB7, pred_test_ResNet50V2, pred_test_IncResNetV2, pred_test_IncV3, pred_test_Xcept, pred_test_ResNet152V2), axis=1)
# pred_test.shape
pred_train = np.concatenate((pred_val_EffNetB7, pred_val_IncV3, pred_val_Xcept), axis=1)
pred_train.shape
pred_test = np.concatenate((pred_test_EffNetB7, pred_test_IncV3, pred_test_Xcept), axis=1)
pred_test.shape
valid_labels.shape
from sklearn.linear_model import Ridge

Ridge = Ridge(alpha=1, random_state=241)
Ridge.fit(pred_train, valid_labels)
predictions = Ridge.predict(pred_test)
sub.loc[:, 'healthy':] = predictions
sub.to_csv('submission_predict_ridge.csv', index=False)
sub.head()
from sklearn.neural_network import MLPClassifier

MLP_clf = MLPClassifier(max_iter=400)
MLP_clf.fit(pred_train, valid_labels)
predictionMLP = MLP_clf.predict(pred_test)
predictionMLP.shape
sub.loc[:, 'healthy':] = predictionMLP
sub.to_csv('submission_3models_MLPReg.csv', index=False)
sub.head()
# LB 0.961 regression, 0.925 classifier
sub1 = pred_test_EffNetB7
sub2 = pred_test_IncV3
sub3 = pred_test_DenseNet201
sub4 = pred_test_Xcept
sub5 = pred_test_ResNet152V2
ent1 = entropy(sub1, base=2, axis = 1)
ent2 = entropy(sub2, base=2, axis = 1)
ent3 = entropy(sub3, base=2, axis = 1)
ent4 = entropy(sub4, base=2, axis = 1)
ent5 = entropy(sub5, base=2, axis = 1)
entropies = np.array([ent1, ent2, ent3, ent4, ent5]).transpose()
entropies.shape

selected = np.argmin(entropies, axis = 1)
submission_size = len(selected)
for i in range(submission_size):
    if selected[i] ==0:
        sub.loc[i, 'healthy' : ] = sub1
    elif selected[i] ==1:
        sub.loc[i, 'healthy' : ] = sub2
    elif selected[i] == 2:
        sub.loc[i, 'healthy' : ] = sub3
    elif selected[i] == 3:
        sub.loc[i, 'healthy' : ] = sub4
    elif selected[i] == 4:
        sub.loc[i, 'healthy' : ] = sub5
sub.to_csv('submission.csv', index=False)