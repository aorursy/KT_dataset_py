!pip install -q efficientnet
#libraries

import numpy as np
import pandas as pd 
from kaggle_datasets import KaggleDatasets
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import efficientnet.tfkeras as efn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
sns.set()

import re, math, cv2, PIL
import os
os.listdir('../input/chest-xray-pneumonia/chest_xray/')
#getting paths and files

home_dir = '../input/chest-xray-pneumonia/chest_xray/'
nor_train = os.listdir(home_dir + 'train/NORMAL')
pne_train = os.listdir(home_dir + 'train/PNEUMONIA')
nor_val = os.listdir(home_dir + 'val/NORMAL')
pne_val = os.listdir(home_dir + 'val/PNEUMONIA')
print("Normal pictures")
plt.figure(figsize=[20, 8])
for i in range(10):
    img = cv2.imread(home_dir + 'train/NORMAL/' + nor_train[i])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    plt.subplot(2, 5, i+1)
    plt.axis('off')
    plt.imshow(img)
plt.show()

print("Pneumonia pictures")
plt.figure(figsize=[20, 8])
for i in range(10):
    img = cv2.imread(home_dir + 'train/PNEUMONIA/'+ pne_train[i])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    plt.subplot(2, 5, i+1)
    plt.axis('off')
    plt.imshow(img)
plt.show()
DEVICE = 'TPU'

#IMAGE SIZES
IMG_SIZE = 256

#batch size and epochs
BATCH_SIZE = 16
EPOCHS = 8

# Test Time Augmentation
TTA = 5
if DEVICE == "TPU":
    print("connecting to TPU...")
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        print("Could not connect to TPU")
        tpu = None

    if tpu:
        try:
            print("initializing  TPU ...")
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.experimental.TPUStrategy(tpu)
            print("TPU initialized")
        except _:
            print("failed to initialize TPU")
    else:
        DEVICE = "GPU"
if DEVICE != "TPU":
    print("Using default strategy for CPU and single GPU")
    strategy = tf.distribute.get_strategy()

if DEVICE == "GPU":
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    

AUTO     = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync
print(f'REPLICAS: {REPLICAS}')
gcs_path = KaggleDatasets().get_gcs_path('chest-xray-pneumonia')
#method for getting files paths
def get_files_labels(folder):
    normal = os.listdir(home_dir + f'{folder}/NORMAL')
    pneumonia = os.listdir(home_dir + f'{folder}/PNEUMONIA')
    normal_labels = np.int64(np.zeros(len(normal)))
    pneumonia_labels = np.int64(np.ones(len(pneumonia)))
    labels = np.concatenate([normal_labels, pneumonia_labels])
    files_path = [gcs_path + f'/chest_xray/{folder}/NORMAL/{file_name}' for file_name in normal] + [gcs_path + f'/chest_xray/{folder}/PNEUMONIA/{file_name}' for file_name in pneumonia]
    files_path = np.array(files_path)
    return files_path, labels
# get paths
train_files, train_labels = get_files_labels('train')
valid_files, valid_labels = get_files_labels('val')
test_files, test_labels = get_files_labels('test')
train_files = np.concatenate([train_files, valid_files])
train_labels = np.concatenate([train_labels, valid_labels])
train_files, valid_files, train_labels, valid_labels = train_test_split(train_files, train_labels, shuffle=True, test_size=0.2)
# imbalance Valid Data
pd.Series(valid_labels).value_counts()
#ploting

imbalance = pd.DataFrame([[len(nor_train), 'Normal'], [len(pne_train), 'Pneumonia']], columns=['Numbers', 'Case'])
imbalance['Percentage'] = round(((imbalance['Numbers'] / sum(imbalance['Numbers'])) * 100),2)
plt.figure(figsize=[6,6])
sns.barplot(x=imbalance['Case'], y=imbalance['Percentage'])
plt.title('Checking imbalancing', fontsize=16)
plt.show()
# configuration for augmentation

ROT_ = 180.0
SHR_ = 2.0
HZOOM_ = 8.0
WZOOM_ = 8.0
HSHIFT_ = 8.0
WSHIFT_ = 8.0
def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    # returns 3x3 transformmatrix which transforms indicies
        
    # CONVERT DEGREES TO RADIANS
    rotation = math.pi * rotation / 180.
    shear    = math.pi * shear    / 180.

    def get_3x3_mat(lst):
        return tf.reshape(tf.concat([lst],axis=0), [3,3])
    
    # ROTATION MATRIX
    c1   = tf.math.cos(rotation)
    s1   = tf.math.sin(rotation)
    one  = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')
    
    rotation_matrix = get_3x3_mat([c1,   s1,   zero, 
                                   -s1,  c1,   zero, 
                                   zero, zero, one])    
    # SHEAR MATRIX
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)    
    
    shear_matrix = get_3x3_mat([one,  s2,   zero, 
                                zero, c2,   zero, 
                                zero, zero, one])        
    # ZOOM MATRIX
    zoom_matrix = get_3x3_mat([one/height_zoom, zero,           zero, 
                               zero,            one/width_zoom, zero, 
                               zero,            zero,           one])    
    # SHIFT MATRIX
    shift_matrix = get_3x3_mat([one,  zero, height_shift, 
                                zero, one,  width_shift, 
                                zero, zero, one])
    
    return K.dot(K.dot(rotation_matrix, shear_matrix), 
                 K.dot(zoom_matrix,     shift_matrix))
def transform(image, DIM=256):    
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image randomly rotated, sheared, zoomed, and shifted
    XDIM = DIM%2 #fix for size 331
    
    rot = ROT_ * tf.random.normal([1], dtype='float32')
    shr = SHR_ * tf.random.normal([1], dtype='float32') 
    h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / HZOOM_
    w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / WZOOM_
    h_shift = HSHIFT_ * tf.random.normal([1], dtype='float32') 
    w_shift = WSHIFT_ * tf.random.normal([1], dtype='float32') 

    # GET TRANSFORMATION MATRIX
    m = get_mat(rot,shr,h_zoom,w_zoom,h_shift,w_shift) 

    # LIST DESTINATION PIXEL INDICES
    x   = tf.repeat(tf.range(DIM//2, -DIM//2,-1), DIM)
    y   = tf.tile(tf.range(-DIM//2, DIM//2), [DIM])
    z   = tf.ones([DIM*DIM], dtype='int32')
    idx = tf.stack( [x,y,z] )
    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(m, tf.cast(idx, dtype='float32'))
    idx2 = K.cast(idx2, dtype='int32')
    idx2 = K.clip(idx2, -DIM//2+XDIM+1, DIM//2)
    
    # FIND ORIGIN PIXEL VALUES           
    idx3 = tf.stack([DIM//2-idx2[0,], DIM//2-1+idx2[1,]])
    d    = tf.gather_nd(image, tf.transpose(idx3))
        
    return tf.reshape(d,[DIM, DIM,3])
def prepare_img(img, augment, dim):
    img = tf.io.read_file(img)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [dim, dim])
    img = tf.cast(img, tf.float32) / 255.
    
    if augment:
        img = transform(img, DIM=dim)
        img = tf.image.random_brightness(img, 0.1)
        img = tf.image.random_flip_left_right(img)
    
    img = tf.reshape(img, [dim,dim, 3])
    return img
def get_dataset(files, augment = False, shuffle = False, repeat = False, 
                label=None, return_image_names=True):
    if label is not None:
        data = tf.data.Dataset.from_tensor_slices((files, label))
    else:
        data = tf.data.Dataset.from_tensor_slices((files))
    
    data.cache()
    
    if repeat:
        data = data.repeat()
    
    if shuffle:
        data = data.shuffle(1024*3)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        data = data.with_options(opt)
    
    if label is not None:
        data = data.map(lambda img, target: (prepare_img(img, augment, IMG_SIZE), target), num_parallel_calls=AUTO)
    else:
        data = data.map(lambda img: (prepare_img(img, augment, IMG_SIZE)), num_parallel_calls=AUTO)
    
    data = data.batch(BATCH_SIZE * REPLICAS)
    data = data.prefetch(AUTO)
    return data
#get images

train_dataset = get_dataset(train_files, label=train_labels, augment=True, repeat=True, shuffle=True)
valid_dataset = get_dataset(valid_files, label=valid_labels)
test_dataset = get_dataset(test_files, label=test_labels)
def build_model(dim, output_bias=None):
    
    #data is imbalance. For imbalance data accuracy metrics is a bad choice. So I'll choose other metrics.  The metrics 
    #Which are specifically for imbalance data.
    
    METRICS = [
      tf.metrics.TruePositives(name='tp'),
      tf.metrics.FalsePositives(name='fp'),
      tf.metrics.TrueNegatives(name='tn'),
      tf.metrics.FalseNegatives(name='fn'), 
      tf.metrics.BinaryAccuracy(name='accuracy'),
      tf.metrics.Precision(name='precision'),
      tf.metrics.Recall(name='recall'),
     # tf.metrics.AUC(name='auc'),
    ]
    
    if output_bias is not None:
        output_bias = tf.initializers.Constant(output_bias)
    
    input_layer = tf.keras.Input((dim, dim, 3), name='ImgIn')
    
    base_model = efn.EfficientNetB6(input_shape=(dim, dim, 3), weights='imagenet', 
                          include_top=False)
    
    x = base_model(input_layer)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs= input_layer, outputs = x)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.BinaryCrossentropy()
    
    model.compile(optimizer=optimizer, loss= loss, metrics= METRICS)
    return model
def get_lr_callback(batch_size=8):
    lr_start   = 0.000005
    lr_max     = 0.00000125 * REPLICAS * batch_size
    lr_min     = 0.000001
    lr_ramp_ep = 5
    lr_sus_ep  = 0
    lr_decay   = 0.8
   
    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
            
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
            
        else:
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
            
        return lr
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
    return lr_callback
# model check point
sv = tf.keras.callbacks.ModelCheckpoint( 'pneumonia_detection_model.h5', monitor= 'val_loss', verbose=0, 
                                            save_best_only=True, save_weights_only=True, mode='min', save_freq='epoch')
# Counts data
#labels are in nd_array so first i'll convert them to pandas series than convert them.
#than use value_counts() 1 indicates pneumonia cases and 0 for normal cases.

COUNT_NORMAL = pd.Series(train_labels).value_counts()[0]
COUNT_PNEUMONIA = pd.Series(train_labels).value_counts()[1]
COUNT_TRAIN = len(train_files)
#initializing bias
initial_bias = np.log([COUNT_PNEUMONIA/COUNT_NORMAL])
initial_bias
#initializing weights

weight_for_0 = (1/COUNT_NORMAL)*(COUNT_TRAIN)/2.0
weight_for_1 = (1/COUNT_PNEUMONIA)*(COUNT_TRAIN)/2.0

class_weight = {0:weight_for_0, 1:weight_for_1}

print(f'Weight for Normal Class: {weight_for_0}')
print(f'Weight for Pneumonia Class: {weight_for_1}')
with strategy.scope():
    model = build_model(IMG_SIZE)
model.summary()
history = model.fit(train_dataset, epochs=EPOCHS, callbacks=[sv, get_lr_callback(BATCH_SIZE)], 
                    steps_per_epoch=COUNT_TRAIN/BATCH_SIZE//REPLICAS, validation_data=valid_dataset, verbose=1, 
                    class_weight=class_weight)
#evaluating test data without tta
loss, tp, fp, tn, fn, acc, precision, recall = model.evaluate(test_dataset)
print(f'Model accuracy without TTA: {round(acc,4)}')
TEST_COUNTS = len(test_labels)
STEPS = TTA * TEST_COUNTS / BATCH_SIZE / REPLICAS

pred = model.predict(get_dataset(test_files, label=None, repeat=True, augment=True), steps=STEPS, verbose=1)[:TTA * TEST_COUNTS]
pred_tta = np.mean(pred.reshape((TEST_COUNTS, TTA), order='F'), axis=1)
accuracy = accuracy_score(y_pred=pred_tta.round(), y_true=test_labels)
confusion_matrix(y_pred=pred_tta.round(), y_true=test_labels)
print(f'Model accuracy with TTA: {round(accuracy,4)}')