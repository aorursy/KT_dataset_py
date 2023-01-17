# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
filesPath = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
#         print(os.path.join(dirname, filename))
        filesPath.append(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
filesPath[:4]
root = "/kaggle/input/camvid/"
mainPath = "/kaggle/input/camvid/CamVid/"
mainList = os.listdir(mainPath)
print(mainList)
from skimage import io
import matplotlib.pyplot as plt
def oneHot(labels):
    x = np.zeros([360,480,12])
    for i in range(360):
        for j in range(480):
            x[i,j,labels[i][j]] = 1
    return x
def dataReader(fileType):
    """
    fileType: Either 'train', 'test' or 'val'  (must be a string)
    returns: two numpy array of input and target
    """
    
    _input = []
    _target = []
    
    txtFile = fileType + '.txt'
    txtPath = mainPath + txtFile
    with open(txtPath, 'r') as f:
        for line in f:
            line = line.strip().split()
            _input.append(io.imread(root + line[0][7:]))
            _target.append(oneHot(io.imread(root + line[1][7:])))
    
    assert len(_input) == len(_target)
    
    return np.array(_input), np.array(_target)
        
    
val_data, val_label = dataReader('val')
val_data.shape
val_label.shape
io.imshow(val_data[82])
train_data, train_label = dataReader('train')
train_data.shape
train_label.shape
import tensorflow as tf
# detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
import keras
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import adam
from keras.activations import relu, softmax
import keras.utils.vis_utils as vutil
filterSize = 32
with tpu_strategy.scope():
    segNet = Sequential([
        #down
        Conv2D(filterSize, 3, input_shape= (360, 480, 3), padding="same"), 
        BatchNormalization(), 
        Activation('relu'), 
        MaxPooling2D(pool_size=(2, 2)),
    
        Conv2D(filterSize * 2, 3, padding="same"), 
        BatchNormalization(), 
        Activation('relu'), 
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(filterSize * 3, 3, padding="same"), 
        BatchNormalization(), 
        Activation('relu'), 
        MaxPooling2D(pool_size=(2, 2)),
        
#         Conv2D(filterSize * 4, 3, padding="same"), 
#         BatchNormalization(), 
#         Activation('relu'), 
#         MaxPooling2D(pool_size=(2, 2)),

        #up
#         UpSampling2D(size=(2, 2)),
#         Conv2D(filterSize * 4, 3, padding="same"), 
#         BatchNormalization(), 
#         Activation('relu'),
        
        UpSampling2D(size=(2, 2)),
        Conv2D(filterSize * 3, 3, padding="same"), 
        BatchNormalization(), 
        Activation('relu'),

        UpSampling2D(size=(2, 2)),
        Conv2D(filterSize * 2, 3, padding="same"), 
        BatchNormalization(), 
        Activation('relu'),

        UpSampling2D(size=(2, 2)),
        Conv2D(filterSize, 3, padding="same"), 
        BatchNormalization(), 
        Activation('relu'), 

        #final Layers
        Conv2D(12, (1,1), padding = 'valid'), 
        Activation('softmax')
    ])
    
    # compile model
    segNet.compile(optimizer= 'adam', metrics= ['accuracy'], 
              loss= 'categorical_crossentropy')
segNet.summary()
vutil.plot_model(segNet, show_shapes= True, show_layer_names=True)
from keras.callbacks import EarlyStopping
earlyStop = EarlyStopping(monitor= 'val_loss', patience= 10)
hist = segNet.fit(train_data, train_label, batch_size= 6, 
                  epochs = 50, 
                  verbose= 2, 
                  validation_data = [val_data, val_label], 
                  callbacks= [earlyStop])
segNet.save_weights('model_weight.hdf5')
import matplotlib.pyplot as plt
%matplotlib inline

Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road_marking = [255,69,0]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

label_colours = np.array([Sky, Building, Pole, Road, 
                          Pavement,Tree, SignSymbol, 
                          Fence, Car,  Pedestrian, 
                          Bicyclist ,  Unlabelled])

# getting 20 row of the test data (ram limitation of kaggle)
test_data = []
test_label = []

with open('/kaggle/input/camvid/CamVid/test.txt', 'r') as f:
    _file = f.readlines()
    
for idx in range(20):
    line = _file[idx].split()
    test_data.append(io.imread(root + line[0][7:]))
    test_label.append(io.imread(root + line[1][7:]))
test_data = np.array(test_data)
test_label = np.array(test_label)
test_data.shape
io.imshow(test_label[1])
pred = segNet.predict(test_data)
io.imshow(np.argmax(pred[1],axis=2))
