# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')

print(train.shape)
print(test.shape)
#transfer (42000, 784) to (42000,28,28)
y_train=train.iloc[:,0]
print(y_train.unique())
x_train=train.drop('label',axis=1)/255.0
print(x_train.shape)
x_train_cp=np.reshape(x_train.values,(42000,28,28,1))
print(x_train_cp.shape)
#how to check the dataframe type of a dataset?

######
def Digmodel(input_shape): # CONV -> BN -> RELU Block and 1 Dense layer
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    #X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32,4,strides = 1, name = 'conv0')(X_input)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(10, activation='softmax', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='HappyModel')

    return model
#creat model-compile-fit
digmodel = Digmodel((28,28,1))
digmodel.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
y_train_cp=pd.get_dummies(y_train)
from keras.callbacks import EarlyStopping
stop=EarlyStopping(monitor='val_loss',patience=3)
Id=np.array(list(range(1,28001)))
digmodel.fit(x_train_cp, y_train_cp,validation_split=0.1,epochs=40, batch_size=200,callbacks=[stop])
x_test_cp=np.reshape(test.values,(28000,28,28,1))/255.0
preds = digmodel.predict(x_test_cp)
preds[1,:]
maxidx=np.argmax(preds,axis=1)
print(maxidx.shape)
print(Id.shape)

df = pd.DataFrame({'ImageId':Id, 'Label':maxidx})
df.head()
Dec18=pd.DataFrame(df)
Dec18.to_csv('Dec18.csv',header=True,index=False)
