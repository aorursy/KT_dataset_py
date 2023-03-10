# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import openslide
import os
import cv2
import PIL
from IPython.display import Image, display
from keras.applications.vgg16 import VGG16,preprocess_input
# Plotly for the interactive viewer (see last section)
import plotly.graph_objs as go
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model,load_model
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten,BatchNormalization,Activation
from keras.layers import GlobalMaxPooling2D
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Option 2: Load images using skimage (requires that tifffile is installed)
import skimage.io
import random
import seaborn as sns


# General packages
import matplotlib
import matplotlib.pyplot as plt
import PIL
from IPython.display import Image, display

# Plotly for the interactive viewer (see last section)
import plotly.graph_objs as go

from keras.models import Sequential, Model, load_model
from keras import applications
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Input

from keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from sklearn import preprocessing, metrics
from sklearn.model_selection import cross_validate
from sklearn.metrics import cohen_kappa_score 
import gc
import skimage.io
from sklearn.model_selection import KFold
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import tensorflow as tf
from tensorflow.python.keras import backend as K
sess = K.get_session()
train=pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/train.csv')
train.head()
img=openslide.OpenSlide('/kaggle/input/prostate-cancer-grade-assessment/train_images/2fd1c7dc4a0f3a546a59717d8e9d28c3.tiff')
display(img.get_thumbnail(size=(512,512)))
img.dimensions
patch = img.read_region((18500,4100), 0, (256, 256))

# Display the image
display(patch)
# Close the opened slide after use
img.close()
train['isup_grade'].value_counts()
train['gleason_score'].unique()
train[(train['isup_grade'] == 2) & (train['gleason_score'] == '4+3')]
train.drop([7273],inplace=True)
train['gleason_score'] = train['gleason_score'].apply(lambda x: "0+0" if x=="negative" else x)
print(len(train[train['gleason_score']=='negative']['isup_grade']))
temp = train.groupby('isup_grade').count()['image_id'].reset_index().sort_values(by='image_id',ascending=False)
temp.style.background_gradient(cmap='Purples')
temp = train.groupby('gleason_score').count()['image_id'].reset_index().sort_values(by='image_id',ascending=False)
temp.style.background_gradient(cmap='Reds')
labels=[]
data=[]
data_dir='/kaggle/input/panda-resized-train-data-512x512/train_images/train_images/'
for i in range(train.shape[0]):
    data.append(data_dir + train['image_id'].iloc[i]+'.png')
    labels.append(train['isup_grade'].iloc[i])
df=pd.DataFrame(data)
df.columns=['images']
df['isup_grade']=labels
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test  = train_test_split(df['images'],df['isup_grade'], test_size=0.05, random_state=1234)
X_train, X_val, y_train, y_val = train_test_split(df['images'],df['isup_grade'], test_size=0.35, random_state=1234)


train=pd.DataFrame(X_train)
train.columns=['images']
train['isup_grade']=y_train

validation=pd.DataFrame(X_val)
validation.columns=['images']
validation['isup_grade']=y_val

test=pd.DataFrame(X_test)
test.columns=['images']
test['isup_grade']=y_test


train['isup_grade']=train['isup_grade'].astype(str)
validation['isup_grade']=validation['isup_grade'].astype(str)
test['isup_grade']=test['isup_grade'].astype(str)
train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,horizontal_flip=True)
val_datagen=train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen=train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train,
    x_col='images',
    y_col='isup_grade',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

validation_generator = val_datagen.flow_from_dataframe(
    validation,
    x_col='images',
    y_col='isup_grade',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')
test_generator = test_datagen.flow_from_dataframe(
    test,
    x_col='images',
    y_col='isup_grade',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')
def vgg16_model( num_classes=None):

    model = VGG16(weights='/kaggle/input/keras-pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=(224, 224, 3))
    x=Flatten()(model.output)
    output=Dense(num_classes,activation='softmax')(x)
    model=Model(model.input,output)
    return model

vgg_conv=vgg16_model(6)
vgg_conv.summary()
def kappa_score(y_true, y_pred):
    
    y_true=tf.math.argmax(y_true)
    y_pred=tf.math.argmax(y_pred)
    return tf.compat.v1.py_func(cohen_kappa_score ,(y_true, y_pred),tf.double)
opt = SGD(lr=0.001)
vgg_conv.compile(loss='categorical_crossentropy',optimizer=opt,metrics=[kappa_score, 'accuracy'])
nb_epochs = 50
batch_size=32
nb_train_steps = train.shape[0]//batch_size
nb_val_steps=validation.shape[0]//batch_size
print("Number of training and validation steps: {} and {}".format(nb_train_steps,nb_val_steps))
history = vgg_conv.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_steps,
    epochs=nb_epochs,
    validation_data=validation_generator,
    validation_steps=nb_val_steps)
# Check Performance
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
kappa = history.history['kappa_score']
val_kappa = history.history['val_kappa_score']

epochs = range(len(loss))

plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

plt.plot(epochs, loss, 'b', label='kappa')
plt.plot(epochs, val_loss, 'r', label='val_kappa')
plt.title('Training and validation kappa')
plt.legend()

plt.show()
# submission code from https://www.kaggle.com/frlemarchand/high-res-samples-into-multi-input-cnn-keras
def predict_submission(df, path):
    
    df["image_path"] = [path+image_id+".tiff" for image_id in df["image_id"]]
    df["isup_grade"] = 0
    predictions = []
    for idx, row in df.iterrows():
        print(row.image_path)
        img=skimage.io.imread(str(row.image_path))
        img = cv2.resize(img, (224,224))
        img = cv2.resize(img, (224,224))
        img = img.astype(np.float32)/255.
        img=np.reshape(img,(1,224,224,3))
        prediction=vgg_conv.predict(img)
        predictions.append(np.argmax(prediction))
            
    df["isup_grade"] = predictions
    df = df.drop('image_path', 1)
    return df[["image_id","isup_grade"]]
test_path = "../input/prostate-cancer-grade-assessment/test_images/"
submission_df = pd.read_csv("../input/prostate-cancer-grade-assessment/sample_submission.csv")

if os.path.exists(test_path):
    test_df = pd.read_csv("../input/prostate-cancer-grade-assessment/test.csv")
    submission_df = predict_submission(test_df, test_path)

submission_df.to_csv('submission.csv', index=False)
submission_df.head()