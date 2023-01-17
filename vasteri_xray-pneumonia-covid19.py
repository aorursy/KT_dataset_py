# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
os.listdir('../input/chest-xray-pneumonia/chest_xray')
train_set = '../input/chest-xray-pneumonia/chest_xray'
train = os.path.join(train_set)
from fastai.metrics import error_rate
from fastai.vision import *
tfms = get_transforms(max_rotate=1, max_zoom=.1)
data = ImageDataBunch.from_folder(train_set, train='train', valid='test', ds_tfms=tfms, size=128, bs=32).normalize(imagenet_stats)
data.show_batch(figsize=(10,10), rows=3)
data.classes
data.batch_stats
#import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
IMG_SIZE = (64, 64)
datagen = ImageDataGenerator(samplewise_center=True, 
                              samplewise_std_normalization=True, 
                              horizontal_flip = True, 
                              vertical_flip = False, 
                              height_shift_range= 0.05, 
                              width_shift_range=0.1, 
                              rotation_range=5, 
                              shear_range = 0.1,
                              fill_mode = 'reflect',
                              zoom_range=0.15)
path = "/kaggle/input/chest-xray-pneumonia/chest_xray/"
train_generator = datagen.flow_from_directory(os.path.join(path,'train'),
        target_size=IMG_SIZE,
        color_mode = 'grayscale',
        batch_size=32,
        class_mode='binary')

x_val, y_val = next(datagen.flow_from_directory(os.path.join(path,'val'),
        target_size=IMG_SIZE,
        color_mode = 'grayscale',
        batch_size=32,
        class_mode='binary')) # one big batch

x_test, y_test = next(datagen.flow_from_directory((os.path.join(path,'test')),
        target_size=IMG_SIZE,
        color_mode = 'grayscale',
        batch_size=180,
        class_mode='binary')) # one big batch
x_test.shape
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

# Instantiate model
model = Sequential()
# Adding feature detection
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=x_test.shape[1:]))

# Adding Max Pooling
model.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
model.add(Flatten())

# Adding connections
model.add(Dense(128, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

# Compiling
model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                           metrics = ['binary_accuracy', 'mae'])
model.summary()
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('xrays_pneumonia_cnn')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)

early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=5)
callbacks_list = [checkpoint, early]
model.fit_generator(train_generator, 
                    steps_per_epoch=100, 
                    validation_data = (x_val, y_val), 
                    epochs = 1, 
                    callbacks = callbacks_list)
# Save full model
model.save('xrays_pneumonia_cnn') 
scores = model.evaluate(x_test, y_test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("val_loss:", scores[0])
print("val_mean_absolute_error:", scores[2])
model.fit_generator(train_generator, 
                    steps_per_epoch=100, 
                    validation_data = (x_val, y_val), 
                    epochs = 11, 
                    callbacks = callbacks_list)
model.load_weights(weight_path)
scores = model.evaluate(x_test, y_test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("val_loss:", scores[0])
print("val_mean_absolute_error:", scores[2])
pred_Y = model.predict(x_test, batch_size = 32, verbose = True)
print(pred_Y[:15])
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
# Compute ROC curve and ROC area for each class

num_classes = 0

fpr = dict()
tpr = dict()
roc_auc = dict()
fpr, tpr, _ = roc_curve(y_test, pred_Y)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
lw = 2
plt.plot(fpr, tpr, color='darkorange', 
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
import cv2
import csv
covid_preds = '/kaggle/input/covid-chest-xray/images'
covid_images = list(os.listdir(covid_preds))
path, dirs, files = next(os.walk(covid_preds))
file_count = len(files)

for image in covid_images:
    img = cv2.imread(os.path.join(covid_preds,image), cv2.IMREAD_GRAYSCALE,)
    try:
        resize = cv2.resize(img, (64,64))
    except cv2.error as e:
        print("Invalid frame")
        cv2.waitKey()

covid_test = []
for image in covid_images:
    img = cv2.imread(os.path.join(covid_preds,image),cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64,64))
    covid_test.append(img)
    
covid_test = np.array(covid_test)
covid_test = covid_test.reshape(file_count,64,64,1)
covid_prediction = model.predict(covid_test)
predict_class = []
for p in covid_prediction:
    if p <= 0.5:
        predict_class.append(0)
    else:
        predict_class.append(1)
with open("my_prediction", "w", newline= "") as f:
    thewriter = csv.writer(f)
    thewriter.writerow(["Image name", "prediction"])
    for i in range(len(covid_prediction)):
        if covid_images[i] ==1:
            thewriter.writerow([covid_images[i],"Positive"])
        else :
            thewriter.writerow([covid_images[i],"Negative"])

covid_diagnosis = pd.read_csv('my_prediction')
covid_diagnosis.head(20)
real_values = pd.read_csv("/kaggle/input/covid-chest-xray/metadata.csv")
real_values["finding"].unique()
# model_urls = "https://download.pytorch.org/models/resnet34-333f7ec4.pth"
!mkdir -p /kaggle/working/torch/models
for model_name in model_urls:
    model_filename=os.path.basename(model_urls[model_name])
    dst=os.path.expanduser(os.path.join("/kaggle/working/torch/models/",model_filename))
    src=os.path.expanduser(os.path.join("/kaggle/working/",model_name,model_name+".pth"))
    if os.path.exists(src) and not os.path.exists(dst):
        print("{}: {} -> {}".format(model_name, src, dst))
        os.symlink(src, dst)
!ls /kaggle/working/torch/models/
# from keras.applications.resnet50 import ResNet50
# !ls /kaggle/working/keras-pretrained-models/