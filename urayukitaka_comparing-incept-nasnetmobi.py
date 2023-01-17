# Basic library

import numpy as np 

import pandas as pd 

import gc



# Dir check

import os
!conda install efficientnet.tfkeras
!pip install efficientnet
# OpenCV

import cv2 # Open cv



# Data preprocessing

from sklearn.model_selection import train_test_split # ML preprocessing



# Visualization

from matplotlib import pyplot as plt

import seaborn as sns



# Validation

from sklearn.metrics import roc_auc_score



# Karas

import keras

from IPython.display import SVG

from keras.utils import model_to_dot

from keras.applications import InceptionV3, MobileNet

import efficientnet.tfkeras as efn

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model 

from keras.models import Sequential 

from keras.models import Input 

from keras.models import load_model

from keras.layers import Dense 

from keras.layers import Conv2D 

from keras.layers import Flatten

from keras.layers import MaxPool2D

from keras.layers import Dropout 

from keras.layers import BatchNormalization

from keras.layers import Activation 

from keras.layers import GlobalAveragePooling2D

from keras.optimizers import Adam 

from keras.callbacks import ModelCheckpoint

from keras.callbacks import EarlyStopping
sample_submission = pd.read_csv("../input/plant-pathology-2020-fgvc7/sample_submission.csv")

test = pd.read_csv("../input/plant-pathology-2020-fgvc7/test.csv")

train = pd.read_csv("../input/plant-pathology-2020-fgvc7/train.csv")
# image loading

img_size=299

train_image = []



for name in train["image_id"]:

    path = '../input/plant-pathology-2020-fgvc7/images/'+name+'.jpg' 

    img=cv2.imread(path) 

    image = cv2.resize(img, (img_size,img_size), interpolation=cv2.INTER_AREA)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

    train_image.append(image)

    

train["img_data"] = train_image
# Visualization some sample

col = ["healthy", "multiple_diseases", "rust", "scab"]

fig, ax = plt.subplots(4,4, figsize=(15,15))

for c in col:

    for i in range(4):

        if c == col[0]:

            sample = train[train[c]==1]

            ax[0,i].set_axis_off()

            ax[0,i].imshow(sample["img_data"].values[i])

            ax[0,i].set_title("{}".format(c))

        elif c == col[1]:

            sample = train[train[c]==1]

            ax[1,i].set_axis_off()

            ax[1,i].imshow(sample["img_data"].values[i])

            ax[1,i].set_title("{}".format(c))

        elif c == col[2]:

            sample = train[train[c]==1]

            ax[2,i].set_axis_off()

            ax[2,i].imshow(sample["img_data"].values[i])

            ax[2,i].set_title("{}".format(c))

        else:

            sample = train[train[c]==1]

            ax[3,i].set_axis_off()

            ax[3,i].imshow(sample["img_data"].values[i])

            ax[3,i].set_title("{}".format(c))
# image loading

img_size=299

test_image = []



for name in test["image_id"]:

    path = '../input/plant-pathology-2020-fgvc7/images/'+name+'.jpg'

    img = cv2.imread(path)

    image = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    test_image.append(image)
fig, ax = plt.subplots(1,4, figsize=(15,6))

for i in range(4):

    ax[i].set_axis_off()

    ax[i].imshow(test_image[i])
class preprocessing():

    # image data:Series data, target:target data dateframe, size:image size

    def __init__(self, image_data, target, size):

        self.image = image_data

        self.target = target

        self.size = size

        pass

    

    # Dimension change and create train and val data

    # test_size:split size

    def dataset(self, test_size, random_state):   

        self.test_size = test_size

        self.random_state = random_state

        

        # Data dimension

        X_Train = np.ndarray(shape=(len(self.image), self.size, self.size, 3), dtype=np.float32)

        # Change to np.ndarray

        for i in range(len(self.image)):

            X_Train[i]=self.image[i]

            i=i+1

    

        # Scaling

        X_Train = X_Train/255



        # change to np.array

        self.target = np.array(self.target.values)

        

        # split train and val data

        X_train, X_val, y_train, y_val = train_test_split(X_Train, self.target, test_size=self.test_size, random_state=self.random_state)

        return X_train, X_val, y_train, y_val
# data

image_data = train["img_data"]

target = train[['healthy', 'multiple_diseases', 'rust', 'scab']]

size = img_size



# preprocessing

test_size=0.2

random_state=20



prepro = preprocessing(image_data, target, size)

X_train, X_val, y_train, y_val = prepro.dataset(test_size, random_state)
def define_model(model_name, size):

    model = model_name(include_top=False, weights="imagenet")

    

    inputs = Input(shape=(size, size, 3))

    x = model(inputs)

    x = GlobalAveragePooling2D()(x)

    x = Dense(512, activation='relu')(x)

    x = Dense(256, activation='relu')(x)

    output = Dense(4, activation="softmax", name="root")(x)

        

    model = Model(inputs, output)

        

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

        

    return model
def exe_model(model, X_train, y_train, X_val, y_val, save_file):

    save_file = str(save_file)

    batch_size=16

    valid_samples=32

    train_samples = len(X_train) - valid_samples

    

    # Data augmentation

    datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.2,

                                 height_shift_range=0.2, horizontal_flip=True)

    datagen.fit(X_train)

    

    # early stopping and model checkpoint

    es = EarlyStopping(monitor='val_loss', patience=15, verbose=1)

    mc = ModelCheckpoint(save_file, monitor="val_loss", verbose=1, save_best_only=True)

    

    hist = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),

                               steps_per_epoch=train_samples/batch_size, 

                               epochs=100, callbacks=[es, mc], 

                               validation_data=datagen.flow(X_val, y_val, batch_size=batch_size),

                               validation_steps=valid_samples/batch_size)

    return hist
def train_curve(hist_data):

    train_loss = hist_data.history["loss"]

    val_loss = hist_data.history["val_loss"]

    train_acc = hist_data.history["accuracy"]

    val_acc = hist_data.history["val_accuracy"]

    

    fig, ax = plt.subplots(1,2, figsize=(20,6))

    # loss

    ax[0].plot(range(len(train_loss)), train_loss, label="train_loss")

    ax[0].plot(range(len(val_loss)), val_loss, label="val_loss")

    ax[0].set_xlabel("epochs")

    ax[0].set_ylabel("loss")

    ax[0].set_yscale("log")

    ax[0].legend()

    # accuracy

    ax[1].plot(range(len(train_acc)), train_acc, label="train_acc")

    ax[1].plot(range(len(val_acc)), val_acc, label="val_acc")

    ax[1].set_xlabel("epochs")

    ax[1].set_ylabel("accuracy")

    ax[1].set_yscale("log")

    ax[1].legend()
SVG(model_to_dot(InceptionV3(), dpi=70).create(prog='dot', format='svg'))
# model compile

incept = define_model(InceptionV3, 299)

incept.summary()
SVG(model_to_dot(Model(incept.layers[0].input, incept.layers[4].output), dpi=70).create(prog='dot', format='svg'))
# save file

save_file = "incept_v1"

# Execute model

hist_incept = exe_model(incept, X_train, y_train, X_val, y_val, save_file)
# ROC AUC score

y_pred_incept = load_model(save_file).predict(X_val)



# print ROC AUC score

print("ROC AUC score:{}".format(roc_auc_score(y_true=y_val, y_score=y_pred_incept, average="weighted").round(3)))
# Training curve

train_curve(hist_incept)
SVG(model_to_dot(MobileNet(), dpi=70).create(prog='dot', format='svg'))
# model compile

mobilen = define_model(MobileNet, 299)

mobilen.summary()
SVG(model_to_dot(Model(mobilen.layers[0].input, mobilen.layers[4].output), dpi=70).create(prog='dot', format='svg'))
# save file

save_file = "mobilen_v1"

# Execute model

hist_mobilen = exe_model(mobilen, X_train, y_train, X_val, y_val, save_file)
# ROC AUC score

y_pred_mobilen = load_model(save_file).predict(X_val)



# print ROC AUC score

print("ROC AUC score:{}".format(roc_auc_score(y_true=y_val, y_score=y_pred_mobilen, average="weighted").round(3)))
# Training curve

train_curve(hist_mobilen)
del X_train, X_val, y_train, y_val

gc.collect()
# Data dimension

X_Test = np.ndarray(shape=(len(test_image), 299, 299, 3), dtype=np.float32)

# Change to np.ndarray

for i in range(len(test_image)):

    X_Test[i]=test_image[i]

    i=i+1

# Scaling

X_Test = X_Test/255
# prediction Inception

Y_test_incept = load_model("incept_v1").predict(X_Test)
# Create submit data

col = ["healthy", "multiple_diseases", "rust", "scab"]



# Inception V3

Y_test_incept = pd.DataFrame(Y_test_incept, columns=col)

submit_incept = pd.DataFrame({})

submit_incept["image_id"] = test["image_id"]

submit_incept[col] = Y_test_incept

submit_incept.to_csv('submit_incept.csv', index=False)
# prediction Inception

Y_test_mobilen = load_model("mobilen_v1").predict(X_Test)
# Create submit data

col = ["healthy", "multiple_diseases", "rust", "scab"]



# Inception V3

Y_test_mobilen = pd.DataFrame(Y_test_mobilen, columns=col)

submit_mobilen = pd.DataFrame({})

submit_mobilen["image_id"] = test["image_id"]

submit_mobilen[col] = Y_test_mobilen

submit_mobilen.to_csv('submit_mobilen.csv', index=False)