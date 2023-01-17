! pip install keras.applications
# Basic library

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Dir check

import os
# OpenCV

import cv2 # Open cv



# Data preprocessing

from sklearn.model_selection import train_test_split # ML preprocessing



# Karas

from keras.applications import DenseNet121 # RestNet number 101

from keras.preprocessing.image import ImageDataGenerator # data augmentation

from keras.models import Model # Define model

from keras.models import Sequential # For define simple neural network

from keras.models import Input # Define Input

from keras.models import load_model

from keras.layers import Dense # Define neural network layer

from keras.layers import Conv2D # Define convolution layer

from keras.layers import Flatten # multidimensional lists into one dimension

from keras.layers import MaxPool2D # Define max pooling layer

from keras.layers import Dropout # Dropout method

from keras.layers import BatchNormalization # BatchNormalization method

from keras.layers import Activation # Define activation

from keras.layers import GlobalAveragePooling2D

from keras.optimizers import Adam # Optimizer

from keras.callbacks import ModelCheckpoint # call back

from keras.callbacks import EarlyStopping



# Visualization

from matplotlib import pyplot as plt

import seaborn as sns



# Validation

from sklearn.metrics import roc_auc_score
sample_submission = pd.read_csv("../input/plant-pathology-2020-fgvc7/sample_submission.csv")

test = pd.read_csv("../input/plant-pathology-2020-fgvc7/test.csv")

train = pd.read_csv("../input/plant-pathology-2020-fgvc7/train.csv")
sample_submission.head()
train.head()
test.head()
# image loading

img_size=256

train_image = []



for name in train["image_id"]:

    path = '../input/plant-pathology-2020-fgvc7/images/'+name+'.jpg' # difine path

    img=cv2.imread(path) # reading the image

    image = cv2.resize(img, (img_size,img_size), interpolation=cv2.INTER_AREA) # Resize the image (100,100), decreasing size:cv2.INTER_AREA

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Change to color array

    train_image.append(image) # listing tha datas
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

img_size=256

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
gblur_img = [cv2.GaussianBlur(img, (3,3),0) for img in train_image]



train["gblur"] = gblur_img
# Visualization some sample

col = ["healthy", "multiple_diseases", "rust", "scab"]

fig, ax = plt.subplots(4,4, figsize=(15,15))

for c in col:

    for i in range(4):

        if c == col[0]:

            sample = train[train[c]==1]

            ax[0,i].set_axis_off()

            ax[0,i].imshow(sample["gblur"].values[i])

            ax[0,i].set_title("{}".format(c))

        elif c == col[1]:

            sample = train[train[c]==1]

            ax[1,i].set_axis_off()

            ax[1,i].imshow(sample["gblur"].values[i])

            ax[1,i].set_title("{}".format(c))

        elif c == col[2]:

            sample = train[train[c]==1]

            ax[2,i].set_axis_off()

            ax[2,i].imshow(sample["gblur"].values[i])

            ax[2,i].set_title("{}".format(c))

        else:

            sample = train[train[c]==1]

            ax[3,i].set_axis_off()

            ax[3,i].imshow(sample["gblur"].values[i])

            ax[3,i].set_title("{}".format(c))
edete_img = [cv2.Canny(img, 100, 200) for img in train_image]



train["edete"] = edete_img
# Visualization some sample

col = ["healthy", "multiple_diseases", "rust", "scab"]

fig, ax = plt.subplots(4,4, figsize=(15,15))

for c in col:

    for i in range(4):

        if c == col[0]:

            sample = train[train[c]==1]

            ax[0,i].set_axis_off()

            ax[0,i].imshow(sample["edete"].values[i])

            ax[0,i].set_title("{}".format(c))

        elif c == col[1]:

            sample = train[train[c]==1]

            ax[1,i].set_axis_off()

            ax[1,i].imshow(sample["edete"].values[i])

            ax[1,i].set_title("{}".format(c))

        elif c == col[2]:

            sample = train[train[c]==1]

            ax[2,i].set_axis_off()

            ax[2,i].imshow(sample["edete"].values[i])

            ax[2,i].set_title("{}".format(c))

        else:

            sample = train[train[c]==1]

            ax[3,i].set_axis_off()

            ax[3,i].imshow(sample["edete"].values[i])

            ax[3,i].set_title("{}".format(c))
eqhist_img = []

for img in train_image:

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    h,s,v = cv2.split(img)

    h = cv2.equalizeHist(h)

    s = cv2.equalizeHist(s)

    v = cv2.equalizeHist(v)

    hsv = cv2.merge((h,s,v))

    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    eqhist_img.append(img)



train["eqhist"] = eqhist_img
# Visualization some sample

col = ["healthy", "multiple_diseases", "rust", "scab"]

fig, ax = plt.subplots(4,4, figsize=(15,15))

for c in col:

    for i in range(4):

        if c == col[0]:

            sample = train[train[c]==1]

            ax[0,i].set_axis_off()

            ax[0,i].imshow(sample["eqhist"].values[i])

            ax[0,i].set_title("{}".format(c))

        elif c == col[1]:

            sample = train[train[c]==1]

            ax[1,i].set_axis_off()

            ax[1,i].imshow(sample["eqhist"].values[i])

            ax[1,i].set_title("{}".format(c))

        elif c == col[2]:

            sample = train[train[c]==1]

            ax[2,i].set_axis_off()

            ax[2,i].imshow(sample["eqhist"].values[i])

            ax[2,i].set_title("{}".format(c))

        else:

            sample = train[train[c]==1]

            ax[3,i].set_axis_off()

            ax[3,i].imshow(sample["eqhist"].values[i])

            ax[3,i].set_title("{}".format(c))
clahe_img = []

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))

for img in train_image:

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    h,s,v = cv2.split(img)

    h = clahe.apply(h)

    s = clahe.apply(s)

    v = clahe.apply(v)

    hsv = cv2.merge((h,s,v))

    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    clahe_img.append(img)



train["clahe"] = clahe_img
# Visualization some sample

col = ["healthy", "multiple_diseases", "rust", "scab"]

fig, ax = plt.subplots(4,4, figsize=(15,15))

for c in col:

    for i in range(4):

        if c == col[0]:

            sample = train[train[c]==1]

            ax[0,i].set_axis_off()

            ax[0,i].imshow(sample["clahe"].values[i])

            ax[0,i].set_title("{}".format(c))

        elif c == col[1]:

            sample = train[train[c]==1]

            ax[1,i].set_axis_off()

            ax[1,i].imshow(sample["clahe"].values[i])

            ax[1,i].set_title("{}".format(c))

        elif c == col[2]:

            sample = train[train[c]==1]

            ax[2,i].set_axis_off()

            ax[2,i].imshow(sample["clahe"].values[i])

            ax[2,i].set_title("{}".format(c))

        else:

            sample = train[train[c]==1]

            ax[3,i].set_axis_off()

            ax[3,i].imshow(sample["clahe"].values[i])

            ax[3,i].set_title("{}".format(c))
# Difine function

def create_rgb_df(sample_df):

    create_df = pd.DataFrame({})

    # Create each list

    red_mean = []

    red_std = []

    green_mean = []

    green_std = []

    blue_mean = []

    blue_std = []

    

    for i in range(len(sample_df)):

        red_m = sample_df.values[i][:,:,0].mean()

        red_s = sample_df.values[i][:,:,0].std()

        green_m = sample_df.values[i][:,:,1].mean()

        green_s = sample_df.values[i][:,:,1].std()

        blue_m = sample_df.values[i][:,:,2].mean()

        blue_s = sample_df.values[i][:,:,2].std()

        # Append to list

        red_mean.append(red_m)

        red_std.append(red_s)

        green_mean.append(green_m)

        green_std.append(green_s)

        blue_mean.append(blue_m)

        blue_std.append(blue_s)



    create_df["red_mean"] = red_mean

    create_df["red_std"] = red_std

    create_df["green_mean"] = green_mean

    create_df["green_std"] = green_std

    create_df["blue_mean"] = blue_mean

    create_df["blue_std"] = blue_std

    

    return create_df
col = ["healthy", "multiple_diseases", "rust", "scab"]



# healthy

sampling_1 = train[train[col[0]]==1]["img_data"]

sampling_2 = train[train[col[0]]==1]["clahe"]

healthy_df_base = create_rgb_df(sampling_1)

healthy_df_clahe = create_rgb_df(sampling_2)



# multiple_diseases

sampling_1 = train[train[col[1]]==1]["img_data"]

sampling_2 = train[train[col[1]]==1]["clahe"]

multi_df_base = create_rgb_df(sampling_1)

multi_df_clahe = create_rgb_df(sampling_2)



# rust

sampling_1 = train[train[col[2]]==1]["img_data"]

sampling_2 = train[train[col[2]]==1]["clahe"]

rust_df_base = create_rgb_df(sampling_1)

rust_df_clahe = create_rgb_df(sampling_2)



# scab

sampling_1 = train[train[col[3]]==1]["img_data"]

sampling_2 = train[train[col[3]]==1]["clahe"]

scab_df_base = create_rgb_df(sampling_1)

scab_df_clahe = create_rgb_df(sampling_2)
# Visualization of Red std distribution

fig, ax = plt.subplots(2,2, figsize=(20, 12))

# healthy

sns.distplot(healthy_df_base["red_std"], label="base", ax=ax[0,0])

sns.distplot(healthy_df_clahe["red_std"], label="clahe", ax=ax[0,0])

ax[0,0].set_title("Red color distribution, for healthy")

ax[0,0].legend()



# multi

sns.distplot(multi_df_base["red_std"], label="base", ax=ax[0,1])

sns.distplot(multi_df_clahe["red_std"], label="clahe", ax=ax[0,1])

ax[0,1].set_title("Red color distribution, for multilple desease")

ax[0,1].legend()



# rust

sns.distplot(rust_df_base["red_std"], label="base", ax=ax[1,0])

sns.distplot(rust_df_clahe["red_std"], label="clahe", ax=ax[1,0])

ax[1,0].set_title("Red color distribution, for rust")

ax[1,0].legend()



# scab

sns.distplot(scab_df_base["red_std"], label="base", ax=ax[1,1])

sns.distplot(scab_df_clahe["red_std"], label="clahe", ax=ax[1,1])

ax[1,1].set_title("Red color distribution, for scab")

ax[1,1].legend()
# Visualization of Blue std distribution

fig, ax = plt.subplots(2,2, figsize=(20, 12))

# healthy

sns.distplot(healthy_df_base["blue_std"], label="base", ax=ax[0,0])

sns.distplot(healthy_df_clahe["blue_std"], label="clahe", ax=ax[0,0])

ax[0,0].set_title("Blue color distribution, for healthy")

ax[0,0].legend()



# multi

sns.distplot(multi_df_base["blue_std"], label="base", ax=ax[0,1])

sns.distplot(multi_df_clahe["blue_std"], label="clahe", ax=ax[0,1])

ax[0,1].set_title("Blue color distribution, for multilple desease")

ax[0,1].legend()



# rust

sns.distplot(rust_df_base["blue_std"], label="base", ax=ax[1,0])

sns.distplot(rust_df_clahe["blue_std"], label="clahe", ax=ax[1,0])

ax[1,0].set_title("Blue color distribution, for rust")

ax[1,0].legend()



# scab

sns.distplot(scab_df_base["blue_std"], label="base", ax=ax[1,1])

sns.distplot(scab_df_clahe["blue_std"], label="clahe", ax=ax[1,1])

ax[1,1].set_title("Blue color distribution, for scab")

ax[1,1].legend()
# Visualization of Red mean distribution

fig, ax = plt.subplots(2,2, figsize=(20, 12))

# healthy

sns.distplot(healthy_df_base["green_std"], label="base", ax=ax[0,0])

sns.distplot(healthy_df_clahe["green_std"], label="clahe", ax=ax[0,0])

ax[0,0].set_title("Green color distribution, for healthy")

ax[0,0].legend()



# multi

sns.distplot(multi_df_base["green_std"], label="base", ax=ax[0,1])

sns.distplot(multi_df_clahe["green_std"], label="clahe", ax=ax[0,1])

ax[0,1].set_title("Green color distribution, for multilple desease")

ax[0,1].legend()



# rust

sns.distplot(rust_df_base["green_std"], label="base", ax=ax[1,0])

sns.distplot(rust_df_clahe["green_std"], label="clahe", ax=ax[1,0])

ax[1,0].set_title("Green color distribution, for rust")

ax[1,0].legend()



# scab

sns.distplot(scab_df_base["green_std"], label="base", ax=ax[1,1])

sns.distplot(scab_df_clahe["green_std"], label="clahe", ax=ax[1,1])

ax[1,1].set_title("Green color distribution, for scab")

ax[1,1].legend()
class prepro_DenseNet():

    # image data:Series data, target:target data dateframe, size:image size

    def __init__(self, image_data, target, size):

        self.image = image_data

        self.target = target

        self.size = size

        pass

    

    # Dimension change and create train and val data

    # test_size:split size

    def preprocessing(self, test_size, random_state):   

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

        self.X_train = X_train

        self.X_val = X_val

        self.y_train = y_train

        self.y_val = y_val 

            

    def define_DenseNet121(self):

        densenet = DenseNet121(include_top=False, weights="imagenet")

        

        inputs = Input(shape=(self.size, self.size, 3))

        x = densenet(inputs)

        x = GlobalAveragePooling2D()(x)

        x = Dense(1024, activation='relu')(x)

        x = Dropout(0.3)(x)

        x = Dense(512, activation='relu')(x)

        output = Dense(4, activation="softmax", name="root")(x)

        

        model = Model(inputs, output)

        

        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.0001, decay=0.0001)

        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

        self.model = model

        

        return self.model, self.model.summary()

        

    def exe_DenseNet121(self, batch_size, epochs, save_file):

        self.save_file = str(save_file)

        self.batch_size = batch_size

        self.epochs = epochs

        # Datagen

        datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)

        datagen.fit(self.X_train)

        # early stopping and model checkpoint

        es_cb = EarlyStopping(monitor="val_loss", patience=10, verbose=1)

        cp_cb = ModelCheckpoint("{}".format(self.save_file), monitor="val_loss", verbose=1, save_best_only=True)

        

        history_ = self.model.fit_generator(datagen.flow(self.X_train, self.y_train, batch_size=self.batch_size),

                                                         steps_per_epoch=len(self.X_train) / self.batch_size, 

                                                         epochs=self.epochs, 

                                                         validation_data=datagen.flow(self.X_val, self.y_val, batch_size=self.batch_size), 

                                                         callbacks=[es_cb, cp_cb])

        self.history_ = history_

        

    def roc_auc_score(self):

        # prediction

        y_pred = load_model("{}".format(self.save_file)).predict(self.X_val)

        # print roc_auc score

        print("roc_auc score:{}".format(roc_auc_score(y_true=self.y_val, y_score=y_pred, average="weighted").round(3)))

        

    def visualization(self):

        # loss and accuracy 

        train_loss = self.history_.history["loss"]

        val_loss = self.history_.history["val_loss"]



        train_acc = self.history_.history["accuracy"]

        val_acc = self.history_.history["val_accuracy"]



        # Visualization

        fig, ax = plt.subplots(1,2,figsize=(20,6))

        ax[0].plot(range(len(train_loss)), train_loss, label="train_loss")

        ax[0].plot(range(len(val_loss)), val_loss, label="val_loss")

        ax[0].set_xlabel("epoch")

        ax[0].set_ylabel("loss")

        ax[0].legend()



        ax[1].plot(range(len(train_acc)), train_acc, label="train_acc")

        ax[1].plot(range(len(val_acc)), val_acc, label="val_acc")

        ax[1].set_xlabel("epoch")

        ax[1].set_ylabel("accuracy")

        ax[1].legend()
# data

image_data = train["img_data"]

target = train[['healthy', 'multiple_diseases', 'rust', 'scab']]

size = img_size



# preprocessing

test_size=0.2

random_state=20

save_file = "dense121_v1"



# Densenet

batch_size = 32

epochs = 100
# Execution

base = prepro_DenseNet(image_data, target, size)

base.preprocessing(test_size, random_state)

base.define_DenseNet121()
base.exe_DenseNet121(batch_size, epochs, save_file)
# roc_auc score

base.roc_auc_score()
# visualization

base.visualization()
# data

image_data = train["clahe"]

target = train[['healthy', 'multiple_diseases', 'rust', 'scab']]

size = img_size



# preprocessing

test_size=0.2

random_state=20

save_file = "dense121_v2"



# Densenet

batch_size = 32

epochs = 100



# Execution

eqhist = prepro_DenseNet(image_data, target, size)

eqhist.preprocessing(test_size, random_state)

eqhist.define_DenseNet121()
eqhist.exe_DenseNet121(batch_size, epochs, save_file)
# roc_auc score

eqhist.roc_auc_score()
# visualization

eqhist.visualization()
# Data dimension

X_Test = np.ndarray(shape=(len(test_image), 256, 256, 3), dtype=np.float32)

# Change to np.ndarray

for i in range(len(test_image)):

    X_Test[i]=test_image[i]

    i=i+1

# Scaling

X_Test = X_Test/255
Y_pred = load_model("dense121_v1").predict(X_Test)

Y_pred.shape
Y_pred = pd.DataFrame(Y_pred, columns=col)

test[col] = Y_pred

submit = test

submit.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
Y_pred = load_model("dense121_v2").predict(X_Test)

Y_pred.shape
Y_pred = pd.DataFrame(Y_pred, columns=col)

test[col] = Y_pred

submit = test

submit.to_csv('my_submission2.csv', index=False)

print("Your submission was successfully saved!")