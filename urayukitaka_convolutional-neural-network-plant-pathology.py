# Basic library

import numpy as np 

import pandas as pd 



# Data preprocessing

import cv2 # Open cv

from sklearn.model_selection import train_test_split



# Visualization

from matplotlib import pyplot as plt



# Machine learning library

import keras

from keras.models import Sequential, Model, load_model

from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization, Activation, Input

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras.preprocessing.image import ImageDataGenerator
# dataframe data

sample_submission = pd.read_csv("../input/plant-pathology-2020-fgvc7/sample_submission.csv")

test = pd.read_csv("../input/plant-pathology-2020-fgvc7/test.csv")

train = pd.read_csv("../input/plant-pathology-2020-fgvc7/train.csv")
# train image data

size = 64

train_image_data = []



# loading

for _id in train["image_id"]:

    path = '../input/plant-pathology-2020-fgvc7/images/'+_id+'.jpg'

    img = cv2.imread(path)

    image = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    train_image_data.append(image)
# test image data

size = 64

test_image_data = []



# loading

for _id in test["image_id"]:

    path = '../input/plant-pathology-2020-fgvc7/images/'+_id+'.jpg'

    img = cv2.imread(path)

    image = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    test_image_data.append(image)
# sample_submission data

sample_submission.head()
# train data

train.head()
# data information

def data_info(data):

    print("-"*20, "data_info", "-"*20)

    print(data.info())

    print("-"*20, "data_info", "-"*20)



data_info(train)
# test data

test.head()
# train image data size

len(train_image_data)
# visualization, train_data

fig, ax = plt.subplots(1,3,figsize=(10,10))

for i in range(3):

    ax[i].imshow(train_image_data[i])
# visualization, test_data

fig, ax = plt.subplots(1,3,figsize=(10,10))

for i in range(3):

    ax[i].imshow(test_image_data[i])
# Data dimension

X_Train = np.ndarray(shape=(len(train_image_data), size, size, 3),

                     dtype=np.float32)

# Change to nu.ndarray

i=0

for image in train_image_data:

    X_Train[i]=train_image_data[i]

    i=i+1

    

# Scaling

X_Train = X_Train/255



# Checking dimension

print("Train_shape:{}".format(X_Train.shape))
# Data dimension adjust

X_Test = np.ndarray(shape=(len(test_image_data), size, size, 3),

                     dtype=np.float32)

# Change to np.ndarray

i=0

for image in test_image_data:

    X_Test[i]=test_image_data[i]

    i=i+1

    

# Scaling

X_Test = X_Test/255



# Checking dimension

print("Train_shape:{}".format(X_Test.shape))
y = train.iloc[:,1:]



# change to np.array

y = np.array(y.values)

print("y_shape:{}".format(y.shape))
# data split

X_train, X_val, y_train, y_val = train_test_split(X_Train,

                                                  y,

                                                  test_size=0.2,

                                                  random_state=10)
# target data

y_train1 = [y[0] for y in y_train]

y_train2 = [y[1] for y in y_train]

y_train3 = [y[2] for y in y_train]

y_train4 = [y[3] for y in y_train]



# val data

y_val1 = [y[0] for y in y_val]

y_val2 = [y[1] for y in y_val]

y_val3 = [y[2] for y in y_val]

y_val4 = [y[3] for y in y_val]



# convert class vectors to binary class metrices

y_train1 = keras.utils.to_categorical(y_train1, 2)

y_train2 = keras.utils.to_categorical(y_train2, 2)

y_train3 = keras.utils.to_categorical(y_train3, 2)

y_train4 = keras.utils.to_categorical(y_train4, 2)



y_val1 = keras.utils.to_categorical(y_val1, 2)

y_val2 = keras.utils.to_categorical(y_val2, 2)

y_val3 = keras.utils.to_categorical(y_val3, 2)

y_val4 = keras.utils.to_categorical(y_val4, 2)
def define_model():

    inputs = Input(shape=(size, size, 3))

    

    # 1st layer

    x = BatchNormalization()(inputs)

    x = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1))(x)

    x = BatchNormalization()(x)

    x = Activation("relu")(x)

    x = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1))(x)

    x = BatchNormalization()(x)

    x = Activation("relu")(x)

    x = MaxPool2D(pool_size=(2,2))(x)

    x = Dropout(0.2)(x)

    

    # 2nd layer

    x = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1))(x)

    x = BatchNormalization()(x)

    x = Activation("relu")(x)

    x = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1))(x)

    x = BatchNormalization()(x)

    x = Activation("relu")(x)

    x = MaxPool2D(pool_size=(2,2))(x)

    x = Dropout(0.2)(x)

    

    # 3rd layer

    x = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1))(x)

    x = BatchNormalization()(x)

    x = Activation("relu")(x)

    x = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1))(x)

    x = BatchNormalization()(x)

    x = Activation("relu")(x)

    x = MaxPool2D(pool_size=(2,2))(x)

    x = Dropout(0.2)(x)

    

    # Flatten

    x = Flatten()(x)

    

    # Dens layer

    x = Dense(1024, activation='relu')(x)

    x = Dropout(0.2)(x)

    x = Dense(1024, activation='relu')(x)

    x = Dropout(0.2)(x)

    

    output1 = Dense(2, activation="softmax", name='output1')(x)

    output2 = Dense(2, activation="softmax", name='output2')(x)

    output3 = Dense(2, activation="softmax", name='output3')(x)

    output4 = Dense(2, activation="softmax", name='output4')(x)

    

    multiModel = Model(inputs, [output1, output2, output3, output4])

    

    # initiate Adam optimizer

    opt = keras.optimizers.adam(lr=0.0001, decay=0.00001)

    

    # Compile

    multiModel.compile(loss={'output1':'categorical_crossentropy',

                            'output2':'categorical_crossentropy',

                            'output3':'categorical_crossentropy',

                            'output4':'categorical_crossentropy'},

                      optimizer=opt,

                      metrics=["accuracy"])

    return multiModel
# data augmentation, This is dropped because GPU is down.

datagen = ImageDataGenerator(rotation_range=360,

                             width_shift_range=0.2,

                             height_shift_range=0.2,

                             horizontal_flip=True)

datagen.fit(X_train)



# define early stopping

es_cb = EarlyStopping(monitor='val_loss',

                    patience=15,

                    verbose=1)

cp_cb = ModelCheckpoint("cnn_model_02.h5",

                        monitor='val_loss',

                        verbose=1,

                        save_best_only=True)

# parameters

batch_size = 8

epochs = 100



# train model

model = define_model()

history = model.fit(X_train,

                   {'output1':y_train1,

                    'output2':y_train2,

                    'output3':y_train3,

                    'output4':y_train4},

                   batch_size=batch_size,

                   epochs=epochs,

                   validation_data=(X_val,

                                   {'output1':y_val1,

                                    'output2':y_val2,

                                    'output3':y_val3,

                                    'output4':y_val4}),

                   callbacks=[es_cb, cp_cb])
# train_loss

train1_loss = history.history["output1_loss"]

train2_loss = history.history["output2_loss"]

train3_loss = history.history["output3_loss"]

train4_loss = history.history["output4_loss"]



# val_loss

val1_loss = history.history["val_output1_loss"]

val2_loss = history.history["val_output2_loss"]

val3_loss = history.history["val_output3_loss"]

val4_loss = history.history["val_output4_loss"]



# train_accuracy

train1_acc = history.history["output1_accuracy"]

train2_acc = history.history["output2_accuracy"]

train3_acc = history.history["output3_accuracy"]

train4_acc = history.history["output4_accuracy"]



# val_accuracy

val1_acc = history.history["val_output1_accuracy"]

val2_acc = history.history["val_output2_accuracy"]

val3_acc = history.history["val_output3_accuracy"]

val4_acc = history.history["val_output4_accuracy"]



# Visualization

fig, ax = plt.subplots(2,4,figsize=(25,10))

plt.subplots_adjust(wspace=0.3)



# train1 loss

ax[0,0].plot(range(len(train1_loss)), train1_loss, label='train1_loss')

ax[0,0].plot(range(len(val1_loss)), val1_loss, label='val1_loss')

ax[0,0].set_xlabel('epoch', fontsize=16)

ax[0,0].set_ylabel('loss', fontsize=16)

ax[0,0].set_yscale('log')

ax[0,0].legend(fontsize=16)



# train2 loss

ax[0,1].plot(range(len(train2_loss)), train2_loss, label='train2_loss')

ax[0,1].plot(range(len(val2_loss)), val2_loss, label='val2_loss')

ax[0,1].set_xlabel('epoch', fontsize=16)

ax[0,1].set_ylabel('loss', fontsize=16)

ax[0,1].set_yscale('log')

ax[0,1].legend(fontsize=16)



# train3 loss

ax[0,2].plot(range(len(train3_loss)), train3_loss, label='train3_loss')

ax[0,2].plot(range(len(val2_loss)), val3_loss, label='val3_loss')

ax[0,2].set_xlabel('epoch', fontsize=16)

ax[0,2].set_ylabel('loss', fontsize=16)

ax[0,2].set_yscale('log')

ax[0,2].legend(fontsize=16)



# train4 loss

ax[0,3].plot(range(len(train4_loss)), train4_loss, label='train4_loss')

ax[0,3].plot(range(len(val4_loss)), val4_loss, label='val4_loss')

ax[0,3].set_xlabel('epoch', fontsize=16)

ax[0,3].set_ylabel('loss', fontsize=16)

ax[0,3].set_yscale('log')

ax[0,3].legend(fontsize=16)



# train1 accuracy

ax[1,0].plot(range(len(train1_acc)), train1_acc, label='train1_accuracy')

ax[1,0].plot(range(len(val1_acc)), val1_acc, label='val1_accuracy')

ax[1,0].set_xlabel('epoch', fontsize=16)

ax[1,0].set_ylabel('accuracy', fontsize=16)

ax[1,0].set_yscale('log')

ax[1,0].legend(fontsize=16)



# train2 accuracy

ax[1,1].plot(range(len(train2_acc)), train2_acc, label='train2_accuracy')

ax[1,1].plot(range(len(val2_acc)), val2_acc, label='val2_accuracy')

ax[1,1].set_xlabel('epoch', fontsize=16)

ax[1,1].set_ylabel('accuracy', fontsize=16)

ax[1,1].set_yscale('log')

ax[1,1].legend(fontsize=16)



# train3 accuracy

ax[1,2].plot(range(len(train3_acc)), train3_acc, label='train3_accuracy')

ax[1,2].plot(range(len(val3_acc)), val3_acc, label='val3_accuracy')

ax[1,2].set_xlabel('epoch', fontsize=16)

ax[1,2].set_ylabel('accuracy', fontsize=16)

ax[1,2].set_yscale('log')

ax[1,2].legend(fontsize=16)



# train4 accuracy

ax[1,3].plot(range(len(train4_acc)), train4_acc, label='train4_accuracy')

ax[1,3].plot(range(len(val4_acc)), val4_acc, label='val4_accuracy')

ax[1,3].set_xlabel('epoch', fontsize=16)

ax[1,3].set_ylabel('accuracy', fontsize=16)

ax[1,3].set_yscale('log')

ax[1,3].legend(fontsize=16)
model = load_model('cnn_model_02.h5')
predict = model.predict(X_Test)

healthy = [y_test[1] for y_test in predict[0]]

multiple_diseases = [y_test[1] for y_test in predict[1]]

rust = [y_test[1] for y_test in predict[2]]

scab = [y_test[1] for y_test in predict[3]]
submit = pd.DataFrame({"image_id":test["image_id"],

                    "healthy":healthy,

                    "multiple_diseases":multiple_diseases,

                    "rust":rust,

                    "scab":scab})

submit.tail()
submit.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")