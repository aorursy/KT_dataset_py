import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")
df_train.head()
df_test.head()
import matplotlib.pyplot as plt



print("Pandas Version:", pd.__version__)



def prepare_dataset(df):

    X = df

    Y = None

    if 'label' in df.columns:

        # to_numpy()는 0.24.0 이상부터 지원

        Y = df["label"].values.reshape([-1, 1])

        X = df.drop("label", axis=1)

    

    # to_numpy()는 0.24.0 이상부터 지원

    X = X.values

    return X, Y



train_X, train_Y = prepare_dataset(df_train)

submit_X, _ = prepare_dataset(df_test)



ntest = 10

fig, ax = plt.subplots(nrows=1, ncols=ntest, figsize=(20, 3))



indices = np.random.randint(0, train_X.shape[0], ntest)

for idx, i in enumerate(indices):    

    ax[idx].imshow(train_X[i].reshape(28, 28))

    ax[idx].set_title(train_Y[i])

    

plt.show()
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization



def make_model(input_shape, label_cnt, dropout_r = 0.3):

    model = Sequential()

    

    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='Same', activation='relu', input_shape=input_shape))

    model.add(BatchNormalization())

              

    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='Same', activation='relu', input_shape=input_shape))

    model.add(BatchNormalization())

              

    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='Same', activation='relu', input_shape=input_shape))

    model.add(BatchNormalization())

    

    model.add(Flatten())

    

    model.add(Dense(512, activation="relu"))

    model.add(BatchNormalization())

    model.add(Dense(128, activation="relu"))

    model.add(BatchNormalization())

    

    # 출력 크기를 label의 one-hot encoding한 사이즈와 동일하게 맞춰준다.

    model.add(Dense(label_cnt, activation='softmax'))

              

    return model
from keras.callbacks import ReduceLROnPlateau

'''

monitor : 판단할 log의 이름

factor : 변경할 비율 (new_lr = lr * factor)

patience : 해당 값 만큼의 epoch이 지날 동안 정확도가 좋아지지 않으면 적용된다.

'''

lr_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.25, min_lr=0.000001)
from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(

    featurewise_center=False,  # set input mean to 0 over the dataset

    samplewise_center=False,  # set each sample mean to 0

    featurewise_std_normalization=False,  # divide inputs by std of the dataset

    samplewise_std_normalization=False,  # divide each input by its std

    zca_whitening=False,  # apply ZCA whitening

    rotation_range=16,  # randomly rotate images in the range (degrees, 0 to 180)

    zoom_range = 0.15, # Randomly zoom image 

    width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)

    height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)

    horizontal_flip=False,  # randomly flip images

    vertical_flip=False)  # randomly flip images



train_X, _ = prepare_dataset(df_train)

datagen.fit(train_X.reshape(-1, 28, 28, 1))
from keras.utils.np_utils import to_categorical



from sklearn.model_selection import KFold

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split





width = 28

height = 28

channel = 1

classes = 10



# 데이터 생성

train_X, train_Y = prepare_dataset(df_train)

submit_X, _ = prepare_dataset(df_test)



# reshape to image

train_X = train_X.reshape(-1, width, height, channel)

submit_X = submit_X.reshape(-1, width, height, channel)



train_X = train_X / 255.0

submit_X = submit_X / 255.0



train_Y = to_categorical(train_Y, num_classes = classes)

print("train_Y.shape=", train_Y.shape)



# 모델 생성

model = make_model((width, height, channel), label_cnt = classes, dropout_r=0.3)

model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])



batch_size = 400

total_batch = int(train_X.shape[0] / batch_size)

print("total_batch=",total_batch)



epochs = 30

n_splits = 10



use_generator = True



# 모델 훈련 및 평가

history = None

if use_generator:

    #미리 데이터를 쪼개서 validation data로 사용하거나, 그냥 validation data도 generator로 넣거나 한다.

    #X_train, X_val, Y_train, Y_val = train_test_split(train_X, train_Y, test_size = 0.1, random_state=84)

    history = model.fit_generator(datagen.flow(train_X, train_Y, batch_size=batch_size),

                                  epochs = epochs,

                                  #validation_data = (X_val, Y_val),

                                  validation_data = datagen.flow(train_X, train_Y, batch_size=batch_size),

                                  verbose = 2,

                                  steps_per_epoch=total_batch,

                                  validation_steps=total_batch,

                                  callbacks=[lr_reduction])

else:     

    history = model.fit(train_X, train_Y, batch_size=batch_size,

                        epochs=epochs, validation_split=0.1)



# show statistics:

fig, ax = plt.subplots(1, 1, figsize=(12,8))

ax.plot(history.history['acc'], 'b-', label="accuracy")

ax.set_title("accuracy")

plt.show()



# test 데이터로 submission.csv생성:

submit_Y = model.predict(submit_X.reshape(-1, width, height, channel))

submit_Y = np.argmax(submit_Y, axis=1)              

ids = [x for x in range(1, submit_Y.shape[0] + 1)]

pd_submit = pd.DataFrame({'ImageId':ids, 'Label':submit_Y})

pd_submit.to_csv("submission.csv", index=False)
