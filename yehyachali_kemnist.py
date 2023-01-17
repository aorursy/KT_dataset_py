import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from keras import optimizers

from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from keras.models import Model

from keras.layers import Conv2D, Input, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.utils.np_utils import to_categorical



from sklearn.model_selection import train_test_split
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

submission = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")

X = train.drop(['label'],1).values

Y = train['label'].values

x_test = test.values



X = X/255.

x_test = x_test/255.



X = X.reshape(-1,28,28,1)

x_test = x_test.reshape(-1,28,28,1)



Y = to_categorical(Y)

x_train, x_valid, y_train, y_valid = train_test_split(X,Y, test_size=0.2)
def get_model():

    In = Input(shape=(28,28,1))

    l = Conv2D(32, (3,3), activation="relu",padding="same")(In)

    l = Conv2D(32, (3,3), activation="relu",padding="same")(l)

    l = BatchNormalization()(l)

    l = MaxPooling2D((2,2))(l)

    l = Dropout(0.2)(l)

    

    l = Conv2D(64, (3,3), activation="relu", padding="same")(l)

    l = Conv2D(64, (3,3), activation="relu", padding="same")(l)

    l = BatchNormalization()(l)

    l = MaxPooling2D((2,2))(l)

    l = Dropout(0.2)(l)

    

    l= Conv2D(128, (3,3), activation="relu", padding="same")(l)

    l= Conv2D(128, (3,3), activation="relu", padding="same")(l)

    l = BatchNormalization()(l)

    

#     l = BatchNormalization()(l)

    l = Dropout(0.2)(l)

    l = Flatten()(l)

    l = Dense(256, activation="relu")(l)

    l = Dropout(0.2)(l)

    Out = Dense(10, activation="softmax")(l)

    model = Model(In, Out)

    model.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"])

    return model

    

model = get_model()

model.summary()
traing = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images



validg = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

#         rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

#         zoom_range = 0.08, # Randomly zoom image 

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images



epochs = 200

batch_size = 32



train_generator = traing.flow(x_train, y_train, batch_size=batch_size)

valid_generator = validg.flow(x_valid, y_valid, batch_size=batch_size)
# history = model.fit_generator(train_generator, epochs=epochs, steps_per_epoch = x_train.shape[0]//batch_size,

#                     validation_data = valid_generator, validation_steps = x_valid.shape[0]//batch_size)

# history = model.fit_generator(train_generator, epochs=epochs, steps_per_epoch = X.shape[0]//batch_size)

history = model.fit(train_generator, 

                    epochs=epochs,

                    steps_per_epoch = x_train.shape[0]//batch_size,

                    validation_data=valid_generator,

                    validation_steps = x_valid.shape[0]//batch_size,

                    verbose=1)
plt.figure(figsize=(10,10))

plt.plot(history.history["val_accuracy"])

plt.plot(history.history["accuracy"])

plt.legend(["val_accuracy", "accuracy"])

plt.show()
preds = model.predict(x_test, verbose=1)

preds = np.array([np.argmax(i) for i in preds])

preds
submission['Label'] = preds

submission.to_csv("submission.csv", index=False)

submission.head()