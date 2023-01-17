#import os

#print(os.listdir("./"))
from copy import deepcopy

import os

import pandas as pd

import csv

import numpy as np

import keras

from keras.models import Model, Sequential

from keras.optimizers import Adam

from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

from keras.callbacks import ReduceLROnPlateau

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

from sklearn.metrics import confusion_matrix

import itertools

from keras.callbacks import ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

start_size = 28



data = pd.read_csv("../input/train.csv")

df = pd.DataFrame(data)



all_linear_data = df.as_matrix(columns=df.columns[1:])

all_labels = df.as_matrix(columns=df.columns[:1])

    

all_image_data = all_linear_data.reshape((df.shape[0], start_size, start_size, 1))

all_image_data = all_image_data / 255.0

all_labels = keras.utils.to_categorical(all_labels, 10)



validation_start = int((1 - 0.1) * all_image_data.shape[0])



train_data = all_image_data[0:validation_start]

train_labels = all_labels[0:validation_start]

validation_train_data = all_image_data[validation_start:]

validation_labels = all_labels[validation_start:]



test_data = pd.read_csv("../input/test.csv")

test_df = pd.DataFrame(test_data)



all_linear_test_image_data = test_df.as_matrix(columns=df.columns[1:])

all_test_image_data = all_linear_test_image_data.reshape((test_df.shape[0], start_size, start_size, 1))

all_test_image_data  = all_test_image_data  / 255.
def get_lr_metrics(optimizer):

    def lr(y_true, y_pred):

        return optimizer.lr

    return lr



def get_model():

    model = Sequential()



    model.add(Conv2D(32,kernel_size=3,activation='relu',input_shape=(28,28,1)))

    model.add(BatchNormalization())

    model.add(Conv2D(32,kernel_size=3,activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.4))



    model.add(Conv2D(64,kernel_size=3,activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(64,kernel_size=3,activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.4))



    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.4))

    model.add(Dense(10, activation='softmax'))

    

    return model





model = get_model()

opimizer = Adam(1e-4)

model.compile(optimizer=opimizer, loss="categorical_crossentropy",

              metrics=["accuracy", get_lr_metrics(opimizer)])
datagen_args = dict(rotation_range=20,

                    width_shift_range=0.1,

                    height_shift_range=0.1,

                    shear_range=0.1,

                    zoom_range=0.1)

datagen = ImageDataGenerator(**datagen_args)





learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',

                                            patience=5,

                                            verbose=1,

                                            factor=0.85,

                                            min_lr=1e-10)



weights_fname = 'mnist'

checkpoints = ModelCheckpoint(weights_fname + '-best.h5', monitor='val_acc', verbose=1,

                              save_best_only=True, save_weights_only=True, mode='max', period=1)



history = model.fit_generator(datagen.flow(train_data, train_labels, batch_size=64),

                              epochs=60, steps_per_epoch=train_data.shape[0]//64, # 0.9969 on 53th epoch

                              validation_data=(validation_train_data, validation_labels),

                              callbacks=[checkpoints, learning_rate_reduction],

                              verbose=0)
fig, ax = plt.subplots(2, 1)



ax[0].plot(history.history['loss'], color='b', label='Training loss')

ax[0].plot(history.history['val_loss'], color='r', label='validation loss', axes=ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['acc'], color='b', label='Training accuracy')

ax[1].plot(history.history['val_acc'], color='r', label='Validation accuracy')

legend = ax[1].legend(loc='best', shadow=True)

def plot_confusion_matrix(cm,

                          classes,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    plt.figure()

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)

     

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment='center',

                 color='white' if cm[i, j] > thresh else 'black')

    

    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()

    

train_predicted_ohe = model.predict(all_image_data)

train_predicted_numbers = np.argmax(train_predicted_ohe, axis=1)

trained_true_numbers = np.argmax(all_labels, axis=1)



predict_ohe = model.predict(all_test_image_data)



confusion_mtx = confusion_matrix(trained_true_numbers, train_predicted_numbers)

plot_confusion_matrix(confusion_mtx, classes=range(10))
def show_num_errors():

    all_errors = confusion_mtx.sum()

    for ind in range(10):

        all_errors = all_errors - confusion_mtx[ind, ind]



    print('------------------------------')

    print(f'all errors: {all_errors} from {all_image_data.shape[0]} ( {100.0 * all_errors / all_image_data.shape[0]} percent)')

    print('------------------------------')

    

show_num_errors()
def display_all_problem_images():

    # display errors for each cipher

    for cipher in range(10):

        number_errors = confusion_mtx[cipher, :].sum() - confusion_mtx[cipher, cipher]



        for index in range(10):

            num_errors = confusion_mtx[cipher, index]

            if (index == cipher) or (num_errors == 0):

                continue



            print(f"True label is {cipher}, but predicted is {index}")



            num_rows = 1 + num_errors // 10        

            plt.figure(figsize=(10 * 1, num_rows * 1))

            plt.axis('off')    



            current_error = 0

            current_index = 0

            while current_error < num_errors:

                if  (trained_true_numbers[current_index] == cipher) and (train_predicted_numbers[current_index] == index):

                    ax = plt.subplot(num_rows, 10, current_error + 1)

                    ax.xaxis.set_major_locator(ticker.NullLocator())

                    ax.xaxis.set_minor_locator(ticker.NullLocator())

                    ax.yaxis.set_major_locator(ticker.NullLocator())

                    ax.yaxis.set_minor_locator(ticker.NullLocator())                

                    plt.imshow(all_image_data[current_index].reshape(start_size, start_size), cmap='gray')

                    current_error = current_error + 1

                current_index = current_index + 1



            plt.show()

            

#display_all_problem_images()
for index in range(0, 10):

    

    print(f'index is {index} from 0-9')

    

    model = get_model()

    opimizer = Adam(1e-4)

    model.compile(optimizer=opimizer, loss="categorical_crossentropy",

              metrics=["accuracy", get_lr_metrics(opimizer)])    

    

    model.load_weights(weights_fname + '-best.h5')

    

    history = model.fit_generator(datagen.flow(train_data, train_labels, batch_size=64),

                                  epochs=55, steps_per_epoch=train_data.shape[0]//64,

                                  validation_data=(validation_train_data, validation_labels),

                                  callbacks=[checkpoints, learning_rate_reduction],

                                  initial_epoch=0,

                                  verbose=0)

    

    train_predicted_ohe_current = model.predict(all_image_data)

    train_predicted_ohe = train_predicted_ohe + train_predicted_ohe_current

    train_predicted_numbers = np.argmax(train_predicted_ohe, axis=1)



    confusion_mtx = confusion_matrix(trained_true_numbers, train_predicted_numbers)

    plot_confusion_matrix(confusion_mtx, classes=range(10))

    

    show_num_errors()

    

    predict_ohe_current = model.predict(all_test_image_data)

    predict_ohe = predict_ohe + predict_ohe_current
display_all_problem_images()
results = np.argmax(predict_ohe, axis=1)

results = pd.Series(results, name="Label")

submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), results], axis=1)
# import the modules we'll need

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "mnist_results1.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(submission)