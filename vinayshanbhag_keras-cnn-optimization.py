# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

import matplotlib

import matplotlib.image as mpimg

import numpy as np

from numpy import random



from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

from keras.utils import np_utils

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau, Callback

from keras import regularizers

from keras.optimizers import Adam





## visualize model using GraphViz

#import os

#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

#from keras.utils import plot_model



def display_images(X, y=[], rows=5, columns=5, cmap="gray"):

    """ Display images and labels

    """

    fig, ax = plt.subplots(rows,columns, figsize=(6,6))

    for row in range(rows):

        for column in range(columns):

            ax[row][column].imshow(X[(row*columns)+column].reshape(28,28), cmap=cmap)

            ax[row][column].set_axis_off()

            if len(y):ax[row][column].set_title("{}:{}".format("label",np.argmax(y[(row*columns)+column])))

    fig.tight_layout()



%matplotlib inline
df = pd.read_csv("../input/train.csv")

#df = pd.read_csv("train.csv")

df.sample(1)
X_train, X_val, y_train, y_val = train_test_split(df.iloc[:,1:].values, df.iloc[:,0].values, test_size = 0.4)

X_cv, X_test, y_cv, y_test = train_test_split(X_val, y_val, test_size = 0.5)

print("X_train:{}\ny_train:{}\n\nX_cv:{}\ny_cv:{}\n\nX_test:{}\ny_test:{}".format(X_train.shape, y_train.shape, X_cv.shape, y_cv.shape, X_test.shape, y_test.shape))
width = 28

height = 28

channels = 1

X_train = X_train.reshape(X_train.shape[0], width, height, channels)

X_cv = X_cv.reshape(X_cv.shape[0], width, height, channels)

X_test = X_test.reshape(X_test.shape[0], width, height, channels)



# convert output classes to one hot representation

y_train = np_utils.to_categorical(y_train, num_classes=10)

y_cv = np_utils.to_categorical(y_cv, num_classes=10)

y_test = np_utils.to_categorical(y_test, num_classes=10)



X_train = X_train.astype('float32')

X_cv = X_cv.astype('float32')

X_test = X_test.astype('float32')



# Scale features (pixel values) from 0-255, to 0-1 

X_train /= 255

X_cv /= 255

X_test /= 255

print("Reshaped:")

print("X_train:{}\ny_train:{}\n\nX_cv:{}\ny_cv:{}\n\nX_test:{}\ny_test:{}".format(X_train.shape, y_train.shape, X_cv.shape, y_cv.shape, X_test.shape, y_test.shape))
display_images(X_train, y_train)
batch_size=84

epochs=5 # Change to 30

verbose=2



class LossHistory(Callback):

    def on_train_begin(self, logs={}):

        self.losses = []



    def on_batch_end(self, batch, logs={}):

        self.losses.append(logs.get('loss'))





def create_model():

    model = Sequential()

    model.add(Conv2D(32, (5,5), padding="same", activation='relu', input_shape=(width, height, channels) ))

    model.add(Conv2D(32, (5,5), padding="same", activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.4))

    model.add(Conv2D(64, (3,3), padding="same", activation='relu'))

    model.add(Conv2D(64, (3,3), padding="same", activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(384, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))

    

    opt = "adam" #Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.compile(loss='categorical_crossentropy',

                  optimizer=opt,

                  metrics=['accuracy'])

    return model



def plot_metrics(h, title=""):

    """ Plot training metrics - loss and accuracy, for each epoch, 

        given a training history object

    """

    fig, axes = plt.subplots(1,2, figsize=(10,5))

      

    axes[0].plot(h.history['loss'], color="lightblue", label="Training", lw=2.0)

    axes[0].plot(h.history['val_loss'], color="steelblue", label="Validation", lw=2.0)



    axes[0].set_title("{} (Loss)".format(title))

    axes[0].set_xlabel("Epoch")

    axes[0].set_xticks(np.arange(len(h.history["loss"]), 2))

    axes[0].set_ylabel("Loss")

    

    axes[1].plot(h.history['acc'], color="lightblue", label="Training", lw=2.0)

    axes[1].plot(h.history['val_acc'], color="steelblue", label="Validation", lw=2.0)

    

    axes[1].set_title("{} (Accuracy)".format(title))

    axes[1].set_xlabel("Epoch")

    axes[1].set_xticks(np.arange(len(h.history["acc"]), 2))

    axes[1].set_ylabel("Accuracy")

    



    for axis in axes:

        axis.ticklabel_format(useOffset=False)

        axis.spines["top"].set_visible(False)

        axis.spines["right"].set_visible(False)

        axis.legend(loc='best', shadow=False)

    fig.tight_layout()

    

def plot_losses(batch_hist, title=""):

    fig, ax1 = plt.subplots()



    ax1.semilogx(batch_hist.losses)

    ax1.set_title("{} (Batch Loss)".format(title))  

    

    ax1.spines["top"].set_visible(False)

    ax1.spines["right"].set_visible(False)



    plt.show()
model0 = create_model()



# Visualize model using GraphViz

#plot_model(model0, show_shapes=True, show_layer_names=False,to_file='model.png')



model0_batch_hist = LossHistory()



model0_metrics = model0.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, 

          validation_data = (X_cv, y_cv), verbose = verbose, shuffle=True, callbacks=[model0_batch_hist])



#model0.save_weights("model0.h5")
learning_rate_controller = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=verbose, factor=0.3, min_lr=0.00001, epsilon=0.001)
model1 = create_model()

model1_batch_hist = LossHistory()

model1_metrics = model1.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, 

          validation_data = (X_cv, y_cv), verbose = verbose, shuffle=True, callbacks=[learning_rate_controller,model1_batch_hist])

#model1.save_weights("model1.h5")
idg = ImageDataGenerator(

        rotation_range=10,

        zoom_range = 0.05, 

        width_shift_range=0.05,

        height_shift_range=0.05,

        horizontal_flip=False,

        vertical_flip=False, data_format="channels_last")
image_data = idg.flow(X_train,y_train, batch_size=25).next()

print("Sample images from ImageDataGenerator:")

display_images(image_data[0], image_data[1])
model2 = create_model()

model2_batch_hist = LossHistory()

model2_metrics = model2.fit_generator(idg.flow(X_train,y_train, batch_size=batch_size),

                    epochs = epochs,

                    steps_per_epoch=X_train.shape[0]//batch_size,

                    validation_data=(X_cv,y_cv),

                    callbacks=[learning_rate_controller,model2_batch_hist],                         

                    verbose = verbose)

#model2.save_weights("model2.h5")
plot_losses(model0_batch_hist, "CNN")

plot_losses(model1_batch_hist, "CNN with Learning Rate Annealer")

plot_losses(model2_batch_hist, "CNN with Augmented Data")
plot_metrics(model0_metrics,"Convolutional Neural Network")

plot_metrics(model1_metrics,"CNN with Learning Rate Annealer\n")

plot_metrics(model2_metrics,"CNN with Annealer and Data Augmentation\n")
models = [model0, model1, model2]

metrics = [model0_metrics, model1_metrics, model2_metrics]

names = ["Convolutional Neural Network", "CNN + Learning Rate Annealing", "CNN + LR + Data Augmentation"

         ]

data = []

for i, m in enumerate(zip(names, metrics, models)):

    data.append([m[0], "{:0.2f}".format(m[1].history["acc"][-1]*100), "{:0.2f}".format(m[1].history["val_acc"][-1]*100), "{:0.2f}".format(m[2].evaluate(X_test, y_test, verbose=0)[1]*100)])



results = pd.DataFrame(data, columns=("Model","Training Accuracy","Validation Accuracy", "Test Accuracy"))

from IPython.display import display, HTML

display(HTML(results.to_html(index=False)))

plt.bar(np.arange(len(results["Model"].values)),results["Training Accuracy"].values.astype("float64"), 0.2, color="lightblue")

plt.bar(np.arange(len(results["Model"].values))+0.2,results["Validation Accuracy"].values.astype("float64"), 0.2, color="steelblue")

plt.bar(np.arange(len(results["Model"].values))+0.4,results["Test Accuracy"].values.astype("float64"), 0.2, color="navy")

plt.ylim(97, 100)

plt.xticks(np.arange(len(results["Model"].values))+0.2, ["CNN","CNN+LR", "CNN+LR+Aug"])

plt.legend(["Training","Validation", "Test"],loc=(1,0.5))

g = plt.gca()

g.spines["top"].set_visible(False)

g.spines["right"].set_visible(False)

plt.title("Accuracy")