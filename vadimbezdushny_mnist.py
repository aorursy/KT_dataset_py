# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import itertools

from sklearn import model_selection

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import datasets, svm, metrics

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from sklearn.metrics import confusion_matrix

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

%matplotlib inline

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
ds_train = pd.read_csv('../input/digit-recognizer/train.csv')

ds_test = pd.read_csv('../input/digit-recognizer/test.csv')

val_size = 0.1

X_all = ds_train.loc[:, ds_train.columns != 'label'].to_numpy()

y_all = ds_train['label'].to_numpy()



test = ds_test.to_numpy()
sns.countplot(y_all)
X_all = X_all / 255.0 # Normalization

X_all = X_all.reshape(-1,28,28,1)



test = test / 255.0

test = test.reshape(-1,28,28,1)



y_all = to_categorical(y_all, num_classes = 10)





X_train, X_val, y_train, y_val = model_selection.train_test_split(X_all, y_all, test_size=val_size, random_state=1)
nrows = 5

ncols = 5

n = 0

fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True, figsize=(15,15))

for row in range(nrows):

    for col in range(ncols):

        ax[row,col].imshow((X_train[n]).reshape((28,28)), cmap = 'binary')

        ax[row,col].set_title("Label :{}".format(np.argmax(y_train[n])))

        n += 1
optimizer = RMSprop(lr=0.001, rho=0.9, decay=0.0)
def get_simple_model():

    model = Sequential()

    model.add(Flatten(input_shape=(28,28,1)))

    model.add(Dense(64, activation = 'sigmoid'))

    model.add(Dropout(0.2))

    model.add(Dense(24, activation = 'sigmoid'))

    model.add(Dropout(0.2))

    model.add(Dense(10, activation = 'softmax'))

    

    model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics=["accuracy"])

    return model
def train_model(model, epochs = 10, batch_size = 256):

    model.summary()

    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=[X_val, y_val])    

    return model, history



def show_history(history):

    plt.plot(history.history['accuracy'])

    plt.plot(history.history['val_accuracy'])

    plt.title('model accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()

    

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('model loss')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()
simple_model, history_sm = train_model(get_simple_model())

show_history(history_sm)
def get_cnn_model():

    model = Sequential()



    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (28,28,1)))

    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu'))

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Dropout(0.25))





    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(Dropout(0.25))





    model.add(Flatten())

    model.add(Dense(256, activation = "relu"))

    model.add(Dropout(0.5))

    model.add(Dense(10, activation = "softmax"))

    

    model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics=["accuracy"])

    return model
cnn_model, history_cnn = train_model(get_cnn_model())

show_history(history_cnn)
Y_pred = cnn_model.predict(X_val)

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

Y_true = np.argmax(y_val,axis = 1) 



cm = confusion_matrix(Y_true, Y_pred_classes) 

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

classes = range(10)

tick_marks = np.arange(len(classes))

plt.xticks(tick_marks, classes)

plt.yticks(tick_marks, classes)



thresh = cm.max() / 2.

for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

    plt.text(j, i, cm[i, j],

             horizontalalignment="center",

             color="white" if cm[i, j] > thresh else "black")



plt.ylabel('True label')

plt.xlabel('Predicted label')
errors = (Y_pred_classes != Y_true)



Y_pred_classes_errors = Y_pred_classes[errors]

Y_pred_errors = Y_pred[errors]

Y_true_errors = Y_true[errors]

X_val_errors = X_val[errors]



def display_errors(errors_index,img_errors,pred_errors, obs_errors):

    n = 0

    nrows = 5

    ncols = 5

    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True, figsize=(15,15))

    for row in range(nrows):

        for col in range(ncols):

            error = errors_index[n]

            ax[row,col].imshow((img_errors[error]).reshape((28,28)), cmap = 'binary')

            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))

            n += 1





Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)

true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))

delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors

sorted_dela_errors = np.argsort(delta_pred_true_errors)

most_important_errors = sorted_dela_errors[-25:]





display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)
def test_submission(model):

    results = model.predict(test)

    results = np.argmax(results, axis=1)

    results = pd.Series(results, name="Label")

    submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

    submission.to_csv("cnn_mnist_datagen.csv",index=False)
cnn_model_2, _ = train_model(get_cnn_model(), epochs = 30)

test_submission(cnn_model_2)