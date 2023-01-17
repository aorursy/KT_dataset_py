import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        break

print(os.path.join(dirname, filename))



# Any results 
import tensorflow as tf

print(tf.__version__)
import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt





import time

import datetime



from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.optimizers import SGD, Adam, Nadam

from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten

from tensorflow.keras.initializers import VarianceScaling
def date_time(x):

  """Function to get Date-Time formats"""

  if x==1:

    return 'Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())

  if x==2:    

    return 'Timestamp: {:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now())

  if x==3:  

    return 'Date now: %s' % datetime.datetime.now()

  if x==4:  

    return 'Date today: %s' % datetime.date.today() 
def show(original,orig_label,augmented,predicted,classified):

  """ Function to preview and compare all three images """

  fig,(ax1,ax2,ax3) = plt.subplots(1,3,squeeze=True, figsize=(10,10))

  ax1.imshow(original.reshape(28, 28),cmap='gray')

  ax2.imshow(augmented.reshape(28, 28),cmap='gray')

  ax3.imshow(predicted.reshape(28, 28),cmap='gray')

  ax1.set_title("Original \n %s"%orig_label)

  ax2.set_title("Augmented \n Noise")

  ax3.set_title("Predicted \n %s"%classified)

  ax1.axis('off')

  ax2.axis('off')

  ax3.axis('off')
def load_data(path):

    with np.load(path) as f:

        x_train, y_train = f['x_train'], f['y_train']

        x_test, y_test = f['x_test'], f['y_test']

        return (x_train, y_train), (x_test, y_test)



(x_train, y_train), (x_val, y_val) = load_data('../input/mnist-numpy/mnist.npz')
from imgaug import augmenters as iaa

seq = iaa.Sequential(iaa.GaussianBlur(sigma=(0, 3.0)))

#seq = iaa.Sequential([iaa.SaltAndPepper(0.2)])



x_train_aug = seq.augment_images(x_train)

x_val_aug = seq.augment_images(x_val)
x_train = x_train/255.0; x_val = x_val/255.0

x_train = x_train.reshape(-1, 28, 28, 1); x_val = x_val.reshape(-1, 28, 28, 1)



x_train_aug = x_train_aug/255.0; x_val_aug = x_val_aug/255.

x_train_aug = x_train_aug.reshape(-1, 28, 28, 1); x_val_aug = x_val_aug.reshape(-1, 28, 28, 1)
class Autoencoder():

    def __init__(self):

        self.img_rows = 28

        self.img_cols = 28

        self.channels = 1

        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        

        optimizer = Adam(lr=0.001)

        

        self.autoencoder_model = self.build_model()

        self.autoencoder_model.compile(loss='mse', optimizer=optimizer)

        self.autoencoder_model.summary()

    

    def build_model(self):

        input_layer = Input(shape=self.img_shape)

        

        # encoder

        encoded = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)

        encoded = MaxPooling2D((2, 2), padding='same')(encoded)

        encoded = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)

        encoded = MaxPooling2D((2, 2), padding='same')(encoded)

        encoded = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)

        encoded = MaxPooling2D((2, 2), padding='same')(encoded)

        

        # decoder

        decoded = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)

        decoded = UpSampling2D((2, 2))(decoded)

        decoded = Conv2D(32, (3, 3), activation='relu', padding='same')(decoded)

        decoded = UpSampling2D((2, 2))(decoded)

        decoded = Conv2D(64, (3, 3), activation='relu')(decoded)

        decoded = UpSampling2D((2, 2))(decoded)      

        output_layer = Conv2D(1, (3, 3), padding='same')(decoded)

        

        return Model(input_layer, output_layer)

    

    def train_model(self, x_train, y_train, x_val, y_val, epochs, batch_size=20):

        early_stopping = EarlyStopping(monitor='val_loss',

                                       min_delta=0,

                                       patience=10,

                                       verbose=1, 

                                       mode='auto')

        model_checkpoint = ModelCheckpoint('Denoising_autoencoder.h5', 

                                           monitor='val_loss', 

                                           mode='min', 

                                           verbose=1, 

                                           save_best_only=True)

        history = self.autoencoder_model.fit(x_train, y_train,

                                             batch_size=batch_size,

                                             epochs=epochs,

                                             validation_data=(x_val, y_val),

                                             callbacks=[early_stopping, model_checkpoint])

        plt.plot(history.history['loss'])

        plt.plot(history.history['val_loss'])

        plt.title('Model loss')

        plt.ylabel('Loss')

        plt.xlabel('Epoch')

        plt.legend(['Train', 'Test'], loc='upper left')

        plt.show()

    def eval_model(self, x_test):

        preds = self.autoencoder_model.predict(x_test)

        return preds
ae = Autoencoder()

start_time = time.time()

print(date_time(1))



ae.train_model(x_train_aug, x_train, x_val_aug, x_val, epochs=500, batch_size=2048)

elapsed_time = time.time() - start_time

elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))



print("\nElapsed Time: " + elapsed_time)

print("Completed Model Trainning", date_time(1))
#from google.colab import drive

#drive.mount('/content/drive')
# if not trained in this notebook use trained file 

try:

  if ae:

    predictions = ae.eval_model(x_val)

except Exception:

    #from my drive

    autoencoder = tf.keras.models.load_model("Denoising_autoencoder.h5")

    predictions = autoencoder.predict(x_val)

    print("Done")

#except:

#  print("Model not found")
n=0

show(original=x_val[n],orig_label="",augmented=x_val_aug[n],predicted=predictions[n],classified="")
class Classify_AE():

  def __init__(self,model):

    self.model = model

    self.classify_ae = self.build_model()

    self.classify_ae.compile(loss='sparse_categorical_crossentropy',optimizer=Adam(lr=1e-4),metrics=['accuracy'])

    self.classify_ae.summary()

  def build_model(self):

    autoencoder = self.model

    hidden_representation = Sequential()

    hidden_representation.add(autoencoder.layers[0]) 

    hidden_representation.add(autoencoder.layers[1]) 

    hidden_representation.add(autoencoder.layers[2]) 

    hidden_representation.add(autoencoder.layers[3]) 

    hidden_representation.add(Flatten())

    hidden_representation.add(Dense(512, activation='relu'))

    hidden_representation.add(Dense(512, activation='relu'))

    hidden_representation.add(Dense(10, activation='softmax'))

    for layer in hidden_representation.layers[:-3]:

      layer.trainable = False

    return hidden_representation

  def train_model(self, x_train, y_train, x_val, y_val, epochs=20, batch_size=20):

    early_stopping = EarlyStopping(monitor='val_loss',

                                   min_delta=0,

                                   patience=10,

                                   verbose=1, 

                                   mode='auto')

    model_checkpoint = ModelCheckpoint('classification_autoencoder.h5', 

                                        monitor='val_accuracy', 

                                        mode='max', 

                                        verbose=1, 

                                        save_best_only=True)

    history = self.classify_ae.fit(x_train, y_train, shuffle = True,

                                   batch_size=batch_size,

                                   epochs=epochs,

                                   validation_data=(x_val, y_val),

                                   workers=4,

                                   callbacks=[early_stopping, model_checkpoint])

    

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('Model loss')

    plt.ylabel('Loss')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Test'], loc='upper left')

    plt.figure()



    plt.plot(history.history['accuracy'])

    plt.plot(history.history['val_accuracy'])

    plt.title('Model Accuracy')

    plt.ylabel('Accuracy')

    plt.xlabel('Epochs')

    plt.legend(['Train','Test'],loc='upper right')

    plt.show()

  def eval_model(self, x_test):

    preds = self.classify_ae.predict(x_test)

    return preds
model = tf.keras.models.load_model("Denoising_autoencoder.h5")

cae = Classify_AE(model)

start_time = time.time()



print(date_time(1))

cae.train_model(x_train, y_train, x_val, y_val, epochs=30, batch_size=2048)

elapsed_time = time.time() - start_time

elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))



print("\nElapsed Time: " + elapsed_time)

print("Completed Model Trainning", date_time(1))
# if not trained in this notebook use trained file 

try:

  if cae:

    predicts = cae.eval_model(x_val)

    p_predicts = cae.eval_model(predictions)

except Exception:

    #from my drive

    Classify_autoencoder = tf.keras.models.load_model("classification_autoencoder.h5")

    predicts = Classify_autoencoder.predict(x_val)

    p_predicts = Classify_autoencoder.predict(predictions)

    print("Done")

#except:

#  print("Model not found")
#if predicts:

val_pred = [np.argmax(value) for value in predicts]

p_pred = [np.argmax(value) for value in p_predicts] 
n=0

show(original=x_val[n],orig_label=y_val[n],augmented=x_val_aug[n],predicted=predictions[n],classified=p_pred[n])
n=5

show(original=x_val[n],orig_label=y_val[n],augmented=x_val_aug[n],predicted=predictions[n],classified=p_pred[n])
from sklearn.metrics import classification_report, confusion_matrix
test_df = pd.DataFrame({'True label': y_val,'Predicted Label': val_pred})

test_df.to_csv('submission.csv', header=True, index=False)

test_df
import itertools

def plot_confusion_matrix(cm, classes,                         

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    

    ticksize = 18

    titlesize = ticksize + 8

    labelsize = ticksize + 5



    figsize = (18, 5)

    params = {'figure.figsize' : figsize,

              'axes.labelsize' : labelsize,

              'axes.titlesize' : titlesize,

              'xtick.labelsize': ticksize,

              'ytick.labelsize': ticksize}



    plt.rcParams.update(params)

    

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    

    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    

    plt.show()
# compute the confusion matrix

confusion_mtx = confusion_matrix(y_val, val_pred) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10))
LABELS=[str(x) for x in range(10)]

print(classification_report(y_val, 

                            val_pred, 

                            target_names = LABELS))
from IPython.display import FileLinks

FileLinks('.')