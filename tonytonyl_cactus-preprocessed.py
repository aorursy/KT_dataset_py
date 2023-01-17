%matplotlib inline



import pandas as pd

import numpy as np

import pylab as plt

import seaborn as sns; sns.set()



import random # Utile pour creer un jeu de validation

random.seed(0)



import tensorflow as tf

from tensorflow import keras



import sys
print("python", sys.version)

for module in np, pd, tf, keras:

    print(module.__name__, module.__version__)
assert sys.version_info >= (3, 5) # Python ≥3.5 required

assert tf.__version__ >= "2.0"    # TensorFlow ≥2.0 required
train_dir = "../input/cactus-dataset/cactus/train.csv"

train_data = pd.read_csv(train_dir)

train_data.has_cactus = train_data.has_cactus.astype(str) # Pour appliquer le preprocessing il faut transformer les variables has_cactus en str
train_data.has_cactus.value_counts()
print(train_data.shape)
import matplotlib.pyplot as mlp

import matplotlib.image as mpimg



img = mpimg.imread("../input/cactus-dataset/cactus/train/000c8a36845c0208e833c79c1bffedd1.jpg")

plt.axis("off")

imgplot = mlp.imshow(img)
#nb_train = np.random.rand(len(train_data)) < 0.8

#train_set = train_data[nb_train]

#valid_set = train_data[~nb_train]



#datagen = keras.preprocessing.image.ImageDataGenerator(

#    rescale=1./255,

#    shear_range = 0.2,

#    zoom_range = 0.2,

#    horizontal_flip = True,

#    vertical_flip = True,

#    preprocessing_function = keras.applications.xception.preprocess_input)



#train_generator = datagen.flow_from_dataframe(

#    dataframe = train_set,

#    directory = train_dir,

#    x_col = 'id',

#    y_col = 'has_cactus',

#    target_size = (32,32),

#    batch_size = 64,

#    class_mode = 'binary')



#valid_generator = datagen.flow_from_dataframe(

#    dataframe = valid_set,

#    directory = train_dir,

#    x_col = 'id',

#    y_col = 'has_cactus',

#    target_size = (32,32),

#    batch_size = 64,

#    class_mode = 'binary')
from tensorflow.keras.preprocessing.image import ImageDataGenerator



def generator(train_data, directory, batch_size, target_size, class_mode):

    

    x_col = 'id'

    y_col = 'has_cactus'

    

    train_datagen = ImageDataGenerator(

        rescale = 1./255, 

        horizontal_flip = True, 

        vertical_flip = True, 

        validation_split = 0.2)



    train_generator = train_datagen.flow_from_dataframe(

        train_data, 

        directory = directory, 

        x_col = x_col, 

        y_col = y_col, 

        target_size = target_size, 

        class_mode = class_mode, 

        batch_size = batch_size, 

        shuffle = True, 

        subset = 'training')



    valid_generator = train_datagen.flow_from_dataframe(

        train_data, 

        directory = directory, 

        x_col = x_col, 

        y_col = y_col, 

        target_size = target_size, 

        class_mode = class_mode, 

        batch_size = batch_size, 

        shuffle = True, 

        subset = 'validation')

    

    return train_generator, valid_generator
directory = "../input/cactus-dataset/cactus/train"

batch_size = 64

target_size = (32,32) # On a des images 32x32

class_mode = 'binary' # Binary puisque qu'on a un vecteur contenant des 0 ou des 1



train_generator, valid_generator = generator(train_data, directory, batch_size, target_size, class_mode)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, Activation

from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, GlobalAveragePooling2D
model = Sequential([

    Conv2D(32, (3, 3), padding = 'same', activation = 'relu', input_shape = (32,32,3)),

    BatchNormalization(),

    Conv2D(32, (3, 3), padding = 'same', activation = 'relu', input_shape = (32,32,3)),

    BatchNormalization(),

    MaxPooling2D(),

    

    Conv2D(64, (3, 3), padding = 'same', activation = 'relu'),

    BatchNormalization(),

    MaxPooling2D(),

    

    Conv2D(128, (3, 3), padding = 'same', activation = 'relu'),

    BatchNormalization(),

    MaxPooling2D(),

    

    Conv2D(256, (3, 3), padding = 'same', activation = 'relu'),

    BatchNormalization(),

    MaxPooling2D(),

    

    GlobalAveragePooling2D(),

    

    Dense(256, activation = 'relu'),

    Dropout(0.5),

    

    Dense(1, activation = 'sigmoid')

]) # padding = 'same' permet de ne pas diminuer la taille des images

   # sigmoid au lieu de softmax puisqu'il est utilisé pour une regression logistique a 2 classes (classification), de plus la somme des proba ne doit pas etre égale a 1



model.compile(loss = 'binary_crossentropy',

              optimizer = 'adam',

              metrics = ['accuracy']) # On utilise une crossentropy binaire et non pas une sparse_categorical_crossentropy

                                      # On utilise l'optimizer Adam, plus rapide que SGD : en ayant effectuer plusieurs test on peut supposer qu'il converge correctement

model.summary()
#model = keras.models.Sequential([

    

#    keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[32, 32, 3]),

#    keras.layers.BatchNormalization(),

#    keras.layers.SeparableConv2D(filters=32, kernel_size=3, padding="same", activation="relu"),

#    keras.layers.BatchNormalization(),

#    keras.layers.MaxPool2D(pool_size=2),

#    keras.layers.Dropout(rate=0.4),

    

#    keras.layers.SeparableConv2D(filters=64, kernel_size=3, padding="same", activation="relu"),

#    keras.layers.BatchNormalization(),

#    keras.layers.SeparableConv2D(filters=64, kernel_size=3, padding="same", activation="relu"),

#    keras.layers.BatchNormalization(),

#    keras.layers.MaxPool2D(pool_size=2),

#    keras.layers.Dropout(rate=0.4),

    

#    keras.layers.Flatten(),

#    keras.layers.Dense(128, activation="relu"),

#    keras.layers.Dense(1, activation="sigmoid")

#])



#model.compile(optimizer= keras.optimizers.SGD(lr=1e-4, momentum=0.9), loss='binary_crossentropy', 

#              metrics=['accuracy'])



#epochs = 40

#history = model.fit_generator(train_generator,

#          validation_data=valid_generator,

#          epochs=epochs,

#          callbacks = [EarlyStopping(patience=10)])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.utils import class_weight



class_weights = class_weight.compute_class_weight('balanced', np.unique(train_generator.classes), train_generator.classes)



callbacks = [EarlyStopping(monitor = 'val_loss', patience = 20),

             ReduceLROnPlateau(patience = 10, verbose = 1),

             ModelCheckpoint(filepath = 'best_model.h5', monitor = 'val_loss', verbose = 0, save_best_only = True)]



history = model.fit_generator(train_generator,

          validation_data = valid_generator,

          epochs = 100,

          verbose = 1,

          shuffle = True,

          callbacks = callbacks,

          class_weight = class_weights)
pd.DataFrame(history.history).plot()

plt.show()
model.load_weights("best_model.h5")
def test_gen(test_dir, target_size, batch_size, class_mode):

    test_datagen = ImageDataGenerator(

        rescale = 1./255)



    test_generator = test_datagen.flow_from_directory(

        directory = test_dir,

        target_size = target_size, 

        batch_size = batch_size,

        class_mode = class_mode,

        shuffle = False)  

    return test_generator
test_dir = "../input//cactus-dataset/cactus/test/"

target_size = (32,32)

batch_size = 1

class_mode = None



test_generator = test_gen(test_dir, target_size, batch_size, class_mode)
#pred = model.predict_generator(test_generator,verbose=1)

#pred_binary = [0 if value < 0.50 else 1 for value in pred]  
def submission():

    sample_submission = pd.read_csv("../input/cactus-dataset/cactus/sample_submission.csv")



    filenames = [path.split('/')[-1] for path in test_generator.filenames] # On récupere les noms des images pour en faire une colonne sur notre csv final

    proba = list(model.predict_generator(test_generator)[:,0]) # On recupere les probabilités prédites par notre modele sur le jeu test (sur lequel on a fait de la data augmentation)



    sample_submission.id = filenames

    sample_submission.has_cactus = proba



    sample_submission.to_csv('submission.csv', index=False)

    return sample_submission



sample_submission = submission()

sample_submission.head()