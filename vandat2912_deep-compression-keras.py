!pip install kerassurgeon

!pip install -q tensorflow-model-optimization
import numpy as np

import pandas

import warnings

warnings.filterwarnings("ignore")



from keras.datasets import mnist

import keras.backend as K

from keras.layers import (

    Conv2D, MaxPool2D, BatchNormalization, Dense, Input, Lambda, Activation, Flatten, Dropout

)

from keras.models import Model

from keras.regularizers import l2
import cv2
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

from sklearn.utils import shuffle
import tensorflow as tf
BATCH_SIZE = 64

IMG_HEIGHT = 28

IMG_WIDTH = 28

IMG_CHANNEL = 1
K.image_data_format()
class ALexNet:

    def __init__(self, img_width, img_height, img_channel, reg=1e-4):

        self.img_height = img_height

        self.img_width = img_width

        self.img_channel = img_channel

        self.reg = reg

        self.channel_axis = -1

        self.data_format = "channels_last"

        

        if K.image_data_format() == "channels_first":

            self.channel_axis = 0

            self.data_format = "channels_first"

#         super(self, ALexNet).__init__(img_width, img_height, img_channel)

        

    def build(self):

        input_shape = (self.img_height, self.img_width, self.img_channel)

        inputs = Input(shape=input_shape, name="input")

        

        x = Conv2D(96, kernel_size=(11, 11), strides=(4, 4), padding="same", kernel_regularizer=l2(self.reg))(inputs)

        x = BatchNormalization(axis=self.channel_axis)(x)

        x = Activation("relu")(x)

        x = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)

        

        x = Conv2D(256, kernel_size=(5, 5), strides=(1, 1), padding="same", kernel_regularizer=l2(self.reg))(x)

        x = BatchNormalization(axis=self.channel_axis)(x)

        x = Activation("relu")(x)

        x = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)

        

        x = Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_regularizer=l2(self.reg))(x)

        x = BatchNormalization(axis=self.channel_axis)(x)

        x = Activation("relu")(x)

    

        x = Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_regularizer=l2(self.reg))(x)

        x = BatchNormalization(axis=self.channel_axis)(x)

        x = Activation("relu")(x)

        

        x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_regularizer=l2(self.reg))(x)

        x = BatchNormalization(axis=self.channel_axis)(x)

        x = Activation("relu")(x)

        

        x = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_regularizer=l2(self.reg))(x)

        x = BatchNormalization(axis=self.channel_axis)(x)

        x = Activation("relu")(x)

        

        x = Flatten()(x)

        x = Dense(9216, kernel_regularizer=l2(self.reg))(x)

        x = Activation("relu")(x)

        x = Dropout(0.3)(x)

        x = Dense(4096, kernel_regularizer=l2(self.reg))(x)

        x = Activation("relu")(x)

        x = Dropout(0.3)(x)

        x = Dense(4096, kernel_regularizer=l2(self.reg))(x)

        x = Activation("relu")(x)

        x = Dropout(0.3)(x)

        x = Dense(1000, kernel_regularizer=l2(self.reg))(x)

        x = Activation("relu")(x)

        x = Dropout(0.3)(x)

        x = Dense(10)(x)

        x = Activation("softmax")(x)

        

        model = Model(inputs=inputs, outputs=x)

        

        return model
alexnet = ALexNet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL)
model = alexnet.build()
model.summary()
(X_train, y_train), (X_test, y_test) = mnist.load_data()
y_train.shape
y_train
lbl = LabelBinarizer()
lbl.fit(y_train)
y_train = lbl.transform(y_train)
y_test = lbl.transform(y_test)
lbl.classes_
X_train, X_test = np.expand_dims(X_train, axis=-1), np.expand_dims(X_test, axis=-1)
class DataGeneration:

    def __init__(self,

                 X_train, y_train,

                 X_test, y_test,

                 img_height=IMG_HEIGHT, 

                 img_width=IMG_WIDTH, 

                 img_channel=IMG_CHANNEL,

                 batch_size=BATCH_SIZE):

        self.img_height = img_height

        self.img_width = img_width

        self.img_channel = img_channel

        self.batch_size = batch_size

        self.X_train = X_train

        self.y_train = y_train

        self.X_test = X_test

        self.y_test = y_test

        self.current_train = 0

        self.current_test = 0

    

    def load_batch(self, task="train"):

        self.current_index = self.current_train

        self.max_length = self.X_train.shape[0]

        self.X = self.X_train

        self.y = self.y_train

        

        if task != "train":

            self.current_index = self.current_test

            self.max_length = self.X_test.shape[0]

            self.X = self.X_test

            self.y = self.y_test

        

        if (self.current_index + self.batch_size > self.max_length):

            self.current_index = 0

            

            if task != "train":

                self.current_test = 0

            else:

                self.current_train = 0

            

            self.X, self.y = shuffle(self.X, self.y, random_state=42)

        

        batch_X = self.X[self.current_index:self.current_index+self.batch_size]

        batch_y = self.y[self.current_index:self.current_index+self.batch_size]

        

        if task != "train":

            self.current_test += self.batch_size

        else:

            self.current_train += self.batch_size

        

#         batch_X = batch_X /255

        return batch_X, batch_y

    

    def generator(self, task="train"):

        while True:

            batch_X, batch_y = self.load_batch(task=task)

            

            yield (batch_X, batch_y)
data_generator = DataGeneration(X_train, y_train, X_test, y_test)
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import os
opt = Adam(learning_rate=1e-3, decay=1e-4)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
checkpoint_path = "trained_weight/"



if not os.path.exists(checkpoint_path):

    os.mkdir(checkpoint_path)
file_path = os.path.join(checkpoint_path, "best_weights.h5")



checkpoint = ModelCheckpoint(filepath=file_path, monitor="val_loss", save_weights_only=True, save_best_only=True, verbose=0)
def learning_rate_schedule(epoch):

    init_lr = 1e-3

    max_epoch = 100

    poly = 1

    

    return float(init_lr*(1 - epoch/max_epoch)**poly)
# model.load_weights("trained_weight/best_weights.h5")
model.fit_generator(data_generator.generator(),

                    steps_per_epoch=data_generator.X_train.shape[0] // data_generator.batch_size,

                    validation_data=data_generator.generator(task="val"),

                    validation_steps=data_generator.X_test.shape[0] // data_generator.batch_size,

                    epochs=10,

                    initial_epoch=0,

                    verbose=1, 

                    callbacks=[checkpoint, LearningRateScheduler(learning_rate_schedule)]

                   )
import tensorflow as tf
model.evaluate(X_test, y_test)
for layer in model.layers:

    print(layer.name)
model.get_layer("dropout_1").get_config()
import tensorflow_model_optimization as tfmot

import tempfile
model = alexnet.build()
model.load_weights("trained_weight/best_weights.h5")
def apply_pruning(layer):

    num_epochs = 100

    end_step = np.ceil(X_train.shape[0] / BATCH_SIZE).astype(np.int32)*num_epochs

    

    pruning_params = {

        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,

                                                                 final_sparsity=0.80,

                                                                 begin_step=0,

                                                                 end_step=end_step),

        "pool_size": (1, 1),

        "block_pooling_type": "AVG",

    }

    

    if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):

        if layer.name not in ["conv2d_1", "conv2d_2"]:

            return tfmot.sparsity.keras.prune_low_magnitude(layer)

    

    return layer
# num_epochs = 100

# end_step = np.ceil(X_train.shape[0] / BATCH_SIZE).astype(np.int32)*num_epochs



# pruning_params = {

#     'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,

#                                                              final_sparsity=0.80,

#                                                              begin_step=0,

#                                                              end_step=end_step),

#     "pool_size": (1, 1),

#     "block_pooling_type": "AVG",

# }
# model_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)



model_pruning = tf.keras.models.clone_model(

    model,

    clone_function=apply_pruning

)
model_pruning.summary()
logdir = tempfile.mkdtemp()



# checkpoint = ModelCheckpoint(os.path.join(checkpoint_path, "best_weights_pruning.h5"), 

#                              monitor="val_loss", 

#                              save_best_only=True, 

#                              save_weights_only=True, 

#                              verbose=1)
callbacks = [

    tfmot.sparsity.keras.UpdatePruningStep(),

    tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),

#     checkpoint

]
new_opt = Adam(learning_rate=1e-3/100, decay=1e-4)
model_pruning.compile(optimizer=new_opt, 

                      loss=tf.keras.losses.categorical_crossentropy,

                      metrics=["accuracy"])
model_pruning.fit_generator(data_generator.generator(),

                            steps_per_epoch=data_generator.X_train.shape[0] // data_generator.batch_size,

                            validation_data=data_generator.generator(task="val"),

                            validation_steps=data_generator.X_test.shape[0] // data_generator.batch_size,

                            epochs=10,

                            initial_epoch=0,

                            verbose=1, 

                            callbacks=callbacks,

                           )
# model_pruning.summary()
model_for_export = tfmot.sparsity.keras.strip_pruning(model_pruning)
_, pruned_keras_file = tempfile.mkstemp(".h5")

tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)
# print('Saved pruned Keras model to:', pruned_keras_file)
converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)

pruned_tflite_model = converter.convert()



_, pruned_tflite_file = tempfile.mkstemp('.tflite')
with open(pruned_tflite_file, 'wb') as f:

    f.write(pruned_tflite_model)



print('Saved pruned TFLite model to:', pruned_tflite_file)
def get_gzipped_model_size(file):

    import os

    import zipfile



    _, zipped_file = tempfile.mkstemp('.zip')

    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:

        f.write(file)



    return os.path.getsize(zipped_file)
print("Size of Keras model: %.2f bytes" % (get_gzipped_model_size("trained_weight/best_weights.h5")))

print("Size of pruned Keras model: %.2f bytes" % (get_gzipped_model_size(pruned_keras_file)))

print("Size of pruned TFlite model: %.2f bytes" % (get_gzipped_model_size(pruned_tflite_file)))
converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_and_pruned_tflite_model = converter.convert()
_, quantized_and_pruned_tflite_file = tempfile.mkstemp(".tflite")
with open(quantized_and_pruned_tflite_file, "wb") as f:

    f.write(quantized_and_pruned_tflite_model)
# print('Saved quantized and pruned TFLite model to:', quantized_and_pruned_tflite_file)

print("Size of baseline Keras model: %.2f bytes" % (get_gzipped_model_size("trained_weight/best_weights.h5")))

print("Size of pruned and quantized TFlite model: %.2f bytes" % (get_gzipped_model_size(quantized_and_pruned_tflite_file)))