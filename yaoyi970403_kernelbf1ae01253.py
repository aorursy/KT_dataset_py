from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import layers, models, Model, Sequential

import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping,ModelCheckpoint

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense

import tensorflow as tf

import json

import os
im_height = 224

im_width = 224

batch_size = 256

epochs = 10
# create direction for saving weights

if not os.path.exists("save_weights"):

    os.makedirs("save_weights")
image_path = "../input/104-flowers-garden-of-eden/jpeg-224x224/"

train_dir = image_path + "train"

validation_dir = image_path + "val"



train_image_generator = ImageDataGenerator( rescale=1./255, 

                                            rotation_range=40, 

                                            width_shift_range=0.2,

                                            height_shift_range=0.2, 

                                           shear_range=0.2,

                                            zoom_range=0.2,

                                            horizontal_flip=True, 

                                            fill_mode='nearest')



train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,

                                                           batch_size=batch_size,

                                                           shuffle=True,

                                                           target_size=(im_height, im_width),

                                                           class_mode='categorical')

    

total_train = train_data_gen.n





validation_image_generator = ImageDataGenerator(rescale=1./255)



val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,

                                                              batch_size=batch_size,

                                                              shuffle=False,

                                                              target_size=(im_height, im_width),

                                                              class_mode='categorical')

    

total_val = val_data_gen.n
covn_base = tf.keras.applications.DenseNet121(weights='imagenet',include_top=False,input_shape=(224,224,3))

covn_base.trainable = True



print(len(covn_base.layers))



for layers in covn_base.layers[:-50]:

    layers.trainable = False

    

    

model = tf.keras.Sequential()

model.add(covn_base)

model.add(tf.keras.layers.GlobalAveragePooling2D())

model.add(tf.keras.layers.Dense(104,activation='softmax'))

model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),

              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),

              metrics=["accuracy"])
reduce_lr = ReduceLROnPlateau(

                                monitor='val_loss', 

                                factor=0.1, 

                                patience=2, 

                                mode='auto',

                                verbose=1

                             )



checkpoint = ModelCheckpoint(

                                filepath='./save_weights/myNASNetMobile.ckpt',

                                monitor='val_acc', 

                                save_weights_only=False, 

                                save_best_only=True, 

                                mode='auto',

                                period=1

                            )



history = model.fit(x=train_data_gen,

                    steps_per_epoch=total_train // batch_size,

                    epochs=epochs,

                    validation_data=val_data_gen,

                    validation_steps=total_val // batch_size,

                    callbacks=[checkpoint, reduce_lr])
model.save_weights('./save_weights/myNASNetMobile.ckpt',save_format='tf')
# plot loss and accuracy image

history_dict = history.history

train_loss = history_dict["loss"]

train_accuracy = history_dict["accuracy"]

val_loss = history_dict["val_loss"]

val_accuracy = history_dict["val_accuracy"]



# figure 1

plt.figure()

plt.plot(range(epochs), train_loss, label='train_loss')

plt.plot(range(epochs), val_loss, label='val_loss')

plt.legend()

plt.xlabel('epochs')

plt.ylabel('loss')



# figure 2

plt.figure()

plt.plot(range(epochs), train_accuracy, label='train_accuracy')

plt.plot(range(epochs), val_accuracy, label='val_accuracy')

plt.legend()

plt.xlabel('epochs')

plt.ylabel('accuracy')

plt.show()
class_indices = train_data_gen.class_indices

inverse_dict = dict((val, key) for key, val in class_indices.items())
from PIL import Image

import numpy as np

# load image

img = Image.open("../input/104-flowers-garden-of-eden/jpeg-224x224/test/003882deb.jpeg")

# resize image to 224x224

img = img.resize((im_width, im_height))
# scaling pixel value to (0-1)

img1 = np.array(img) / 255.



# Add the image to a batch where it's the only member.

img1 = (np.expand_dims(img1, 0))



result = np.squeeze(model.predict(img1))

predict_class = np.argmax(result)

print(inverse_dict[int(predict_class)],result[predict_class])

plt.title([inverse_dict[int(predict_class)],result[predict_class]])

plt.imshow(img)