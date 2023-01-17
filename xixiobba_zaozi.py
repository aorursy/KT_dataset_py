import tensorflow as tf

import os

import numpy as np

import matplotlib.pyplot as plt
!pip install keras_efficientnets
tf.__version__
!pip install keras.applications
from keras_efficientnets import EfficientNetB3
base_dir = os.path.join(os.path.dirname('../input/newzaozi/'), 'zaozi')

print (base_dir)
IMAGE_SIZE = (328,552)

BATCH_SIZE = 16

train_dir = base_dir+'/train'

val_dir = base_dir+'/val'



datagen = tf.keras.preprocessing.image.ImageDataGenerator(

    rescale=1./255,

    rotation_range=20)



train_generator = datagen.flow_from_directory(

    train_dir,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE)



val_generator = datagen.flow_from_directory(

    val_dir,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE)
for image_batch, label_batch in train_generator:

  break

image_batch.shape, label_batch.shape
print (train_generator.class_indices)



labels = '\n'.join(sorted(train_generator.class_indices.keys()))



with open('labels.txt', 'w') as f:

  f.write(labels)
!cat labels.txt
IMG_SHAPE = IMAGE_SIZE + (3,)



# Create the base model from the pre-trained model MobileNet V2

base_model = EfficientNetB3(input_shape=IMG_SHAPE,include_top=False, weights='imagenet')
base_model.trainable = False
model = tf.keras.Sequential([

  base_model,

  tf.keras.layers.Conv2D(64, 3, activation='tanh'),

  tf.keras.layers.Dropout(0.2),

  tf.keras.layers.GlobalAveragePooling2D(),

  tf.keras.layers.Dense(3, activation='softmax',kernel_regularizer='l2')

])
model.compile(optimizer=tf.keras.optimizers.Adam(), 

              loss='categorical_crossentropy', 

              metrics=['accuracy'])
epochs = 6



history = model.fit(train_generator, 

                    steps_per_epoch=len(train_generator), 

                    epochs=epochs, 

                    validation_data=val_generator, 

                    validation_steps=len(val_generator))
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



loss = history.history['loss']

val_loss = history.history['val_loss']



plt.figure(figsize=(8, 8))

plt.subplot(2, 1, 1)

plt.plot(acc, label='Training Accuracy')

plt.plot(val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.ylabel('Accuracy')

plt.ylim([min(plt.ylim()),1])

plt.title('Training and Validation Accuracy')



plt.subplot(2, 1, 2)

plt.plot(loss, label='Training Loss')

plt.plot(val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.ylabel('Cross Entropy')

plt.ylim([0,1.0])

plt.title('Training and Validation Loss')

plt.xlabel('epoch')

plt.show()
base_model.trainable = True
# Let's take a look to see how many layers are in the base model

print("Number of layers in the base model: ", len(base_model.layers))
# # Fine tune from this layer onwards

# fine_tune_at = 100



# # Freeze all the layers before the `fine_tune_at` layer

# for layer in base_model.layers[:fine_tune_at]:

#   layer.trainable =  False
model.compile(loss='categorical_crossentropy',

              optimizer = tf.keras.optimizers.Adam(1e-3),

              metrics=['accuracy'])
history_fine = model.fit(train_generator, 

                         steps_per_epoch=len(train_generator), 

                         epochs=20, 

                         validation_data=val_generator, 

                         validation_steps=len(val_generator))
acc = history_fine.history['accuracy']

val_acc = history_fine.history['val_accuracy']



loss = history_fine.history['loss']

val_loss = history_fine.history['val_loss']



plt.figure(figsize=(8, 8))

plt.subplot(2, 1, 1)

plt.plot(acc, label='Training Accuracy')

plt.plot(val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.ylabel('Accuracy')

plt.ylim([min(plt.ylim()),1])

plt.title('Training and Validation Accuracy')



plt.subplot(2, 1, 2)

plt.plot(loss, label='Training Loss')

plt.plot(val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.ylabel('Cross Entropy')

plt.ylim([0,1.0])

plt.title('Training and Validation Loss')

plt.xlabel('epoch')

plt.show()
saved_model_dir = '../output/kaggle/working/'

model.save('my_model.h5', saved_model_dir)
# saved_model_dir = '../output/kaggle/working/'

# tf.saved_model.save(model, saved_model_dir)



# converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# tflite_model = converter.convert()



# with open('model.tflite', 'wb') as f:

#   f.write(tflite_model)
from keras.models import load_model
model_tf = load_model('./my_model.h5')
#model_tf.summary()
from keras.preprocessing import image

import PIL
label = ['kong','lanzao','liekou']

file_path = '../input/newzaozi/zaozi/val/liekou/shoot_07_22_09_15_48_343_2.bmp'

PIL.Image.open(file_path)
img = image.load_img(file_path, target_size=(328, 552))

x = image.img_to_array(img)/255

x = np.expand_dims(x, axis=0)

x.shape

pred = label[np.argmax(model_tf.predict(x))]
pred