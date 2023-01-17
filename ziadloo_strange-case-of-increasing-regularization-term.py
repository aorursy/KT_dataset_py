import os
import shutil
from random import shuffle

# make sure there's no dataset folder from previous executions
if os.path.exists('./dataset'):
    shutil.rmtree('./dataset')

# create folder structure
if not os.path.exists('./dataset'):
    os.makedirs('./dataset')

if not os.path.exists('./dataset/train'):
    os.makedirs('./dataset/train')
if not os.path.exists('./dataset/train/dog'):
    os.makedirs('./dataset/train/dog')
if not os.path.exists('./dataset/train/cat'):
    os.makedirs('./dataset/train/cat')

if not os.path.exists('./dataset/validation'):
    os.makedirs('./dataset/validation')
if not os.path.exists('./dataset/validation/dog'):
    os.makedirs('./dataset/validation/dog')
if not os.path.exists('./dataset/validation/cat'):
    os.makedirs('./dataset/validation/cat')

if not os.path.exists('./dataset/test'):
    os.makedirs('./dataset/test')
if not os.path.exists('./dataset/test/unlabeled'):
    os.makedirs('./dataset/test/unlabeled')

# select indices as for training and validation in random
class_size = 12500
cat_indices = list(range(class_size))
dog_indices = list(range(class_size))

validation_to_train_ratio = 1. / 5.

shuffle(cat_indices)
shuffle(dog_indices)

cat_train_indices = cat_indices[int(validation_to_train_ratio * class_size) : ]
dog_train_indices = dog_indices[int(validation_to_train_ratio * class_size) : ]
cat_validation_indices = cat_indices[0 : int(validation_to_train_ratio * class_size)]
dog_validation_indices = dog_indices[0 : int(validation_to_train_ratio * class_size)]

for index in cat_train_indices:
    src = '../input/train/cat.%d.jpg' % index
    dst = './dataset/train/cat/cat.%s.jpg' % str(index).zfill(5)
    shutil.copyfile(src, dst)

for index in dog_train_indices:
    src = '../input/train/dog.%d.jpg' % index
    dst = './dataset/train/dog/dog.%s.jpg' % str(index).zfill(5)
    shutil.copyfile(src, dst)

for index in cat_validation_indices:
    src = '../input/train/cat.%d.jpg' % index
    dst = './dataset/validation/cat/cat.%s.jpg' % str(index).zfill(5)
    shutil.copyfile(src, dst)

for index in dog_validation_indices:
    src = '../input/train/dog.%d.jpg' % index
    dst = './dataset/validation/dog/dog.%s.jpg' % str(index).zfill(5)
    shutil.copyfile(src, dst)

for index in range(12500):
    src = '../input/test/%d.jpg' % (index + 1)
    dst = './dataset/test/unlabeled/%s.jpg' % str(index).zfill(5)
    shutil.copyfile(src, dst)
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory('./dataset/train',
                                                    target_size=(150, 150),
                                                    batch_size=100,
                                                    shuffle = True,
                                                    class_mode='binary')
validation_generator = test_datagen.flow_from_directory('./dataset/validation',
                                                       target_size=(150, 150),
                                                       batch_size=100,
                                                       shuffle = False,
                                                       class_mode='binary')
test_generator = test_datagen.flow_from_directory('./dataset/test',
                                                  target_size=(150, 150),
                                                  batch_size=100,
                                                  shuffle = False,
                                                  class_mode='binary')
from keras import models
from keras import layers
from keras import optimizers
from keras.applications import VGG16
from keras import regularizers
import keras.backend as K

conv_base = VGG16(weights = 'imagenet',
             include_top = False,
             input_shape = (150, 150, 3))
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l1_l2(0.02)))
model.add(layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(0.01)))

conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

def pure_loss(y_true, y_pred):
    return K.binary_crossentropy(y_true, y_pred)

model.compile(loss='binary_crossentropy', optimizer=optimizers.Adadelta(), metrics=['acc', pure_loss])

history = model.fit_generator(train_generator,
                              steps_per_epoch = 50,
                              epochs = 100,
                              validation_data = validation_generator,
                              validation_steps = 50)
prediction = model.predict_generator(test_generator)
print(prediction)
import csv

csvData = [['id', 'label']]
for i, j in enumerate(prediction):
    csvData.append([i + 1, j[0]])
csvFile = open('./submission.csv', 'w')
with csvFile:
   writer = csv.writer(csvFile)
   writer.writerows(csvData)

# delete the dataset folder
if os.path.exists('./dataset'):
    shutil.rmtree('./dataset')
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
pure_loss = history.history['pure_loss']
val_pure_loss = history.history['val_pure_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize = (12, 8))
plt.plot(epochs, acc, 'ro', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure(figsize = (12, 8))
plt.plot(epochs, loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.figure(figsize = (12, 8))
plt.plot(epochs, pure_loss, 'ro', label='Training pure loss')
plt.plot(epochs, val_pure_loss, 'b', label='Validation pure loss')
plt.title('Training and validation pure loss')
plt.legend()

plt.show()