import matplotlib.pyplot as plt
import numpy as np
from keras import optimizers, regularizers
from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import InceptionV3
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
data_dir = '../input/art-movements/dataset/dataset/'
RESOLUTION = 150
BATCH_SIZE = 64

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.3)

train_generator = train_datagen.flow_from_directory(
        data_dir + 'train/',
        target_size=(RESOLUTION, RESOLUTION),
        batch_size=BATCH_SIZE,
        class_mode='categorical', subset="training")

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.3)

val_generator = val_datagen.flow_from_directory(
        data_dir + 'train/',
        target_size=(RESOLUTION, RESOLUTION),
        batch_size=BATCH_SIZE,
        class_mode='categorical', subset="validation")

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        data_dir + 'test/',
        target_size=(RESOLUTION, RESOLUTION),
        batch_size=BATCH_SIZE,
        class_mode='categorical')
# Class labels
train_generator.class_indices
## Imagenet InceptionV3 weights
weights_v3_path = '../input/keras-pretrained-models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
conv_base = InceptionV3(include_top=False, input_shape=(150, 150, 3), weights=weights_v3_path)

## Freezing first 32. layers (Edges and basic shapes)
for layer in conv_base.layers[:32]:
    layer.treinable = False

model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
# adjust here to 3 class
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-4), metrics=['acc'])
model.summary()
N_TRAIN = 540
N_VAL = 228
history = model.fit_generator(
    train_generator,
    steps_per_epoch=(N_TRAIN // BATCH_SIZE),
    epochs=40,
    validation_data=val_generator,
    validation_steps=(N_VAL // BATCH_SIZE)
)
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend(loc='best')

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc='best')
# saving first model
models = []
models.append(model)
conv_base = InceptionV3(include_top=False, input_shape=(150, 150, 3), weights=None)

## Freezing first 32. layers (Edges and basic shapes)
for layer in conv_base.layers[:32]:
    layer.treinable = False

model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
# adjust here to 3 class
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-4), metrics=['acc'])
N_TRAIN = 540
N_VAL = 228
history = model.fit_generator(
    train_generator,
    steps_per_epoch=(N_TRAIN // BATCH_SIZE),
    epochs=40,
    validation_data=val_generator,
    validation_steps=(N_VAL // BATCH_SIZE)
)
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend(loc='best')

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc='best')
models.append(model)
loss, acc = models[0].evaluate_generator(test_generator, steps=len(test_generator))
print('transfer learning model - loss:{:.2f} - acc:{:.2f}'.format(loss, acc))
loss, acc = models[1].evaluate_generator(test_generator, steps=len(test_generator))
print('without transfer learning model - loss:{:.2f} - acc:{:.2f}'.format(loss, acc))

# let's use the first model, the accuracy is better.
model = models[0]
model.save('inception_v3_art.h5')
# using first model, the accuracy is better.
model = models[0]
model.pop()
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-4), metrics=['acc'])

model.summary()
Y_pred = model.predict_generator(test_generator, steps=len(test_generator))
np.save('features_inception_v3.npy', Y_pred)
# for name, arr_ in zip(test_generator.filenames, Y_pred):
#     name = name.split('/')[1]
#     name = name.split('.')[0]
#     np.savetxt('{}'.format(name), arr_)
