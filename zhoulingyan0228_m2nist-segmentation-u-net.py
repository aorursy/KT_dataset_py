import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import keras
import sklearn
import sklearn.preprocessing
from sklearn.model_selection import train_test_split

segmented = np.load("../input/segmented.npy")
_, HEIGHT, WIDTH, N_CLASSES = segmented.shape
combined = np.load("../input/combined.npy").reshape((-1, HEIGHT, WIDTH, 1))/255
inputs=keras.layers.Input((HEIGHT, WIDTH,1))
x=keras.layers.ZeroPadding2D(((0, 0), (0, 96-WIDTH)))(inputs)
layers = []
for n, k, s in [(32, 5, 1),(64, 5, 1),(128, 5, 1),(128, 3, 1),(128, 3, 1)]:
    x=keras.layers.Conv2D(n, kernel_size=k, strides=s, padding='same')(x)
    x=keras.layers.LeakyReLU()(x)
    x=keras.layers.Conv2D(n, kernel_size=k, strides=s, padding='same')(x)
    x=keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    x=keras.layers.LeakyReLU()(x)
    x=keras.layers.BatchNormalization()(x)
    layers.append(x)
layers.pop()
for n, k, s in [(128, 3, 1),(128, 3, 1)]:
    x=keras.layers.Conv2D(n, kernel_size=k, strides=s, padding='same')(x)
    x=keras.layers.LeakyReLU()(x)
    x=keras.layers.BatchNormalization()(x)
for n, k, s in reversed([(N_CLASSES, 5, 2),(64, 5, 2),(64, 5, 2),(128, 5, 2),(128, 5, 2)]):
    x=keras.layers.Conv2DTranspose(n, kernel_size=k, strides=s, padding='same')(x)
    if len(layers)>0:
        l = layers.pop()
        x=keras.layers.concatenate([l, x])
    x=keras.layers.LeakyReLU()(x)
    x=keras.layers.BatchNormalization()(x)
x=keras.layers.Conv2DTranspose(N_CLASSES, kernel_size=5, strides=1, padding='same')(x)
x=keras.layers.Cropping2D(((0, 0), (0, 96-WIDTH)))(x)
outputs = keras.layers.Activation('softmax')(x)
model = keras.models.Model(inputs=inputs, outputs=outputs)
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(0.001),
              metrics=['accuracy'])

datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=10.0,
                                                       width_shift_range=2,
                                                       height_shift_range=2,
                                                       shear_range=0.0,
                                                       zoom_range=0.1,
                                                       data_format='channels_last',
                                                       validation_split=0.1
                                                      )

epochs = 30
batch_size = 50
model.fit_generator(zip(datagen.flow(combined, batch_size=batch_size, subset='training', seed=1), datagen.flow(segmented, batch_size=batch_size, subset='training', seed=1)),
                    epochs=epochs, 
                    steps_per_epoch = len(combined)//batch_size,
                    validation_data=zip(datagen.flow(combined, batch_size=batch_size, subset='validation', seed=1), datagen.flow(segmented, batch_size=batch_size, subset='validation', seed=1)),
                    validation_steps=50,
                    #callbacks=[keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=2, verbose=0, mode='auto')],
                    verbose=2
         )
N_TEST = 10
SEED = np.random.randint(0, 1000)
originals = next(datagen.flow(combined, batch_size=N_TEST, subset='validation', seed=SEED))
ground_truth = next(datagen.flow(segmented, batch_size=N_TEST, subset='validation', seed=SEED))
predicted = model.predict_on_batch(originals)
predicted = np.round(predicted).astype(np.int)
plt.figure(figsize=(20, 5))
np.set_printoptions(threshold=np.nan)
for i in range(N_TEST):
    plt.subplot(4, N_TEST, i+1)
    plt.imshow(originals[i].reshape((HEIGHT, WIDTH)))
    plt.subplot(4, N_TEST, i+1+N_TEST)
    plt.imshow(np.argmax(predicted[i], axis=2), vmax=10, vmin=0)
    plt.subplot(4, N_TEST, i+1+2*N_TEST)
    plt.imshow(np.argmax(ground_truth[i], axis=2), vmax=10, vmin=0)
    plt.subplot(4, N_TEST, i+1+3*N_TEST)
    plt.imshow(np.any(predicted[i]-ground_truth[i], axis=2))