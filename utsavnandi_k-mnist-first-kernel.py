import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib notebook

import seaborn as sns

import tensorflow as tf

tf.random.set_seed(42)

np.random.seed(43)

#from sklearn.model_selection import train_test_split
!nvidia-smi
df_train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

df_val = pd.read_csv('/kaggle/input/Kannada-MNIST/Dig-MNIST.csv')

df_test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')
y = tf.keras.utils.to_categorical(df_train['label'].values)

y_val = tf.keras.utils.to_categorical(df_val['label'].values)

y_not_enc = df_train['label'].values



df_train = df_train.drop(['label'], axis=1)

df_val = df_val.drop(['label'], axis=1)



X = df_train.values.reshape(-1, 28, 28, 1)

X_val = df_val.values.reshape(-1, 28, 28, 1)



#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, stratify=y_not_enc, random_state=42)
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.,

                                   rotation_range=15,

                                   width_shift_range=0.25,

                                   height_shift_range=0.25,

                                   shear_range=0.2,

                                   zoom_range=0.2,

                                   horizontal_flip=False)

datagen.fit(X)

valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)
model = tf.keras.models.Sequential()



model.add(tf.keras.layers.Conv2D(filters = 16, kernel_size = 3, strides = 1, activation = 'selu', kernel_initializer='glorot_normal', input_shape = (28, 28, 1)))

model.add(tf.keras.layers.Conv2D(filters = 16, kernel_size = 3, strides = 1, activation = 'selu', kernel_initializer='glorot_normal', padding='same', input_shape = (28, 28, 1)))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2))

model.add(tf.keras.layers.Dropout(0.2))



model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, strides = 1, activation = 'selu', kernel_initializer='glorot_normal', padding='same'))

model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, strides = 1, activation = 'selu', kernel_initializer='glorot_normal', padding='same'))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2))

#model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.AlphaDropout(0.25))



model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, strides = 1, activation = 'selu', kernel_initializer='glorot_normal', padding='same'))

model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, strides = 1, activation = 'selu', kernel_initializer='glorot_normal', padding='same'))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2))

model.add(tf.keras.layers.AlphaDropout(0.25))

#model.add(tf.keras.layers.Dropout(0.2))



model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(units = 1024, activation = 'relu'))

model.add(tf.keras.layers.Dense(units = 10, activation = 'softmax'))



model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0), 

              loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.summary()
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

history = model.fit_generator(datagen.flow(X, y, batch_size=256), 

                              steps_per_epoch=256, epochs=50, 

                              validation_data=valid_datagen.flow(X_val, y_val),

                              validation_steps=50,

                              callbacks=[reduce_lr]).history
plt.plot(history['accuracy'])

plt.plot(history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train'], loc='upper left') # 'val'

plt.show()
sub = pd.DataFrame(df_test['id'], columns=['id'])

df_test = df_test.drop(columns=['id'])

X_val = df_test.values.reshape(-1, 28, 28, 1) / 255

sub['label'] = np.argmax(model.predict(tf.cast(X_val, tf.float32)), axis=1)
sub.to_csv('/kaggle/working/submission.csv', index=False)