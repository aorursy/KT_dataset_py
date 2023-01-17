# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from keras.callbacks import EarlyStopping


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
train = train.reindex(np.random.permutation(train.index))
label = train.label.values
train.drop(['label'], axis = 1, inplace = True)
X_train = train/255
X_train = X_train.values.reshape(len(X_train),28,28,1)
initializer = tf.keras.initializers.TruncatedNormal()
model = keras.Sequential([
      tf.keras.layers.Conv2D(42, 3, activation='relu', padding='same', kernel_initializer=initializer, bias_initializer=initializers.Zeros()),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Conv2D(42, 3, activation='relu', padding='same'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Conv2D(64, 4, activation='relu', padding='same'),
      tf.keras.layers.Conv2D(64, 4, activation='relu', padding='same'),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10),
])


early_stopping_monitor = EarlyStopping(patience=4, restore_best_weights=True)
model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

history = model.fit(X_train, label, validation_split = 0.2, batch_size = 30, epochs = 40, callbacks=[early_stopping_monitor])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()
    
dot_img_file = '/kaggle/working/model_1.png'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True, show_layer_names=True)

model.summary()
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
test = test/255
test = test.values.reshape(len(test),28,28,1)
test.shape


probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test)
predictions[0]
ids = []
for i, z in enumerate(predictions):
    ids.append(np.argmax(z))
ids = pd.DataFrame(ids, columns=['Label'])
ids['ImageId'] = ids.index+1
ids = ids[['ImageId','Label']]
ids
ids.to_csv('/kaggle/working/predict.csv', index = False)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
