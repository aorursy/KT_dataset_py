# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
fetal_health = pd.read_csv('/kaggle/input/fetal-health-classification/fetal_health.csv')
fetal_health.head()
fetal_health['fetal_health'].value_counts()
import seaborn as sns

from sklearn import model_selection

import matplotlib.pyplot as plt



import tensorflow as tf
print(tf.__version__)
fetal_health.describe().T
fetal_health.info()
plt.figure(figsize=(10,10))

sns.heatmap(fetal_health.corr())
X = fetal_health.drop(['fetal_health'], axis=1).values

y = fetal_health['fetal_health'].values
seed = 42



X_train, X_val, y_train, y_val = model_selection.train_test_split(X, y, test_size=0.3, random_state=seed)

X_val, X_test, y_val, y_test = model_selection.train_test_split(X_val, y_val, test_size=0.5, random_state=seed)
print(X_train.shape)

print(X_val.shape)

print(X_test.shape)
rows, feat = X_train.shape



model = tf.keras.models.Sequential([

    tf.keras.layers.Input(shape=(feat,),),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(2048, activation='relu', kernel_initializer='normal'),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(1024, activation='relu', kernel_initializer='normal'),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(512, activation='relu', kernel_initializer='normal'),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(256, activation='relu', kernel_initializer='normal'),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(128, activation='relu', kernel_initializer='normal'),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(64, activation='relu', kernel_initializer='normal'),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(4, activation='softmax')

])
print(model.summary)
earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)



adam = tf.keras.optimizers.Adam(lr=0.0001)



model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
hist = model.fit(X_train, y_train,

                validation_data = (X_val, y_val),

                epochs=600,

                batch_size=32,

                callbacks=[earlystop],

                verbose=1)
plt.plot(hist.history['loss'], label='training loss')

plt.plot(hist.history['val_loss'], label='validation loss')

plt.legend()
plt.plot(hist.history['accuracy'], label='training accuracy')

plt.plot(hist.history['val_accuracy'], label='validation accuracy')

plt.legend()
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score



categorical_pred = np.argmax(model.predict(X_test), axis=1)



print('Results for Categorical Mode')

print(accuracy_score(y_test, categorical_pred))

print(classification_report(y_test, categorical_pred))
print(confusion_matrix(y_test, categorical_pred))
predictions = model.predict(X_test)



parr = []



for pred in predictions:

    parr.append(np.argmax(pred))

parr = np.array(parr).reshape(-1)

print("F1 score {}".format(f1_score(y_test, parr, average='micro')))